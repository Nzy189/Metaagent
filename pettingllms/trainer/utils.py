
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
import uuid
import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer
from pettingllms.misc import colorful_print

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import async_server_class
from pettingllms.utils.logger_config import get_multi_logger
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RequestState:
    """跟踪单个请求的状态"""
    request_id: str
    server: ray.actor.ActorHandle
    created_time: datetime
    response_received_time: Optional[datetime] = None
    timeout_seconds: float = 10.0  # 响应后10秒超时
    is_timeout_cleanup_scheduled: bool = False
    is_cleaned_up: bool = False


def initialize_llm_servers(
    worker_group,
    server_class,
    server_config,
    *,
    reuse_existing: bool = False,
    lifetime_detached: bool = False,
    actor_name: str = "async_llm_server",
    write_registry_path: Optional[str] = None,
    strict_reuse: bool = True,
):
    print(f"DEBUG: Starting initialize_llm_servers, worker_group={worker_group}")
    
    if worker_group is None:
        world_size=1
        name_prefix="actor_rollout"
    else:
        world_size=worker_group.world_size
        name_prefix=worker_group.name_prefix
    
    print(f"DEBUG: world_size={world_size}, name_prefix={name_prefix}")
    
    rollout_tp_size = server_config.actor_rollout_ref.rollout.tensor_model_parallel_size
    rollout_dp_size = world_size // rollout_tp_size
    
    # 当 world_size 小于 tp_size 时，确保至少启动 1 个 server
    if rollout_dp_size < 1:
        print(
            f"DEBUG: rollout_dp_size computed as 0 (world_size={world_size}, tp_size={rollout_tp_size}), fallback to 1"
        )
        rollout_dp_size = 1
    
    print(f"DEBUG: rollout_tp_size={rollout_tp_size}, rollout_dp_size={rollout_dp_size}")

    async_llm_servers = [None] * rollout_dp_size
    server_addresses = [None] * rollout_dp_size
    created_new_server = False

    # Start all server instances, restart if address already in use.
    unready_dp_ranks = set(range(rollout_dp_size))
    print(f"DEBUG: unready_dp_ranks={unready_dp_ranks}")
    
    while len(unready_dp_ranks) > 0:
        print(f"DEBUG: Processing unready_dp_ranks: {unready_dp_ranks}")
        
        if worker_group is None:
            print(f"DEBUG: Creating server for worker_group=None case")
            servers = {}
            if reuse_existing:
                try:
                    existing = ray.get_actor(actor_name)
                    servers = {0: existing}
                    print(f"DEBUG: Reused existing server actor '{actor_name}': {existing}")
                except Exception:
                    if strict_reuse:
                        raise RuntimeError(
                            f"Actor '{actor_name}' not found while reuse_existing=True. "
                            f"Please launch it first via the launcher script."
                        )
                    servers = {}

            if not servers:
                options_kwargs = dict(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=__import__("ray._raylet")._raylet.NodeID.from_hex(ray.nodes()[0]["NodeID"]),
                        soft=False,
                    ),
                    name=actor_name,
                )

                # 将关键环境变量传入 actor 进程，避免行为不一致
                runtime_env = {"env": {}}
                for key in [
                    "VLLM_GPU_MEMORY_UTILIZATION",
                    "VLLM_USE_V1",
                    "VLLM_WORKER_MULTIPROC_METHOD",
                    "CUDA_LAUNCH_BLOCKING",
                ]:
                    if key in os.environ:
                        runtime_env["env"][key] = os.environ[key]
                if len(runtime_env["env"]) > 0:
                    options_kwargs["runtime_env"] = runtime_env

                # 不要为 Actor 预占 GPU（vLLM 引擎会通过 placement group 按 tp_size 申请 GPU）
                if torch.cuda.is_available():
                    options_kwargs["num_gpus"] = 0
                    print(f"DEBUG: Do not reserve GPU for actor; leave GPUs to vLLM placement group")

                if lifetime_detached:
                    options_kwargs["lifetime"] = "detached"

                print(f"DEBUG: Creating server with options: {options_kwargs}")
                server = server_class.options(**options_kwargs).remote(server_config, 1, 0, "actor_rollout")
                servers = {0: server}
                created_new_server = True
                print(f"DEBUG: Created server: {server}")
        else:
            print(f"DEBUG: Creating servers for worker_group case")
            register_center = ray.get_actor(f"{name_prefix}_register_center")
            workers_info = ray.get(register_center.get_worker_info.remote())
            assert len(workers_info) == world_size

            servers = {}

            # Step 1: try to reuse existing named actors when requested
            if reuse_existing:
                print("DEBUG: Trying to reuse existing async_llm_server_{rank} actors")
                ranks_to_check = list(unready_dp_ranks)
                for rollout_dp_rank in ranks_to_check:
                    actor_name_rank = f"async_llm_server_{rollout_dp_rank}"
                    try:
                        existing = ray.get_actor(actor_name_rank)
                        servers[rollout_dp_rank] = existing
                        print(f"DEBUG: Reused existing server actor '{actor_name_rank}': {existing}")
                    except Exception:
                        if strict_reuse:
                            raise RuntimeError(
                                f"Actor '{actor_name_rank}' not found while reuse_existing=True. "
                                f"Please launch it first via the launcher script or disable strict_reuse."
                            )
                        # not found, will create below

            # Step 2: create the rest that were not found
            ranks_to_create = [r for r in unready_dp_ranks if r not in servers]
            if ranks_to_create:
                for rollout_dp_rank in ranks_to_create:
                    actor_name_rank = f"async_llm_server_{rollout_dp_rank}"
                    try:
                        server_handle = server_class.options(
                            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                                node_id=workers_info[rollout_dp_rank * rollout_tp_size],
                                soft=False,
                            ),
                            name=actor_name_rank,
                        ).remote(server_config, rollout_dp_size, rollout_dp_rank, name_prefix)
                        servers[rollout_dp_rank] = server_handle
                        created_new_server = True
                        print(f"DEBUG: Created server '{actor_name_rank}': {server_handle}")
                    except ValueError as ve:
                        # Name already taken; attempt to reuse instead of failing
                        try:
                            existing = ray.get_actor(actor_name_rank)
                            servers[rollout_dp_rank] = existing
                            print(f"DEBUG: Name collision for '{actor_name_rank}', reused existing actor: {existing}")
                        except Exception:
                            print(f"DEBUG: Failed to create or reuse actor '{actor_name_rank}': {ve}")
                            raise

        for rollout_dp_rank, server in servers.items():
            print(f"DEBUG: Processing server for rank {rollout_dp_rank}")
            try:
                print(f"DEBUG: Getting server address for rank {rollout_dp_rank}")
                address = ray.get(server.get_server_address.remote())
                print(f"DEBUG: Got address {address} for rank {rollout_dp_rank}")
                server_addresses[rollout_dp_rank] = address
                async_llm_servers[rollout_dp_rank] = server
                unready_dp_ranks.remove(rollout_dp_rank)
                print(f"DEBUG: Successfully initialized server for rank {rollout_dp_rank}")
            except Exception as e:
                print(f"Failed to get server address for rank {rollout_dp_rank}: {e}")
                print(f"DEBUG: Exception details: {type(e).__name__}: {str(e)}")
                # 清理失败的 server
                try:
                    ray.kill(server)
                except:
                    pass
                # 重新抛出异常，让外层重试逻辑处理
                raise e
        
    print(f"DEBUG: All servers initialized, starting engine init")
    # 只有在所有服务器都就绪后才初始化引擎
    valid_servers = [server for server in async_llm_servers if server is not None]
    print(f"DEBUG: Found {len(valid_servers)} valid servers")
    
    if valid_servers and created_new_server:
        print(f"DEBUG: Initializing engines for {len(valid_servers)} servers")
        ray.get([server.init_engine.remote() for server in valid_servers])
        print(f"DEBUG: Engine initialization completed")
    
    # 返回有效的服务器列表而不是包含 None 的列表
    valid_addresses = [addr for addr in server_addresses if addr is not None]
    
    # 写出注册信息（可选）
    if write_registry_path is not None:
        try:
            import json
            with open(write_registry_path, "w") as f:
                json.dump({
                    "actor_names": [actor_name for _ in valid_servers],
                    "addresses": valid_addresses,
                }, f)
            print(f"DEBUG: Wrote server registry to {write_registry_path}")
        except Exception as e:
            print(f"DEBUG: Failed to write server registry: {e}")

    print(f"DEBUG: Returning {len(valid_servers)} servers and {len(valid_addresses)} addresses")
    return valid_servers, valid_addresses

        # All server instances are ready, init AsyncLLM engine.
        

    

class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    - Request timeout management: automatically cleanup requests that are not retrieved within timeout
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000, request_timeout_seconds: float = 10.0):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
            request_timeout_seconds (float, optional): seconds to wait after response before cleanup. Defaults to 10.0.
        """
        self.config = config
        self.request_timeout_seconds = request_timeout_seconds
        
        # 过滤掉 None 的服务器句柄
        valid_server_handles = [server for server in server_handles if server is not None]
        
        if not valid_server_handles:
            raise ValueError("No valid server handles provided. Please check server initialization.")
            
        self.server_handles = valid_server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in self.server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
        
        # 请求状态跟踪
        self.active_requests: Dict[str, RequestState] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_running = False
        self._cleanup_task_started = False
        
        # 初始化多日志系统
        self.multi_logger = get_multi_logger()

        # 请求计数与并发保护
        self._sent_count = 0
        self._completed_count = 0
        self._counter_lock = asyncio.Lock()
        # 完成统计（仅成功产生结果的请求参与平均）
        self._sum_prompt_len = 0
        self._sum_answer_len = 0
        self._completed_with_result_count = 0

    def _start_cleanup_task(self):
        """启动后台清理任务"""
        if self._cleanup_task_started or self._cleanup_running:
            return
            
        try:
            # 检查是否有运行的事件循环
            loop = asyncio.get_running_loop()
            self._cleanup_running = True
            self._cleanup_task_started = True
            # 创建清理任务但不等待它
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            # 没有运行的事件循环，稍后在generate方法中启动
            pass

    async def _periodic_cleanup(self):
        """定期清理过期的请求"""
        while self._cleanup_running:
            try:
                current_time = datetime.now()
                expired_requests = []
                
                for request_id, req_state in self.active_requests.items():
                    # 如果响应已接收且超过超时时间，则需要清理
                    if (req_state.response_received_time is not None and 
                        not req_state.is_cleaned_up and
                        current_time - req_state.response_received_time > timedelta(seconds=req_state.timeout_seconds)):
                        expired_requests.append(request_id)
                
                # 清理过期请求
                for request_id in expired_requests:
                    await self._cleanup_request(request_id)
                
                # 每5秒检查一次
                await asyncio.sleep(5)
                
            except Exception as e:
                self.multi_logger.log_error(
                    env_idx=-1,
                    rollout_idx=-1,
                    error_source="request_cleanup",
                    error=e,
                    context_data={"error_message": str(e)},
                    additional_info={"cleanup_task": "periodic_cleanup_failed"}
                )
                await asyncio.sleep(5)  # 发生错误时也等待一段时间

    async def _cleanup_request(self, request_id: str):
        """清理单个过期请求"""
        if request_id not in self.active_requests:
            return
            
        req_state = self.active_requests[request_id]
        if req_state.is_cleaned_up:
            return
            
        try:
            # 尝试中止服务器上的请求
            if hasattr(req_state.server, 'abort_request'):
                await req_state.server.abort_request.remote(request_id)
            
            req_state.is_cleaned_up = True
            
            self.multi_logger.log_model_interaction(
                env_idx=-1,
                rollout_idx=-1,
                policy_name="system",
                prompt="",
                response="",
                extra_data={
                    "event": "request_timeout_cleanup",
                    "request_id": request_id,
                    "created_time": req_state.created_time.isoformat(),
                    "response_received_time": req_state.response_received_time.isoformat() if req_state.response_received_time else None,
                    "timeout_seconds": req_state.timeout_seconds
                }
            )
            
            # 从活动请求中移除
            del self.active_requests[request_id]
            
        except Exception as e:
            self.multi_logger.log_error(
                env_idx=-1,
                rollout_idx=-1,
                error_source="request_cleanup",
                error=e,
                context_data={"request_id": request_id, "error_message": str(e)},
                additional_info={"cleanup_action": "cleanup_request_failed"}
            )

    def stop_cleanup_task(self):
        """停止清理任务"""
        self._cleanup_running = False
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        # 检查是否有可用的服务器
        if not self.weighted_serveres:
            raise RuntimeError("No available servers. Please check server initialization.")
        
        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        rollout_idx: int,
        turn_idx: int,
        agent_idx: int,
        dpr_prompt:DataProto,
        sampling_params: Optional[dict[str, Any]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        image_data: Optional[list[Any]] = None,
        application_id: Optional[str] = None,
        env_idx: Optional[int] = None,
        #rollout_idx: Optional[int] = None,
        policy_name: Optional[str] = None,
        timeout: Optional[float] = 60.0
    ) -> DataProto:
        """Generate tokens from prompt ids or DataProto.

        Args:
            dpr_prompt: llm_inputs.batch = TensorDict({
            "input_ids":
            "attention_mask":   
            "position_ids": 
            "responses":
            prompt_ids (List[int], optional): List of prompt token ids (for legacy usage).
            sampling_params (Dict[str, Any], optional): Sampling parameters (for legacy usage).
            application_id (str, optional): Application ID for new usage.

        Returns:
            DataProto: DataProto format output consistent with Router's generate_sequences.
        """
        
        # 确保清理任务已启动（在事件循环中）
        self._start_cleanup_task()

        if application_id is None:
            application_id = str(uuid.uuid4())
        else:
            application_id = str(application_id)

        unique_request_id = f"{application_id}_{uuid.uuid4().hex[:8]}"
        
        # 选择服务器并创建请求状态跟踪
        server = self._choose_server(unique_request_id)
        request_state = RequestState(
            request_id=unique_request_id,
            server=server,
            created_time=datetime.now(),
            timeout_seconds=self.request_timeout_seconds
        )
        self.active_requests[unique_request_id] = request_state
 
        self.multi_logger.log_model_interaction(
            env_idx=env_idx if rollout_idx is not None else -1,  # 添加缺少的env_idx参数
            rollout_idx=rollout_idx if rollout_idx is not None else -1,
            policy_name=policy_name if policy_name is not None else "unknown",
            prompt="",  
            response="",  
            extra_data={
                "event": "generate_start",
                "dpr_prompt_shape": str(dpr_prompt.batch['input_ids'].shape) if hasattr(dpr_prompt, 'batch') else None,
                "sampling_params": sampling_params,
                "application_id": application_id
            }
        )
        
        # Ensure sampling_params is a dictionary (vLLM requires mapping, not None)
        if sampling_params is None:
            sampling_params = {}
        
        # Extract prompt_ids from DataProto and convert to list
        prompt_ids = dpr_prompt.batch['input_ids'][0].tolist() 
        
        # Remove padding tokens
        original_length = len(prompt_ids)
        while prompt_ids and prompt_ids[0] == 151643:
            prompt_ids.pop(0)
            
        self.multi_logger.log_model_interaction(
            env_idx=env_idx if rollout_idx is not None else -1,  # 添加缺少的env_idx参数
            rollout_idx=rollout_idx if rollout_idx is not None else -1,
            policy_name=policy_name if policy_name is not None else "unknown",
            prompt="",
            response="",
            extra_data={
                "event": "prompt_preprocessing",
                "original_length": original_length,
                "final_length": len(prompt_ids),
                "first_10_tokens": prompt_ids[:10] if prompt_ids else [],
                "server_selected": str(server)
            }
        )
        
        # Ensure we have valid tokens
        if not prompt_ids:
            error_msg = "No valid tokens found after removing padding"
            self.multi_logger.log_model_interaction(
                env_idx=env_idx if rollout_idx is not None else -1,  # 添加缺少的env_idx参数
                rollout_idx=rollout_idx if rollout_idx is not None else -1,
                policy_name=policy_name if policy_name is not None else "unknown",
                prompt="",
                response="",
                extra_data={
                    "event": "generate_error",
                    "error": error_msg
                }
            )
            raise ValueError(error_msg) 
        
        # Use direct await on Ray remote call - this is the correct async pattern!
        import asyncio
        
        sent_marked = False
        try:

            self.multi_logger.log_model_interaction(
                env_idx=env_idx if rollout_idx is not None else -1,  # 添加缺少的env_idx参数
                rollout_idx=rollout_idx if rollout_idx is not None else -1,
                policy_name=policy_name if policy_name is not None else "unknown",
                prompt=str(prompt_ids[:20]) + "..." if len(prompt_ids) > 20 else str(prompt_ids),
                response="",
                extra_data={
                    "event": "server_generate_start",
                    "prompt_ids_length": len(prompt_ids),
                    "timeout": timeout
                }
            )
            
            # 发送前更新“已发送”计数
            async with self._counter_lock:
                self._sent_count += 1
                sent_marked = True
                if self._sent_count % 20 == 0:
                    print(f"[AsyncLLMServerManager] 已发送请求数达到 {self._sent_count}")

            # Directly await the Ray remote call with timeout
            #output = await asyncio.wait_for(
             #   server.generate.remote(
              #      prompt_ids=prompt_ids,
              #      sampling_params=sampling_params,
              #      request_id=unique_request_id,  # Use unique request ID
               # ),
                #timeout=timeout
            #)
            output = await server.generate.remote(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=unique_request_id,  # Use unique request ID
            )
            
            # 标记请求已收到响应，开始倒计时清理
            if unique_request_id in self.active_requests:
                self.active_requests[unique_request_id].response_received_time = datetime.now()

        except asyncio.TimeoutError:
            # 清理失败的请求
            if unique_request_id in self.active_requests:
                del self.active_requests[unique_request_id]
                
            error_msg = f"Generate request timed out after 20 seconds for request {unique_request_id} (app_id: {application_id})"
            self.multi_logger.log_model_interaction(
                env_idx=env_idx if rollout_idx is not None else -1,  # 添加缺少的env_idx参数
                rollout_idx=rollout_idx if rollout_idx is not None else -1,
                policy_name=policy_name if policy_name is not None else "unknown",
                prompt="",
                response="",
                extra_data={
                    "event": "generate_timeout",
                    "timeout": timeout,
                    "application_id": application_id
                }
            )
            raise TimeoutError(error_msg)
        except Exception as e:
            # 清理失败的请求
            if unique_request_id in self.active_requests:
                del self.active_requests[unique_request_id]
                
           
            raise
        finally:
            # 成功或失败后，都统计“已完成”
            if sent_marked:
                async with self._counter_lock:
                    self._completed_count += 1
                    if self._completed_count % 20 == 0:
                        if self._completed_with_result_count > 0:
                            avg_prompt = self._sum_prompt_len / self._completed_with_result_count
                            avg_answer = self._sum_answer_len / self._completed_with_result_count
                            print(
                                f"[AsyncLLMServerManager] 已完成请求数达到 {self._completed_count} | 当前平均 prompt 长度={avg_prompt:.2f}, answer 长度={avg_answer:.2f} (基于 {self._completed_with_result_count} 个成功请求)"
                            )
                        else:
                            print(
                                f"[AsyncLLMServerManager] 已完成请求数达到 {self._completed_count} | 当前平均 prompt/answer 长度：暂无成功请求"
                            )
        
    
        response_str = tokenizer.decode(output, skip_special_tokens=True)
        
      

        # 统计本次成功请求的 prompt/answer 长度（计入平均）
        prompt_len_for_stats = len(prompt_ids)
        # Transform vLLM output to DataProto
        # Response ids from vLLM (output is list[int])
        if not isinstance(output, list):
            raise TypeError(
                f"Unexpected output type from server.generate: {type(output)}; expected list[int]"
            )
        response_ids_generated = output
        answer_len_for_stats = len(response_ids_generated)

        # 累加到成功统计
        async with self._counter_lock:
            self._sum_prompt_len += prompt_len_for_stats
            self._sum_answer_len += answer_len_for_stats
            self._completed_with_result_count += 1

        # Read lengths from config with sensible fallbacks
        rollout_cfg = getattr(self.config, "actor_rollout_ref", None)
        rollout_cfg = getattr(rollout_cfg, "rollout", None)
        prompt_max_len = int(getattr(rollout_cfg, "prompt_length", len(prompt_ids)))
        response_max_len = int(getattr(rollout_cfg, "response_length", len(response_ids_generated)))

        # Truncate to fit
        prompt_ids_tail = prompt_ids[-prompt_max_len:]
        response_ids_tail = response_ids_generated[:response_max_len]

        # Build tensors: prompts left-pad, responses right-pad
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 1
        pad_token_id = 0

        # prompts
        prompts_tensor = torch.full((batch_size, prompt_max_len), pad_token_id, dtype=torch.long, device=device)
        if len(prompt_ids_tail) > 0:
            prompts_tensor[0, -len(prompt_ids_tail) :] = torch.tensor(
                prompt_ids_tail, dtype=torch.long, device=device
            )
        prompt_attention_mask = torch.zeros((batch_size, prompt_max_len), dtype=torch.long, device=device)
        if len(prompt_ids_tail) > 0:
            prompt_attention_mask[0, -len(prompt_ids_tail) :] = 1

        # responses
        responses_tensor = torch.full((batch_size, response_max_len), pad_token_id, dtype=torch.long, device=device)
        if len(response_ids_tail) > 0:
            responses_tensor[0, : len(response_ids_tail)] = torch.tensor(
                response_ids_tail, dtype=torch.long, device=device
            )
        response_attention_mask = torch.zeros((batch_size, response_max_len), dtype=torch.long, device=device)
        if len(response_ids_tail) > 0:
            response_attention_mask[0, : len(response_ids_tail)] = 1

        # merge
        input_ids_tensor = torch.cat([prompts_tensor, responses_tensor], dim=1)
        attention_mask_tensor = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids_full = compute_position_id_with_mask(attention_mask_tensor)

        batch_dict = {
            "prompts": prompts_tensor,
            "responses": responses_tensor,
            "response_mask": response_attention_mask,
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "position_ids": position_ids_full,
        }
        non_tensors = {
            "rollout_idx": np.array([rollout_idx] * input_ids_tensor.shape[0], dtype=object),
            "turn_idx": np.array([turn_idx] * input_ids_tensor.shape[0], dtype=object),
            "agent_idx": np.array([agent_idx] * input_ids_tensor.shape[0], dtype=object),
        }
        output_dpr = DataProto.from_dict(tensors=batch_dict, non_tensors=non_tensors)
        
        return output_dpr, response_str
    
    def manually_cleanup_request(self, request_id: str):
      
        if request_id in self.active_requests:
            self.active_requests[request_id].is_cleaned_up = True
            del self.active_requests[request_id]
            
            self.multi_logger.log_model_interaction(
                env_idx=-1,
                rollout_idx=-1,
                policy_name="system",
                prompt="",
                response="",
                extra_data={
                    "event": "manual_request_cleanup",
                    "request_id": request_id
                }
            )
    
    def get_active_requests_count(self) -> int:
        """获取当前活跃请求的数量"""
        return len(self.active_requests)
    
    def get_pending_cleanup_count(self) -> int:
        """获取等待清理的请求数量"""
        current_time = datetime.now()
        pending_count = 0
        
        for req_state in self.active_requests.values():
            if (req_state.response_received_time is not None and 
                not req_state.is_cleaned_up and
                current_time - req_state.response_received_time > timedelta(seconds=req_state.timeout_seconds)):
                pending_count += 1
                
        return pending_count


def convert_prompt_to_dpr(tokenizer, processor, prompts, max_prompt_length, multi_modal=False, **kwargs):
    """
    Convert prompt dict to veRL's DataProto.
    
    Args:
        tokenizer: HF tokenizer, must support apply_chat_template and __call__ tokenization
        chat_parser: Reserved (currently unused)
        prompts: dict, {"text": str, "image": None or image path}
        max_prompt_length: Maximum prompt length (left padding)
        multi_modal: Whether multimodal (if True, should also pass processor and other necessary information)
        kwargs: Optional parameters, such as processor, meta_info, etc.
    Returns:
        DataProto: Contains tensor and non-tensor information
    """
    from verl.protocol import DataProto, union_two_dict
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import pad_sequence_to_length
    import numpy as np
    import torch

    if not isinstance(prompts, dict) or "text" not in prompts:
        raise ValueError("prompts must be a dictionary containing 'text' key: {'text': str, 'image': Optional[path]} ")

    text = prompts.get("text", "") or ""
    image_path = prompts.get("image", None)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        chat = np.array([
            {"content": text, "role": "user"}
        ])

        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False
        )
        

        inputs = tokenizer(
            prompt_with_chat_template,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # Multimodal (optional): depends on externally provided processor
        multi_modal_inputs = None
        if multi_modal and image_path is not None and "processor" in kwargs:
            
            image_inputs = processor.image_processor([image_path], return_tensors="pt")
            multi_modal_inputs = {k: v for k, v in image_inputs.items()}
           

        # Pad to a unified length
        input_ids = pad_sequence_to_length(
            input_ids,
            max_seq_len=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
        )
        attention_mask = pad_sequence_to_length(
            attention_mask,
            max_seq_len=max_prompt_length,
            pad_token_id=0,
            left_pad=True,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        data.non_tensor_batch["formatted_prompts"] = np.array([prompt_with_chat_template])
        if multi_modal_inputs is not None:
            data.non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

        # Merge meta_info if provided
        meta_info = kwargs.get("meta_info")
        if meta_info:
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data
    finally:
        tokenizer.padding_side = old_padding_side


def convert_dpr_to_response(tokenizer, chat_parser, dpr, max_prompt_length, multi_modal=False, **kwargs):
    try:
        attn = dpr.batch["attention_mask"][0, max_prompt_length :]
        tokens = dpr.batch["responses"][0]

        # Find last index where attention == 1
        non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            trimmed = tokens[:0]  # empty
        else:
            last_valid_idx = non_pad_indices[-1].item()
            trimmed = tokens[: last_valid_idx + 1]  # include the last valid token

        response = tokenizer.decode(trimmed, skip_special_tokens=False)

        pad_token = tokenizer.pad_token if tokenizer.pad_token else ""
        eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
        response = response.replace(pad_token, "").replace(eos_token, "")
        
        # Ensure we always return a string
        return response if response is not None else ""
    except Exception as e:
        print(f"Error in convert_dpr_to_response: {e}")
        return ""

from math import prod
from typing import List, Dict, Iterator, Tuple, Optional

def compute_strides(sizes: List[int]) -> List[int]:
    """Row-major（最后一维变化最快）的stride。"""
    n = len(sizes)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return strides

def tuple_to_index(t: List[int], sizes: List[int]) -> int:
    """(t0,...,tN-1) -> rollout 索引。"""
    strides = compute_strides(sizes)
    return sum(t[i] * strides[i] for i in range(len(sizes)))

def index_to_tuple(idx: int, sizes: List[int]) -> List[int]:
    """rollout 索引 -> (t0,...,tN-1)。"""
    t = []
    strides = compute_strides(sizes)
    for i, s in enumerate(strides):
        q, idx = divmod(idx, s)
        t.append(q)
    return t

def agent_sample_rollouts(
    agent_idx: int, sample_idx: int, sizes: List[int]
) :
    strides = compute_strides(sizes)
    stride = strides[agent_idx]         # block_len
    period = sizes[agent_idx] * stride
    total = prod(sizes)
    result=[]

    for base in range(0, total, period):
        start = base + sample_idx * stride
       
        for k in range(stride):
            result.append(start + k)
    return result


def build_reverse_mapping(
    agent_names: List[str],
    sizes: List[int],
    batch_size: int,
) -> Dict[str, List[Dict[str, object]]]:
    total = prod(sizes)
    strides = compute_strides(sizes)
    out: Dict[str, List[Dict[str, object]]] = {}
    total=1
    for _ in sizes:
        total*=_

    for batch_idx in range(batch_size):
        out[batch_idx] = {}
        for i, name in enumerate(agent_names):
            stride = strides[i]
            period = sizes[i] * stride
            items = []
            for s in range(sizes[i]):
                starts = list(range(s * stride, total, period))
                
                rollouts = []
                for st in starts:
                    rollouts.extend(range(st+batch_idx*total, st+batch_idx*total + stride))
                entry= rollouts
                items.append(entry)
            out[batch_idx][name] = items
    return out

# ---- 示例用法 ----
if __name__ == "__main__":
    agent_names = ["agent1", "agent2", "agent3"]
    sizes = [2, 3, 2]  # 分别采样 a1=2, a2=3, a3=2 -> total=12

    print("总 rollouts:", prod(sizes))

    rm_full = build_reverse_mapping(agent_names, sizes, batch_size=32)
    print("agent3, sample=0 的具体索引：", rm_full[1]["agent3"][0])
