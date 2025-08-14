
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


from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import async_server_class


def initialize_llm_servers(worker_group,server_class,server_config):

    rollout_tp_size = server_config.actor_rollout_ref.rollout.tensor_model_parallel_size
    rollout_dp_size = worker_group.world_size // rollout_tp_size

    register_center = ray.get_actor(f"{worker_group.name_prefix}_register_center")
    workers_info = ray.get(register_center.get_worker_info.remote())
    assert len(workers_info) == worker_group.world_size

    async_llm_servers = [None] * rollout_dp_size
    server_addresses = [None] * rollout_dp_size

    if server_config.actor_rollout_ref.rollout.agent.custom_async_server:
        server_class = server_class(
            rollout_backend=server_config.actor_rollout_ref.rollout.name,
            rollout_backend_module=server_config.actor_rollout_ref.rollout.agent.custom_async_server.path,
            rollout_backend_class=server_config.actor_rollout_ref.rollout.agent.custom_async_server.name,
        )
    else:
        server_class = server_class(rollout_backend=server_config.actor_rollout_ref.rollout.name)

    # Start all server instances, restart if address already in use.
    unready_dp_ranks = set(range(rollout_dp_size))
    while len(unready_dp_ranks) > 0:
        servers = {
            rollout_dp_rank: server_class.options(
                # make sure AsyncvLLMServer colocates with its corresponding workers
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=workers_info[rollout_dp_rank * rollout_tp_size],
                    soft=False,
                ),
                name=f"async_llm_server_{rollout_dp_rank}",
            ).remote(server_config, rollout_dp_size, rollout_dp_rank, worker_group.name_prefix)
            for rollout_dp_rank in unready_dp_ranks
        }
    

        for rollout_dp_rank, server in servers.items():
            try:
                address = ray.get(server.get_server_address.remote())
                server_addresses[rollout_dp_rank] = address
                async_llm_servers[rollout_dp_rank] = server
                unready_dp_ranks.remove(rollout_dp_rank)
            except Exception:
                ray.kill(server)
                print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")
    
    return async_llm_servers, server_addresses

        # All server instances are ready, init AsyncLLM engine.
        

    

class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        dpr_prompt:DataProto,
        sampling_params: Optional[dict[str, Any]] = None,
        image_data: Optional[list[Any]] = None,
        application_id: Optional[str] = None,
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
            DataProto: 与 Router 的 generate_sequences 一致的 DataProto 格式输出。
        """
        print(f"=== DEBUG: AsyncLLMServerManager.generate 调用 ===")
      
        print(f"dpr_prompt: {dpr_prompt.batch['input_ids']}")
        print(f"sampling_params: {sampling_params}")
        print(f"application_id: {application_id}")
 
        application_id=uuid.uuid4()
        
       
     
        
        server = self._choose_server(application_id)
        print(f"选择的服务器: {server}")
        
        output = await server.generate.remote(
            request_id=application_id,
            prompt_ids=dpr_prompt.batch['input_ids'],
            sampling_params=sampling_params,
        )
        print(f"output: {output}")
        """The output  of server.engine.generate

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.

    transform to DataProto

    """
            

def convert_prompt_to_dpr(tokenizer, chat_parser, processor, prompts, max_prompt_length, multi_modal=False, **kwargs):
    """
    将 prompt dict 转换为 veRL 的 DataProto。
    
    Args:
        tokenizer: HF tokenizer，需支持 apply_chat_template 与 __call__ 分词
        chat_parser: 预留（当前未使用）
        prompts: dict，{"text": str, "image": None 或 图片路径}
        max_prompt_length: 最长 prompt 长度（左侧 padding）
        multi_modal: 是否多模态（若 True，应同时传入 processor 等必要信息）
        kwargs: 可选参数，如 processor、meta_info 等
    Returns:
        DataProto: 包含张量与非张量信息
    """
    from verl.protocol import DataProto, union_two_dict
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import pad_sequence_to_length
    import numpy as np
    import torch

    if not isinstance(prompts, dict) or "text" not in prompts:
        raise ValueError("prompts 必须是包含 'text' 键的字典: {'text': str, 'image': Optional[path]} ")

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

    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token
    response = response.replace(pad_token, "").replace(eos_token, "")

