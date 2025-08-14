import asyncio
import logging
from copy import deepcopy

import aiohttp
import numpy as np
import torch
from openai.types.chat import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout import vLLMRollout,vLLMAsyncRollout
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
from verl.utils import hf_tokenizer
from verl.experimental.agent_loop import AgentLoopManager
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
import ray
from verl.single_controller.ray.base import (
    RayWorkerGroup,
    RayResourcePool,
    RayClassWithInitArgs,
    create_colocated_worker_cls,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

logger = logging.getLogger(__name__)


def _repeat_interleave(value: torch.Tensor | np.ndarray, repeats: int) -> torch.Tensor | np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)


async def poll_completions_openai(address: str, **completions_request) -> ChatCompletion | dict:
    # OpenAI-compatible chat completion endpoint
    # Build multiple address candidates to be robust to IPv4/IPv6 binding nuances
    headers = {
        "Content-Type": "application/json",
    }

    # Remove non-OpenAI fields if present
    completions_request.pop("meta_info", None)
    completions_request.pop("extra_headers", None)

    # Normalize address candidates
    def _format_host(host: str) -> str:
        # Bracket IPv6 literals
        if ":" in host and not host.startswith("["):
            return f"[{host}]"
        return host

    try:
        host, port = address.rsplit(":", 1)
    except ValueError:
        host, port = address, "80"
    candidates = [f"{_format_host(host)}:{port}"]
    # Local fallbacks if the server is on the same node
    candidates.append(f"127.0.0.1:{port}")
    candidates.append(f"[::1]:{port}")

    max_retries = 3
    retry_delay = 1  # seconds

    last_exc = None
    for retry in range(max_retries):
        for cand in candidates:
            base_url = f"http://{cand}/v1/chat/completions"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        base_url,
                        json=completions_request,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=2700),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {error_text}")
                        return await response.json()
            except Exception as e:
                last_exc = e
                continue
        if retry == max_retries - 1:
            break
        await asyncio.sleep(retry_delay)
        retry_delay *= 2

    if last_exc:
        raise last_exc
    raise Exception("All retries failed")


class Router:
    """
    Least-used routing by address. Each model/policy has its own Router instance.
    """

    def __init__(self, config, tokenizer, addresses: list[str]):
        # A set of "ip:port" addresses
        self.addresses = addresses or []
        self.tensor_parallel_size = config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 1)
        self._lock = asyncio.Lock()
        # Track usage count for each address
        self._usage: dict[str, int] = {addr: 0 for addr in self.addresses}
        # Pin application_id to an address to improve data locality
        self._application_id_to_address: dict[str, str] = {}

        self.counter = 0
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

    async def get_address(self, application_id: str) -> str:
        """
        Select the least-used address and increment its usage count.
        """
        async with self._lock:
            if not self._usage:
                raise RuntimeError("Router has no available addresses")

            min_address, min_usage = min(self._usage.items(), key=lambda x: x[1])
            if application_id not in self._application_id_to_address:
                self._application_id_to_address[application_id] = min_address
                self._usage[min_address] = self._usage.get(min_address, 0) + 1
            else:
                # Try to keep data locality while roughly balancing load
                cur_address = self._application_id_to_address[application_id]
                cur_usage = self._usage.get(cur_address, 0)
                if (min_usage == 0 or cur_usage - min_usage >= 4) and cur_usage > 0:
                    self._application_id_to_address[application_id] = min_address
                    self._usage[min_address] = self._usage.get(min_address, 0) + 1
                else:
                    self._usage[cur_address] = self._usage.get(cur_address, 0) + 1
        return self._application_id_to_address[application_id]

    async def release_address(self, addr: str, application_id: str) -> None:
        """ Decrement the usage count for a server address when done."""
        async with self._lock:
            if addr in self._usage:
                self._usage[addr] = max(0, self._usage.get(addr, 0) - 1)

    async def generate_sequences(self, batch: DataProto, application_id: str, **sampling_params):
        kwargs = dict(
            n=self.config.actor_rollout_ref.rollout.n,
            max_tokens=self.config.actor_rollout_ref.rollout.response_length,  # Changed from max_completion_tokens
            temperature=self.config.actor_rollout_ref.rollout.temperature,
            top_p=self.config.actor_rollout_ref.rollout.top_p,
            logprobs=1,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        if is_validate:
            kwargs.update(
                {
                    #'top_k': self.config.val_kwargs.top_k,
                    "top_p": self.config.actor_rollout_ref.rollout.val_kwargs.top_p,
                    "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        if batch.meta_info.get("max_tokens", None) is not None:
            kwargs["max_tokens"] = batch.meta_info["max_tokens"]

        if batch.meta_info.get("agent_rollout", False):
            kwargs["n"] = 1

        kwargs.update(sampling_params)

        address = await self.get_address(application_id)

        tasks = []
        batch_size = len(batch.non_tensor_batch["formatted_prompts"])
        batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]

        for batch_index, formatted_prompt in enumerate(batch.non_tensor_batch["formatted_prompts"]):
            # Convert to Chat Completions request with single user message
            self.counter += 1
            messages = [{"role": "user", "content": formatted_prompt}]
            tasks.append(
                self.submit_completions(
                    address=address,
                    model=self.model_name,
                    messages=messages,
                    **kwargs,
                )
            )

        # Potential blocking: asyncio.gather can block if any task takes too long
        logger.debug("Sending total requests: %s", self.counter)
        completions_list = await asyncio.gather(*tasks)
        await self.release_address(address, application_id)  # Release the address when done

        for batch_index, completions in enumerate(completions_list):
            comps = []
            for choice in completions.get("choices", []):
                token_ids = []
                logprobs = choice.get("logprobs")
                if isinstance(logprobs, dict):
                    # vLLM chat logprobs may return structured tokens under content with token_id
                    content_tokens = logprobs.get("content")
                    if isinstance(content_tokens, list) and len(content_tokens) > 0:
                        # Each item may be a dict with token or token_id
                        for tk in content_tokens:
                            if isinstance(tk, dict) and "token_id" in tk:
                                token_ids.append(int(tk["token_id"]))
                    elif "tokens" in logprobs:
                        # Legacy support: tokens like ["id:123", ...]
                        for t in logprobs.get("tokens", []):
                            try:
                                token_ids.append(int(str(t).split(":")[1]))
                            except Exception:
                                continue
                if not token_ids:
                    # Fallback: tokenize the returned text/content
                    text = None
                    if isinstance(choice.get("message"), dict):
                        text = choice["message"].get("content", "")
                    if text is None:
                        text = choice.get("text", "")
                    if text is None:
                        text = ""
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                comps.append(token_ids)
            batch_response_ids[batch_index] = comps

        return await self.postprocess_batch(batch, batch_response_ids, kwargs["n"])

    async def submit_completions(self, address, model, **kwargs):
        # Potential blocking: network I/O can block
        return await poll_completions_openai(address=address, model=model, **kwargs)

    async def postprocess_batch(self, batch: DataProto, response_ids: list[list[int]], n: int) -> DataProto:
        # NOTE: For Completion API, batch_completions is a list of lists of strings (not dictionaries)
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        idx = batch.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = batch.batch["attention_mask"]
        position_ids = batch.batch["position_ids"]
        non_tensor_batch = deepcopy(batch.non_tensor_batch)

        # Flatten to list.
        # Flatten the list of lists of token IDs
        response = []
        for r_ids in response_ids:
            if r_ids is not None:  # Ensure we don't process None values
                for r in r_ids:
                    response.append(r)
        assert len(response) == len(non_tensor_batch["formatted_prompts"]) * n
        response_tensor = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.actor_rollout_ref.rollout.response_length).to(idx.device)

        if n > 1:
            idx = _repeat_interleave(idx, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            for key, val in non_tensor_batch.items():
                non_tensor_batch[key] = _repeat_interleave(val, n)

        batch_size = len(idx)
        seq = torch.cat([idx, response_tensor], dim=-1)

        response_length = response_tensor.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response_tensor, eos_token=self.eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        output = TensorDict(
            {
                "prompts": idx,
                "responses": response_tensor,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=output, meta_info=batch.meta_info)
    
def test_router_simple():
    from omegaconf import OmegaConf, open_dict
    config = OmegaConf.load("pettingllms/config/code/ppo_trainer/base.yaml")
    test_serve_num=3
    addresses = []
    if not ray.is_initialized():
        ray.init(num_cpus=4)
    for _ in range(test_serve_num):
        server = AsyncvLLMServer.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=__import__("ray._raylet")._raylet.NodeID.from_hex(ray.nodes()[0]["NodeID"]),
                        soft=False,
                    ),
            name=f"async_llm_server_{_}",
        ).remote(config, 3, _, "actor_rollout")
        address = ray.get(server.get_server_address.remote())
        addresses.append(address)
    print(addresses)
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    router = Router(config, tokenizer, addresses)
    print(router.get_address("test"))




def test_router():
    import os
    import gc
    import torch.distributed as dist
    from transformers import AutoConfig
    from omegaconf import OmegaConf, open_dict
    from verl.utils import hf_tokenizer
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout


    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RAY_LOCAL_WORLD_SIZE", "1")
        os.environ.setdefault("RAY_LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    #tokenizer = hf_tokenizer(model_path, trust_remote_code=True)


    ppo_config = OmegaConf.load("pettingllms/config/code/ppo_trainer/base.yaml")
    with open_dict(ppo_config):
        ppo_config.actor_rollout_ref.model.path = model_path
        ppo_config.actor_rollout_ref.model.tokenizer_path = model_path
        ppo_config.actor_rollout_ref.model.use_shm = False
        ppo_config.actor_rollout_ref.rollout.mode = "async"
        ppo_config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        if ppo_config.actor_rollout_ref.rollout.get("agent") is None:
            ppo_config.actor_rollout_ref.rollout.agent = OmegaConf.create({})
        if ppo_config.actor_rollout_ref.rollout.agent.get("num_workers") is None:
            ppo_config.actor_rollout_ref.rollout.agent.num_workers = 1
        if ppo_config.actor_rollout_ref.rollout.agent.get("custom_async_server", "__MISSING__") == "__MISSING__":
            ppo_config.actor_rollout_ref.rollout.agent.custom_async_server = None


    # rollout_config.engine_kwargs.vllm.compilation_config.level = 0
    # rollout_config.engine_kwargs.vllm.compilation_config.use_cudagraph = False
    with open_dict(ppo_config):
        if ppo_config.get("engine_kwargs") is None:
            ppo_config.engine_kwargs = OmegaConf.create({})
        if ppo_config.engine_kwargs.get("vllm") is None:
            ppo_config.engine_kwargs.vllm = OmegaConf.create({})
        if ppo_config.engine_kwargs.vllm.get("compilation_config") is None:
            ppo_config.engine_kwargs.vllm.compilation_config = OmegaConf.create({})
        ppo_config.engine_kwargs.vllm.compilation_config.level = 0
        ppo_config.engine_kwargs.vllm.compilation_config.use_cudagraph = False


    #model_hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)


    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    if not ray.is_initialized():
        ray.init(num_cpus=1)

    use_gpu = torch.cuda.is_available()
    resource_pool = RayResourcePool(
        process_on_nodes=[1], use_gpu=use_gpu, name_prefix="actor_rollout", max_colocate_count=1
    )
    actor_remote = ray.remote(AsyncActorRolloutRefWorker)
    actor_cia = RayClassWithInitArgs(
        cls=actor_remote,
        config=ppo_config.actor_rollout_ref,
        role="actor_rollout",
    )
    # 使用共置包装 + spawn，模拟训练器创建流程，避免注册中心边缘问题
    worker_dict_cls = create_colocated_worker_cls({"actor_rollout": actor_cia})
    wg_top = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        name_prefix="actor_rollout",
    )
    spawned = wg_top.spawn(prefix_set={"actor_rollout"})
    worker_group = spawned["actor_rollout"]
    worker_group.init_model()


    rollout_manager = AgentLoopManager(
        config=ppo_config,
        worker_group=worker_group
    )
 

    addresses = getattr(rollout_manager, "server_addresses", [])
    print("addresses:", addresses)


if __name__ == "__main__":
    test_router_simple()