
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
from readline import add_history
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
import uuid
import hydra
from copy import deepcopy
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
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from dataclasses import dataclass
from openai.types.completion import Completion
from datetime import datetime, timedelta
import aiohttp


# =================== 全局共享资源管理 ===================
# 全局共享ClientSession，避免频繁创建销毁连接
_shared_session = None
_session_lock = asyncio.Lock()

# 全局并发控制Semaphore，限制同时进行的LLM请求数量
_llm_request_semaphore = None
_semaphore_lock = asyncio.Lock()


async def get_shared_session() -> aiohttp.ClientSession:

    global _shared_session
    
    async with _session_lock:
        if _shared_session is None or _shared_session.closed:
            connector = aiohttp.TCPConnector(
                limit=200,  
                limit_per_host=100,  
                ttl_dns_cache=300,  
                force_close=False,  
                enable_cleanup_closed=True  
            )
            
            timeout = aiohttp.ClientTimeout(
                total=300,  
                connect=30,  
                sock_read=300  
            )
            
            _shared_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            print(f"[Session] Created shared aiohttp session with connection pool (limit=200, per_host=100)")
    
    return _shared_session


async def get_llm_semaphore(max_concurrent: int = 50) -> asyncio.Semaphore:

    global _llm_request_semaphore
    
    async with _semaphore_lock:
        if _llm_request_semaphore is None:
            _llm_request_semaphore = asyncio.Semaphore(max_concurrent)
            print(f"[Semaphore] Created global LLM request semaphore with max_concurrent={max_concurrent}")
    
    return _llm_request_semaphore


async def cleanup_shared_session():
    
    global _shared_session
    async with _session_lock:
        if _shared_session is not None and not _shared_session.closed:
            await _shared_session.close()
            _shared_session = None
            print("[Session] Closed shared aiohttp session")



async def poll_completions_openai(address: str, **completions_request) -> Completion:
   
    session = await get_shared_session()
    semaphore = await get_llm_semaphore(max_concurrent=50)
    
    if address.startswith(('http://', 'https://')):
        base_url = f"{address}/v1/completions"
    else:
        base_url = f"http://{address}/v1/completions"
    
    headers = {
        "Content-Type": "application/json",
    }


    completions_request.pop("meta_info", None)
    completions_request.pop("extra_headers", None)
    async with semaphore:
        try:
            async with session.post(
                base_url, 
                json=completions_request, 
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                result = await response.json()
                return result
                
        except asyncio.TimeoutError as e:
            error_msg = f"Request timeout to {address}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
            
        except aiohttp.ClientError as e:
            error_msg = f"Client error when requesting {address}: {e}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error when requesting {address}: {e}"
            print(f"[ERROR] {error_msg}")
            raise


async def submit_completions(
    address: str, 
    model: str, 
    prompt: str, 
    max_retries: int = 3,
    initial_retry_delay: float = 1.0,
    **kwargs
):
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = await poll_completions_openai(
                address=address, 
                model=model, 
                prompt=prompt, 
                **kwargs
            )
            
            # 成功则返回
            if attempt > 0:
                print(f"[Retry] Request succeeded on attempt {attempt + 1}/{max_retries}")
            return result
            
        except Exception as e:
            last_exception = e
            
  
            if attempt == max_retries - 1:
                print(f"[ERROR] All {max_retries} retry attempts failed for {address}")
                print(f"[ERROR] Final error: {e}")
                raise e
            

            retry_delay = initial_retry_delay * (2 ** attempt)
            print(f"[Retry] Attempt {attempt + 1}/{max_retries} failed: {e}")
            print(f"[Retry] Retrying in {retry_delay:.1f}s... (address={address})")
            
            await asyncio.sleep(retry_delay)
    
    if last_exception:
        raise last_exception
    else:
        raise Exception(f"Request failed after {max_retries} attempts")

async def postprocess_batch(batch: DataProto, response_ids: list[list[int]], n: int,pad_token_id,eos_token_id,max_response_length,max_prompt_length) -> DataProto:
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
    
    # Truncate prompts if they exceed max_prompt_length
    if idx.size(1) > max_prompt_length:
        print(f"[WARNING] Truncating prompt from {idx.size(1)} to {max_prompt_length}")
        # For left-padded prompts, keep the rightmost tokens (most recent context)
        idx = idx[:, -max_prompt_length:]
        attention_mask = attention_mask[:, -max_prompt_length:]
        position_ids = position_ids[:, -max_prompt_length:]

    # Flatten to list.
    # Flatten the list of lists of token IDs
    response = []
    for r_ids in response_ids:
        if r_ids is not None and len(r_ids) > 0:  # Ensure we don't process None or empty values
            for r in r_ids:
                # Handle empty response
                if r is None or len(r) == 0:
                    print(f"[WARNING] Empty response detected, using EOS token as fallback")
                    response.append([eos_token_id])
                    continue
                    
                # Truncate each response if it exceeds max_response_length
                if len(r) > max_response_length:
                    print(f"[WARNING] Truncating response from {len(r)} to {max_response_length}")
                    r = r[:max_response_length]
                response.append(r)
        else:
            # Fallback for None or empty r_ids
            print(f"[WARNING] None or empty r_ids detected, using EOS token as fallback")
            for _ in range(n):  # Add n empty responses for this batch
                response.append([eos_token_id])
    
    # Ensure we have the expected number of responses
    expected_count = len(non_tensor_batch["formatted_prompts"]) * n
    if len(response) != expected_count:
        print(f"[WARNING] Response count mismatch: expected {expected_count}, got {len(response)}")
        # Pad with empty responses if needed
        while len(response) < expected_count:
            print(f"[WARNING] Adding fallback empty response")
            response.append([eos_token_id])
        # Truncate if too many
        if len(response) > expected_count:
            print(f"[WARNING] Too many responses, truncating to {expected_count}")
            response = response[:expected_count]
    
    response_tensor = pad_2d_list_to_length(response, pad_token_id, max_length=max_response_length).to(idx.device)

    batch_size = len(idx)
    
    # Debug info before concatenation
    try:
        # Ensure the concatenated sequence doesn't exceed max total length
        max_total_length = max_prompt_length + max_response_length
        current_prompt_length = idx.size(1)
        current_response_length = response_tensor.size(1)
        
        # If total would exceed limit, adjust response length
        if current_prompt_length + current_response_length > max_total_length:
            new_response_length = max_total_length - current_prompt_length
            if new_response_length > 0:
                print(f"[WARNING] Total sequence too long ({current_prompt_length + current_response_length}), truncating response to {new_response_length}")
                response_tensor = response_tensor[:, :new_response_length]
            else:
                print(f"[ERROR] Prompt is already too long ({current_prompt_length}), setting response to empty")
                response_tensor = torch.full((batch_size, 1), pad_token_id, dtype=response_tensor.dtype, device=response_tensor.device)
        
        seq = torch.cat([idx, response_tensor], dim=-1)

        response_length = response_tensor.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response_tensor, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
    except Exception as e:
        print(f"[ERROR postprocess_batch] Tensor concatenation failed: {e}")
        print(f"[ERROR] Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

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

async def llm_async_generate(
    rollout_idx: int,
    turn_idx: int,
    agent_idx: int,
    prompt_dpr: DataProto,
    ppo_trainer_config: DictConfig,
    address: Optional[str] = None,
    model_name: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    enable_thinking: Optional[bool] = False,
    image_data: Optional[list[Any]] = None,
    application_id: Optional[str] = None,
    env_idx: Optional[int] = None,
    #rollout_idx: Optional[int] = None,
    policy_name: Optional[str] = None,
    timeout: Optional[float] = 60.0,    
    mode: Optional[str] = "train",
    #override_temperature: Optional[float] = None,
    lora_id: Optional[str] = None,
    agent_config: Optional[DictConfig] = None,
) -> DataProto:
    """Generate tokens from prompt ids or DataProto.

    Returns:
        DataProto: DataProto format output consistent with Router's generate_sequences.
    """

    if application_id is None:
        application_id = str(uuid.uuid4())
    else:
        application_id = str(application_id)

    unique_request_id = f"{application_id}_{uuid.uuid4().hex[:8]}"

    # Debug: print input address/model info
    try:
        print(f"[LLM][llm_async_generate] request_id={unique_request_id} address={address} model={model_name} mode={mode}")
    except Exception:
        pass


    if mode == "train":
        default_temp = 1.0
    else:  # mode == "validate" or other
        default_temp = 0.6 if enable_thinking else 0.0
    
    if agent_config is not None:
        if mode == "train":
            temp = getattr(agent_config, 'train_temperature', default_temp)
        else:  # mode == "validate" or other
            temp = getattr(agent_config, 'val_temperature', default_temp)
    else:
        temp = default_temp
    
    try:
        print(f"[LLM][llm_async_generate] enable_thinking={enable_thinking} mode={mode} temperature={temp}")
    except Exception:
        pass

    top_p=0.95
    kwargs={
        "n":1,
        "temperature":temp,
        "top_p":top_p,
        "max_tokens":ppo_trainer_config.data.max_response_length,
        "top_k":-1,
        "logprobs":1,
    }
    batch_size = len(prompt_dpr.non_tensor_batch["formatted_prompts"])
    batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]
    text = ""  # Initialize text variable for return value

    # Determine which model to use: LoRA adapter or base model
    # In vllm, LoRA is selected by specifying the lora_name as the model parameter
    if lora_id is not None:
        actual_model = lora_id  # Use LoRA adapter name as model
        try:
            print(f"[LLM][llm_async_generate] Using LoRA adapter: {lora_id}")
        except Exception:
            pass
    else:
        actual_model = model_name  # Use base model name
        try:
            print(f"[LLM][llm_async_generate] Using base model: {model_name}")
        except Exception:
            pass

    tasks = []
    for batch_index, formatted_prompt in enumerate(prompt_dpr.non_tensor_batch["formatted_prompts"]):
        # For Completion API, we need to convert the conversation to a prompt string
        
        tasks.append(
            submit_completions(
                address=address,
                model=actual_model,  # Pass LoRA name or base model name
                prompt=formatted_prompt,
                max_retries=3, 
                initial_retry_delay=1.0,  
                **kwargs,
            )
        )


    try:
        print(f"[LLM][batch_request] Starting {len(tasks)} requests for rollout_idx={rollout_idx}, turn_idx={turn_idx}")
    except Exception:
        pass
    
    # Use return_exceptions=True to handle failures gracefully
    start_time = time.time()
    completions_list = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed_time = time.time() - start_time
    
    # 统计成功和失败的请求
    success_count = sum(1 for c in completions_list if not isinstance(c, Exception))
    error_count = len(completions_list) - success_count
    
    try:
        print(f"[LLM][batch_request] Completed in {elapsed_time:.2f}s: {success_count} success, {error_count} errors")
    except Exception:
        pass
    
    for batch_index, completions in enumerate(completions_list):
        comps = []
        
        # Handle exceptions from API calls
        if isinstance(completions, Exception):
            print(f"[ERROR] API call failed for batch {batch_index}: {completions}")
            # Return empty token list as fallback
            comps.append([tokenizer.eos_token_id])  # At least add EOS token
            batch_response_ids[batch_index] = comps
            continue
            
        # Handle None or invalid responses
        if completions is None:
            print(f"[WARNING] Got None response for batch {batch_index}")
            comps.append([tokenizer.eos_token_id])
            batch_response_ids[batch_index] = comps
            continue
            
        # Normal processing
        try:
            choices = completions.get("choices", [])
            if not choices:
                print(f"[WARNING] No choices in response for batch {batch_index}")
                comps.append([tokenizer.eos_token_id])
            else:
                for choice in choices:
                    token_ids = choice.get("logprobs", {}).get("tokens", [])
                    text = choice.get("text", "")
                    if token_ids:
                        token_ids = [int(t.split(":")[1]) for t in token_ids]
                        comps.append(token_ids)
                    else:
                        # Fallback: if no token_ids, add EOS
                        print(f"[WARNING] No token_ids in choice for batch {batch_index}")
                        comps.append([tokenizer.eos_token_id])
        except Exception as e:
            print(f"[ERROR] Failed to process response for batch {batch_index}: {e}")
            comps.append([tokenizer.eos_token_id])
            
        batch_response_ids[batch_index] = comps
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    max_response_length=ppo_trainer_config.data.max_response_length
    max_prompt_length=ppo_trainer_config.data.max_prompt_length
    output_dpr = await postprocess_batch(prompt_dpr, batch_response_ids, kwargs["n"], pad_token_id, eos_token_id,max_response_length,max_prompt_length)
    output_dpr.non_tensor_batch["rollout_idx"] = np.array([rollout_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["turn_idx"] = np.array([turn_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["agent_idx"] = np.array([agent_idx] * output_dpr.batch.shape[0], dtype=object)
    return output_dpr, text
    
    

    

def convert_prompt_to_format(tokenizer, enable_thinking, prompts,**kwargs):
    """
    Convert prompt dict to veRL's DataProto.
    
    Args:
        tokenizer: HF tokenizer, must support apply_chat_template and __call__ tokenization
        enable_thinking: Whether to enable thinking mode for chat template
        prompts: dict, {"text": str, "image": None or image path}
        kwargs: Optional parameters, such as processor, meta_info, etc.
    Returns:
        DataProto: Contains tensor and non-tensor information
    """

    if not isinstance(prompts, dict) or "text" not in prompts:
        raise ValueError("prompts must be a dictionary containing 'text' key: {'text': str, 'image': Optional[path]} ")

    text = prompts.get("text", "") or ""
    image_path = prompts.get("image", None)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
  
    chat = np.array([
        {"content": text, "role": "user"}
    ])

    prompt_with_chat_template = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking
    )

    return prompt_with_chat_template
    

    

def convert_prompt_to_dpr(tokenizer, processor, prompts, max_prompt_length, multi_modal=False, enable_thinking=False, **kwargs):
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
            enable_thinking=enable_thinking
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
        
        # Truncate if prompt exceeds max_prompt_length
        if input_ids.size(1) > max_prompt_length:
            print(f"[WARNING] convert_prompt_to_dpr: Truncating prompt from {input_ids.size(1)} to {max_prompt_length}")
            # Keep the rightmost tokens (most recent context) for better quality
            input_ids = input_ids[:, -max_prompt_length:]
            attention_mask = attention_mask[:, -max_prompt_length:]
            # Regenerate the prompt string from truncated tokens to ensure consistency
            prompt_with_chat_template = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(f"[WARNING] Regenerated prompt after truncation, new length: {len(prompt_with_chat_template)} chars")

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
