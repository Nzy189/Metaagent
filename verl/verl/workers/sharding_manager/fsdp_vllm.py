# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
from typing import Union
import os
import time
from collections import OrderedDict
from typing import List

import torch
from peft import PeftModel
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from vllm import AsyncLLMEngine

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class FSDPVLLMShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = 'dummy_hf',
        layered_summon: bool = True
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine

        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            try:
                self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None
            except:
                self.model_runner = self.inference_engine.engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None
            
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = 'dummy' not in load_format
        if is_version_ge(pkg='vllm', minver='0.7.3'):
            VLLMHijack.hijack()
        
        # Store multi_lora configuration for __enter__ method
        self.multi_lora = False
        self.lora_num = 1

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self, multi_lora: bool = None, lora_num: int = None):
        # Use passed parameters or fall back to instance attributes
        if multi_lora is not None:
            self.multi_lora = multi_lora
        if lora_num is not None:
            self.lora_num = lora_num
        
        def __collect_lora_params(adapter_name: str = None)->OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            
            Args:
                adapter_name: The name of the adapter to collect. If None, uses the active adapter.
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    if not self.base_sync_done:
                        raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                    # Pass adapter_name to extract the correct adapter's parameters
                    lora_params = layered_summon_lora_params(self.module, adapter_name=adapter_name if adapter_name else "default")
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
                            # Pass adapter_name to get_peft_model_state_dict to avoid 'default' KeyError
                            lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module, adapter_name=adapter_name)
                            lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu() 
                                        for name, param in lora_params.items()}
                        else:
                            model = self.module._fsdp_wrapped_module.base_model.model
                            orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device else 'cuda'
                            model = model.to('cpu')
                            for name, param in model.state_dict().items():
                                if any(x in name for x in ['_flat_param', 'lora_']):
                                    continue
                                name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                                lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                            model = model.to(orig_dev)
                    torch.cuda.empty_cache()
            else:
                if self.base_sync_done:
                    # Pass adapter_name to get_peft_model_state_dict to avoid 'default' KeyError
                    lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module, adapter_name=adapter_name)
                else:
                    model = self.module._fsdp_wrapped_module.base_model.model
                    orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device else 'cuda'
                    model = model.to('cpu')
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ['_flat_param', 'lora_']):
                            continue
                        name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                        lora_params[name] = param.detach().cpu()
                    model = model.to(orig_dev)
            return lora_params

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        get_torch_device().empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        peft_config = None
        adapter_name = None
        lora_id = None

        # Store all LoRA adapters info for multi-LoRA mode
        all_lora_adapters = []
        
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            # Get all available adapters
            available_adapters = list(self.module._fsdp_wrapped_module.peft_config.keys())
            
            # Check if this is multi-LoRA mode based on configuration
            # Priority: explicit multi_lora flag > auto-detection from adapters
            if self.multi_lora and self.lora_num > 1:
                is_multi_lora = True
                logger.info(f"Multi-LoRA mode enabled via configuration: lora_num={self.lora_num}")
            else:
                # Fallback to auto-detection (no 'default' adapter, only lora_1, lora_2, etc.)
                is_multi_lora = (
                    'default' not in available_adapters and 
                    all(name.startswith('lora_') for name in available_adapters)
                )
                if is_multi_lora:
                    logger.info(f"Multi-LoRA mode auto-detected from adapters: {available_adapters}")
            
            if is_multi_lora:
                # Multi-LoRA mode: sync all LoRA adapters to vLLM
                print(f"Multi-LoRA mode detected. Available adapters: {available_adapters}")
                
                # Get the current active adapter to restore later
                original_active_adapter = self.module._fsdp_wrapped_module.active_adapter
                print(f"The current active adapter: {original_active_adapter}")
                # Collect parameters for all adapters
                for adapter_name in available_adapters:
                    if adapter_name.startswith('lora_'):
                        lora_id = int(adapter_name.split('_')[-1])
                        peft_config = self.module._fsdp_wrapped_module.peft_config.get(adapter_name)
                        
                        # Switch to this adapter to collect its parameters
                        self.module._fsdp_wrapped_module.set_adapter(adapter_name)
                        params = __collect_lora_params(adapter_name=adapter_name)
                        
                        all_lora_adapters.append({
                            'adapter_name': adapter_name,
                            'lora_id': lora_id,
                            'peft_config': peft_config,
                            'params': params
                        })
                        print(f"Collected LoRA adapter '{adapter_name}' (lora_id={lora_id})")
                
                # Restore the original active adapter
                if original_active_adapter in available_adapters:
                    self.module._fsdp_wrapped_module.set_adapter(original_active_adapter)
                else:
                    # If original was default, switch to lora_1
                    self.module._fsdp_wrapped_module.set_adapter('lora_1')
                
                # For single update_params call, use the first adapter (will be updated in loop later)
                adapter_name = all_lora_adapters[0]['adapter_name']
                lora_id = all_lora_adapters[0]['lora_id']
                peft_config = all_lora_adapters[0]['peft_config']
                params = all_lora_adapters[0]['params']
                
            else:
                # Single LoRA mode or default adapter exists
                active_adapter = self.module._fsdp_wrapped_module.active_adapter
                
                if 'default' in available_adapters:
                    # Use default adapter
                    peft_config = self.module._fsdp_wrapped_module.peft_config.get('default', None)
                    adapter_name = 'default'
                    lora_id = None
                elif available_adapters:
                    # Use the first available adapter
                    adapter_name = available_adapters[0]
                    if adapter_name.startswith('lora_'):
                        lora_id = int(adapter_name.split('_')[-1])
                    else:
                        lora_id = None
                    peft_config = self.module._fsdp_wrapped_module.peft_config.get(adapter_name)
                    logger.warning(f"No 'default' adapter found. Using adapter: '{adapter_name}'")
                else:
                    raise ValueError("No LoRA adapters found in peft_config")
                
                params = __collect_lora_params(adapter_name=adapter_name)
        else:
            params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            # In multi-LoRA mode, sync all adapters to vLLM
            if all_lora_adapters:
                print(f"Syncing {len(all_lora_adapters)} LoRA adapters to vLLM...")
                for lora_info in all_lora_adapters:
                    self.update_params(
                        lora_info['params'], 
                        peft_config=lora_info['peft_config'], 
                        lora_id=lora_info['lora_id'], 
                        adapter_name=lora_info['adapter_name']
                    )
                    del lora_info['params']  # Free memory after each sync
                logger.info(f"Successfully synced all {len(all_lora_adapters)} LoRA adapters to vLLM")
            else:
                # Single LoRA mode: sync only the active adapter
                self.update_params(params, peft_config=peft_config, lora_id=lora_id, adapter_name=adapter_name)
                del params
            
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            get_torch_device().empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None, lora_id=None, adapter_name=None):
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                # Use provided lora_id if available (multi-LoRA mode), otherwise generate one
                if lora_id is not None and adapter_name:
                    lora_int_id = lora_id
                    lora_name = adapter_name
                else:
                    lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                    lora_name = f"{lora_int_id}"
                self.inference_engine.worker.remove_lora(lora_int_id)

                lora_reqest = TensorLoRARequest(
                    lora_name=lora_name,
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                self.inference_engine.worker.add_lora(lora_reqest)
                print(f"vLLM synced LoRA '{lora_name}' (id={lora_int_id}), loaded_params: {len(updated_params)}")
                return
            else:
                def replace_lora_wrapper(k):
                    stacked_params = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        return k.replace(".weight", ".base_layer.weight")
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        return k.replace(".bias", ".base_layer.bias")
                    return k
                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        patch_vllm_moe_model_weight_loader(model)
        device = get_torch_device().current_device()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in updated_params.items()))

        self.base_sync_done = True
        print(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
