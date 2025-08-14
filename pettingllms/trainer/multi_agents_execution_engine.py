import asyncio
import concurrent.futures
import logging
import time
import json
import traceback
import uuid
try:
    from verl.protocol import DataProto
except Exception:  # fallback when verl is a src tree: verl/verl/protocol.py
    from verl import DataProto
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import openai
import torch
from openai.types import Completion
from pettingllms.trainer.multiagentssys_register import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING, ENV_BATCH_CLASSES
from functools import partial
import multiprocessing

from pettingllms.multi_agent_env.base.env import Env, EnvBatch
from pettingllms.misc import colorful_print
from pettingllms.parser.chat_template.parser import ChatTemplateParser
from pettingllms.router.router import Router
from pettingllms.trainer.utils import convert_prompt_to_dpr, convert_dpr_to_response
from threading import Thread


logger = logging.getLogger(__name__)




class MultiAgentsExecutionEngine:
    def _load_config_parameters(self):
        """Load parameters from config with fallback to defaults"""
        
        
        # Data configuration - direct access with fallbacks
        if hasattr(self.config, 'data') and self.config.data is not None:
            self.max_prompt_length = getattr(self.config.data, 'max_prompt_length', 10240)
            self.max_response_length = getattr(self.config.data, 'max_response_length', 2048)
        else:
            self.max_prompt_length = 10240
            self.max_response_length = 2048
        # Multi-agent interaction configuration - direct access with fallbacks
        if hasattr(self.config, 'multi_agent_interaction') and self.config.multi_agent_interaction is not None:
            self.turn_order = getattr(self.config.multi_agent_interaction, 'turn_order', ['code_generator', 'test_generator'])
            self.num_interacting_agents = getattr(self.config.multi_agent_interaction, 'num_interacting_agents', 2)
            self.shared_observation = getattr(self.config.multi_agent_interaction, 'shared_observation', True)
        else:
            self.turn_order = ['code_generator', 'test_generator']
            self.num_interacting_agents = 2
            self.shared_observation = True
        
        # Rollout configuration - direct access with fallbacks
        if hasattr(self.config, 'data') and self.config.data is not None:
            self.sample_temperature = getattr(self.config.data, 'sample_temperature', 0.7)
            self.gen_batch_size = getattr(self.config.data, 'gen_batch_size', 64)
            self.gen_n_samples = getattr(self.config.data, 'gen_n_samples', 8)
        else:
            self.sample_temperature = 0.7
            self.gen_batch_size = 64
            self.gen_n_samples = 8
    def __init__(
        self,
        config,
        tokenizer_dict=None,
        processor_dict=None,
        server_manager_dict=None,
        agent_policy_mapping=None,
        env_args=None,
        max_workers=64,
        **kwargs,
    ):
        

        self.config = config
        self.tokenizer_dict = tokenizer_dict
        self.processor_dict = processor_dict or {}
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.env_args = env_args or {}
        self.max_workers = max_workers
        # Read parameters from config with fallback to defaults
        self._load_config_parameters()
        self.n_cpu = multiprocessing.cpu_count()

        # Environment configuration - direct access
        if hasattr(self.config, 'env') and self.config.env is not None:
            self.max_turns = getattr(self.config.env, 'max_turns', 8)
            env_name = getattr(self.config.env, 'name', None)
            if env_name is None:
                raise ValueError("env.name is not set in the config.env")
        else:
            raise ValueError("env is not set in the config")
            
        print(f"env_name: {env_name}")
        
        # Store env_name for later use
        self.env_name = env_name
        self.env_class = ENV_CLASS_MAPPING[env_name]
        self.agent_class_list = [AGENT_CLASS_MAPPING[agent_name] for agent_name in self.turn_order]
        self._init_agents_and_envs()
        #self._init_agents()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        # rollout_engine_dict is not maintained in this class to avoid referencing a non-existent attribute
        self.server_manager_dict = server_manager_dict or {}
        self.chat_parser_dict={}
        #for key,value in self.router_dict.items():
        #    self.chat_parser_dict[key]=ChatTemplateParser.get_parser(self.tokenizer_dict[key], disable_thinking=False)
        

    def _init_agents_and_envs(self):
       
        # Check for batched_init in config.env
        if hasattr(self.config, 'env') and self.config.env is not None:
            batched_init = getattr(self.config.env, 'batched_init', True)
        else:
            batched_init = getattr(self.config, 'batched_init', True)
        if batched_init == False:
            with multiprocessing.Pool(self.n_cpu // 4) as pool: # Only use 1/4 of the cores to avoid conflicts
                func=partial(self.__init_one_env_instance, env_args=self.env_args)
                self.envs = pool.map(func, range(self.gen_batch_size*self.gen_n_samples))
            self.env_list = self.envs
            
        else:
            self.env_batch_class=ENV_BATCH_CLASSES[self.env_name]
            self.envs_batch=self.env_batch_class(env_idx_list=range(self.gen_batch_size), rollout_idx_list=range(self.gen_batch_size*self.gen_n_samples), samples=self.gen_n_samples, max_turns=self.max_turns, config=self.config)
            self.envs=self.envs_batch.env_list
            self.env_list = self.envs

        # Initialize the agent group for each rollout
        self.agent_groups = [
            [agent_cls(rollout_idx=rollout_idx) for agent_cls in self.agent_class_list]
            for rollout_idx in range(len(self.env_list))
        ]
        
        
        
    
    def __init_one_env_instance(self, rollout_idx, env_args):
        env = self.env_class( env_idx=rollout_idx % self.gen_batch_size,rollout_idx=rollout_idx,max_turns=self.max_turns, **env_args)
        
        return env
    

        
    async def generate_single_rollout(self, rollout_idx, timing_raw, meta_info):
        """
        Generate a single rollout, adapted for multi-agent interaction in the code testing environment.
        
        Args:
            env: Code testing environment instance
            timing_raw: Timing record dictionary
            meta_info: Meta information
            
        Returns:
            DataProto: DataProto object containing trajectory data
        """
        rollout_id = str(uuid.uuid4())
        trajectory_per_task_dict = {}
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()


        env = self.env_list[rollout_idx] if hasattr(self, "env_list") else self.envs[rollout_idx]
        agent_group = self.agent_groups[rollout_idx]
        

        
        print(f"=== DEBUG: 开始多轮对话，最大轮数: {self.max_turns} ===")
        print(f"转换顺序: {self.turn_order}")
        print(f"可用的 tokenizer: {list(self.tokenizer_dict.keys())}")
        print(f"可用的 server_manager: {list(self.server_manager_dict.keys())}")
        
        for turn_idx in range(self.max_turns):
            print(f"\n=== DEBUG: 第 {turn_idx + 1} 轮对话 ===")
            for agent_idx, agent_name in enumerate(self.turn_order):
                print(f"\n--- Agent {agent_idx}: {agent_name} ---")
                current_agent = agent_group[agent_idx]
                current_agent.update_from_env(env)
                prompt = current_agent.current_prompt
                print(f"Agent prompt: {prompt}")
                
                # Select the policy name; if not provided, fall back to any available policy
                policy_name = self.agent_policy_mapping.get(agent_name) if self.agent_policy_mapping else None
                if policy_name is None:
                    policy_name = next(iter(self.server_manager_dict.keys())) if self.server_manager_dict else next(iter(self.tokenizer_dict.keys()))
                
                print(f"使用的策略: {policy_name}")
                print(f"tokenizer 类型: {type(self.tokenizer_dict[policy_name])}")
                
                # Convert to DataProto format
                print("=== DEBUG: 转换 prompt 到 DataProto 格式 ===")
                
                dpr_prompt = convert_prompt_to_dpr(self.tokenizer_dict[policy_name], 
                        self.chat_parser_dict.get(policy_name), 
                        self.processor_dict.get(policy_name) if isinstance(self.processor_dict, dict) else None,
                        prompt, 
                        self.max_prompt_length,
                       multi_modal=False
                   )
                
                # Generate responses
                print("=== DEBUG: 调    用 server_manager.generate ===")
                print(f"application_id: {rollout_id}")
                try:
                    output_dpr = await self.server_manager_dict[policy_name].generate(
                        dpr_prompt, 
                        application_id=rollout_id
                    )
                    print(f"server_manager.generate 成功，output 类型: {type(output_dpr)}")
                    if hasattr(output_dpr, 'batch'):
                        print(f"output batch keys: {list(output_dpr.batch.keys())}")
                except Exception as e:
                    print(f"server_manager.generate 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Convert response format
                print("=== DEBUG: 转换响应格式 ===")
                try:
                    response = convert_dpr_to_response(
                        self.tokenizer_dict[policy_name], 
                        self.chat_parser_dict.get(policy_name), 
                        output_dpr, 
                        self.max_prompt_length
                    )
                    print(f"响应转换成功: {response}")
                except Exception as e:
                    print(f"响应转换失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                current_agent.update_from_model(response)
                
                env.step(agent_name, current_agent.action)
                
                current_agent.calculate_reward(env,mode="sum")

                output_dpr.non_tensor_batch["reward"] = [current_agent.reward]
                # union returns a new DataProto, so reassign back
                trajectory_per_task_dict[policy_name] = trajectory_per_task_dict[policy_name].union(output_dpr)
                
                # 打印轨迹信息
                print(f"\n=== 轨迹信息 ===")
                print(f"current agent name: {agent_name}")
                print(f"current agent prompt: {prompt}")
                print(f"current agent response: {response}")
                print(f"current agent reward: {current_agent.reward}")
                print(f"current agent action: {current_agent.action}")
                print(f"current agent observation: {env.observation}")
                print(f"=== 轨迹信息结束 ===\n")
        return trajectory_per_task_dict
        
       
    
class AsyncMultiAgentsExecutionEngine(MultiAgentsExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_server_manager_simple(trainer_config,config):
    import ray
    import os
    from verl.utils import hf_tokenizer
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
    from verl.utils import hf_tokenizer


    os.environ["VERL_VLLM_DISTRIBUTED_BACKEND"] = "none"

    os.environ["VLLM_USE_V1"] = "1"

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print(f"=== DEBUG: 开始测试 ===")
    print(f"trainer_config type: {type(trainer_config)}")
    print(f"config type: {type(config)}")
    print(f"Model path: {trainer_config.actor_rollout_ref.model.path}")

    test_serve_num=1
    if not ray.is_initialized():
        ray.init(num_cpus=4)
    for _ in range(test_serve_num):
        options_kwargs = dict(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=__import__("ray._raylet")._raylet.NodeID.from_hex(ray.nodes()[0]["NodeID"]),
                        soft=False,
                    ),
            name=f"async_llm_server_{_}",
        )
        # Allocate a GPU to the server actor if available, otherwise vLLM will see CUDA but cannot set device
        if torch.cuda.is_available():
            options_kwargs["num_gpus"] = 1
        server = AsyncvLLMServer.options(**options_kwargs).remote(trainer_config, 1, _, "actor_rollout")
        print(f"begin to init engine")

        
        ray.get(server.init_engine.remote(), timeout=60)
        if server.engine is None:
            raise ValueError("engine is not initialized")

        print(f"begin to init server manager")
        from pettingllms.trainer.utils import AsyncLLMServerManager
        from verl.utils import hf_tokenizer

        server_manager = AsyncLLMServerManager(config=trainer_config, server_handles=[server])

        model_path_local =trainer_config.actor_rollout_ref.model.path
        tokenizer_local = hf_tokenizer(model_path_local, trust_remote_code=True)
        
        print(f"=== DEBUG: Tokenizer信息 ===")
        print(f"tokenizer_local type: {type(tokenizer_local)}")
        print(f"tokenizer_local has keys method: {hasattr(tokenizer_local, 'keys')}")
        
        # 修复：创建正确的 tokenizer_dict（字典格式）
        tokenizer_dict = {"code_generator": tokenizer_local}
        
        prompt_text = "Hello"
        prompt_ids = tokenizer_local.encode(prompt_text, add_special_tokens=False)
        print(f"prompt_ids: {prompt_ids}")
        print(f"begin to generate")
        server_manager_dict={}
        server_manager_dict["code_generator"]=server_manager

        print("=== DEBUG: 初始化多智能体执行引擎 ===")
        print(f"tokenizer_dict keys: {list(tokenizer_dict.keys())}")
        print(f"server_manager_dict keys: {list(server_manager_dict.keys())}")
        
        # 修复：传递正确的 tokenizer_dict
        multi_agent_execution_engine = MultiAgentsExecutionEngine(
            config=config, 
            tokenizer_dict=tokenizer_dict, 
            server_manager_dict=server_manager_dict
        )
        
        async def _run_manager_gen():
            print("=== DEBUG: 开始生成单个rollout ===")
            try:
                result = await multi_agent_execution_engine.generate_single_rollout(0, None, None)
                print(f"=== DEBUG: 生成完成，结果类型: {type(result)} ===")
                return result
            except Exception as e:
                print(f"=== DEBUG: generate_single_rollout 内部错误: {e} ===")
                import traceback
                traceback.print_exc()
                raise

        try:
            dpr = asyncio.run(_run_manager_gen())
            print(f"[AsyncLLMServerManager.generate] 成功生成！")
            
            # 检查返回结果的结构
            if isinstance(dpr, dict):
                print(f"返回的是字典，键: {list(dpr.keys())}")
                for key, value in dpr.items():
                    print(f"  {key}: {type(value)}")
                    if hasattr(value, 'batch') and 'prompts' in value.batch:
                        print(f"    prompts shape: {tuple(value.batch['prompts'].shape)}")
                    if hasattr(value, 'batch') and 'responses' in value.batch:
                        print(f"    responses shape: {tuple(value.batch['responses'].shape)}")
            else:
                print(f"返回类型: {type(dpr)}")
                if hasattr(dpr, 'batch'):
                    print(f"batch keys: {list(dpr.batch.keys())}")
                    
        except Exception as e:
            print(f"[AsyncLLMServerManager.generate] FAILED: {e}")
            print(f"=== DEBUG: 详细错误信息 ===")
            import traceback
            traceback.print_exc()

    return None



def test_rollout_engine_simple():
    # 按照新的 PPO 初始化思路，仅验证 ServerManager 能生成 DataProto
    from omegaconf import OmegaConf
    trainer_config = OmegaConf.load("pettingllms/config/code/ppo_trainer/base.yaml")
    config=OmegaConf.load("pettingllms/config/code/code_test.yaml")
    _ = test_server_manager_simple(trainer_config,config)

if __name__ == "__main__":
    test_rollout_engine_simple()