import asyncio
import concurrent.futures
import logging
import time
import json
import traceback
import uuid
from tqdm.asyncio import tqdm
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
from pettingllms.trainer.utils import convert_prompt_to_dpr, convert_dpr_to_response
from pettingllms.utils.logger_config import get_multi_logger
from threading import Thread
from pettingllms.trainer.utils import build_reverse_mapping
from pettingllms.multi_agent_env.code.code_utils import get_ray_docker_worker_cls


logger = logging.getLogger(__name__)




class MultiAgentsExecutionEngine:
    def _load_config_parameters(self):
        """Load parameters from config with fallback to defaults"""
        
        
        # Data configuration - direct access with fallbacks
        if hasattr(self.config, 'data') and self.config.data is not None:
            self.max_prompt_length = getattr(self.config.data, 'max_prompt_length', 1024)
            self.max_response_length = getattr(self.config.data, 'max_response_length', 1024)
        else:
            self.max_prompt_length = 1024
            self.max_response_length = 1024
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
            self.gen_n_samples = getattr(self.config.data, 'gen_n_samples', 1)
        else:
            self.sample_temperature = 0.7
            self.gen_batch_size = 64
            self.gen_n_samples = 1
            
        # Timeout configuration - direct access with fallbacks
        if hasattr(self.config, 'timeout') and self.config.timeout is not None:
            self.generate_timeout = getattr(self.config.timeout, 'generate_timeout', 60.0)
            self.step_timeout = getattr(self.config.timeout, 'step_timeout', 30.0)
        else:
            self.generate_timeout = 120.0  # 60 seconds for generation
            self.step_timeout = 20.0      # 30 seconds for environment step
        
    def __init__(
        self,
        config,
        tokenizer_dict=None,
        processor_dict=None,
        server_manager_dict=None,
        agent_policy_mapping=None,
        env_args=None,
        max_workers=1000,
        **kwargs,
    ):
        

        self.config = config
        self.tokenizer_dict = tokenizer_dict
        self.processor_dict = processor_dict or {}
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.env_args = env_args or {}
        self.max_workers = max_workers
        
        self.multi_logger = get_multi_logger()
        
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
        self.mode=self.config.mode
        
        
        # agent_configs is a dict with keys like "agent_0", "agent_1"
        # Each agent config has a "name" field that corresponds to turn_order
        self.agent_configs_raw = self.config.agent_policy_configs.agent_configs
        
        # Create a mapping from agent name to agent config
        self.agent_config_dict = {}
        for agent_key, agent_config in self.agent_configs_raw.items():
            agent_name = agent_config.name
            self.agent_config_dict[agent_name] = agent_config
        self.sample_mode=self.config.sample_mode
        # Calculate total sample_num and sample_num_list based on turn_order
        self.sample_num = 1
        self.sample_num_list = []
        if self.sample_mode=="env":
            self.sample_num=self.gen_n_samples
        else:
            for agent_name in self.turn_order:
                if agent_name in self.agent_config_dict and hasattr(self.agent_config_dict[agent_name], 'sample_num'):
                    agent_sample_num = self.agent_config_dict[agent_name].sample_num
                    self.sample_num *= agent_sample_num
                    self.sample_num_list.append(agent_sample_num)
                else:
                    self.sample_num *= 1
                    self.sample_num_list.append(1)
        print(f"sample_num: {self.sample_num}")
        print(f"sample_num_list: {self.sample_num_list}")
        print(f"agent_config_dict keys: {list(self.agent_config_dict.keys())}")
        #self.env_worker_size=self.gen_batch_size*self.sample_num

        

        #self._init_agents()

        #self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        # rollout_engine_dict is not maintained in this class to avoid referencing a non-existent attribute
        self.server_manager_dict = server_manager_dict or {}
        self.chat_parser_dict={}
        #self._init_agents_and_envs()
        #for key,value in self.router_dict.items():
        #    self.chat_parser_dict[key]=ChatTemplateParser.get_parser(self.tokenizer_dict[key], disable_thinking=False)
        

    def init_agents_and_envs(self):
       
        # Check for batched_init in config.env
        if hasattr(self.config, 'env') and self.config.env is not None:
            batched_init = getattr(self.config.env, 'batched_init', True)
        else:
            batched_init = getattr(self.config, 'batched_init', True)
        if batched_init == False:
            with multiprocessing.Pool(self.n_cpu // 4) as pool: # Only use 1/4 of the cores to avoid conflicts
                func=partial(self.__init_one_env_instance, env_args=self.env_args)
                self.envs = pool.map(func, range(self.gen_batch_size*self.sample_num))
           
            
        else:
            self.env_workers = None
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
                RayDockerWorker = get_ray_docker_worker_cls()
                if RayDockerWorker is not None:
                    num_workers = self.gen_batch_size * self.sample_num
                    self.env_workers = [RayDockerWorker.remote(_) for _ in range(num_workers)]
                    self.multi_logger.log_rollout_summary(
                        -1, -1, "env_workers",
                        f"init {num_workers} env workers"
                    )
            except Exception:
                self.env_workers = None

            self.env_batch_class=ENV_BATCH_CLASSES[self.env_name]
            self.rollout_idx_list=range(self.gen_batch_size*self.sample_num)
            self.envs_batch=self.env_batch_class(
                env_idx_list=range(self.gen_batch_size),
                rollout_idx_list=range(self.gen_batch_size*self.sample_num),
                samples=self.sample_num,
                max_turns=self.max_turns,
                config=self.config,
                mode=self.mode
            )
            self.envs=self.envs_batch.env_list
        

        
        
        # Initialize the agent group for each rollout
        self.agent_groups_dict = {}
        self.env_rollout_mapping={}
        self.agent_rollout_mapping={}
        
        if self.sample_mode=="env":

            for env_idx in range(self.gen_batch_size):
                self.env_rollout_mapping[env_idx] = [_ for _ in range(env_idx*self.sample_num, (env_idx+1)*self.sample_num)]
            for agent_idx, agent_name in enumerate(self.turn_order):
                agent_class = self.agent_class_list[agent_idx]
                self.agent_groups_dict[agent_name] = [agent_class(env_idx=rollout_idx//self.sample_num, agent_sample_idx=rollout_idx%self.sample_num) for rollout_idx in range(self.gen_batch_size*self.sample_num)]
        else:
            for agent_idx, agent_name in enumerate(self.turn_order):
                agent_class = self.agent_class_list[agent_idx]
                # Get sample_num from agent_config_dict
                if agent_name in self.agent_config_dict and hasattr(self.agent_config_dict[agent_name], 'sample_num'):
                    agent_sample_num = self.agent_config_dict[agent_name].sample_num
                else:
                    agent_sample_num = 1
                self.agent_groups_dict[agent_name] = [[agent_class(env_idx=env_idx, agent_sample_idx=sample_idx) for sample_idx in range(agent_sample_num)] for env_idx in range(self.gen_batch_size)]

            for env_idx in range(self.gen_batch_size):
                self.env_rollout_mapping[env_idx] = [_ for _ in range(env_idx*self.sample_num, (env_idx+1)*self.sample_num)]
            self.agent_rollout_mapping = build_reverse_mapping(self.turn_order, self.sample_num_list, batch_size=self.gen_batch_size)
    def __init_one_env_instance(self, rollout_idx, env_args):
        env = self.env_class( env_idx=rollout_idx % self.gen_batch_size,rollout_idx=rollout_idx,max_turns=self.max_turns, **env_args)
        
        return env
                   
            
    async def generate_single_rollout(self, rollout_idx):
        """
        Generate a single rollout, adapted for multi-agent interaction in the code testing environment.
        
        Args:
            env: Code testing environment instance
            timing_raw: Timing record dictionary
            meta_info: Meta information
            
        Returns:
            DataProto: DataProto object containing trajectory data
        """
        trajectory_per_task_dict = {}
        env_idx = rollout_idx// self.sample_num
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()


        env = self.env_list[rollout_idx] if hasattr(self, "env_list") else self.envs[rollout_idx]
        agent_group = []
        for agent_name in self.turn_order:
            agent_group.append(self.agent_groups_dict[agent_name][rollout_idx])


        self.multi_logger.log_async_event(env_idx,
            rollout_idx, "rollout_start", 
            f"Starting multi-turn conversation, max turns: {self.max_turns}",
            {
                "turn_order": self.turn_order,
                "available_tokenizers": list(self.tokenizer_dict.keys()),
                "available_server_managers": list(self.server_manager_dict.keys())
            }
        )
        
        for turn_idx in range(self.max_turns):
            self.multi_logger.log_async_event(
                env_idx, rollout_idx, "turn_start",
                f"Starting turn {turn_idx + 1}",
                {"turn_idx": turn_idx + 1}
            )
            
            for agent_idx, agent_name in enumerate(self.turn_order):
                current_agent = agent_group[agent_idx]
                current_agent.update_from_env(env)
                prompt = current_agent.current_prompt
                
                # Select the policy name; if not provided, fall back to any available policy
                policy_name = self.agent_policy_mapping.get(agent_name) if self.agent_policy_mapping else None
                if policy_name is None:
                    policy_name = next(iter(self.server_manager_dict.keys())) if self.server_manager_dict else next(iter(self.tokenizer_dict.keys()))
                

                # Convert to DataProto format
                dpr_prompt = convert_prompt_to_dpr( self.tokenizer_dict[policy_name], 
                        self.chat_parser_dict.get(policy_name), 
                        prompt, 
                        self.max_prompt_length,
                       multi_modal=False
                   )
                
                # Generate responses
                generation_success = True
                output_dpr = None
                response_str = None
                try:
                    output_dpr,response_str = await asyncio.wait_for(
                        self.server_manager_dict[policy_name].generate(
                            rollout_idx=rollout_idx, 
                            turn_idx=turn_idx, 
                            agent_idx=agent_idx,
                            dpr_prompt=dpr_prompt, 
                            application_id=str(uuid.uuid4()),
                            tokenizer=self.tokenizer_dict[policy_name],
                            env_idx=rollout_idx // self.sample_num,
                            policy_name=policy_name,
                            timeout=self.generate_timeout
                        ),
                        timeout=self.generate_timeout
                    )
                except asyncio.TimeoutError:
                    self.multi_logger.log_env_agent_info(
                        env_idx, rollout_idx, turn_idx + 1, agent_name,
                        f"❌ Generation timed out after {self.generate_timeout}s",
                        {"error": "timeout", "timeout_seconds": self.generate_timeout}
                    )
                    generation_success = False
                    output_dpr = None
                    response_str = None
                except Exception as e:
                    self.multi_logger.log_env_agent_info(
                        env_idx, rollout_idx, turn_idx + 1, agent_name,
                        f"Failed to generate response: {e}",
                        {"error": str(e), "traceback": traceback.format_exc()}
                    )
                    generation_success = False
                    output_dpr = None
                    response_str = None
               
                # Skip processing if generation failed
                if not generation_success:
                    continue
                
                # Only proceed if we have a valid response
                if response_str is None:
                    continue
                    
                current_agent.update_from_model(response_str)
                
                step_success = True
                try:
                    await asyncio.wait_for(
                        current_agent.step(self.envs[rollout_idx],env_worker=self.env_workers[rollout_idx]),
                        timeout=self.step_timeout
                    )
                except asyncio.TimeoutError:
                    self.multi_logger.log_env_agent_info(
                        env_idx, rollout_idx, turn_idx + 1, agent_name,
                        f"❌ Environment step timed out after {self.step_timeout}s",
                        {"error": "timeout", "timeout_seconds": self.step_timeout}
                    )
                    step_success = False
                
                # Skip processing if step failed
                if not step_success:
                    continue

                # Only process trajectory if both generation and step succeeded
                if output_dpr is not None:
                    output_dpr.non_tensor_batch["reward"] = [current_agent.agent_reward]
                    output_dpr.non_tensor_batch["agent_name"] = [agent_name]  # Add agent name for metrics tracking
               
                    if trajectory_per_task_dict[policy_name].batch is None:
                        # If empty, assign directly
                        trajectory_per_task_dict[policy_name] = output_dpr
                    else:
                        # Use concat instead of union, because each response content is different
                        trajectory_per_task_dict[policy_name] = DataProto.concat([
                            trajectory_per_task_dict[policy_name], 
                            output_dpr
                        ])
                    
                
         
                self.multi_logger.log_env_agent_info(
                    env_idx, rollout_idx, turn_idx + 1, agent_name,
                    "Trajectory information updated",
                    {
                        "agent_name": agent_name,
                        "agent_response": response_str,
                        "agent_reward_history": current_agent.reward_history,
                        "agent_action": str(current_agent.current_action),
                       # "env_state": env.state,
                    
                       
                    }
                )
                if agent_name == self.turn_order[-1]:
                    self.multi_logger.log_env_agent_info(
                        env_idx, rollout_idx, turn_idx + 1, agent_name,
                        "Trajectory information updated",
                        {
                            "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                          
                            "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.generated_test_vs_golden_code_match_ratio,
                       
                        }
                    )
        
        
            finish=True
            for agent in agent_group:
                if not agent.done:
                    finish=False
                    break
            if finish:
                agent_rewards={agent_name: agent.reward_history for agent_name, agent in zip(self.turn_order, agent_group)}
                
                self.multi_logger.log_rollout_summary(
                    env_idx, rollout_idx, "success!!",
                    f"Rollout {rollout_idx} completed successfully",
                    extra_data={
                        "turn_idx": turn_idx,
                        "agent_rewards": agent_rewards,
                        "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                        "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                    }
                )
                
                break
        agent_rewards={agent_name: agent.reward_history for agent_name, agent in zip(self.turn_order, agent_group)}
        self.multi_logger.log_rollout_summary(
                env_idx, rollout_idx, "rollout_complete",
                f"Rollout {rollout_idx} completed successfully",
                extra_data={
                    "turn_idx": 4,
                    "agent_rewards": agent_rewards,
                    "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                    "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                }
            )


        
       
        #trajectory_per_task_dict = self._assign_consistent_uids(trajectory_per_task_dict)
        
        return trajectory_per_task_dict

        
 
    async def generate_env_idx_rollout_agent_flow(self, env_idx, mode="train"):
        """
        Generate a single rollout, adapted for multi-agent interaction in the code testing environment.
        
        Args:
            env: Code testing environment instance
            timing_raw: Timing record dictionary
            meta_info: Meta information
            
        Returns:
            DataProto: DataProto object containing trajectory data
        """
       # init trajectory_per_task_dict
        trajectory_per_task_dict = {}
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()


        
        # Get the seed environment for this env_idx
        
        
        

        
        self.multi_logger.log_async_event(
            env_idx, self.env_rollout_mapping[env_idx][0], "rollout_start", 
            f"Starting multi-turn conversation for env {env_idx}, max turns: {self.max_turns}",
            {
                "env_idx": env_idx,
                "turn_order": self.turn_order,
                "available_tokenizers": list(self.tokenizer_dict.keys()),
                "available_server_managers": list(self.server_manager_dict.keys())
            }
        )
       
        for turn_idx in range(self.max_turns):
            self.multi_logger.log_async_event(
                env_idx, self.env_rollout_mapping[env_idx][0], "turn_start",
                f"Starting turn {turn_idx + 1}",
                {"turn_idx": turn_idx + 1}
            )
            
            # agent upate from envs and generate response
            for agent_idx, agent_name in enumerate(self.turn_order):
                if agent_name in self.agent_config_dict and hasattr(self.agent_config_dict[agent_name], 'sample_num') and mode=="train":
                    agent_sample_num = self.agent_config_dict[agent_name].sample_num  
                else:
                    agent_sample_num = 1
                async def process_single_sample(agent_name,sample_idx,application_id):
                    current_agent = self.agent_groups_dict[agent_name][env_idx][sample_idx]
                    current_agent.update_from_env(self.envs[self.agent_rollout_mapping[env_idx][agent_name][sample_idx][0]])
                    prompt = current_agent.current_prompt
                
                    # Select the policy name; if not provided, fall back to any available policy
                    policy_name = self.agent_policy_mapping.get(agent_name) if self.agent_policy_mapping else None
                    if policy_name is None:
                        policy_name = next(iter(self.server_manager_dict.keys())) if self.server_manager_dict else next(iter(self.tokenizer_dict.keys()))
                

                    # Convert to DataProto format
                    dpr_prompt = convert_prompt_to_dpr(self.tokenizer_dict[policy_name], 
                            self.processor_dict.get(policy_name) if isinstance(self.processor_dict, dict) else None,
                            prompt, 
                            self.max_prompt_length,
                        multi_modal=False
                    )

                    

                    
                    # Generate responses
                    generation_success = True
                    output_dpr = None
                    response_str = None
                    self.multi_logger.log_model_interaction(
                        env_idx, self.agent_rollout_mapping[env_idx][agent_name][sample_idx][0], policy_name,
                        f"Generating response for agent {agent_name} (sample {sample_idx})",
                        {
                            "agent_name": agent_name,
                            "sample_idx": sample_idx,
                            "rollout_id_list": self.agent_rollout_mapping[env_idx][agent_name][sample_idx],
                            "prompt": prompt,
                            
                        }
                    )
                    
                    output_dpr,response_str = await self.server_manager_dict[policy_name].generate(
                                dpr_prompt, 
                                application_id=application_id,
                                tokenizer=self.tokenizer_dict[policy_name],
                                env_idx=env_idx,
                                rollout_idx=self.agent_rollout_mapping[env_idx][agent_name][sample_idx][0],
                                policy_name=policy_name,
                                timeout=self.generate_timeout
                            )
                    self.multi_logger.log_model_interaction(
                        env_idx, self.agent_rollout_mapping[env_idx][agent_name][sample_idx][0], policy_name,
                        f"Got response for agent {agent_name} (sample {sample_idx})",
                        {
                            "agent_name": agent_name,
                            "sample_idx": sample_idx,
                            "rollout_id_list": self.agent_rollout_mapping[env_idx][agent_name][sample_idx],
                            "response_str": response_str,
                            
                        }
                    )
                    current_agent.update_from_model(response_str)
                    agent_rollout_list=[]
                    self.multi_logger.log_model_interaction(
                        -1, -1, "rollout_start",
                        f"Starting multi-turn conversation for env {env_idx}, max turns: {self.max_turns}",
                        {   
                            "agent_name": agent_name,
                            "sample_idx": sample_idx,
                            "rollout_idx": self.agent_rollout_mapping[env_idx][agent_name][sample_idx],
                        }
                    )
                    # env step to combine env groups - run all rollout steps concurrently
                    step_tasks = []
                    task_meta = []  # (rollout_idx, env)
                    for rollout_idx in self.agent_rollout_mapping[env_idx][agent_name][sample_idx]:
                        env = self.envs[rollout_idx]
                        agent_rollout_list.append(env)
                       
                        step_tasks.append(
                            asyncio.wait_for(
                                env.step(agent_name, current_agent.current_action,env_worker=self.env_workers[rollout_idx]),
                                timeout=self.step_timeout
                            )
                        )
                       
                        task_meta.append((rollout_idx, env))


                    # Await all steps to finish before proceeding
                    step_results = await asyncio.gather(*step_tasks, return_exceptions=True)
                

                    # Log failures but do not early-return until all steps are attempted
                    failure_count = 0
                    for (rollout_idx, env), result in zip(task_meta, step_results):
                        if isinstance(result, asyncio.TimeoutError):
                            self.multi_logger.log_rollout_summary(
                                env_idx, rollout_idx, turn_idx + 1, agent_name,
                                f"❌ Environment step timed out after {self.step_timeout}s",
                                {"error": "timeout", "timeout_seconds": self.step_timeout}
                            )
                            failure_count += 1
                        elif isinstance(result, Exception):
                            self.multi_logger.log_env_agent_info(
                                env_idx, rollout_idx, turn_idx + 1, agent_name,
                                f"Failed to execute environment step: {result}",
                                {"error": str(result), "traceback": traceback.format_exc()}
                            )
                            failure_count += 1

                    # If all steps failed, skip processing for this sample
                    if failure_count == len(step_results):
                        return None
                            
                    # TODO: calculate reward for test generator
                    current_agent.calculate_reward(agent_rollout_list)
                    for rollout_idx in self.agent_rollout_mapping[env_idx][agent_name][sample_idx]:
                        env=self.envs[rollout_idx]
                        self.multi_logger.log_env_agent_info(
                            env_idx, rollout_idx, turn_idx + 1, agent_name,
                            f"Agent {agent_name} (sample {sample_idx}) completed action",
                            {
                                "agent_name": agent_name,
                                "sample_idx": sample_idx,
                                "agent_response": response_str[:200] + "..." if len(response_str) > 200 else response_str,  # Truncate long responses
                                "agent_reward": current_agent.agent_reward,
                                "agent_action":  str(current_agent.current_action),  # Truncate long actions
                                "reward_history_dict": current_agent.reward_history,
                                "env_state": env.state
                            }
                        )
                        # Log environment state only once per turn after all agents have acted
                        if agent_name == self.turn_order[-1]:
                            self.multi_logger.log_env_agent_info(
                                env_idx, rollout_idx, turn_idx + 1, "environment",
                                "Environment state after all agents acted",
                                {
                                    "env_state_summary": {
                                        "done": getattr(env, 'done', False),
                                        "state_type": str(type(env.state)),
                                        "has_mismatch_cases": hasattr(env.state, 'golden_truth_test_vs_generated_code_mismatch_cases'),
                                        "has_match_cases": hasattr(env.state, 'golden_truth_test_vs_generated_code_match_cases')
                                    }
                                }
                            )
                    
                    return {
                        'sample_idx': sample_idx,
                        'current_agent': current_agent,
                        #'env_list': env_list,
                        'output_dpr': output_dpr,
                        'response_str': response_str,
                        'policy_name': policy_name
                    }
                
                
                
                # 并行执行所有sample
                sample_tasks = [
                    asyncio.create_task(process_single_sample(agent_name,sample_idx,application_id=uuid.uuid4()), name=f"sample_{sample_idx}")
                    for sample_idx in range(agent_sample_num)
                ]
                
                
                sample_results = await asyncio.gather(*sample_tasks, return_exceptions=True)
                
            
                for result in sample_results:
                    if result is None or isinstance(result, Exception):
                        continue  
                    
                    current_agent = result['current_agent']
                    #env_list = result['env_list']
                    output_dpr = result['output_dpr']
                    policy_name = result['policy_name']
                    # Only process trajectory if both generation and step succeeded
                    if output_dpr is not None:
                        output_dpr.non_tensor_batch["reward"] = [current_agent.agent_reward]
                        output_dpr.non_tensor_batch["agent_name"] = [agent_name]  # Add agent name for metrics tracking
                    
                        if trajectory_per_task_dict[policy_name].batch is None:
                            # If empty, assign directly
                            trajectory_per_task_dict[policy_name] = output_dpr
                        else:
                            # Use concat instead of union, because each response content is different
                            trajectory_per_task_dict[policy_name] = DataProto.concat([
                                trajectory_per_task_dict[policy_name], 
                                output_dpr
                            ])
                    
                
               
            env_rollout_list=[]
            last_agent_name=self.turn_order[-1]
            current_agent=self.agent_groups_dict[last_agent_name][env_idx][0]
            for rollout_idx in self.env_rollout_mapping[env_idx]:
                env_rollout_list.append(self.envs[rollout_idx])
            new_env_seed_idx=current_agent.select_env(env_rollout_list)
            if new_env_seed_idx==-1:
                new_env_seed=None
                break
            else:
                new_env_seed=self.envs[new_env_seed_idx]
                for rollout_idx in self.env_rollout_mapping[env_idx]:
                    self.envs[rollout_idx]=new_env_seed
                # Check if environment is done (e.g., all tests passed)
                if hasattr(new_env_seed, 'done') and new_env_seed.done:
                    termination_reason = getattr(new_env_seed, 'termination_reason', 'environment_done')
                    self.multi_logger.log_async_event(
                        env_idx, self.env_rollout_mapping[env_idx][0], "early_termination",
                        f"Environment completed early due to: {termination_reason}",
                        {"termination_reason": termination_reason, "completed_turn": turn_idx + 1, "completed_agent": agent_name}
                    )
                    
                    # Calculate final rewards for all agents
                    agent_rewards = {}
                    for agent_name_final in self.turn_order:
                        # Get the last sample of each agent for final reward
                        final_agent = self.agent_groups_dict[agent_name_final][env_idx][-1]
                        agent_rewards[agent_name_final] = final_agent.agent_reward
                    
                    self.multi_logger.log_rollout_summary(
                        rollout_idx=rollout_idx,
                        agent_rewards=agent_rewards,
                        termination_reason=termination_reason,
                        extra_data={
                            "completed_turn": turn_idx + 1,
                            "completed_agent": agent_name,
                            "env_state": new_env_seed.state,
                            "mismatch_cases_count": len(new_env_seed.state.golden_truth_test_vs_generated_code_mismatch_cases) if hasattr(new_env_seed.state, 'golden_truth_test_vs_generated_code_mismatch_cases') else 0,
                            "match_cases_count": len(new_env_seed.state.golden_truth_test_vs_generated_code_match_cases) if hasattr(new_env_seed.state, 'golden_truth_test_vs_generated_code_match_cases') else 0,
                            "reward_history_dict": current_agent.reward_history
                        }
                    )

                    
                    

                    # For the main agent action log, use one of the rollout_idx from env_rollout_mapping
                    
                      
                
                
                

     
        
        self.multi_logger.log_async_event(
            env_idx, self.env_rollout_mapping[env_idx][0], "rollout_complete",
            f"Environment {env_idx} rollout completed successfully",
            {}
        )
        
        # 为 trajectory_per_task_dict 分配一致的 uid
        #trajectory_per_task_dict = self._assign_consistent_uids(trajectory_per_task_dict)
        
        return trajectory_per_task_dict

    async def generate_multiple_rollouts_concurrent(self, rollout_indices):
        max_concurrent_tasks=self.max_workers
        
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def run_env_rollouts_with_semaphore(rollout_idx):
            try:
                result = await self.generate_single_rollout(rollout_idx)
                return result
            except Exception as e:
                # Log the error but don't raise it, let the caller handle it
                self.multi_logger.log_async_event(
                    -1, rollout_idx, "rollout_error",
                    f"Rollout {rollout_idx} failed: {e}",
                    {"error": str(e), "rollout_idx": rollout_idx}
                )
                # Return empty result instead of raising
                empty_result = {}
                for policy_name in self.tokenizer_dict.keys():
                    empty_result[policy_name] = DataProto()
                return empty_result
        
        tasks = [
            asyncio.create_task(
                run_env_rollouts_with_semaphore(rollout_idx), 
                name=f"env_{rollout_idx}_rollouts"
            )
            for rollout_idx in range(self.gen_batch_size*self.sample_num)
        ]
        
        aggregated_results = {}
        for policy_name in self.tokenizer_dict.keys():
            aggregated_results[policy_name] = DataProto()
        
        completed_count = 0
        failed_count = 0
        
  
        task_pbar = tqdm(total=len(tasks), desc="Rollouts", position=1, leave=False)
        
        try:
        
            for completed_task in asyncio.as_completed(tasks):
                try:
                    
                    rollout_result = await completed_task
        
                    for policy_name, policy_data in rollout_result.items():
                        print(policy_data)
                        if policy_data.batch is not None:  
                            if aggregated_results[policy_name].batch is None:
                                aggregated_results[policy_name] = policy_data
                            else:
                                aggregated_results[policy_name] = DataProto.concat([
                                    aggregated_results[policy_name], 
                                    policy_data
                                ])
                    
                    completed_count += 1
                    
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)})")
                except Exception as e:
                    failed_count += 1
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)}, {failed_count} failed)")
                    
                    self.multi_logger.log_async_event(
                        -1, -1, "task_error",
                        f"Task failed with error: {e}",
                        {
                            "failed_count": failed_count,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "traceback": traceback.format_exc()
                        }
                    )
                    
                    continue
                    
        except Exception as e:
            # Log Ray status when encountering errors
            self.multi_logger.log_ray_status(context="during_error")
            
            self.multi_logger.log_async_event(
                -1, -1, "concurrent_batch_error",
                f"Concurrent execution encountered error: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            for task in tasks:
                if not task.done():
                    task_name = task.get_name()
                    self.multi_logger.log_async_event(
                        -1, -1, "task_cancel",
                        f"Cancelling task {task_name}"
                    )
                    task.cancel()
            raise

        task_pbar.close()
        
        self.multi_logger.log_async_event(
            -1, -1, "concurrent_batch_complete",
            "Concurrent execution completed",
            {
                "successfully_processed": completed_count,
                "total_env_groups": len(tasks),
                "total_rollouts": len(rollout_indices),
                "failed": failed_count,
                "success_rate": f"{completed_count}/{len(tasks)}",
                "aggregated_policies": list(aggregated_results.keys()),
            }
        )
        
        # Log Ray status after concurrent execution
        self.multi_logger.log_ray_status(context="after_concurrent_batch")
        
        # 为 aggregated_results 分配基于 rollout_idx, turn_idx, agent_idx 的相同 uid
        aggregated_results = self._assign_consistent_uids(aggregated_results)
        
        return aggregated_results
    
    def _assign_consistent_uids(self, aggregated_results, filter_ratio=0.0):
        import uuid
        import numpy as np
        from collections import defaultdict
        
        uid_mapping = {}
        all_rewards = []
        uid_reward_groups = defaultdict(list)  # 用于过滤: uid -> [(sample_index, policy_name, reward_value)]
        
        # 分配UID并收集数据
        for policy_name, data_proto in aggregated_results.items():
            if data_proto.batch is None or len(data_proto) == 0:
                continue

            non_tensor_batch = data_proto.non_tensor_batch
            
            # 检查必要字段，缺失则跳过
            if not all(key in non_tensor_batch for key in ["rollout_idx", "turn_idx", "agent_idx"]):
                data_proto.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(data_proto))], dtype=object
                )
                continue
            
            rollout_indices = non_tensor_batch["rollout_idx"]
            turn_indices = non_tensor_batch["turn_idx"] 
            agent_indices = non_tensor_batch["agent_idx"]
            rewards = non_tensor_batch.get("reward", [])
            
            uids = []
            for i in range(len(data_proto)):
                # 生成UID key
                key = (rollout_indices[i]//self.sample_num, (turn_indices[i]>1), agent_indices[i])
                if key not in uid_mapping:
                    uid_mapping[key] = str(uuid.uuid4())
                uid = uid_mapping[key]
                uids.append(uid)
                
                # 收集reward数据
                if len(rewards) > 0 and filter_ratio > 0:
                    reward_val = float(rewards[i]) if rewards[i] is not None else 0.0
                    uid_reward_groups[uid].append((i, policy_name, reward_val))
                
                if len(rewards) > 0:
                    reward_val = float(rewards[i]) if rewards[i] is not None else 0.0
                    all_rewards.append(reward_val)
            
            data_proto.non_tensor_batch["uid"] = np.array(uids, dtype=object)
        
        # 执行过滤
        sample_to_remove = set()
        if filter_ratio > 0:
            for uid, samples in uid_reward_groups.items():
                if len(samples) > 1:
                    # 计算偏差并排序
                    rewards_in_group = [s[2] for s in samples]
                    group_mean = np.mean(rewards_in_group)
                    samples_with_deviation = [(s[0], s[1], abs(s[2] - group_mean)) for s in samples]
                    samples_with_deviation.sort(key=lambda x: x[2], reverse=True)
                    
                    # 移除偏差最大的部分
                    num_to_remove = int(len(samples_with_deviation) * filter_ratio)
                    for i in range(num_to_remove):
                        sample_idx, policy_name, _ = samples_with_deviation[i]
                        sample_to_remove.add((policy_name, sample_idx))
        
        # 应用过滤
        if sample_to_remove:
            for policy_name, data_proto in aggregated_results.items():
                if data_proto.batch is None:
                    continue
                
                keep_indices = [i for i in range(len(data_proto)) 
                               if (policy_name, i) not in sample_to_remove]
                
                if len(keep_indices) < len(data_proto):
                    # 过滤batch数据
                    if data_proto.batch:
                        for key, value in data_proto.batch.items():
                            if hasattr(value, '__getitem__'):
                                data_proto.batch[key] = value[keep_indices]
                    
                    # 过滤non_tensor数据
                    if data_proto.non_tensor_batch:
                        for key, value in data_proto.non_tensor_batch.items():
                            if isinstance(value, np.ndarray):
                                data_proto.non_tensor_batch[key] = value[keep_indices]
                            elif hasattr(value, '__getitem__'):
                                data_proto.non_tensor_batch[key] = [value[i] for i in keep_indices]
        
        # 记录统计信息
        if all_rewards:
            summary = {
                "total_samples": len(all_rewards),
                "mean_reward": float(np.mean(all_rewards)),
                "std_reward": float(np.std(all_rewards)),
                "filtered_samples": len(sample_to_remove) if filter_ratio > 0 else 0
            }
            
            self.multi_logger.log_async_event(
                -1, -1, "reward_summary",
                f"UID assignment and reward statistics complete",
                summary
            )
        
        return aggregated_results
        
       
    
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
    
    test_multi_logger = get_multi_logger()
    test_multi_logger.log_async_event(
        -1, -1, "test_start",
        "Starting test_server_manager_simple",
        {
            "trainer_config_type": str(type(trainer_config)),
            "config_type": str(type(config)),
            "model_path": trainer_config.actor_rollout_ref.model.path
        }
    )
    
    # Log Ray status before initialization
    test_multi_logger.log_ray_status(context="before_ray_init")

    
    if not ray.is_initialized():
        # 先从注册文件读取地址，其次使用 address=auto，最后本地起实例
        import json
        registry_path = os.environ.get("VLLM_REGISTRY_PATH", "logs/ray_vllm_registry.json")
        ray_address = os.environ.get("RAY_ADDRESS", None)
        ray_namespace = os.environ.get("RAY_NAMESPACE", None)
        if ray_address is None and os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)
                    ray_address = registry.get("ray_address")
                    if ray_namespace is None:
                        ray_namespace = registry.get("namespace")
                    if ray_address:
                        print(f"Loaded Ray address from registry: {ray_address} (namespace={ray_namespace})")
            except Exception as e:
                print(f"Failed to load registry at {registry_path}: {e}")

        try:
            if ray_address:
                ray.init(address=ray_address, namespace=ray_namespace)
                print(f"Connected to Ray cluster at {ray_address} (namespace={ray_namespace})")
            else:
                ray.init(address="auto", namespace=ray_namespace)
                print(f"Connected to existing Ray cluster via address=auto (namespace={ray_namespace})")
        except Exception:
            # Use all available CPUs instead of hardcoded 4
            import multiprocessing
            num_cpus = multiprocessing.cpu_count()
            ray.init(num_cpus=224)
            print(f"Started a local Ray instance for testing with {num_cpus} CPUs")
    
    # Log Ray status after initialization
    test_multi_logger.log_ray_status(context="after_ray_init")
    
    server_list=[]
    print(f"begin to init server list (reuse existing named actor)")
    # 严格复用：直接查找已存在的命名 actor，避免任何新引擎启动
    try:
        server = ray.get_actor("async_llm_server")
    except Exception as e:
        raise RuntimeError(
            "未找到已启动的 vLLM 命名 actor 'async_llm_server'。请先运行启动脚本 "
            "pettingllms/scripts/launch_vllm_servers.py 并确保与当前连接的 Ray 集群一致。"
        ) from e

    # 获取地址仅用于日志
    try:
        addr = ray.get(server.get_server_address.remote())
        print(f"Found existing async_llm_server at {addr}")
    except Exception:
        addr = None

    server_list = [server]
    server_addresses = [addr]
    print(f"Reused {len(server_list)} server(s): {[s is not None for s in server_list]}")    

    test_multi_logger.log_async_event(
        -1, -1, "server_manager_init_start", 
        f"Starting to init server manager for server"
    )
    from pettingllms.trainer.utils import AsyncLLMServerManager
    from verl.utils import hf_tokenizer



    model_path_local =trainer_config.actor_rollout_ref.model.path
    tokenizer_local = hf_tokenizer(model_path_local, trust_remote_code=True)
    server_manager = AsyncLLMServerManager(config=trainer_config, server_handles=server_list)
    
    
    tokenizer_dict = {"code_generator": tokenizer_local}
    server_manager_dict={}
    server_manager_dict["code_generator"]=server_manager
    

    test_multi_logger.log_async_event(
        -1, -1, "multi_agent_engine_init_start",
        "Initializing Multi-Agent Execution Engine",
        {
            "tokenizer_dict_keys": list(tokenizer_dict.keys()),
            "server_manager_dict_keys": list(server_manager_dict.keys())
        }
    )
    
    # Fix: Pass correct tokenizer_dict
    multi_agent_execution_engine = MultiAgentsExecutionEngine(
        config=config, 
        tokenizer_dict=tokenizer_dict, 
        server_manager_dict=server_manager_dict
    )
    multi_agent_execution_engine.init_agents_and_envs()
    
    test_rollout_indices = range(len(multi_agent_execution_engine.envs))   
    test_multi_logger.log_async_event(
        -1, -1, "concurrent_test_start",
        "Testing concurrent rollout execution with class method",
        {"test_rollout_count": len(list(test_rollout_indices))}
    )
    
    try:
        concurrent_results = asyncio.run(
            multi_agent_execution_engine.generate_multiple_rollouts_concurrent(
                test_rollout_indices
            )
        )
        
    except Exception as e:
        test_multi_logger.log_async_event(
            -1, -1, "concurrent_test_error",
            "Class method concurrent execution failed",
            {"error": str(e), "traceback": traceback.format_exc()}
        )



       

    return None



import hydra
from omegaconf import DictConfig

def test_rollout_engine_simple(config_path=None):
    from omegaconf import OmegaConf
    import sys
    import argparse
    
    if config_path is None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Multi-Agent Execution Engine')
        parser.add_argument('--config', '-c', type=str, 
                          default="pettingllms/config/code/code_eval.yaml",
                          help='Path to config file')
        parser.add_argument('--trainer_config', '-t', type=str,
                          default="pettingllms/config/code/ppo_trainer/eval.yaml", 
                          help='Path to trainer config file')
        args = parser.parse_args()
        config_path = args.config
        trainer_config_path = args.trainer_config
    else:
        trainer_config_path = "pettingllms/config/code/ppo_trainer/eval.yaml"
    
    trainer_config = OmegaConf.load(trainer_config_path)
    config = OmegaConf.load(config_path)
    _ = test_server_manager_simple(trainer_config, config)

@hydra.main(config_path="../config/code", config_name="code_eval", version_base=None)
def run_benchmark_with_hydra(config: DictConfig):
    """使用 Hydra 运行 benchmark，可以通过命令行覆盖任何配置参数"""
    from omegaconf import OmegaConf
    
    # 加载 trainer 配置
    trainer_config = OmegaConf.load("pettingllms/config/code/ppo_trainer/eval.yaml")
    
    print(f"运行 benchmark: {config.env.benchmark}")
    print(f"实验名称: {config.get('experiment_name', 'code_test')}")
    print("="*50)
    
    # 运行测试
    _ = test_server_manager_simple(trainer_config, config)

if __name__ == "__main__":
    # 如果直接运行该文件，使用 Hydra 版本
    run_benchmark_with_hydra()