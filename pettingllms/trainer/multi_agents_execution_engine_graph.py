import asyncio
import logging
import time
import json
import traceback
import uuid
from tqdm.asyncio import tqdm
import random
try:
    from verl.protocol import DataProto
except Exception:  # fallback when verl is a src tree: verl/verl/protocol.py
    from verl import DataProto
import torch
import numpy as np
from pettingllms.trainer.multiagentssys_register import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING, ENV_BATCH_CLASS_MAPPING, ENV_WORKER_CLASS_MAPPING
from functools import partial
import multiprocessing
from pettingllms.utils.performance import create_timer
import copy
from pettingllms.trainer.async_generate import convert_prompt_to_dpr, llm_async_generate
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.multi_agent_env.code.code_worker import get_ray_docker_worker_cls
from pettingllms.utils.openai import init_patch_context, patch_all, wrap_autogen_graph


logger = logging.getLogger(__name__)




class MultiAgentsExecutionEngine:
    def _load_config_parameters(self):
        self.max_prompt_length = getattr(self.config.training, 'max_prompt_length', 1024)
        self.max_response_length = getattr(self.config.training, 'max_response_length', 1024)
        self.turn_order = self.config.multi_agent_interaction.turn_order
        self.num_interacting_agents = self.config.multi_agent_interaction.num_interacting_agents
        self.parallel = getattr(self.config.multi_agent_interaction, 'parallel', False)
        self.generate_timeout = getattr(self.config.training, 'generate_timeout', 300.0)
        # Multi-modal support configuration
        self.enable_multimodal = getattr(self.config.training, 'enable_multimodal', False)
        if self.num_interacting_agents != len(self.turn_order):
            raise ValueError("num_interacting_agents must be equal to the length of turn_order")
          
        
    def __init__(
        self,
        config,
        ppo_trainer_config_dict=None,
        tokenizer_dict=None,
        processor_dict=None,
        server_address_dict=None,
        agent_policy_mapping=None,
        env_args=None,
        max_workers=1000,
        lora_differ_mode=False,
        agent_lora_mapping=None,
    ):
        
        # Initialize timer for this engine
        self.timer = create_timer("MultiAgentsExecutionEngine")
        self.timer.start("Initializing MultiAgentsExecutionEngine")

        self.config = config
        self.ppo_trainer_config_dict = ppo_trainer_config_dict or {}
        self.tokenizer_dict = tokenizer_dict
        self.processor_dict = processor_dict or {}
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.env_args = env_args or {}
        self.max_workers = max_workers
        self.lora_differ_mode = lora_differ_mode
        self.agent_lora_mapping = agent_lora_mapping or {}
        # Read parameters from config with fallback to defaults
        self.timer.checkpoint("Loading config parameters")
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
        self.experiment_name = self.config.training.experiment_name
        self.env_name = env_name
        self.env_class = ENV_CLASS_MAPPING[env_name]
        self.agent_class_list = [AGENT_CLASS_MAPPING[agent_name] for agent_name in self.turn_order]
        self.agent_configs_raw = self.config.agent_policy_configs.agent_configs
        self.agent_config_dict = {}
        for agent_key, agent_config in self.agent_configs_raw.items():
            agent_name = agent_config.name
            self.agent_config_dict[agent_name] = agent_config
        self.step_timeout = getattr(self.config.training, 'step_timeout', 150.0)
        print(f"agent_config_dict keys: {list(self.agent_config_dict.keys())}")
        self.server_address_dict = server_address_dict 
        self.chat_parser_dict={}
        self.rollout_latency_dict = {}
        self.timer.checkpoint("MultiAgentsExecutionEngine initialization completed")
        
        # Initialize patch context with engine attributes
        init_patch_context(
            server_address_dict=self.server_address_dict,
            tokenizer_dict=self.tokenizer_dict,
            ppo_trainer_config_dict=self.ppo_trainer_config_dict,
            agent_policy_mapping=self.agent_policy_mapping
        )
        
        if self.env_name in ENV_WORKER_CLASS_MAPPING:
                _worker_factory_or_cls = ENV_WORKER_CLASS_MAPPING[self.env_name]
                try:
                    # Get num_workers first to pass to worker factory for CPU calculation
                    num_workers = self.config.training.get("num_workers", 180)
                    # Try to call factory with num_workers parameter if it's callable
                    if callable(_worker_factory_or_cls):
                        try:
                            RayDockerWorker = _worker_factory_or_cls(num_workers=num_workers)
                        except TypeError:
                            # Fallback if factory doesn't accept num_workers
                            RayDockerWorker = _worker_factory_or_cls()
                    else:
                        RayDockerWorker = _worker_factory_or_cls
                except Exception as e:
                    print(f"Failed to create RayDockerWorker from mapping for env '{self.env_name}': {e}")
                    RayDockerWorker = None
        else:
            num_workers = self.config.training.get("num_workers", 180)
            RayDockerWorker = get_ray_docker_worker_cls(num_workers=num_workers)
        print("begin to create Ray docker workers")
        if RayDockerWorker is not None and hasattr(RayDockerWorker, "remote"):
            num_workers = self.config.training.get("num_workers", 180)
            self.num_workers = num_workers

            # Get GPU group ID for worker pool isolation
            import os
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                gpu_ids = sorted([g.strip() for g in cuda_visible.split(",") if g.strip()])
                self.gpu_group_id = f"gpu_{'_'.join(gpu_ids)}"
            else:
                self.gpu_group_id = "gpu_default"

            print(f"GPU group ID: {self.gpu_group_id}")
            self.timer.checkpoint(f"Creating {num_workers} Ray docker workers for GPU group {self.gpu_group_id}")
            self.env_workers = [RayDockerWorker.remote(idx) for idx in range(num_workers)]
        else:
            self.gpu_group_id = "gpu_default"
            print(f"RayDockerWorker is not available or invalid for env '{self.env_name}'. Skipping env workers initialization.")
        

    
    def patch_and_run_autogen_graph(self, graph_module_or_callable):
        """
        Patch and run autogen graph with local vLLM.
        
        Args:
            graph_module_or_callable: Module with main() or callable
            
        Returns:
            Wrapped callable
        """
        patch_all()
        
        if callable(graph_module_or_callable):
            graph_func = graph_module_or_callable
        else:
            graph_func = graph_module_or_callable.main
        
        return wrap_autogen_graph(graph_func)
    
    def get_graph_function(self):
        """
        Get the appropriate graph function from config.training.workflow_function.
        
        Uses AGENT_WORKER_FLOW_FUNCTIONS registry to look up the function by name.
        
        Returns:
            graph_func: The graph function to execute
            
        Raises:
            ValueError: If workflow_function not in config or not in registry
        """
        from pettingllms.trainer.multiagentssys_register import AGENT_WORKER_FLOW_FUNCTIONS
        
        # Get workflow_function name from config
        workflow_function_name = getattr(self.config, 'workflow_function', None)
        if workflow_function_name is None:
            raise ValueError("workflow_function not specified in config")
        
        # Look up in registry
        if workflow_function_name not in AGENT_WORKER_FLOW_FUNCTIONS:
            raise ValueError(
                f"workflow_function '{workflow_function_name}' not found in AGENT_WORKER_FLOW_FUNCTIONS. "
                f"Available functions: {list(AGENT_WORKER_FLOW_FUNCTIONS.keys())}"
            )
        
        graph_func = AGENT_WORKER_FLOW_FUNCTIONS[workflow_function_name]
        if graph_func is None:
            raise ValueError(f"Failed to load workflow_function '{workflow_function_name}' - import returned None")
        
        return graph_func
    
    async def run_autogen_graph_async(self, graph_module_or_callable):
        """Run autogen graph asynchronously."""
        wrapped = self.patch_and_run_autogen_graph(graph_module_or_callable)
        return await wrapped()

    def init_agents_and_envs(self,mode="train",step_idx=0):
        self.multi_logger = get_multi_logger(experiment_name=self.experiment_name)
        self.timer.checkpoint("Starting init_agents_and_envs")
        self.mode=mode
        self.success_rollout_idx_list_dict={}
        self.success_ave_turn_dict={}
        
        # Initialize enable_thinking mapping for each agent
        self.agent_enable_thinking = {}
        # Initialize enable_multimodal mapping for each agent
        self.agent_enable_multimodal = {}
        for agent_name in self.turn_order:
            agent_config = self.agent_config_dict.get(agent_name, None)
            # Read enable_thinking from agent config, default to False
            enable_thinking = getattr(agent_config, 'enable_thinking', False) if agent_config else False
            self.agent_enable_thinking[agent_name] = enable_thinking
            # Read enable_multimodal from agent config, fallback to global setting
            enable_multimodal = getattr(agent_config, 'enable_multimodal', self.enable_multimodal) if agent_config else self.enable_multimodal
            self.agent_enable_multimodal[agent_name] = enable_multimodal
            print(f"Agent '{agent_name}' enable_thinking: {enable_thinking}, enable_multimodal: {enable_multimodal}")
        
        if mode=="validate":
            self.sample_num=self.config.training.validate_sample_num
            self.gen_batch_size=1
            for agent_name in self.turn_order:
                self.success_rollout_idx_list_dict[agent_name]=[]
                self.success_ave_turn_dict[agent_name]=0
        else:
            self.sample_num=self.config.training.train_sample_num
            self.gen_batch_size=self.config.training.train_batch_size

        self.env_batch_class=ENV_BATCH_CLASS_MAPPING[self.env_name]
        env_indices=range(step_idx*self.gen_batch_size, (step_idx+1)*self.gen_batch_size)
        # Convert to list for safety
        env_indices_list = list(env_indices)
        self.envs_batch=self.env_batch_class(
            env_idx_list=range(self.gen_batch_size),
            rollout_idx_list=range(self.gen_batch_size*self.sample_num),
            env_indices=env_indices_list,
            samples=self.sample_num,
            max_turns=self.max_turns,
            config=self.config,
            mode=self.mode
        )
        self.envs=self.envs_batch.env_list
        self.gen_batch_size=len(self.envs)//self.sample_num
        self.env_idx_list=range(len(self.envs)//self.sample_num)
        self.rollout_idx_list=range(len(self.envs))
        self.env_rollout_mapping={}
        for env_idx in range(len(self.env_idx_list)):
            self.env_rollout_mapping[env_idx] = [_ for _ in range(env_idx*self.sample_num, (env_idx+1)*self.sample_num)]
        self.timer.checkpoint("Starting batched env initialization")
            
        self.agent_groups_list = []
        for rollout_idx in range(len(self.envs)):
            self.agent_groups_list_per_rollout = []
            for agent_idx, agent_name in enumerate(self.turn_order):
                agent_class = self.agent_class_list[agent_idx]

                agent_init_params = {'env_idx': rollout_idx, 'agent_sample_idx': rollout_idx, 'rollout_idx': rollout_idx}
                agent_init_params['benchmark'] = getattr(self.config.env, 'benchmark', 'AIME24') if hasattr(self.config, 'env') else 'AIME24'

                self.agent_groups_list_per_rollout.append(agent_class(**agent_init_params))
            self.agent_groups_list.append(self.agent_groups_list_per_rollout)
        
    
                   
            
    async def generate_single_rollout(self, rollout_idx):
        """
        Execute autogen graph workflow with patched model clients for RL training.
        
        The workflow:
        1. Prepare environment and model_client_dict for each agent/policy
        2. Call autogen graph (e.g., code_graph) with patched OpenAIChatCompletionClient
        3. The patched client intercepts generate calls and routes to llm_async_generate
        4. Collect trajectories (DataProto) from each generation for RL training
        5. Add rewards to trajectories based on environment outcomes
        
        Args:
            rollout_idx: Index of the rollout
            
        Returns:
            trajectory_per_task_dict: {policy_name: DataProto} containing trajectories with rewards
        """
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from pettingllms.utils.openai import reset_turn_idx, get_turn_idx, set_rollout_context, get_trajectory_store
        
        trajectory_per_task_dict = {p: DataProto() for p in self.tokenizer_dict.keys()}
        env_idx = rollout_idx // self.sample_num
        env = self.envs[rollout_idx]
        
        # Set rollout context for trajectory collection
        set_rollout_context(rollout_idx, env_idx)
        
        # Reset turn index and clear trajectory store for this rollout
        reset_turn_idx()
        
        # Build model_client_dict: {agent_name: OpenAIChatCompletionClient}
        # Each client is configured with the corresponding policy's server
        model_client_dict = {}
        for agent_name in self.turn_order:
            policy_name = self.agent_policy_mapping.get(agent_name)
            if policy_name is None:
                self.multi_logger.log_env_agent_info(
                    self.mode, env_idx, rollout_idx, 0, agent_name,
                    f"No policy mapping found for agent {agent_name}",
                    {"error": "missing_policy"}
                )
                continue
                
            ppo_cfg = self.ppo_trainer_config_dict[policy_name]
            addresses = self.server_address_dict[policy_name]
            address = random.choice(addresses) if isinstance(addresses, (list, tuple)) else addresses
            
            # Create patched OpenAI client (the patch will intercept its generate calls)
            model_client = OpenAIChatCompletionClient(
                model=policy_name,  # Use policy_name as model identifier
                api_key="dummy",  # Not used due to patch
                base_url=address,  # vLLM server address
            )
            model_client_dict[agent_name] = model_client
        
        # Get the autogen graph workflow function from registry using config.workflow_function
        try:
            graph_func = self.get_graph_function()
        except ValueError as e:
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 0, "system",
                f"Failed to get graph function: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            return trajectory_per_task_dict
        
        # Execute the patched autogen graph
        # The patch will intercept OpenAIChatCompletionClient.create() calls
        # and route them to llm_async_generate, collecting trajectories
        try:
            # Wrap and run the graph
            wrapped_graph = self.patch_and_run_autogen_graph(graph_func)
            result_env = await wrapped_graph(env=env, model_client_dict=model_client_dict)
            
            # After graph execution, collect trajectories from the patch context
            trajectory_store = get_trajectory_store()
            
            # Calculate final reward from environment
            # For code env, reward is based on test pass rate
            final_reward = 0.0
            if hasattr(env, 'state'):
                if hasattr(env.state, 'ground_truth_test_vs_generated_code_match_ratio'):
                    final_reward = env.state.ground_truth_test_vs_generated_code_match_ratio
                elif hasattr(env.state, 'generated_test_vs_generated_code_match_ratio'):
                    final_reward = env.state.generated_test_vs_generated_code_match_ratio
            
            # Add rewards to collected trajectories and aggregate
            for (r_idx, t_idx, policy_name), (output_dpr, response) in trajectory_store.items():
                if r_idx != rollout_idx:
                    continue  # Skip trajectories from other rollouts
                
                # Add reward to trajectory
                # You can implement different reward shaping strategies here
                # For now, use final reward for all turns
                output_dpr.non_tensor_batch["reward"] = np.array([final_reward])
                
                # Handle LoRA if enabled
                agent_name = output_dpr.non_tensor_batch.get("agent_name", [None])[0]
                if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                    batch_size = output_dpr.batch.batch_size[0] if hasattr(output_dpr.batch, 'batch_size') else len(output_dpr.batch)
                    output_dpr.non_tensor_batch["lora_ids"] = np.array(
                        [self.agent_lora_mapping[agent_name]] * batch_size, 
                        dtype=object
                    )
                
                # Concatenate to trajectory dict
                if trajectory_per_task_dict[policy_name].batch is None:
                    trajectory_per_task_dict[policy_name] = output_dpr
                else:
                    trajectory_per_task_dict[policy_name] = DataProto.concat([
                        trajectory_per_task_dict[policy_name], 
                        output_dpr
                    ])
            
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, get_turn_idx(), "system",
                f"Graph execution completed with {get_turn_idx()} turns, final_reward={final_reward}",
                {
                    "success": True, 
                    "turns": get_turn_idx(),
                    "final_reward": final_reward,
                    "trajectory_count": len(trajectory_store)
                }
            )
            
        except Exception as e:
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, get_turn_idx(), "system",
                f"Graph execution failed: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
        
        return trajectory_per_task_dict
            

        
    async def generate_env_idx_rollout(self, env_idx, if_greedy=True,):
        """
        多个相同 env_idx 的 rollout 并行生成，每个节点选择 reward 最大的分支，复制形成树状采样。
        """
        trajectory_per_task_dict = {p: DataProto() for p in self.tokenizer_dict.keys()}
        rollout_idx_list = self.env_rollout_mapping[env_idx]
        envs_list = [self.envs[rollout_idx] for rollout_idx in rollout_idx_list]
        agent_groups = [self.agent_groups_list[rollout_idx] for rollout_idx in rollout_idx_list]

        for node_idx, agent_name in enumerate(self.turn_order):
            policy_name = self.agent_policy_mapping.get(agent_name)
            ppo_cfg = self.ppo_trainer_config_dict[policy_name]
            address_list = self.server_address_dict[policy_name]

            # generate for all rollouts at this node
            results = []
            for idx, rollout_idx in enumerate(rollout_idx_list):
                env = envs_list[idx]
                current_agent = agent_groups[idx][node_idx]
                current_agent.update_from_env(node_idx, env)

                prompt = {"text": current_agent.current_prompt, "image": None}
                prompt_dpr = convert_prompt_to_dpr(
                    self.tokenizer_dict[policy_name],
                    self.processor_dict.get(policy_name),
                    prompt,
                    self.max_prompt_length,
                    multi_modal=False,
                    enable_thinking=False
                )
                if prompt_dpr is None:
                    results.append((idx, None, "", None))
                    continue

                address = random.choice(address_list) if isinstance(address_list, (list, tuple)) else address_list
                model_path = ppo_cfg.actor_rollout_ref.model.path
                model_name = str(model_path) if "checkpoint" in str(model_path) else "/".join(str(model_path).split("/")[-2:])

                try:
                    output_dpr, response = await llm_async_generate(
                        rollout_idx=rollout_idx,
                        turn_idx=0,
                        agent_idx=0,
                        prompt_dpr=prompt_dpr,
                        ppo_trainer_config=ppo_cfg,
                        address=address,
                        model_name=model_name,
                        tokenizer=self.tokenizer_dict[policy_name],
                        enable_thinking=False,
                        image_data=None,
                        application_id=str(uuid.uuid4()),
                        env_idx=env_idx,
                        policy_name=policy_name,
                        timeout=self.generate_timeout,
                        mode=self.mode,
                        lora_id=None,
                        agent_config=None,
                    )
                except Exception as e:
                    print(f"[Engine][ERROR] llm_async_generate failed: {e}")
                    output_dpr, response = None, ""

                response = response or ""
                current_agent.update_from_model(response)
                results.append((idx, output_dpr, response, current_agent))

            # step, reward, trajectory, select best
            rewards = []
            for idx, output_dpr, response, current_agent in results:
                env = envs_list[idx]
                try:
                    env_worker_id = rollout_idx_list[idx] % self.num_workers
                    env_worker = self.env_workers[env_worker_id]
                    if hasattr(env, 'state'):
                        env.state.assigned_worker_id = env_worker_id
                        env.state.gpu_group_id = self.gpu_group_id
                    await asyncio.wait_for(current_agent.step(env, env_worker=env_worker), timeout=self.step_timeout)
                except asyncio.TimeoutError:
                    pass

                if hasattr(current_agent, 'calculate_reward'):
                    current_agent.calculate_reward(env)
                elif hasattr(current_agent, 'agent_reward'):
                    if not hasattr(current_agent, 'reward_history'):
                        current_agent.reward_history = []
                    current_agent.reward_history.append(current_agent.agent_reward)

                reward_val = getattr(current_agent, 'agent_reward', 0) or 0
                rewards.append(reward_val)

                if output_dpr is not None:
                    output_dpr.non_tensor_batch["reward"] = np.array([reward_val])
                    output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name], dtype=object)
                    if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                        batch_size = output_dpr.batch.batch_size[0] if hasattr(output_dpr.batch, 'batch_size') else len(output_dpr.batch)
                        output_dpr.non_tensor_batch["lora_ids"] = np.array([self.agent_lora_mapping[agent_name]] * batch_size, dtype=object)
                    if trajectory_per_task_dict[policy_name].batch is None:
                        trajectory_per_task_dict[policy_name] = output_dpr
                    else:
                        trajectory_per_task_dict[policy_name] = DataProto.concat([trajectory_per_task_dict[policy_name], output_dpr])

            # select best rollout for this node and clone
            if rewards:
                best_i = int(np.argmax(np.asarray(rewards)))
                selected_env = envs_list[best_i]
                selected_agent_group = agent_groups[best_i]
                envs_list = [copy.deepcopy(selected_env) for _ in envs_list]
                agent_groups = [copy.deepcopy(selected_agent_group) for _ in agent_groups]
                if selected_env.done:
                    break

        return trajectory_per_task_dict


    async def generate_multiple_rollouts_concurrent(self, env_idx_list, rollout_mode="tree"):
        rollout_indices=[]
        for env_idx in env_idx_list:
            rollout_indices.extend(self.env_rollout_mapping[env_idx])
        concurrent_timer = create_timer("ConcurrentRollouts")
        concurrent_timer.start(f"Starting concurrent rollouts for {len(rollout_indices)} rollouts")
        
        concurrent_timer.checkpoint("Creating async tasks")
        
        if rollout_mode == "tree" and self.mode != "validate":
            tasks = [
                asyncio.create_task(
                    self.generate_env_idx_rollout(env_idx, if_greedy=getattr(self.config, 'if_greedy', True)), 
                    name=f"env_{env_idx}_rollouts"
                )
                for env_idx in env_idx_list
            ]
        else:
            tasks = [
            asyncio.create_task(
            self.generate_single_rollout(rollout_idx), 
                name=f"env_{rollout_idx}_rollouts"
            )
            for rollout_idx in range(self.gen_batch_size*self.sample_num)
        ]
        
        
        concurrent_timer.checkpoint(f"Created {len(tasks)} async tasks")
        
        aggregated_results = {}
        for policy_name in self.tokenizer_dict.keys():
            aggregated_results[policy_name] = DataProto()
        
        completed_count = 0
        failed_count = 0
  
        task_pbar = tqdm(total=len(tasks), desc="Rollouts", position=1, leave=False)
        
        try:
            concurrent_timer.checkpoint("Starting task execution")
            for completed_task in asyncio.as_completed(tasks):
                try:
                    
                    rollout_result = await completed_task
        
                    for policy_name, policy_data in rollout_result.items():
                        #print(policy_data)
                        if policy_data.batch is not None:  
                            if aggregated_results[policy_name].batch is None:
                                aggregated_results[policy_name] = policy_data
                            else:
                                aggregated_results[policy_name] = DataProto.concat([
                                    aggregated_results[policy_name], 
                                    policy_data
                                ])
                            #print(f"The length of concatenated aggregated_results[policy_name]: {len(aggregated_results[policy_name])}")
                    
                    completed_count += 1
                    
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)})")
                except Exception as e:
                    failed_count += 1
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)}, {failed_count} failed)")
                    
                    self.multi_logger.log_async_event(
                        self.mode, -1, -1, "task_error",
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
            self.multi_logger.log_ray_status(mode=self.mode, context="during_error")
            
            self.multi_logger.log_async_event(
                self.mode, -1, -1, "concurrent_batch_error",
                f"Concurrent execution encountered error: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            for task in tasks:
                if not task.done():
                    task_name = task.get_name()
                    self.multi_logger.log_async_event(
                        self.mode, -1, -1, "task_cancel",
                        f"Cancelling task {task_name}"
                    )
                    task.cancel()
            raise

        task_pbar.close()
        
        concurrent_timer.checkpoint("All tasks completed")
        with open("rollout_data.json", "w", encoding="utf-8") as f:
            json.dump(self.rollout_latency_dict, f, ensure_ascii=False, indent=4)
        if self.mode=="validate":
            for agent_name in self.turn_order:
                success_rate = len(self.success_rollout_idx_list_dict.get(agent_name, [])) / len(tasks)
                self.multi_logger.log_rollout_summary(
                    self.mode, -1, -1, 
                    {agent_name: success_rate}, 
                    "validate_finished",
                    extra_data={"success_rate": success_rate}
                )
            
        self.multi_logger.log_async_event(
            self.mode, -1, -1, "concurrent_batch_complete",
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
        self.multi_logger.log_ray_status(mode=self.mode, context="after_concurrent_batch")
        
        concurrent_timer.end("Concurrent rollouts completed successfully")
        return aggregated_results



