"""
Rollout Serving Manager for coordinating asynchronous rollout generation.
"""
import asyncio
from dataclasses import dataclass, field
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
import hydra
import os
from transformers import AutoTokenizer
from verl.single_controller.ray.base import RayWorkerGroup
from pettingllms.llm_agent.agent_proxy import VllmWrapperWg, ApiCallingWrapperWg
from verl import DataProto
from .async_batch_manager import AsyncBatchManager
from .rollout_consumer import RolloutConsumer, BufferStatus

from pettingllms.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from pettingllms.utils import register_resolvers
register_resolvers()
from .request_generation import generate_initial_request

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment
    turn_id: int = 0 # current turn id







class RolloutServingManager:
    """
    Main serving manager that coordinates multiple producers and consumers
    for asynchronous rollout generation with multiple VllmWrapperWg instances.
    """

    def __init__(self, 
                 config,
                 mode: str = "train",
                 max_queue_size: int = 100,
                 batch_timeout: float = 30.0):
        """
        Initialize the rollout serving manager.
        Args:
            config: Configuration object
            num_producers: Number of producer workers
            num_consumers: Number of consumer workers  
            max_queue_size: Maximum size of request queue
            batch_timeout: Timeout for batch processing
        """
        self.config = config
        self.env_config =getattr(self.config.es_manager, mode)
        self.mode = mode
        self.entry_buffer = asyncio.Queue()
        
        self.output_result=[]
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        
        # 初始化buffer_status
        self.buffer_status = BufferStatus(0, 0)
        
        
        

    async def _init_entry_list(self):
        
        
        env_id = 0
        entry_id = 0
        self.entry_task_sum=0
        for tag, n_group in zip(self.env_config.tags, self.env_config.n_groups):
            env_id += 1
            
            for _ in range(n_group):
                entry_id += 1
                
                
                cfg_template = self.config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                env_obj = REGISTERED_ENVS[env_class](env_config)
                text = generate_initial_request(tag, self.config)
                
                entry_dict = {'tag': tag, 'env_id': env_id, 'entry_id': entry_id,
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj, 'text': text}
                
                self.entry_buffer.put(entry_dict)
                self.entry_task_sum+=1
                self.buffer_status.total_entries_added+=1
    

    async def run(self):
        consumers = [RolloutConsumer(self.config, self.tokenizer) for _ in range(self.config.num_batch_workers)]
        tasks = []
        for consumer in consumers:
            task = asyncio.create_task(
                asyncio.wait_for(
                    consumer.consume(self.entry_buffer, self.output_result),
                    timeout=self.config.max_runtime
                )
            )
            tasks.append(task)
        print("init entry list")
        await self._init_entry_list()
        print("init entry list done")

        await self.entry_buffer.join()
        print("entry buffer join done!!")
        for task in tasks:
            task.cancel()
        print("tasks cancel done!!")
        await asyncio.gather(*tasks, return_exceptions=True)
        print("tasks gather done!!")
        print("tasks results done!!")
        
        
        return self.output_result
        
        
        
        
       
        

                
                
            




    
    
        
       
        
        
   


    
    
        
       
        
        
   
   