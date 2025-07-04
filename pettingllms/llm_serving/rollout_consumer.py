"""
Rollout Consumer for LLM generation and environment step operations.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pettingllms.llm_agent.ctx_manager import ContextManager
from pettingllms.llm_agent.es_manager import EnvStateManager
from verl import DataProto
from .async_batch_manager import AsyncBatchManager, BatchRequest
from verl.workers.rollout.sglang_rollout.sglang_rollout import AsyncEngine
from dataclasses import dataclass

@dataclass
class BufferStatus:
    """Status of a buffer"""
    total_entries_added: int = 0
    total_entries_completed: int = 0





class RolloutConsumer:
    """
    Consumer component that performs LLM generation and environment steps.
    Operates in two phases:
    1. LLM Generation: Generate responses from prompts
    2. Environment Step: Process responses and update environment states
    """
    
    def __init__(self, 
                 config,
                 tokenizer,
                 worker_id: int = 0):
        """
        Initialize the rollout consumer.
        
        Args:
            config: Configuration object
            tokenizer: Tokenizer for text processing
            llm_wrapper: LLM wrapper for generation (VllmWrapperWg or ApiCallingWrapperWg)
            batch_manager: Async batch manager for coordination
            worker_id: Unique identifier for this consumer worker
        """
        self.config = config
        self.tokenizer = tokenizer
        self.llm_engine = AsyncEngine(config, tokenizer)
        self.worker_id = worker_id
        self.buffer_status = BufferStatus()

    def _generate_new_input_text(self, entry: Dict):
        '''
        TODO:
        Generate new input text for the next phase of rollout
        '''
        return entry['text']

    def _two_phase_rollout(self, entry: Dict):
        '''
        Two-phase rollout:
        1. LLM generation
        2. Environment step
        '''
        if entry['status'].num_actions >= entry['max_actions_per_traj']:
            done = True
            return entry,done
        if entry['status'].truncated or entry['status'].terminated:
            done = True
            return entry,done
        
        llm_output_text = self.llm_engine.async_generate(prompt=entry['text'])["text"]
        observation, reward, done, info = entry['env'].step(llm_output_text)
        new_input_text=self._generate_new_input_text(entry)
        entry['text'] = new_input_text
        
        entry['status'].num_actions += 1
        entry['status'].rewards.append(reward)
        entry['status'].turn_id += 1
        
        return entry,done


    async def consume(self, input_queue: asyncio.Queue, output_result: List[Dict]):
        while True:
            try:
                entry = await input_queue.get()
            except asyncio.CancelledError:
                break
            if entry is None:
                break
            entry, done = self._two_phase_rollout(entry)
            if done:
                output_result.append(entry)
                self.buffer_status.total_entries_completed += 1
            else:
                await input_queue.put(entry)
                self.buffer_status.total_entries_added += 1
            input_queue.task_done()
        return self.buffer_status

 