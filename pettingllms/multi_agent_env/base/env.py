from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking only to avoid runtime circular imports
    from pettingllms.multi_agent_env.base.agent import AgentData




@dataclass    
class Env:
    """
    An environment for multi-turn interactions with LLMs.
    The environment provides a series of questions/prompts and evaluates responses using a custom reward function.
    The interaction terminates after reaching the maximum number of turns.
    """

    def __init__(self, env_idx: int, rollout_idx: int, max_turns: int,  config: dict | None = None):
        """
        Initialize the multi-agents environment using the dataclass EnvData.

        Args:
            env_idx: Environment index
            rollout_idx: Rollout index  
            max_turns: Maximum number of turns before terminating
            task: Dictionary containing the task information
            config: Configuration for the system
        """
        env_idx: int
        rollout_idx: int
        max_turns: int
        turn: int = 0
        state: Optional[Any] = None
        is_pass: bool = False
        
        # Save configuration
        self.config = config
        
        # Initialize variables required by step method
        self.history = []
        self.task = None
        self.current_turn = 0
        self.done = False
        self.state = None


        
        

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Response string from the LLM

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append(action)

        # Calculate reward for the current turn using the abstract method
        assert self.task is not None, "Task is not set"
        reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # Increment turn counter
        self.current_turn += 1

        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task


class EnvBatch:
    def __init__(self, env_idx_list: List[int], rollout_idx: List[int], max_turns: int):
        self.env_list=[]
        for env_idx in env_idx_list:
            env=Env(env_idx, max_turns)
            self.env_list.append(env)
