import logging
import copy
import io
import sys
import time
import typing
import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple, List

from pettingllms.multi_agent_env.code.agents.code_agent import CodeGenerationAgent
from pettingllms.multi_agent_env.code.agents.unit_test_agent import UnitTestGenerationAgent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.code_utils import (
        load_problem_batch,
        extract_code_from_response,
        evaluate_code_against_tests,
    )
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CodeEnvState:
    problem: str=None
    golden_code: str=None
    generated_code_history: List[str]=field(default_factory=list)
    generated_code: str=None
    generated_test_input: List[str]=None
    generated_test_output: List[str]=None
    ground_truth_test_input: List[str]=None
    ground_truth_test_output: List[str]=None
    exe_code_generated_test_output: List[str]=None
    exe_code_ground_truth_test_output: List[str]=None
    # Evaluation results: generated test vs generated code
    ground_truth_test_vs_generated_code_mismatch_cases: List[Dict]=None
    ground_truth_test_vs_generated_code_match_cases: List[Dict]=None
    ground_truth_test_vs_generated_code_match_ratio: float=0
    generated_test_vs_generated_code_match_cases: List[Dict]=None
    generated_test_vs_generated_code_mismatch_cases: List[Dict]=None
    generated_test_vs_generated_code_mismatch_cases_history: List[Dict]=field(default_factory=list)
    generated_test_vs_generated_code_match_ratio: float=0
    generated_test_vs_golden_code_match_cases: List[Dict]=None
    generated_test_vs_golden_code_mismatch_cases: List[Dict]=None
    generated_test_vs_golden_code_match_ratio: float=0
    # Final reward assigned at end of workflow (1.0 if all golden tests pass, else 0.0)
    final_reward: float = 0.0

class CodeEnv(Env):
    """
    Environment for code generation and testing tasks with dual-agent interaction.
    
    This environment coordinates between code generation and unit test generation agents,
    similar to how WebEnv coordinates between code and visual agents.
    """

    def __init__(
        self,
        problem: str = None,
        ground_truth_test_input: List[str] = None,
        ground_truth_test_output: List[str] = None,
        env_idx: int = 0,
        rollout_idx: int = 0,
        config: dict | None = None,
    ):
        """
        Initialize the code test environment.
        
        Args:
            problem: The coding problem description
            ground_truth_test_input: Ground truth test inputs
            ground_truth_test_output: Ground truth test outputs
            env_idx: Environment index
            rollout_idx: Rollout index
            config: Configuration dictionary
        """
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, config=config)
        self.state = CodeEnvState(
            problem=problem,
            ground_truth_test_input=ground_truth_test_input,
            ground_truth_test_output=ground_truth_test_output
        )
        self.backend = "ray_docker" 
        # Convenience attribute for engine to read without drilling into state
        self.final_reward: float = 0.0


   

    def reset(self):

        self.state.generated_code=None
        self.state.generated_code_history=[]
        self.state.generated_test_input=None
        self.state.generated_test_output=None
        self.state.exe_code_generated_test_output=None
        self.state.exe_code_ground_truth_test_output=None
        self.state.ground_truth_test_vs_generated_code_mismatch_cases=None
        self.state.ground_truth_test_vs_generated_code_match_cases=None
        self.state.ground_truth_test_vs_generated_code_match_ratio=0
        self.state.generated_test_vs_generated_code_match_cases=None
        self.state.generated_test_vs_generated_code_mismatch_cases=None
        self.state.generated_test_vs_generated_code_mismatch_cases_history=[]
        self.state.generated_test_vs_generated_code_match_ratio=0
        self.state.generated_test_vs_golden_code_match_cases=None
        self.state.generated_test_vs_golden_code_mismatch_cases=None
        self.state.generated_test_vs_golden_code_match_ratio=0
        self.state.final_reward = 0.0
        self.final_reward = 0.0


class CodeEnvBatch:
    """Batch environment for managing multiple CodeEnv instances."""
    
    def __init__(
        self,
        env_indices: List[int],
        config: dict = None,
        mode: str = "train",
        samples: int = 1,
        rollout_idx_list: List[int] = None,
        env_workers: List = None
    ):
        """
        Initialize batch code environment.
        
        Args:
            env_indices: List of environment indices
            config: Configuration dictionary
            mode: Mode of operation ("train", "validate", "test")
            samples: Number of samples per problem
            rollout_idx_list: List of rollout indices
            env_workers: Optional list of environment workers
        """
        self.mode = mode
        self.config = config or {}
        self.samples = samples if mode == "train" else 1
        
        # Load problems based on mode
        if mode == "train":
            difficulty = getattr(config.env, "difficulty", "difficult") if hasattr(config, "env") else "difficult"
            # Ensure env_indices are safe by using list to pass to load function
            safe_env_indices = list(env_indices) if not isinstance(env_indices, list) else env_indices
            self.problem_list = load_problem_batch(
                safe_env_indices,
                benchmark_name="train",
                mode="train",
                difficulty=difficulty
            )
        else:
            benchmark_name = getattr(config.env, "benchmark", "test") if hasattr(config, "env") else "test"
            safe_env_indices = list(env_indices) if not isinstance(env_indices, list) else env_indices
            self.problem_list = load_problem_batch(
                safe_env_indices,
                mode=mode,
                benchmark_name=benchmark_name
            )
        
        # Generate rollout indices if not provided
        if rollout_idx_list is None or mode == "validate":
            rollout_idx_list = list(range(len(self.problem_list) * self.samples))
        
        # Create environment instances
        self.env_list = []
        for i, problem in enumerate(self.problem_list):
            ground_truth_test_input = problem["test_input"]
            ground_truth_test_output = problem["test_output"]
            
            for s in range(self.samples):
                env = CodeEnv(
                    problem=problem["question"],
                    ground_truth_test_input=ground_truth_test_input,
                    ground_truth_test_output=ground_truth_test_output,
                    env_idx=i,
                    rollout_idx=rollout_idx_list[i * self.samples + s],
                    config=config
                )
                self.env_list.append(env)
        
        # Validation check
        if len(self.env_list) != len(rollout_idx_list):
            raise ValueError(
                f"Environment list length mismatch: {len(self.env_list)} != {len(rollout_idx_list)}"
            )