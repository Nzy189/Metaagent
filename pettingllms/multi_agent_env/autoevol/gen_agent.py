from typing import List, Dict, Any, Optional
import json
import logging
import re
import os
import subprocess
import asyncio
import numpy as np
import torch
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.model import compute_position_id_with_mask
from tensordict import TensorDict
logger = logging.getLogger(__name__)
from pettingllms.multi_agent_env.autoevol.reward_function import REWARD_FUNCTIONS
from pettingllms.multi_agent_env.autoevol.utils import load_and_tokenize_jsonl

class MASGenerator(Agent):
    """MAS Designer Agent - designs multi-agent systems"""

    def __init__(self, task_type: str = "math", rollout_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.task_type = task_type.lower()
        self.rollout_idx = rollout_idx

        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)


    def update_from_env(self, env_data: Env):
        """Update agent from environment data and generate Qwen-formatted prompt"""
        self.env_data = env_data

        # Get code generation prompt
        user_prompt_text = env_data.state.problem
        system_prompt_text = "You are an expert in designing Multi-Agent System workflows."

        # Format with Qwen chat template
        prompt_text = (
            f"<|im_start|>system\n{system_prompt_text}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        self.current_prompt = {"text": prompt_text, "image": None}



    def update_from_model(self, response: str):
        code = ""

        code_match = re.search(r"<code>\s*```python(.*?)```\s*</code>", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
            if matches:
                code = matches[-1].strip()
            else:
                
                code = "# Error: Could not extract code from the model response."
                logger.warning("Failed to extract code from model response")


        self.generated_code = code


        self.current_action = code

        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None, output_dir: str = None,
                   server_address: str = None, model_name: str = None, tokenizer=None,
                   max_prompt_length: int = 2048, max_response_length: int = 2048):
        """
        Execute MAS Designer step: generate mas.py, run it with vLLM access, calculate reward.

        Returns:
            Tuple[List[Tuple[DataProto, str]], float]:
                - tokenized_trajectories: List of (DataProto, response_text) tuples
                - final_reward: Reward score from task-specific reward function
        """
        

        # Ensure output directory is provided
        if output_dir is None:
            raise ValueError("output_dir must be provided to step()")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare the code with necessary imports and path setup
        dyevolve_dir = os.path.dirname(os.path.abspath(__file__))

        # Add environment setup at the beginning of the code
        env_setup_code = f"""
import os
import sys

# Set up environment for Ray Worker execution
# Add dyevolve directory to Python path for imports
dyevolve_dir = r'{dyevolve_dir}'
if dyevolve_dir not in sys.path:
    sys.path.insert(0, dyevolve_dir)

# Set up output directory for trajectory saving
output_dir = r'{output_dir}'
os.makedirs(output_dir, exist_ok=True)

# Set environment variables if needed
if hasattr(env_data, 'problem'):
    os.environ['WORKFLOW_QUESTION'] = str(env_data.problem)
"""


        # Combine all parts
        

        # Save generated code to mas.py
        mas_py_path = os.path.join(output_dir, "mas.py")
        self.trajectory_json_path= os.path.join(output_dir, "traj.json")

        trajectory_output_code = f"""
# Automatically save executor conversations after workflow execution
try:
    from ag2_tracer import get_global_tracker
    tracker = get_global_tracker()
    if tracker.agent_conversations:  # Only save if there are conversations
        import os
        from datetime import datetime

        # Use the output_dir from environment setup
        trajectory_file = r'{self.trajectory_json_path}'

        tracker.save_all(filepath=trajectory_file, append=False)
        print(f"\\n[Conversation data saved to {{trajectory_file}}]")
except Exception as e:
    # Silently fail - don't interrupt workflow execution
    print(f"\\n[Warning: Failed to save executor conversations: {{e}}]")
    pass
"""
        full_code = env_setup_code + "\n" + self.generated_code + "\n" + trajectory_output_code
        with open(mas_py_path, 'w') as f:
            f.write(full_code)

        logger.info(f"Saved MAS code to {mas_py_path}")

        # Run the mas.py file in Ray Docker Worker environment
        try:
            # Read the generated MAS code
            with open(mas_py_path, 'r') as f:
                mas_code = f.read()

            # Execute in Ray Docker Worker if available, otherwise fallback to subprocess
            if env_worker is not None:
                # Use Ray Docker Worker for execution
                logger.info(f"Executing MAS code in Ray Docker Worker for rollout {self.rollout_idx}")

                # Import the helper function
                from pettingllms.multi_agent_env.math.math_worker import get_code_execution_output

                # Execute with timeout (5 minutes)
                execution_timeout = 300.0
                output = await get_code_execution_output(
                    code=mas_code,
                    timeout=execution_timeout,
                    ray_actor=env_worker
                )

                # Parse output
                stdout = output
                stderr = ""

                # Check for errors in Ray execution
                if isinstance(stdout, str) and stdout.startswith("error:"):
                    logger.error(f"Ray execution failed: {stdout}")
                    stderr = stdout
                    stdout = ""
                elif stdout == "timeout":
                    logger.error(f"Ray execution timed out for rollout {self.rollout_idx}")
                    raise subprocess.TimeoutExpired(mas_py_path, execution_timeout)

            else:
                # Fallback to subprocess execution
                logger.warning("env_worker is None, falling back to subprocess execution")
                dyevolve_dir = os.path.dirname(os.path.abspath(__file__))
                env = os.environ.copy()
                result = subprocess.run(
                    ['python', mas_py_path],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=output_dir,
                    env=env
                )
                stdout = result.stdout
                stderr = result.stderr

            # Extract summary and trajectory from output
            summary = self._extract_summary(stdout)

            # Extract trajectory from stdout (legacy method)
            trajectory_store = self._extract_trajectory_from_stdout(stdout)
            if trajectory_store:
                logger.info(f"Extracted {len(trajectory_store)} trajectory entries from stdout")
                self.trajectory_store = trajectory_store
            else:
                logger.warning("No trajectory data found in stdout")
                self.trajectory_store = {}

            # Load and tokenize trajectory data from saved JSONL file if tokenizer provided
            tokenized_trajectories = []
            if tokenizer is not None:
                trajectory_file = self.trajectory_json_path
                if not os.path.exists(trajectory_file):
                    logger.warning(f"Trajectory file {trajectory_file} not found, skipping tokenization")
                    self.tokenized_trajectories = []
                else:
                    # Use the new load_and_tokenize_jsonl function from utils
                    tokenized_trajectories = load_and_tokenize_jsonl(
                        trajectory_file, tokenizer, max_prompt_length, max_response_length
                    )
                    if tokenized_trajectories:
                        logger.info(f"Tokenized {len(tokenized_trajectories)} trajectory turns")
                        # Store tokenized trajectories in a new attribute
                        self.tokenized_trajectories = tokenized_trajectories
                    else:
                        logger.warning("No tokenized trajectories generated")
                        self.tokenized_trajectories = []

            # Log stderr if there were errors
            if stderr:
                logger.warning(f"MAS stderr output: {stderr[:500]}")

            # Calculate reward using task-specific reward function
            final_reward = 0.0
            if self.task_type in REWARD_FUNCTIONS:
                reward_func = REWARD_FUNCTIONS[self.task_type]
                final_reward = reward_func(summary, env_data)
                logger.info(f"Rollout {self.rollout_idx}: final_reward={final_reward}")
            else:
                logger.warning(f"No reward function found for task_type={self.task_type}, defaulting to 0.0")
                final_reward = 0.0

            self.agent_reward = final_reward
            self.reward_history.append(final_reward)

            # Return tokenized trajectories and final reward
            return tokenized_trajectories, final_reward

        except subprocess.TimeoutExpired:
            logger.error(f"MAS execution timed out for rollout {self.rollout_idx}")
            return [], 0.0
        except Exception as e:
            logger.error(f"Error executing MAS: {e}")
            return [], 0.0



    def _extract_summary(self, stdout: str) -> str:
        """Extract summary from workflow output"""
        start_marker = "WORKFLOW_SUMMARY_START"
        end_marker = "WORKFLOW_SUMMARY_END"

        if start_marker in stdout and end_marker in stdout:
            start_idx = stdout.find(start_marker) + len(start_marker)
            end_idx = stdout.find(end_marker)
            summary = stdout[start_idx:end_idx].strip()
            return summary
        else:
            lines = [line.strip() for line in stdout.split('\n') if line.strip()]
            return lines[-1] if lines else ""
    
    def _extract_trajectory_from_stdout(self, stdout: str) -> dict:
        """Extract trajectory data from subprocess stdout"""
        import pickle
        import base64

        start_marker = "TRAJECTORY_DATA_START"
        end_marker = "TRAJECTORY_DATA_END"

        if start_marker in stdout and end_marker in stdout:
            start_idx = stdout.find(start_marker) + len(start_marker)
            end_idx = stdout.find(end_marker)
            trajectory_b64 = stdout[start_idx:end_idx].strip()

            trajectory_bytes = base64.b64decode(trajectory_b64)
            trajectory_store = pickle.loads(trajectory_bytes)
            return trajectory_store
        else:
            return {}

