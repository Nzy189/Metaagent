import logging
import copy
from typing import Any, List, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.math.math_worker import get_code_execution_output
from pettingllms.multi_agent_env.stateful.utils import (
    extract_code_from_response,
    extract_actions_from_code_output
)
from pettingllms.multi_agent_env.stateful.prompt import build_tool_prompt

logger = logging.getLogger(__name__)


class ToolAgent(Agent):
    """
    Code generation style tool agent
    - Determines initial/subsequent prompts via benchmark parameter
    - Other logic (execution, parsing, scoring, write-back, completion) remains unchanged
    """

    def __init__(self, rollout_idx: int | None = None, benchmark: str = "plan_path", **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.benchmark = benchmark
        self.agent_reward_history = []
        self.success = False
        self.agent_reward = 0.0
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """Update agent prompt based on environment state"""
        self.env_data = env_data
        state = getattr(env_data, "state", None)

        formatted_prompt = (
            "You are an AI assistant specialized in solving planning problems through code generation. Your task is to analyze the given scenario and generate Python code that produces a sequence of actions to solve the problem.\n\nInstructions:\n1. Write Python code enclosed in ```python and ``` tags\n2. Your code MUST output the result using print() function\n3. Example of correct output format:\n   print(\"**Actions List**:\", [\"U\", \"R\", \"D\", \"L\"])\n"

        )

        # Add history information when turn > 0
        if turn_idx > 0 and hasattr(state, 'code_history'):
            history_formatted_prompt = ""
            history_formatted_prompt += "\n\n=== HISTORY FROM PREVIOUS TURNS ===\n"
            for i in range(len(state.code_history)):
                history_formatted_prompt += f"\n--- Turn {i+1} ---\n"
                if i < len(state.code_history):
                    # Truncate code to 500 characters
                    code_preview = state.code_history[i]
                    if len(code_preview) > 500:
                        code_preview = code_preview[:250] + "\n...(truncated)...\n" + code_preview[-250:]
                    history_formatted_prompt += f"Previous Code:\n{code_preview}\n"
                if i < len(state.execution_history):
                    # Truncate execution output to 300 characters
                    exec_preview = state.execution_history[i]
                    if len(exec_preview) > 300:
                        exec_preview = exec_preview[:150] + "...(truncated)..." + exec_preview[-150:]
                    history_formatted_prompt += f"Previous Execution Output:\n{exec_preview}\n"
                if i < len(state.tool_action_history):
                    history_formatted_prompt += f"Previous Tool Action: {state.tool_action_history[i]}\n"
                if i < len(state.plan_action_history):
                    history_formatted_prompt += f"Previous Plan Action: {state.plan_action_history[i]}\n"
            history_formatted_prompt += "\n=== END OF HISTORY ===\n\n"
            #formatted_prompt += history_formatted_prompt
        formatted_prompt += build_tool_prompt(self.benchmark, turn_idx, state)
        
        if self.benchmark in ("plan_path", "sokoban"):
            formatted_prompt += (
                "4. Actions should be represented as a list of strings: ['U', 'D', 'L', 'R'] (Up, Down, Left, Right)\n"
                "5. You may return either the complete action sequence to reach the goal, or a partial sequence if you're uncertain\n"
                "6. CRITICAL: You MUST use print() to output the result. Do NOT just return a value.\n"
                "7. Please use algorithm like BFS or A* to solve the problem.\n"
                "8. Your code must output the final action sequence in this exact format:\n"
                "   print(\"**Actions List**:\", [\"U\", \"R\", \"D\", \"L\"])\n\n"
                "Example code structure:\n"
                "```python\n"
                "# Your algorithm here (BFS, A*, etc.)\n"
                "actions = ['U', 'R', 'D', 'L']  # Result from your algorithm\n"
                "print(\"**Actions List**:\", actions)  # MUST print the result\n"
                "```\n\n"
                "Note: If your algorithm produces numerical results, convert them using:\n"
                "action_map = {0:'U', 1:'D', 2:'L', 3:'R'}\n"
                "actions = [action_map[num] for num in numerical_results]\n"
                "print(\"**Actions List**:\", actions)\n\n"
            )
        elif self.benchmark == "suduku":
            formatted_prompt += (
                "4. For Sudoku, return the complete grid solution.\n"
                "5. CRITICAL: You MUST use print() to output the result.\n"
                "6. Example: print(\"**Actions List**:\", [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]])\n\n"
            )
        else:
            formatted_prompt += (
                "4. Actions should be represented as a list of strings: ['U', 'D', 'L', 'R'] (Up, Down, Left, Right)\n"
                "5. You may return either the complete action sequence to reach the goal, or a partial sequence if you're uncertain\n"
                "6. CRITICAL: You MUST use print() to output the result.\n"
                "7. Example: print(\"**Actions List**:\", [\"U\", \"R\", \"D\", \"L\"])\n\n"
            )
        
        if self.benchmark in ("plan_path", "sokoban"):
            formatted_prompt += ""
        elif self.benchmark == "suduku":
            formatted_prompt += (
                "Remember: Your code must use print() to output the result:\n"
                "print(\"**Actions List**:\", [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]])\n"
            )
        else:
            formatted_prompt += (
                "Remember: Your code must use print() to output the result:\n"
                "print(\"**Actions List**:\", [\"U\", \"R\", \"D\", \"L\"])  # If solved/no moves needed, use []\n"
            )
        
        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        """Extract code from model response"""
        if response is None:
            self.current_code = ""
            return self.current_code
            
        self.current_code = extract_code_from_response(response)
        return self.current_code

    async def step(self, env_data: Env, env_worker: Any = None):
        """Execute code, parse actions, score and update environment"""
        generated_code = self.current_code or ""
        if self.current_code is None:
            self.agent_reward = 0.0  # No code - failure
        env_data.state.code_generated_action = generated_code

        code_execution_output = None
        has_output = False
        try:
            code_execution_output = await get_code_execution_output(
                generated_code,
                timeout=20.0,
                ray_actor=env_worker,
            )
            env_data.state.code_execution_output = code_execution_output
            # Check if there is valid output (not null, not empty, not error)
            if code_execution_output and code_execution_output.strip() and not code_execution_output.startswith("error:"):
                has_output = True
        except Exception as e:
            code_execution_output = f"error: {e}"
            env_data.state.code_execution_output = code_execution_output

        env_data.state.tool_execution_output = code_execution_output
        env_data.state.tool_code = generated_code

        self.current_action = extract_actions_from_code_output(code_execution_output or "", self.benchmark)

        env_data.state.tool_action = self.current_action

        # Save to history after executing code and getting results
        if hasattr(env_data.state, 'code_history'):
            env_data.state.code_history.append(generated_code if generated_code else "")

        if hasattr(env_data.state, 'execution_history'):
            env_data.state.execution_history.append(code_execution_output if code_execution_output else "")

        if hasattr(env_data.state, 'tool_action_history'):
            env_data.state.tool_action_history.append(copy.deepcopy(self.current_action) if self.current_action else [])

        state = copy.deepcopy(env_data.state)
        state.step(self.current_action)

        # Reward logic:
        # - If code has print output (not null): 0.3
        # - If task is successful (done): 1.0
        # - Otherwise: 0.0
        if hasattr(state, 'done') and state.done:
            self.agent_reward = 1.0  # Success
            self.success = True
        elif has_output:
            self.agent_reward = 0.3  # Code has valid output
        else:
            self.agent_reward = 0.0  # No output or failure
    
    def calculate_reward(self, env_data: Env):
        self.agent_reward = self.agent_reward+env_data.state.reward
        self.reward_history.append(self.agent_reward)
    

    def reset(self):
        """Reset agent state"""
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.done = False
        self.is_pass = False
        self.success = False
        self.agent_reward = 0.0
