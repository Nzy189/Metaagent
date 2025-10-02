import logging
import re
from typing import Any, List, Tuple, Optional
import random

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.stateful.utils import (
    _extract_actions, _extract_path, _actions_to_path, _format_grid
)

from pettingllms.multi_agent_env.math.math_utils import get_code_execution_output
from pettingllms.multi_agent_env.stateful.prompt import build_tool_prompt
import asyncio
logger = logging.getLogger(__name__)
import copy

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[:length] + "..."


def generate_sudoku_example(n: int, seed: int = 42) -> List[List[int]]:
    random.seed(seed)
    
    grid = [[0 for _ in range(n)] for _ in range(n)]
    
    if n == 4:
        grid = [
            [1, 2, 3, 4],
            [3, 4, 1, 2], 
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ]
    elif n == 9:
        grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
    else:
        for i in range(min(n, 3)):  
            for j in range(min(n, 3)):
                grid[i][j] = (i * n + j) % n + 1
    
    return grid


def extract_code_from_response(response: str) -> str:

    if response is None or not isinstance(response, str):
        return ""
    
    
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()  
    
    
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    
    incomplete_python_pattern = r'```python\s*(.*?)$'
    matches = re.findall(incomplete_python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    
    incomplete_code_pattern = r'```\s*(.*?)$'
    matches = re.findall(incomplete_code_pattern, response, re.DOTALL)
    
    if matches:
        code = matches[-1].strip()
        if any(keyword in code for keyword in ['def ', 'import ', 'from ', '=', 'print(', 'return', 'if ', 'for ', 'while ']):
            return code
    
    return response.strip()


def extract_actions_from_code_output(output: str, benchmark: str = "plan_path") -> Optional[List]:
    """
    从代码执行输出中提取动作序列，支持不同benchmark格式。
    
    Args:
        output: 代码执行的输出字符串
        benchmark: benchmark类型，决定期望的action格式
        
    Returns:
        动作序列列表，格式取决于benchmark
    """
    
    if output is None or not isinstance(output, str) or output.startswith("error:"):
        return None
    
    try:
        actions_pattern = r'\*\*Actions List\*\*:\s*(\[.*?\])'
        matches = re.findall(actions_pattern, output, re.DOTALL)
        
        if matches:
            actions_str = matches[-1]
            try:
                actions = eval(actions_str)
                if isinstance(actions, list):
                    if benchmark == "plan_path":
                        if all(isinstance(action, str) and action in ['U', 'D', 'L', 'R'] for action in actions):
                            return actions
                    elif benchmark == "sudoku4x4":
                        
                        if (len(actions) > 0 and isinstance(actions[0], list) and 
                            all(isinstance(row, list) and len(row) > 0 for row in actions)):
                            return actions
                       
                        elif all(isinstance(step, list) and len(step) == 3 for step in actions):
                            return actions
                    else:
                        return actions  
            except:
                pass
        
        
        actions_pattern2 = r'Actions:\s*(\[.*?\])'
        matches2 = re.findall(actions_pattern2, output, re.DOTALL)
        
        if matches2:
            actions_str = matches2[-1]
            try:
                actions = eval(actions_str)
                if isinstance(actions, list):
                    if benchmark == "plan_path":
                        if all(isinstance(action, str) and action in ['U', 'D', 'L', 'R'] for action in actions):
                            return actions
                    elif benchmark == "sudoku4x4":
                        if (len(actions) > 0 and isinstance(actions[0], list)):
                            return actions
                    else:
                        return actions
            except:
                pass
        
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    parsed = eval(line)
                    if isinstance(parsed, list):
                        if benchmark == "plan_path":
                            if all(isinstance(item, str) and item in ['U', 'D', 'L', 'R'] for item in parsed):
                                return parsed
                        elif benchmark == "sudoku4x4":
                            if (len(parsed) > 0 and isinstance(parsed[0], list)):
                                return parsed
                        else:
                            return parsed
                except:
                    continue
        
    except Exception as e:
        logger.warning(f"Failed to extract actions from code output: {e}")
    
    return None


class ToolAgent(Agent):
    """
    Code-generation style planning agent.
    - Only initial/subsequent prompts are determined by benchmark (externalized).
    - Other logic (execution, parsing, scoring, write-back, done determination) remains unchanged.
    """

    def __init__(self, rollout_idx: int | None = None, benchmark: str = "plan_path", **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.benchmark = benchmark  # Key: switch prompts for different tasks
        self.agent_reward_history = []
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)

    def update_from_env(self, turn_idx: int, env_data: Env):
        self.env_data = env_data
        state = getattr(env_data, "state", None)
        formatted_prompt = "You are an AI assistant specialized in solving planning problems through code generation. Very important, use the algorithm like BFS or A* to solve the problem. Do not reasoning directly. Your task is to analyze the given scenario and generate Python code that produces a sequence of actions to solve the problem.\n\nInstructions:\n1. Write Python code enclosed in ```python and ``` tags\n2. Your code should output an action sequence using print() in EXACTLY one of these formats: \n   - **Actions List**: [\"U\",\"D\",\"L\",\"R\"]\n   - Actions: [\"U\",\"D\",\"L\",\"R\"]\n"
        formatted_prompt+= build_tool_prompt(self.benchmark, turn_idx, state)
        if self.benchmark in ("plan_path", "sokoban"):
            formatted_prompt+= "3. Your code must compute moves from the given state; \n2. Output format must be EXACTLY: **Actions List**: [\"U\",\"D\",\"L\",\"R\"] (or empty []).\n3. If you cannot solve fully, output a partial but valid list.\n\n 4. Please use algorithm like BFS or A* to solve the problem. Very important, do not print action list directly. 5. Please print the final result. You must use 'print' to print the final result."
        elif self.benchmark == "sudoku4x4":
            # 从 state 中获取数独大小
            sudoku_size = 4  # 默认大小
            sudoku_size = state.config.map_size
            
            # 生成动态示例
            example_grid = generate_sudoku_example(sudoku_size)
            example_str = str(example_grid).replace(' ', '')  # 移除空格以匹配格式
            
            formatted_prompt+= f"3. For Sudoku, actions should be either:\n   - Complete grid: {example_str} (example for {sudoku_size}x{sudoku_size} sudoku)\n   - Fill steps: [[row,col,value], [row,col,value], ...] (0-indexed)\n4. You may return either format depending on your solving approach\n5. Ensure your code is executable and produces clear output\n\n"
        else:
            formatted_prompt+= "3. Actions should be represented as a list of strings: ['U', 'D', 'L', 'R'] (Up, Down, Left, Right)\n4. You may return either the complete action sequence to reach the goal, or a partial sequence if you're uncertain\n5. Ensure your code is executable and produces clear output\n\n"
        
        if self.benchmark in ("plan_path", "sokoban"):
            formatted_prompt+= f""
            #formatted_prompt+= f"Important: Your code must output the final action sequence in this exact format:\n**Actions List**: [\"U\", \"R\", \"D\", \"L\"] (example for path planning)\n\n"
            #formatted_prompt+= f"Important: Your code must output the final action sequence in this exact format:\n**Actions List**: [\"U\", \"R\", \"D\", \"L\"] (example for path planning)\n\nNote: If your algorithm produces numerical results, convert them using action_map = {{0:'U', 1:'D', 2:'L', 3:'R'}} before outputting.\n"
        elif self.benchmark == "sudoku4x4":
            sudoku_size = 4 
            if state and hasattr(state, 'size'):
                sudoku_size = state.size
            elif state and hasattr(state, 'config'):
                if hasattr(state.config, 'map_size'):
                    sudoku_size = state.config.map_size
                elif isinstance(state.config, dict) and 'map_size' in state.config:
                    sudoku_size = state.config['map_size']
            
            example_grid = generate_sudoku_example(sudoku_size)
            example_str = str(example_grid).replace(' ', '')  
            
            formatted_prompt+= f"Important: Your code must output the final action in one of these exact formats:\n**Actions List**: {example_str} (complete grid) for {sudoku_size}x{sudoku_size} sudoku.\n"
        else:
            formatted_prompt+= f"Important: Your code must output the final action sequence in this exact format:\n**Actions List**: [\"U\", \"R\", \"D\", \"L\"] (example). If solved/no moves needed, output an empty list [].\n"
        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        if response is None:
            self.current_code = ""
            return self.current_code
            
        self.current_code = extract_code_from_response(response)
        return self.current_code

    async def step(self, env_data: Env, env_worker: Any = None):
        # === 以下保持你的原始实现（执行代码 -> 解析 -> 评分 -> 回写） ===
        
        generated_code = self.current_code or ""
        if self.current_code is None:
            self.agent_reward = -1
        env_data.state.code_generated_action = generated_code

        code_execution_output = None
        try:
            code_execution_output = await get_code_execution_output(
                generated_code,
                timeout=20.0,
                ray_actor=env_worker,
            )
            env_data.state.code_execution_output = code_execution_output
        except Exception as e:
            code_execution_output = f"error: {e}"
            env_data.state.code_execution_output = code_execution_output
        if code_execution_output is None:
            self.agent_reward = -2
        env_data.state.tool_execution_output = code_execution_output
        env_data.state.tool_code = generated_code
        self.current_action = extract_actions_from_code_output(code_execution_output or "", self.benchmark)
        
        env_data.state.tool_action = self.current_action
        state = copy.deepcopy(env_data.state)
        state.step(self.current_action)
        if self.benchmark in ("plan_path", "sokoban") and self.current_action is None:
            self.agent_reward = -2
        else:
            self.agent_reward =  state.reward
        self.agent_reward_history.append(self.agent_reward)
        
        
        if hasattr(state, 'done') and env_data.state.done:
            
            if self.benchmark == "plan_path":
                
                if hasattr(state, 'pos') and hasattr(state, 'goal') and state.pos == state.goal:
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)  
            elif self.benchmark == "eight_queens":
                
                if hasattr(state, '_is_solved') and state._is_solved():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
            elif self.benchmark == "blocksworld":
                
                if hasattr(state, '_is_goal_reached') and state._is_goal_reached():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
            elif self.benchmark == "sudoku4x4":
                
                if hasattr(state, '_is_solved') and state._is_solved():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
        
        if self.agent_reward is None:
            self.agent_reward = 0.0
        

    def reset(self):
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.done = False
        self.is_pass = False