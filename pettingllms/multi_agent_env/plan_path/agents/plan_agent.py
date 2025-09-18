import re
import json
import copy
import logging
import ast
from typing import Any, List, Tuple, Dict, Optional

from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.multi_agent_env.plan_path.prompt import build_plan_prompt
from pettingllms.multi_agent_env.plan_path.utils import _extract_actions, _extract_path, _actions_to_path, _format_grid
logger = logging.getLogger(__name__)

def truncatefn(s, length=300):
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= length else s[: length // 2] + "...(truncated)..." + s[-length // 2 :]



def extract_final_action(text: str, benchmark: str = "plan_path") -> List | None:
    """
    Extract the final action that appears on the last line starting with '#### '.
    Support different formats for different benchmarks.
    """
    # 安全检查：确保text不为None
    if text is None or not isinstance(text, str):
        return None
    
    # Find all lines starting with '#### '
    pattern = re.compile(r'(?m)^\s*####\s+(.+)$', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        return None
    
    action_str = matches[-1].strip()
    
    try:
        # 尝试使用ast.literal_eval解析（支持嵌套列表）
        if action_str.startswith('[') and action_str.endswith(']'):
            parsed = ast.literal_eval(action_str)
            if isinstance(parsed, list):
                return parsed
    except (ValueError, SyntaxError):
        pass
    
    try:
        # 尝试直接解析JSON格式
        if action_str.startswith('[') and action_str.endswith(']'):
            return json.loads(action_str)
    except json.JSONDecodeError:
        pass
    
    try:
        # 处理没有引号的简单格式，如 [U,R,D,L]（仅适用于plan_path）
        if benchmark == "plan_path" and action_str.startswith('[') and action_str.endswith(']'):
            # 移除方括号
            inner = action_str[1:-1].strip()
            if inner:
                # 分割并清理每个动作
                actions = [item.strip().strip('"\'') for item in inner.split(',')]
                return [action for action in actions if action]
            else:
                return []
    except Exception:
        pass
    
    # If all failed, return None
    return None


class PlanAgent(Agent):
    """
    Unified PlanWalker:
    - benchmark: plan_path | eight_queens | blocksworld | sudoku4x4
    - Only prompt changes with benchmark; evaluation and write-back pipeline remains consistent.
    """

    def __init__(self, rollout_idx: int | None = None, benchmark: str = "plan_path", **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.benchmark = benchmark
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)
        self.multi_logger = get_multi_logger()
        self.action_list = []
        self.state_list = []

    def reset(self):
        self.action_list = []
        self.state_list = []

    # ===================== Prompt Construction (Externalized) =====================
    def update_from_env(self, turn_idx: int, env_data: Env):
        self.env_data = env_data
        state = getattr(env_data, "state", None)
        formatted_prompt = f"You are a planning and reasoning agent. You will receive: The original task description, The Code Agent’s code, The code execution output. Your job is to reason carefully, decide the final action, and format your response exactly as specified. Instructions: Read the task, inspect the code, and verify the execution output against the task requirements. If the code/output is correct and sufficient, adopt it; otherwise, improve or override it with your own reasoning. Keep your reasoning concise but explicit: justify why the final action is correct. Formatting is mandatory. Please give the final action list after ####. "
        if self.benchmark == "plan_path":
            formatted_prompt+= "Example: #### [U,R,D,L].\n"
        if self.benchmark == "sudoku4x4":
            formatted_prompt+= "Example: #### [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]] or #### [[0,1,2],[0,2,3],[1,0,4]].\n"

       
        formatted_prompt+= build_plan_prompt(self.benchmark,turn_idx, state)
        formatted_prompt+= f"Here is code agent's code: {state.tool_code}.\n"
        formatted_prompt+= f"Here is code agent's execution output: {state.tool_execution_output}. "
        formatted_prompt+= f"Here is code agent's action: {state.tool_action}.\n"
        if turn_idx > 0:
            for i, action in enumerate(self.action_list):
                formatted_prompt+= f"The {i+1}th action is {action}. The {i+1}th state is {self.state_list[i]}.\n"
            formatted_prompt+= f"Please reason step by step and give the final action list after ####."
            if self.benchmark == "plan_path":
                formatted_prompt+= f"Example: #### [U,R,D,L].\n"
            if self.benchmark == "sudoku4x4":
                formatted_prompt+= f"Example: #### [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]] or [[0,1,2],[0,2,3],[1,0,4]].\n"

            
        self.current_prompt = {"text": formatted_prompt, "image": None}

    
    def update_from_model(self, response: str):
        # 安全检查：确保response不为None
        if response is None:
            self.current_action = []
            return self.current_action
            
        self.current_action = extract_final_action(response, self.benchmark)
        if self.current_action is None:
            self.current_action = []
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        env_data.state.plan_action = self.current_action
        state = env_data.state
        self.state_list.append(state)
        state.step(self.current_action)
        self.action_list.append(self.current_action)
        if self.current_action is None or self.current_action == []:
            self.agent_reward = -5
        else:
            self.agent_reward = state.reward
        
        # 检查是否成功完成任务
        if hasattr(state, 'done') and state.done:
            # 根据不同的 benchmark 检查成功条件
            if self.benchmark == "plan_path":
                # PlanPath: 检查是否到达目标位置
                if hasattr(state, 'pos') and hasattr(state, 'goal') and state.pos == state.goal:
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)  # 确保成功时有正奖励
            elif self.benchmark == "eight_queens":
                # EightQueens: 检查是否正确放置了所有皇后
                if hasattr(state, '_is_solved') and state._is_solved():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
            elif self.benchmark == "blocksworld":
                # Blocksworld: 检查是否达到目标配置
                if hasattr(state, '_is_goal_reached') and state._is_goal_reached():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
            elif self.benchmark == "sudoku4x4":
                # Sudoku4x4: 检查是否正确解决数独
                if hasattr(state, '_is_solved') and state._is_solved():
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = max(self.agent_reward, 1.0)
        
        # 确保 agent_reward 不为 None
        if self.agent_reward is None:
            self.agent_reward = 0.0
        