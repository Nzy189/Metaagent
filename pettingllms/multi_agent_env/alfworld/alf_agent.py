import logging
from typing import Any, Dict, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.logger_config import get_multi_logger

# 我们将在 utils.py 里实现下面这些
from .utils import (
    build_prompt_from_obs,
    extract_action_from_text,
    choose_executable_action,
)

logger = logging.getLogger(__name__)


class AlfWorldAgent(Agent):

    def __init__(self, rollout_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)
        self.multi_logger = get_multi_logger()
        self.current_prompt = None
        self.current_action = None

    def update_from_env(self, env_data: Env):
        """将 ALFWorld 的观测组装成 LM 提示词。"""
        self.env_data = env_data
        obs = getattr(env_data, "agent_observations", None) or {}
        self.current_prompt = {"text": build_prompt_from_obs(obs), "image": None}

    def update_from_model(self, response: str):
        """从模型文本中抽取动作（字符串），并尝试对齐 admissible actions。"""
        obs = getattr(self.env_data, "agent_observations", None) or {}
        admissible = obs.get("admissible_actions", []) or []
        # 1) 从输出抽取“意图动作”字符串
        intent = extract_action_from_text(response)
        # 2) 对齐到可执行动作（若需要模糊匹配/模板补全）
        action = choose_executable_action(intent, admissible)
        self.current_action = action
        return action

    async def step(self, env_data: Env, env_worker: Any = None):
        """把 current_action 交给 env 执行，并更新奖励与 done。"""
        action = self.current_action
        obs, reward, done, info = await env_data.step(action, env_worker=env_worker)

        # 简单的增量奖励：与上一步对比（也可改为累计）
        if len(self.reward_history) > 0:
            self.agent_reward = float(reward) - float(self.reward_history[-1])
        else:
            self.agent_reward = float(reward)
        self.reward_history.append(float(reward))
        self.done = bool(done)
        self.current_info = info

    def calculate_reward(self, env_data_list):
        """在本任务里，reward 由环境直接提供，一般无需额外计算。"""
        # 保持兼容接口：返回最近一次奖励
        if self.reward_history:
            return float(self.reward_history[-1])
        return 0.0

    def reset(self):
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.done = False
        self.reward_history.clear()
