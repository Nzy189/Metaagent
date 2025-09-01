import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env.base.env import MultiAgentsEnvironment
from pettingllms.utils.logger_config import get_multi_logger
from alfworld.agents.environment import get_environment
from .utils import load_config_file
logger = logging.getLogger(__name__)

@dataclass
class AlfWorldState:
    task_desc: str = ""
    observation: str = ""
    inventory: str = ""
    admissible_actions: List[str] = None
    last_action: Optional[str] = None
    last_reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = None
    step_count: int = 0


class AlfWorldEnv(MultiAgentsEnvironment):
    """
    """
    def __init__(
        self,
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
        *,
        task_id: int | None = None,   # 新增：用于明确任务编号
        split: str | None = None,     # 新增：可在 Batch 内覆盖 train/valid
    ):
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, max_turns=max_turns, config=config)
        self.multi_logger = get_multi_logger()
        self.backend = "local_textworld"

        self._split = split or self._get_from_config(config, ["env", "split"], default="train")
        self._task_id = task_id if task_id is not None else env_idx
        self._load_raw_env(config)
        self._task_count, self._task_indexer = self._introspect_tasks(self.raw_env)

        # 归一化 task_id 到合法范围（允许超界循环）
        if self._task_count > 0:
            self._task_id = self._task_id % self._task_count

        self.state = AlfWorldState()
        self._reset_episode_inner()

    # ---------------------- 原生加载 ----------------------
    def _load_raw_env(self, config: Optional[dict]):
        split = self._split
        task_type = self._get_from_config(config, ["env", "task_type"], default=None)
        reward_type = self._get_from_config(config, ["env", "reward_type"], default="dense")
        extra = self._get_from_config(config, ["env", "kwargs"], default={}) or {}
        self.raw_env = get_environment("AlfredTWEnv")(split=split, task_type=task_type, reward=reward_type, **extra)

    @staticmethod
    def _get_from_config(cfg: Optional[dict], path: List[str], default=None):
        cur = cfg or {}
        for k in path:
            if not isinstance(cur, dict) or (k not in cur):
                return default
            cur = cur[k]
        return cur

    # ---------------------- 任务枚举/指派 ----------------------
    def _introspect_tasks(self, env) -> tuple[int, dict]:
        """
        返回 (任务总数, 指派方法字典)。
        指派方法字典中可能包含以下键：
          - "set_task_id": callable(idx)
          - "reset_with_index": callable(idx) -> (obs, info)
          - "set_attr_task_id": callable(idx)  # 通过 env.task_id = idx
        另外尝试读取 "gamefiles"/"task_list"/"task_files" 来估计任务总数。
        """
        indexer: Dict[str, Any] = {}

        # 1) 直接方法
        if hasattr(env, "set_task_id") and callable(getattr(env, "set_task_id")):
            indexer["set_task_id"] = env.set_task_id

        # 2) reset(index=…)
        def _reset_with_index(idx):
            return env.reset(index=idx)
        if hasattr(env, "reset"):
            try:
                # 探测是否支持 index 参数（不真正改变状态）
                # 部分实现对探测敏感，这里不执行，只登记方法，失败会在真正调用时回退
                indexer["reset_with_index"] = _reset_with_index
            except Exception:
                pass

        # 3) 通过属性 task_id
        if hasattr(env, "task_id"):
            def _set_attr_task_id(idx):
                setattr(env, "task_id", idx)
            indexer["set_attr_task_id"] = _set_attr_task_id

        # 估计任务数量（多路兜底）
        candidates = []
        for name in ("task_list", "task_files", "gamefiles", "games", "tasks"):
            if hasattr(env, name):
                try:
                    obj = getattr(env, name)
                    if isinstance(obj, (list, tuple)):
                        candidates.append(len(obj))
                except Exception:
                    pass

        # 有些实现把底层放在 env.env / env._env
        for holder in ("env", "_env", "venv"):
            sub = getattr(env, holder, None)
            if sub is None:
                continue
            for name in ("task_list", "task_files", "gamefiles", "games", "tasks"):
                if hasattr(sub, name):
                    try:
                        obj = getattr(sub, name)
                        if isinstance(obj, (list, tuple)):
                            candidates.append(len(obj))
                    except Exception:
                        pass

        task_count = max(candidates) if candidates else -1
        return task_count, indexer

    def _assign_task(self, task_id: int):
        """
        在真正 reset 前切换到底层的第 task_id 个任务。
        多种方式依次尝试，失败则跳过，最终让 reset() 自行决定。
        """
        # 优先专用 API
        if "set_task_id" in self._task_indexer:
            try:
                self._task_indexer["set_task_id"](task_id)
                return
            except Exception as e:
                logger.debug(f"set_task_id({task_id}) failed: {e}")

        # 其次通过属性
        if "set_attr_task_id" in self._task_indexer:
            try:
                self._task_indexer["set_attr_task_id"](task_id)
                return
            except Exception as e:
                logger.debug(f"set_attr_task_id({task_id}) failed: {e}")

        # reset(index=…) 会放在 _reset_episode_inner 里尝试

    # ---------------------- reset/step ----------------------
    def _reset_episode_inner(self):
        self.state = AlfWorldState()
        # 先尽力指派任务
        if self._task_count > 0:
            self._assign_task(self._task_id)

        # 真正 reset：优先尝试 reset(index=task_id)
        obs, info = None, None
        tried = False
        if "reset_with_index" in self._task_indexer and self._task_count > 0:
            try:
                obs, info = self._task_indexer["reset_with_index"](self._task_id)
                tried = True
            except Exception as e:
                logger.debug(f"reset(index={self._task_id}) failed: {e}")

        if not tried:
            obs, info = self.raw_env.reset()

        self._parse_obs_info(obs, info)
        self.state.step_count = 0

    def reset(self):
        self._reset_episode_inner()
        return self._pack_agent_observation()

    def step(self, action: str):
        """基本 step 封装。"""
        self.state.last_action = action
        obs, reward, done, info = self.raw_env.step(action)
        self.state.last_reward = float(reward or 0.0)
        self.state.done = bool(done)
        self._parse_obs_info(obs, info)
        self.state.step_count += 1
        return self._pack_agent_observation()

    def _parse_obs_info(self, obs: Any, info: Dict[str, Any] | None):
        info = info or {}
        if isinstance(obs, dict):
            observation = obs.get("obs", "") or obs.get("observation", "") or ""
            inventory = obs.get("inv", "") or obs.get("inventory", "") or ""
            task_desc = obs.get("task_desc", "") or info.get("task_desc", "")
        else:
            observation = str(obs) if obs is not None else ""
            inventory = info.get("inventory", "")
            task_desc = info.get("task_desc", "")

        admissible = info.get("admissible_commands") or info.get("admissible_actions") or []
        if admissible is None:
            admissible = []

        self.state.task_desc = task_desc or ""
        self.state.observation = observation or ""
        self.state.inventory = inventory or ""
        self.state.admissible_actions = list(admissible)
        self.state.info = info

    def _pack_agent_observation(self) -> Dict[str, Any]:
        s = self.state
        return {
            "task_desc": s.task_desc,
            "observation": s.observation,
            "inventory": s.inventory,
            "admissible_actions": s.admissible_actions or [],
            "last_action": s.last_action,
            "last_reward": s.last_reward,
            "done": s.done,
            "step_count": s.step_count,
        }


from typing import List

class AlfWorldEnvBatch:
    """
   
    """
    def __init__(
        self,
        env_idx_list: List[int],
        rollout_idx_list: List[int],
        samples: int,
        max_turns: int,
        config: dict,
        mode: str = "train",
        *,
        env_workers: List = None,
    ):
        self.mode = mode
        alf_config_path = "pettingllms/multi_agent_env/alfworld/configs/config_tw.yaml"
        config = load_config_file(alf_config_path)
        base_split = (config or {}).get("env", {}).get("split", "train")
        split = "valid" if mode in ("validate", "test") else "train"

        # 准备一个临时环境，只为探测任务总数
        _probe_env = AlfWorldEnv(
            env_idx=0,
            rollout_idx=0,
            max_turns=max_turns,
            config=config,
            split=split,
            task_id=0,
        )
        total_tasks = max(_probe_env._task_count, 0)

        if mode == "train":
            k = len(env_idx_list)
            if total_tasks > 0:
                task_ids = list(range(min(k, total_tasks)))
            else:
                # 无法探测到任务总数时，按 env_idx_list 直接用（让底层 env 自行处理）
                task_ids = list(range(k))
            # 使用传入的 samples
            eff_samples = samples
            eff_rollouts = rollout_idx_list
            if len(eff_rollouts) != len(task_ids) * eff_samples:
                raise ValueError(
                    f"rollout_idx_list 长度不匹配，期望 {len(task_ids)*eff_samples}，实际 {len(eff_rollouts)}"
                )
        else:
            # 验证/测试：加载全部任务；samples=1；自动生成 rollout_idx_list
            eff_samples = 1
            task_ids = list(range(total_tasks)) if total_tasks > 0 else [0]
            eff_rollouts = list(range(len(task_ids) * eff_samples))

        self.env_list = []
        # 实例化所有 env
        for i, task_id in enumerate(task_ids):
            for s in range(eff_samples):
                env = AlfWorldEnv(
                    env_idx=i,
                    rollout_idx=eff_rollouts[i * eff_samples + s],
                    max_turns=max_turns,
                    config=config,
                    split=split,
                    task_id=task_id,
                )
                self.env_list.append(env)

        # 最终一致性检查（与 CodeTestEnvBatch 行为对齐）
        if len(self.env_list) != len(eff_rollouts):
            raise ValueError(
                f"len(self.env_list) != len(rollout_idx_list): {len(self.env_list)} != {len(eff_rollouts)}"
            )
