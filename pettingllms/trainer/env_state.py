from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import copy
import math

# =========================================================
# 1) Eight Queens (N-Queens) EnvState
# =========================================================

class EightQueensEnvState:
    """
    N 皇后（默认 N=8）：
    - 状态：按行布置，每行一个列号，-1 表示未放。
    - 动作：place(row, col) -> 将第 row 行的皇后放到 col（或 col=-1 表示清空该行）。
    - 逐步终止条件：所有行均已布置且无冲突 -> 成功；或步数超限 -> 失败。
    - 一次性评测：传入完整的列数组 solution: List[int]（长度 N），评估可行性和奖励。
    """

    DEFAULT_R_STEP      = -0.01   # 每步轻微惩罚
    DEFAULT_R_INVALID   = -0.10   # 非法放置（越界/行不匹配/冲突）惩罚
    DEFAULT_R_GOAL      = +1.00   # 成功摆放 N 皇后
    DEFAULT_R_OPT       = +0.50   # “最优”加成：按定义为行序逐放、从空到满且无任意覆盖/撤销（即最少 N 步）
    DEFAULT_R_FAIL      = -1.00   # 失败
    DEFAULT_MAX_STEPS   = 10_000

    def __init__(
        self,
        N: int = 8,
        *,
        r_step: float = None,
        r_invalid: float = None,
        r_goal: float = None,
        r_opt: float = None,
        r_fail: float = None,
        max_steps: int = None,
    ):
        self.N = N
        self.r_step     = self.DEFAULT_R_STEP     if r_step     is None else r_step
        self.r_invalid  = self.DEFAULT_R_INVALID  if r_invalid  is None else r_invalid
        self.r_goal     = self.DEFAULT_R_GOAL     if r_goal     is None else r_goal
        self.r_opt      = self.DEFAULT_R_OPT      if r_opt      is None else r_opt
        self.r_fail     = self.DEFAULT_R_FAIL     if r_fail     is None else r_fail
        self.max_steps  = self.DEFAULT_MAX_STEPS  if max_steps  is None else max_steps
        self.reset_agent()

    # ---------- 基础 ----------
    def conflicts(self, state_cols: List[int]) -> int:
        """返回冲突对数（对角线/同行/同列；同行由设计已避免，仅计列/斜线）"""
        N = len(state_cols)
        cnt = 0
        for r1 in range(N):
            c1 = state_cols[r1]
            if c1 < 0:  # 未放置，不产生冲突
                continue
            for r2 in range(r1 + 1, N):
                c2 = state_cols[r2]
                if c2 < 0:
                    continue
                if c1 == c2 or abs(c1 - c2) == abs(r1 - r2):
                    cnt += 1
        return cnt

    def is_goal(self, state_cols: List[int]) -> bool:
        return all(c >= 0 for c in state_cols) and self.conflicts(state_cols) == 0

    def describe(self) -> str:
        return f"EightQueensWorker: Place N queens on {self.N}x{self.N} board, one per row, no conflicts."

    # ---------- 逐步交互 ----------
    def reset_agent(self):
        self.cols: List[int] = [-1] * self.N
        self.steps: int = 0
        self.done: bool = False
        self.total_reward: float = 0.0
        self.invalid_count: int = 0
        self.clean_forward_only: bool = True  # 用于“最优”判定：是否严格从空到满、每行恰一步
        self.rows_placed_once: Set[int] = set()  # 记录每行是否仅放置过一次

    def get_valid_actions(self) -> List[Tuple[int,int]]:
        """
        返回当前所有可选动作 (row, col)，以及 (row, -1) 用于清空该行。
        为了简洁，不提前剪冲突（允许放置但若冲突则记为 invalid）。
        """
        acts = []
        for r in range(self.N):
            for c in range(-1, self.N):  # -1 表示清空该行
                acts.append((r, c))
        return acts

    def step(self, action: Tuple[int,int]):
        """
        action = (row, col)；col=-1 表示清空 row 行
        奖励：r_step 基础步惩罚 + (若冲突/越界) r_invalid；
             若达到目标：+r_goal；若恰好 N 步从空到满且无覆盖/撤销：+r_opt
        """
        if self.done:
            return copy.deepcopy(self.cols), 0.0, True, {"msg": "episode already done"}

        row, col = action
        reward = self.r_step
        valid = True

        if not (0 <= row < self.N) or not (-1 <= col < self.N):
            valid = False
        else:
            prev = self.cols[row]
            self.cols[row] = col
            # 冲突即时判定
            if self.conflicts(self.cols) > 0:
                valid = False

            # “最优”轨迹检测：必须每行只放置一次，且不允许清空/覆盖
            if col == -1:
                self.clean_forward_only = False
            else:
                if prev != -1:
                    # 覆盖或重复设置
                    self.clean_forward_only = False
                else:
                    # 首次设置
                    if row in self.rows_placed_once:
                        self.clean_forward_only = False
                    self.rows_placed_once.add(row)

            if not valid:
                # 回滚以保持状态可解释
                self.cols[row] = prev

        if not valid:
            reward += self.r_invalid
            self.invalid_count += 1

        self.steps += 1

        if self.is_goal(self.cols):
            reward += self.r_goal
            # “最优”定义：恰好 N 步完成、每行恰放一次且无撤销/覆盖
            if self.clean_forward_only and len(self.rows_placed_once) == self.N and self.steps == self.N:
                reward += self.r_opt
            self.done = True
        elif self.steps >= self.max_steps:
            reward += self.r_fail
            self.done = True

        self.total_reward += reward
        info = {
            "valid": valid,
            "steps": self.steps,
            "invalid_count": self.invalid_count,
            "done": self.done,
            "cols": copy.deepcopy(self.cols),
        }
        return copy.deepcopy(self.cols), reward, self.done, info

    # ---------- 一次性评测 ----------
    def reward_for_solution(self, solution: List[int]) -> Tuple[float, Dict[str,Any]]:
        """
        传入长度 N 的列数组（每行一个 col，-1 代表未放），打分：
          - 全部放置且无冲突：r_goal（若恰好 N 个非 -1 且无覆盖语义，可视为最优：+r_opt）
          - 否则：r_fail
        再加上 r_step * L 和 r_invalid * invalid（若出现越界/冲突）
        """
        invalid = 0
        N = self.N
        if len(solution) != N:
            return self.r_fail, {"feasible": False, "optimal": None}

        # 基础非法检查
        for c in solution:
            if not (-1 <= c < N):
                invalid += 1
        conf = self.conflicts(solution)
        invalid += conf

        filled = sum(1 for c in solution if c >= 0)
        feasible = (filled == N and conf == 0)
        L = filled  # 填放次数视为“步数”

        R = self.r_step * L + self.r_invalid * invalid
        if feasible:
            R += self.r_goal
            # “最优”粗定义：恰好 N 次填放
            if filled == N:
                R += self.r_opt
        else:
            R += self.r_fail

        info = {
            "feasible": feasible,
            "optimal": (True if feasible and filled == N else False if feasible else None),
            "conflicts": conf,
            "filled": filled,
            "steps": L,
        }
        return R, info


# =========================================================
# 2) Blocksworld EnvState
# =========================================================

@dataclass
class BWAction:
    kind: str          # "move"
    block: str
    dest: str          # 目标块名 或 "table"

class BlocksworldEnvState:
    """
    经典方块世界（Blocks World）：
    - 状态：若干堆栈（table 上若干 tower），每个 block 名称唯一。
      例：[['A','B'], ['C']] 表示 B 在 A 上，C 单独在桌面，列表末端为顶部。
    - 动作：("move", x, y) 将清空块 x 移到目的 y 顶部（或 y="table" 放到桌面生成新堆）。
      约束：x 必须 clear；若 y != "table"，则 y 必须 clear；禁止将 x 移到自身或其上方堆形成环。
    - 终止：达到给定目标配置（忽略塔的顺序还是严格相等可选）；或步数超限。
    - “最优”：若提供 `min_opt_steps`（外部先验）则与之比较；否则返回 None。
    """

    DEFAULT_R_STEP      = -0.01
    DEFAULT_R_INVALID   = -0.10
    DEFAULT_R_GOAL      = +1.00
    DEFAULT_R_OPT       = +0.50
    DEFAULT_R_FAIL      = -1.00
    DEFAULT_MAX_STEPS   = 10_000

    def __init__(
        self,
        init_stacks: List[List[str]],
        goal_stacks: List[List[str]],
        *,
        strict_goal_match: bool = True,   # True: 堆序和塔内顺序都必须完全一致；False: 忽略塔集合的排列顺序（多集合相等）
        min_opt_steps: Optional[int] = None,  # 若已知最短步，最优时给加成
        r_step: float = None,
        r_invalid: float = None,
        r_goal: float = None,
        r_opt: float = None,
        r_fail: float = None,
        max_steps: int = None,
    ):
        self.init_stacks = copy.deepcopy([list(s) for s in init_stacks])
        self.goal_stacks = copy.deepcopy([list(s) for s in goal_stacks])
        self.strict = strict_goal_match
        self.min_opt_steps = min_opt_steps

        self.r_step     = self.DEFAULT_R_STEP     if r_step     is None else r_step
        self.r_invalid  = self.DEFAULT_R_INVALID  if r_invalid  is None else r_invalid
        self.r_goal     = self.DEFAULT_R_GOAL     if r_goal     is None else r_goal
        self.r_opt      = self.DEFAULT_R_OPT      if r_opt      is None else r_opt
        self.r_fail     = self.DEFAULT_R_FAIL     if r_fail     is None else r_fail
        self.max_steps  = self.DEFAULT_MAX_STEPS  if max_steps  is None else max_steps

        # 快速索引
        self.all_blocks: Set[str] = set(sum(self.init_stacks, []))
        self.reset_agent()

    # ---------- 工具 ----------
    @staticmethod
    def _norm_stacks(stacks: List[List[str]]) -> List[List[str]]:
        """标准化副本：每堆从底到顶；copy"""
        return [list(t) for t in stacks]

    def is_clear(self, stacks: List[List[str]], block: str) -> bool:
        """块是该塔的顶端即可视为 clear"""
        for tower in stacks:
            if tower and tower[-1] == block:
                return True
        return False

    def find_block(self, stacks: List[List[str]], block: str) -> Tuple[int,int]:
        """返回 (塔索引, 层索引)，层索引 0 底部，len-1 顶部；若不存在抛错"""
        for si, tower in enumerate(stacks):
            for li, b in enumerate(tower):
                if b == block:
                    return si, li
        raise ValueError(f"block {block} not found")

    def equal_stacks(self, A: List[List[str]], B: List[List[str]]) -> bool:
        if self.strict:
            return A == B
        # 非严格：视为多集合相等（塔的顺序不敏感）
        def canon(X):
            return sorted([tuple(t) for t in X])
        return canon(A) == canon(B)

    def is_goal(self, stacks: List[List[str]]) -> bool:
        return self.equal_stacks(self._norm_stacks(stacks), self._norm_stacks(self.goal_stacks))

    def describe(self) -> str:
        sg = "strict" if self.strict else "permutation-invariant"
        return f"BlocksworldWorker: move clear blocks among stacks; goal match={sg}."

    # ---------- 逐步交互 ----------
    def reset_agent(self):
        self.stacks: List[List[str]] = self._norm_stacks(self.init_stacks)
        self.steps: int = 0
        self.done: bool = False
        self.total_reward: float = 0.0
        self.invalid_count: int = 0

    def get_valid_actions(self) -> List[BWAction]:
        acts: List[BWAction] = []
        # 枚举所有清空块
        clear_blocks = [tower[-1] for tower in self.stacks if tower]
        for x in clear_blocks:
            # 移到桌面
            acts.append(BWAction("move", x, "table"))
            # 移到其他清空块之上
            for y in clear_blocks:
                if y != x:
                    acts.append(BWAction("move", x, y))
        return acts

    def _apply_move(self, stacks: List[List[str]], x: str, dest: str) -> Tuple[List[List[str]], bool]:
        stacks = self._norm_stacks(stacks)
        if x not in self.all_blocks:
            return stacks, False

        # x 必须 clear
        if not self.is_clear(stacks, x):
            return stacks, False

        # 找到 x
        sx, lx = self.find_block(stacks, x)
        if lx != len(stacks[sx]) - 1:
            return stacks, False

        # 弹出 x
        stacks[sx].pop()
        if len(stacks[sx]) == 0:
            # 清空塔可移除空列表，也可保留，这里保留但允许空塔
            pass

        if dest == "table":
            stacks.append([x])
            return stacks, True

        if dest not in self.all_blocks:
            return stacks, False

        # 目标必须 clear
        if not self.is_clear(stacks, dest):
            return stacks, False

        sd, ld = self.find_block(stacks, dest)
        if ld != len(stacks[sd]) - 1:
            return stacks, False

        if dest == x:
            return stacks, False

        # 放置
        stacks[sd].append(x)
        return stacks, True

    def step(self, action: BWAction):
        """
        执行动作 ("move", block, dest)，dest ∈ {"table"} ∪ blocks
        奖励：r_step + (invalid ? r_invalid) + (若达成目标 r_goal + 若最优 r_opt)
        """
        if self.done:
            return copy.deepcopy(self.stacks), 0.0, True, {"msg": "episode already done"}

        reward = self.r_step
        valid = False

        if action.kind == "move":
            new_stacks, ok = self._apply_move(self.stacks, action.block, action.dest)
            valid = ok
            if ok:
                self.stacks = new_stacks
        else:
            # 仅支持 move
            valid = False

        if not valid:
            reward += self.r_invalid
            self.invalid_count += 1

        self.steps += 1

        if self.is_goal(self.stacks):
            reward += self.r_goal
            if self.min_opt_steps is not None and self.steps == self.min_opt_steps:
                reward += self.r_opt
            self.done = True
        elif self.steps >= self.max_steps:
            reward += self.r_fail
            self.done = True

        self.total_reward += reward
        info = {
            "valid": valid,
            "steps": self.steps,
            "invalid_count": self.invalid_count,
            "done": self.done,
            "stacks": copy.deepcopy(self.stacks),
        }
        return copy.deepcopy(self.stacks), reward, self.done, info

    # ---------- 一次性评测 ----------
    def reward_for_plan(self, plan: List[BWAction]) -> Tuple[float, Dict[str,Any]]:
        stacks = self._norm_stacks(self.init_stacks)
        invalid = 0
        L = 0
        for act in plan:
            L += 1
            if act.kind != "move":
                invalid += 1
                continue
            stacks2, ok = self._apply_move(stacks, act.block, act.dest)
            if ok:
                stacks = stacks2
            else:
                invalid += 1

        feasible = self.is_goal(stacks)
        R = self.r_step * L + self.r_invalid * invalid
        if feasible:
            R += self.r_goal
            if self.min_opt_steps is not None and L == self.min_opt_steps:
                R += self.r_opt
        else:
            R += self.r_fail

        info = {
            "feasible": feasible,
            "optimal": (L == self.min_opt_steps) if (feasible and self.min_opt_steps is not None) else None,
            "steps": L,
            "invalid_steps": invalid,
            "final_stacks": stacks,
        }
        return R, info


# =========================================================
# 3) Sudoku 4x4 EnvState
# =========================================================

class Sudoku4x4EnvState:
    """
    4×4 数独：
    - 取值：1..4
    - 规则：行/列/2×2 宫内不重复
    - 动作：(r, c, v) 将单元格 (r,c) 设为 v（0-based）
    - 初始：给定 puzzle（0 代表空）；目标：填满且满足规则
    - “最优”：若恰好用 initial_empty 个填入步完成（只填空格，从不覆盖/清空）
    """

    DEFAULT_R_STEP      = -0.01
    DEFAULT_R_INVALID   = -0.10
    DEFAULT_R_GOAL      = +1.00
    DEFAULT_R_OPT       = +0.50
    DEFAULT_R_FAIL      = -1.00
    DEFAULT_MAX_STEPS   = 10_000

    def __init__(
        self,
        puzzle: List[List[int]],  # 4x4, 元素 0..4；0 表示空
        *,
        r_step: float = None,
        r_invalid: float = None,
        r_goal: float = None,
        r_opt: float = None,
        r_fail: float = None,
        max_steps: int = None,
    ):
        assert len(puzzle) == 4 and all(len(row) == 4 for row in puzzle), "puzzle must be 4x4"
        self.init_grid = [row[:] for row in puzzle]
        self.r_step     = self.DEFAULT_R_STEP     if r_step     is None else r_step
        self.r_invalid  = self.DEFAULT_R_INVALID  if r_invalid  is None else r_invalid
        self.r_goal     = self.DEFAULT_R_GOAL     if r_goal     is None else r_goal
        self.r_opt      = self.DEFAULT_R_OPT      if r_opt      is None else r_opt
        self.r_fail     = self.DEFAULT_R_FAIL     if r_fail     is None else r_fail
        self.max_steps  = self.DEFAULT_MAX_STEPS  if max_steps  is None else max_steps

        self.initial_empty = sum(1 for r in range(4) for c in range(4) if self.init_grid[r][c] == 0)
        self.reset_agent()

    # ---------- 规则 ----------
    @staticmethod
    def _in_range(v: int) -> bool:
        return 1 <= v <= 4

    def _row_ok(self, grid, r, v) -> bool:
        return all(x != v for x in grid[r])

    def _col_ok(self, grid, c, v) -> bool:
        return all(grid[r][c] != v for r in range(4))

    def _box_ok(self, grid, r, c, v) -> bool:
        br, bc = (r // 2) * 2, (c // 2) * 2
        for rr in range(br, br+2):
            for cc in range(bc, bc+2):
                if grid[rr][cc] == v:
                    return False
        return True

    def is_goal(self, grid: List[List[int]]) -> bool:
        # 所有非零且满足规则
        for r in range(4):
            for c in range(4):
                v = grid[r][c]
                if not self._in_range(v):
                    return False
                # 暂时将该格置 0 再检查唯一性
                tmp = grid[r][c]; grid[r][c] = 0
                ok = self._in_range(tmp) and self._row_ok(grid, r, tmp) and self._col_ok(grid, c, tmp) and self._box_ok(grid, r, c, tmp)
                grid[r][c] = tmp
                if not ok:
                    return False
        return True

    def describe(self) -> str:
        return "Sudoku4x4Worker: fill 4x4 grid with 1..4 obeying row/col/2x2-box uniqueness."

    # ---------- 逐步交互 ----------
    def reset_agent(self):
        self.grid = [row[:] for row in self.init_grid]
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self.invalid_count = 0
        self.clean_fill_only = True  # 仅填初始空格、且不覆盖

    def get_valid_actions(self) -> List[Tuple[int,int,int]]:
        acts = []
        for r in range(4):
            for c in range(4):
                for v in range(1,5):
                    acts.append((r,c,v))
        return acts

    def _can_place(self, r: int, c: int, v: int) -> bool:
        if not (0 <= r < 4 and 0 <= c < 4 and self._in_range(v)):
            return False
        # 检查规则
        if self.grid[r][c] != 0:
            # 非空位也允许写入，但记为 invalid（覆盖）
            # 返回 False 以触发 invalid 逻辑
            return False
        if not (self._row_ok(self.grid, r, v) and self._col_ok(self.grid, c, v) and self._box_ok(self.grid, r, c, v)):
            return False
        return True

    def step(self, action: Tuple[int,int,int]):
        """
        action = (r, c, v)
        合法：该格目前为空，且 v 与行/列/宫约束不冲突
        奖励：r_step + (invalid ? r_invalid) + (goal ? r_goal + opt)
        """
        if self.done:
            return [row[:] for row in self.grid], 0.0, True, {"msg": "episode already done"}

        r, c, v = action
        reward = self.r_step
        valid = self._can_place(r, c, v)

        if valid:
            self.grid[r][c] = v
        else:
            self.invalid_count += 1
            reward += self.r_invalid
            # 若尝试覆盖或非法填数，视为不是“干净的只填空格”
            if 0 <= r < 4 and 0 <= c < 4 and self.grid[r][c] != 0:
                self.clean_fill_only = False

        self.steps += 1

        if self.is_goal(self.grid):
            reward += self.r_goal
            # “最优”：仅填初始空位、无覆盖，且恰好 initial_empty 步完成
            filled_now = sum(1 for r in range(4) for c in range(4) if self.init_grid[r][c] == 0 and self.grid[r][c] != 0)
            if self.clean_fill_only and filled_now == self.initial_empty and self.steps == self.initial_empty:
                reward += self.r_opt
            self.done = True
        elif self.steps >= self.max_steps:
            reward += self.r_fail
            self.done = True

        self.total_reward += reward
        info = {
            "valid": valid,
            "steps": self.steps,
            "invalid_count": self.invalid_count,
            "done": self.done,
            "grid": [row[:] for row in self.grid],
        }
        return [row[:] for row in self.grid], reward, self.done, info

    # ---------- 一次性评测 ----------
    def reward_for_solution(self, grid: List[List[int]]) -> Tuple[float, Dict[str,Any]]:
        if len(grid) != 4 or any(len(row) != 4 for row in grid):
            return self.r_fail, {"feasible": False, "optimal": None}

        # 统计：填入步数、非法步（覆盖/冲突/越界）
        L = 0
        invalid = 0
        clean = True
        # 逐格对比：只统计初始空格 -> 非零 作为一次“填入”
        for r in range(4):
            for c in range(4):
                v0 = self.init_grid[r][c]
                v1 = grid[r][c]
                if v0 == 0 and v1 != 0:
                    L += 1
                    if not self._in_range(v1):
                        invalid += 1
                    else:
                        # 临时检查合法性（在候选 grid 上验证）
                        tmp = grid[r][c]; grid[r][c] = 0
                        ok = self._row_ok(grid, r, v1) and self._col_ok(grid, c, v1) and self._box_ok(grid, r, c, v1)
                        grid[r][c] = tmp
                        if not ok:
                            invalid += 1
                elif v0 != 0 and v1 != v0:
                    # 覆盖了 givens
                    invalid += 1
                    clean = False

        feasible = self.is_goal(grid)
        R = self.r_step * L + self.r_invalid * invalid
        if feasible:
            R += self.r_goal
            if clean and L == self.initial_empty:
                R += self.r_opt
            opt = (clean and L == self.initial_empty)
        else:
            R += self.r_fail
            opt = None

        info = {
            "feasible": feasible,
            "optimal": opt,
            "steps": L,
            "invalid_steps": invalid,
        }
        return R, info
