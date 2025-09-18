#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

# ========== 默认配置（可用命令行覆盖） ==========
GRID_H = GRID_W = 5
TRAIN_SIZE = 1000
TEST_SIZE  = 100
TRAIN_SEED = 123
TEST_SEED  = 456
BLOCK_RATIO = 0.22  # 5x5 建议 0.18~0.28；过大易无路

OUT_DIR = Path("datasets/plan_path")
TRAIN_PATH = OUT_DIR / "train.json"
TEST_PATH  = OUT_DIR / "test.json"



from typing import Any, Dict, Optional, List, Tuple


def _extract_actions(text: str) -> Optional[List[str]]:
    """
    尝试从文本中解析动作序列（U/D/L/R）。优先 JSON，其次宽松匹配。
    支持：
      Actions: ["R","R","D","L"]
      ["U","D","L","R"]
      U R D L U
    """
    # 1) JSON-like 数组
    try:
        # 找到第一个方括号数组片段
        m = re.search(r"\[(?:.|\n)*?\]", text)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                cand = [x.strip().upper() for x in arr]
                if all(x in {"U", "D", "L", "R"} for x in cand):
                    return cand
    except Exception:
        pass

    # 2) 宽松：抓取连续的 U/D/L/R 字母（逗号/空格/换行分隔）
    toks = re.findall(r"[UDLR]", text.upper())
    if toks:
        return toks
    return None
# ========== 基础工具 ==========
def bfs_shortest_path(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    from collections import deque
    H, W = grid.shape
    def inb(r,c): return 0 <= r < H and 0 <= c < W
    def passable(r,c): return int(grid[r,c]) == 0
    if not (inb(*start) and inb(*goal) and passable(*start) and passable(*goal)):
        return None
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            x = cur
            while x is not None:
                path.append(x)
                x = prev[x]
            return list(reversed(path))
        r, c = cur
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if inb(nr,nc) and passable(nr,nc) and (nr,nc) not in prev:
                prev[(nr,nc)] = cur
                q.append((nr,nc))
    return None

def grid_array_to_str_lines(grid: np.ndarray, free_char=".", block_char="#") -> List[str]:
    return ["".join(free_char if int(v)==0 else block_char for v in row) for row in grid]

def to_item(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int], path: List[Tuple[int,int]]) -> Dict:
    return {
        "grid": grid_array_to_str_lines(grid),
        "start": [int(start[0]), int(start[1])],
        "goal": [int(goal[0]),  int(goal[1])],
        "optimal_path": [[int(r), int(c)] for (r,c) in path]
    }



def synthesize(n_samples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    np.random.seed(seed)
    out: List[Dict] = []
    trials = 0
    max_trials = max(2000, n_samples * 50)
    while len(out) < n_samples and trials < max_trials:
        trials += 1
        grid = (np.random.rand(GRID_H, GRID_W) < BLOCK_RATIO).astype(int)
        frees = [(r,c) for r in range(GRID_H) for c in range(GRID_W) if grid[r,c] == 0]
        if len(frees) < 2:  # 必须至少两个可行点
            continue
        s = rng.choice(frees)
        g = rng.choice(frees)
        while g == s:
            g = rng.choice(frees)
        path = bfs_shortest_path(grid, s, g)
        if path is None:
            continue
        out.append(to_item(grid, s, g, path))
    if len(out) < n_samples:
        print(f"[WARN] only generated {len(out)}/{n_samples}. Consider lowering BLOCK_RATIO.")
    return out




def truncatefn(s, length=300):
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= length else s[: length // 2] + "...(truncated)..." + s[-length // 2 :]


def _format_grid(lines: List[str]) -> str:
    return "\n".join(lines)


import re

def _extract_path(text: str) -> Optional[List[Tuple[int, int]]]:
    """
    尝试从文本中解析坐标路径 [[r,c],...]
    支持 JSON，或松散的 [r,c] [r,c] ... 形式。
    """
    # 1) 直接 JSON
    try:
        m = re.search(r"\[(?:.|\n)*?\]", text)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and arr and isinstance(arr[0], (list, tuple)):
                path = []
                for p in arr:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        r, c = int(p[0]), int(p[1])
                        path.append((r, c))
                    else:
                        return None
                return path
    except Exception:
        pass

    # 2) 宽松地抓 [r,c]
    pairs = re.findall(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
    if pairs:
        return [(int(r), int(c)) for r, c in pairs]
    return None


def _actions_to_path(actions: List[str], start: Tuple[int, int], passable_fn, in_bounds_fn) -> List[Tuple[int, int]]:
    """从动作序列构造路径（遇到非法动作用“停留+标记”为非法，但最终评分会惩罚非法）。"""
    pos = start
    path = [pos]
    delta = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    for a in actions:
        if a not in delta:
            # 非法动作：保持原地，让后续 reward_for_path 统计非法步
            path.append(pos)
            continue
        dr, dc = delta[a]
        nr, nc = pos[0] + dr, pos[1] + dc
        if in_bounds_fn(nr, nc) and passable_fn(nr, nc):
            pos = (nr, nc)
        # 若越界/撞墙，保持原地；非法步会在 reward_for_path 中计入
        path.append(pos)
    return path



def main():
  

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = synthesize(TRAIN_SIZE, TRAIN_SEED)
    test  = synthesize(TEST_SIZE,  TEST_SEED)

    TRAIN_PATH.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    TEST_PATH.write_text(json.dumps(test,  ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Saved {len(train)} samples to {TRAIN_PATH}")
    print(f"✅ Saved {len(test)} samples to  {TEST_PATH}")

def load_plan_path_problem_batch(
    env_indices: List[int],
    dataset_name: str = "train", 
    split: str = "train",
    mode: str = "train",
    config: dict = None,
    benchmark_name: str = "PlanPath"
) -> List[Dict[str, Any]]:
    """
    Load a batch of plan path problems based on benchmark name.
    
    Args:
        env_indices: List of environment indices
        dataset_name: Dataset name 
        split: Dataset split
        mode: "train" or "validate"
        config: Configuration dict
        benchmark_name: Benchmark name to determine problem type
        
    Returns:
        A list of problem dicts appropriate for the benchmark
    """
    if benchmark_name == "PlanPath":
        # 对于PlanPath，生成或加载网格路径问题
        if mode == "train":
            # 训练模式：动态生成
            problems = []
            for i in range(len(env_indices)):
                # 生成一个随机的网格路径问题
                grid_problems = synthesize(1, seed=42 + i)
                if grid_problems:
                    problem = grid_problems[0]
                    problems.append({
                        "grid": problem["grid"],
                        "start": tuple(problem["start"]),
                        "goal": tuple(problem["goal"]),
                        "optimal_path": [tuple(p) for p in problem["optimal_path"]]
                    })
            return problems
        else:
            # 验证模式：从文件加载或生成固定问题
            if TEST_PATH.exists():
                with open(TEST_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                problems = []
                for i, idx in enumerate(env_indices):
                    if idx < len(data):
                        problem = data[idx]
                        problems.append({
                            "grid": problem["grid"],
                            "start": tuple(problem["start"]),
                            "goal": tuple(problem["goal"]),
                            "optimal_path": [tuple(p) for p in problem["optimal_path"]]
                        })
                return problems
            else:
                # 如果没有测试文件，生成固定的测试问题
                problems = []
                for i in range(len(env_indices)):
                    grid_problems = synthesize(1, seed=1000 + i)  # 固定种子确保可重现
                    if grid_problems:
                        problem = grid_problems[0]
                        problems.append({
                            "grid": problem["grid"],
                            "start": tuple(problem["start"]),
                            "goal": tuple(problem["goal"]),
                            "optimal_path": [tuple(p) for p in problem["optimal_path"]]
                        })
                return problems
    
    elif benchmark_name == "EightQueens":
        # 对于八皇后问题，返回不同规模的问题
        problems = []
        for i in range(len(env_indices)):
            n = 4 + (i % 5)  # N从4到8
            problems.append({"N": n})
        return problems
    
    elif benchmark_name == "Blocksworld":
        # 对于方块世界问题，生成不同的初始和目标配置
        problems = []
        block_configs = [
            {
                "init_stacks": [["A", "B"], ["C"]],
                "goal_stacks": [["C", "A"], ["B"]]
            },
            {
                "init_stacks": [["A"], ["B", "C"], ["D"]],
                "goal_stacks": [["D", "C", "B", "A"], [], []]
            },
            {
                "init_stacks": [["A", "B", "C"], []],
                "goal_stacks": [[], ["C", "B", "A"]]
            },
        ]
        for i in range(len(env_indices)):
            config_idx = i % len(block_configs)
            problems.append(block_configs[config_idx])
        return problems
    
    elif benchmark_name == "sudoku4x4":
        # 对于sudoku4x4问题，使用seed生成不同的初始状态
        problems = []
        for i in range(len(env_indices)):
            # 每个环境使用不同的seed
            seed = env_indices[i] if i < len(env_indices) else i
            problems.append({"seed": seed})
        return problems
    
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}. Supported: PlanPath, EightQueens, Blocksworld, sudoku4x4")


if __name__ == "__main__":
    main()
