# prompt.py
from __future__ import annotations
import json
from typing import Any

def _truncate(s, length=300):
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= length else s[: length // 2] + "...(truncated)..." + s[-length // 2 :]

# ========== 各 benchmark 的 prompt 生成函数 ==========

def prompt_plan_path(turn_idx: int, state: Any) -> str:
    grid = getattr(state, "grid", None) or []
    start = getattr(state, "start", None)
    goal = getattr(state, "goal", None)

    grid_str = grid
    return (
            "You are a path planner (grid walker).\n\n"
            f"Grid ('.' free, '#' blocked):\n```\n{grid_str}\n```\n"
            f"Start: {start},  Goal: {goal}\n"
            "Rules: moves are U/D/L/R; stay within bounds; do not cross '#'.\n\n"
        )
def prompt_eight_queens(turn_idx: int, state: Any) -> str:
    N = getattr(state, "N", 8)
    positions = getattr(state, "positions", None)
    if turn_idx == 0:
        return (
            f"You are solving the N-Queens puzzle for N={N}.\n"
            "Place one queen per row, no two queens attack each other (no same column or diagonal).\n\n"
            "Output a JSON array of length N where value is the column index (0-based) for each row.\n"
            "Return only a JSON array like:\n```json\n[1, 3, 0, 2]\n```\n"
            "Example for N=4:\n```json\n[1, 3, 0, 2]\n```\n"
        )
    else:
        return (
            f"Refine your N={N} Queens solution.\n"
            f"You have put the queens in the following positions: {positions}\n"
            f"However, it is not a valid solution. Please first analyze why it is not a valid solution and then correct it.\n"
            "Return only a JSON array like:\n```json\n[0,4,7,5,2,6,1,3]\n```\n"
        )

def prompt_blocksworld(turn_idx: int, state: Any) -> str:
    init_stacks = getattr(state, "init_stacks", None) or getattr(state, "stacks", None)
    goal_stacks = getattr(state, "goal_stacks", None)
    current_stacks = getattr(state, "current_stacks", None)

    def fmt(x): return json.dumps(x, ensure_ascii=False)
    if turn_idx == 0:
        return (
            "You are solving a Blocks World rearrangement problem.\n"
            f"Initial stacks (bottom→top): {fmt(init_stacks)}\n"
            f"Goal stacks (bottom→top):    {fmt(goal_stacks)}\n\n"
            "Move rule: move the top (clear) block x to either the table or onto another clear block y.\n"
            "Return a JSON array of actions. Each action is one of:\n"
            "```json\n{\"move\": [\"B\", \"table\"]}\n{\"move\": [\"C\", \"B\"]}\n```\n"
        )
    else:
        return (
            "You have already put the blocks in the following stacks:\n"
            f"Current stacks: {fmt(current_stacks)}\n"
            f"However, it is not a valid solution. Please first analyze why it is not a valid solution and then correct it.\n"
            "Refine your Blocks World plan.\n"
            f"Initial: {fmt(init_stacks)}\nGoal: {fmt(goal_stacks)}\n"
    
            "Return only a JSON array of actions like:\n"
            "[ {\"move\": [\"B\",\"table\"]}, {\"move\": [\"C\",\"B\"]} ]\n"
        )

def prompt_sudoku4x4(turn_idx: int, state: Any) -> str:
    puzzle = getattr(state, "puzzle", None) or getattr(state, "init_grid", None)
    size = getattr(state, "size", 4)  # Dynamically get sudoku size
    
    # Simplify puzzle display, only show observation info
    observation = getattr(state, "observation", "")
    if not observation and puzzle:
        # If no observation, generate simplified display from puzzle
        obs_lines = []
        for row in puzzle:
            obs_lines.append(' '.join(str(x) if x != 0 else '.' for x in row))
        observation = '\n'.join(obs_lines)
    
    if turn_idx == 0:
        return (
            f"Solve the {size}x{size} Sudoku. Fill digits 1..{size}; rows, columns, and sub-grids must have unique digits.\n"
            f"Current puzzle:\n```\n{observation}\n```\n\n"
            f"Output either:\n"
            f"1) **Completed {size}x{size} grid** as JSON array\n"
            f"2) A JSON list of fill steps (r,c,v)\n"
        )
    else:
        return (
            f"Refine your {size}x{size} Sudoku solution.\n"
            f"Current state:\n```\n{observation}\n```\n"
            f"Analyze errors and provide the corrected solution.\n"
        )

# ========== 统一调度 ==========

PROMPT_BUILDERS = {
    "plan_path": prompt_plan_path,
    "eight_queens": prompt_eight_queens,
    "blocksworld": prompt_blocksworld,
    "sudoku4x4": prompt_sudoku4x4,
}

def build_plan_prompt(benchmark: str, turn_idx: int, state: Any) -> str:
    if benchmark not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return PROMPT_BUILDERS[benchmark](turn_idx, state)

def build_tool_prompt(benchmark: str, turn_idx: int, state: Any) -> str:
    if benchmark not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return PROMPT_BUILDERS[benchmark](turn_idx, state)



