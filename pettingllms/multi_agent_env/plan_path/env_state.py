from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, ClassVar
import copy
import math
import random
import numpy as np
from collections import deque

# =========================================================
# 1) Eight Queens (N-Queens) EnvState
# =========================================================

@dataclass
class EnvStateBase:
    # 将基类参数移到最后，或者使用field(init=False)
    tool_action: List[str] = field(default_factory=list, init=False)
    tool_code: str = field(default="", init=False)
    tool_execution_output: str = field(default="", init=False)
    plan_action: List[str] = field(default_factory=list, init=False)
    observation: str = field(default="", init=False)
    
    def __post_init__(self):
        # 在子类的__post_init__中会调用super().__post_init__()
        if not hasattr(self, 'tool_action'):
            self.tool_action = []
        if not hasattr(self, 'tool_code'):
            self.tool_code = ""
        if not hasattr(self, 'tool_execution_output'):
            self.tool_execution_output = ""
        if not hasattr(self, 'plan_action'):
            self.plan_action = []
        if not hasattr(self, 'observation'):
            self.observation = ""
    
    def __str__(self) -> str:
        """Print only essential environment info"""
        # 安全处理 tool_action 可能为 None 的情况
        tool_action_str = ""
        if self.tool_action is not None:
            tool_action_str = self.tool_action[:3] if len(self.tool_action) > 3 else self.tool_action
            tool_action_str = str(tool_action_str) + ('...' if len(self.tool_action) > 3 else '')
        else:
            tool_action_str = "None"
        
        # 安全处理 observation 可能为 None 的情况
        observation_str = ""
        if self.observation is not None:
            observation_str = self.observation[:100] if len(self.observation) > 100 else self.observation
            observation_str = str(observation_str) + ('...' if len(self.observation) > 100 else '')
        else:
            observation_str = "None"
            
        return (
            f"action: {tool_action_str}\n"
            f"observation: {observation_str}"
        )
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict_compact(self) -> dict:
        """Return compact dictionary representation for logging (base class default implementation)"""
        return {
            "tool_action": self.tool_action[:5] if self.tool_action is not None and len(self.tool_action) > 5 else self.tool_action,  # 最多显示5个动作
            "tool_code_lines": len(self.tool_code.split('\n')) if self.tool_code else 0,  # 只显示代码行数
            "tool_output_chars": len(self.tool_execution_output) if self.tool_execution_output is not None else 0,  # 只显示输出字符数
            "plan_action": self.plan_action[:3] if self.plan_action is not None and len(self.plan_action) > 3 else self.plan_action,  # 最多显示3个计划动作
            "observation": (self.observation[:100] + "..." if len(self.observation) > 100 else self.observation) if self.observation is not None else None  # 截断观察信息
        }

@dataclass
class EightQueensEnvState(EnvStateBase):
    """N-Queens problem: Place N queens on an NxN board so they don't attack each other"""
    
    N: int = 8
    cols: List[int] = field(default_factory=lambda: [-1] * 8)
    positions: List[int] = field(default_factory=lambda: [-1] * 8)
    done: bool = False
    step_count: int = 0
    reward: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        # 重新初始化以确保N个位置
        self.cols = [-1] * self.N
        self.positions = [-1] * self.N
        self.reset()

    def reset(self):
        """重置环境"""
        self.cols = [-1] * self.N  # 每行皇后的列位置，-1表示未放置
        self.positions = self.cols[:]  # prompt需要的positions属性
        self.done = False
        self.step_count = 0
        self.reward = 0.0
        self.observation = self.text_observation()
    
    def text_observation(self) -> str:
        """Text observation"""
        board = []
        for r in range(self.N):
            row = ['.'] * self.N
            if self.cols[r] >= 0:
                row[self.cols[r]] = 'Q'
            board.append(''.join(row))
        return '\n'.join(board)
    
    def available_actions(self) -> List[Tuple[int, int]]:
        """可用动作：(row, col)，col=-1表示清空该行"""
        actions = []
        for r in range(self.N):
            for c in range(-1, self.N):
                actions.append((r, c))
        return actions
    
    def _conflicts(self, cols: List[int]) -> int:
        """计算冲突数量"""
        count = 0
        for r1 in range(len(cols)):
            c1 = cols[r1]
            if c1 < 0:
                continue
            for r2 in range(r1 + 1, len(cols)):
                c2 = cols[r2]
                if c2 < 0:
                    continue
                # 同列或同对角线
                if c1 == c2 or abs(c1 - c2) == abs(r1 - r2):
                    count += 1
        return count
    
    def _is_solved(self) -> bool:
        """检查是否解决"""
        return all(c >= 0 for c in self.cols) and self._conflicts(self.cols) == 0

    def step(self, action):
        """执行动作，更新环境状态。动作格式：[col1, col2, ..., colN] JSON数组"""
        if self.done:
            self.reward = 0.0
            return
        
        # 解析动作：期望是包含N个列索引的列表
        if not isinstance(action, list) or len(action) != self.N:
            self.reward = -1.0  # 无效动作格式惩罚
            return
        
        # 检查所有列索引是否合法
        for col in action:
            if not isinstance(col, int) or not (0 <= col < self.N):
                self.reward = -1.0  # 无效列索引惩罚
                return
        
        # 记录之前的状态
        prev_conflicts = self._conflicts(self.cols)
        
        # 设置新的皇后位置
        self.cols = list(action)
        self.positions = self.cols[:]  # 更新positions属性
        
        # 计算奖励
        self.reward = -0.01  # 基础步数惩罚
        
        # 检查冲突
        current_conflicts = self._conflicts(self.cols)
        if current_conflicts > 0:
            # 有冲突的惩罚，但不回滚（允许中间状态）
            self.reward = -0.5 - current_conflicts * 0.1
        else:
            # 无冲突时的奖励
            self.reward += 0.5  # 无冲突奖励
            
            # 检查是否完成
            if self._is_solved():
                self.reward += 2.0  # 成功完成大奖励
                # 效率奖励
                self.step_count += 1
                if self.step_count <= self.N:  # 最优步数
                    self.reward += 0.5
                self.done = True
        
        self.step_count += 1
        self.observation = self.text_observation()
    
    def to_dict_compact(self) -> dict:
        """Return compact dictionary representation for logging"""
        base_dict = super().to_dict_compact()
        placed_queens = sum(1 for c in self.cols if c >= 0)
        conflicts = self._conflicts(self.cols)
        base_dict.update({
            "board_size": f"{self.N}x{self.N}",
            "queens_placed": placed_queens,
            "total_queens": self.N,
            "conflicts": conflicts,
            "done": self.done,
            "steps": self.step_count,
            "reward": round(self.reward, 3)
        })
        return base_dict



# =========================================================
# 2) Blocksworld EnvState
# =========================================================

@dataclass
class BlocksworldEnvState(EnvStateBase):
    """方块世界：移动积木块到指定的堆叠配置"""
    
    init_stacks: List[List[str]]
    goal_stacks: List[List[str]]
    stacks: List[List[str]] = field(default_factory=list)
    current_stacks: List[List[str]] = field(default_factory=list)
    all_blocks: Set[str] = field(default_factory=set)
    done: bool = False
    step_count: int = 0
    reward: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.init_stacks = [list(s) for s in self.init_stacks]
        self.goal_stacks = [list(s) for s in self.goal_stacks]
        self.all_blocks = set(sum(self.init_stacks, []))
        self.reset()

    def reset(self):
        """重置环境"""
        self.stacks = [list(s) for s in self.init_stacks]
        self.current_stacks = [list(s) for s in self.stacks]  # prompt需要的current_stacks属性
        self.done = False
        self.step_count = 0
        self.reward = 0.0
        self.observation = self.text_observation()
    
    def text_observation(self) -> str:
        """Text observation"""
        obs = []
        for i, stack in enumerate(self.stacks):
            if stack:
                obs.append(f"Stack {i}: {' -> '.join(stack)}")
            else:
                obs.append(f"Stack {i}: empty")
        obs.append(f"Goal: {self.goal_stacks}")
        return '\n'.join(obs)
    
    def available_actions(self) -> List[Tuple[str, str]]:
        """可用动作：(block, destination)，destination可以是block名或'table'"""
        actions = []
        # 找到所有可移动的块（栈顶块）
        clear_blocks = []
        for stack in self.stacks:
            if stack:
                clear_blocks.append(stack[-1])
        
        for block in clear_blocks:
            # 移动到桌面
            actions.append((block, "table"))
            # 移动到其他块上
            for other_block in clear_blocks:
                if other_block != block:
                    actions.append((block, other_block))
        return actions
    
    def _is_clear(self, block: str) -> bool:
        """检查块是否在栈顶"""
        for stack in self.stacks:
            if stack and stack[-1] == block:
                return True
        return False
    
    def _find_block(self, block: str) -> Tuple[int, int]:
        """找到块的位置：(stack_index, position_in_stack)"""
        for si, stack in enumerate(self.stacks):
            for pi, b in enumerate(stack):
                if b == block:
                    return si, pi
        raise ValueError(f"Block {block} not found")
    
    def _is_goal_reached(self) -> bool:
        """检查是否达到目标"""
        # 简单比较，忽略空栈
        current = [stack for stack in self.stacks if stack]
        goal = [stack for stack in self.goal_stacks if stack]
        return sorted([tuple(s) for s in current]) == sorted([tuple(s) for s in goal])
    
    def _calculate_goal_similarity(self) -> float:
        """计算当前配置与目标配置的相似度 (0-1)"""
        total_blocks = len(self.all_blocks)
        correct_positions = 0
        
        # 为每个块检查是否在正确的相对位置
        for block in self.all_blocks:
            current_pos = self._get_block_context(block, self.stacks)
            goal_pos = self._get_block_context(block, self.goal_stacks)
            
            if current_pos == goal_pos:
                correct_positions += 1
        
        return correct_positions / total_blocks if total_blocks > 0 else 0.0
    
    def _get_block_context(self, block: str, stacks: List[List[str]]) -> Tuple:
        """获取块的上下文：(下面的块, 上面的块)"""
        for stack in stacks:
            if block in stack:
                idx = stack.index(block)
                below = stack[idx-1] if idx > 0 else None
                above = stack[idx+1] if idx < len(stack)-1 else None
                return (below, above)
        return (None, None)

    def step(self, action):
        """执行动作，更新环境状态。动作格式：[{"move": ["B","table"]}, {"move": ["C","B"]}] JSON数组"""
        if self.done:
            self.reward = 0.0
            return
        
        # 解析动作：期望是包含move操作的字典列表
        if not isinstance(action, list):
            self.reward = -1.0  # 无效动作格式惩罚
            return
        
        total_reward = 0.0
        
        # 执行每个动作
        for move_action in action:
            if not isinstance(move_action, dict) or "move" not in move_action:
                total_reward += -0.5  # 无效动作格式惩罚
                continue
            
            move = move_action["move"]
            if not isinstance(move, list) or len(move) != 2:
                total_reward += -0.5  # 无效move格式惩罚
                continue
            
            block, dest = move
            step_reward = -0.01  # 基础步数惩罚
            
            # 检查块是否可移动（在栈顶）
            if not self._is_clear(block):
                step_reward = -0.3  # 块不在栈顶惩罚
                total_reward += step_reward
                continue
            
            # 记录移动前的相似度
            prev_similarity = self._calculate_goal_similarity()
            
            # 执行移动
            try:
                stack_idx, pos = self._find_block(block)
                
                # 检查目标是否有效
                if dest != "table" and (dest not in self.all_blocks or not self._is_clear(dest)):
                    step_reward = -0.3  # 目标块不可用惩罚
                    total_reward += step_reward
                    continue
                
                # 执行移动
                self.stacks[stack_idx].pop()  # 移除块
                
                if dest == "table":
                    # 移动到桌面（创建新栈）
                    self.stacks.append([block])
                else:
                    # 移动到其他块上
                    dest_stack_idx, _ = self._find_block(dest)
                    self.stacks[dest_stack_idx].append(block)
                
                # 更新current_stacks属性
                self.current_stacks = [list(s) for s in self.stacks]
                
                # 计算移动后的相似度
                current_similarity = self._calculate_goal_similarity()
                
                # 稠密奖励设计
                # 1. 基于目标相似度的改进
                similarity_improvement = current_similarity - prev_similarity
                if similarity_improvement > 0:
                    step_reward += similarity_improvement * 0.5  # 正向进步奖励
                elif similarity_improvement < 0:
                    step_reward += similarity_improvement * 0.3  # 负向进步惩罚
                
                # 2. 基于当前相似度的奖励
                step_reward += current_similarity * 0.1
                
                # 3. 特殊情况奖励
                # 如果块移动到了目标位置的正确上下文中
                target_context = self._get_block_context(block, self.goal_stacks)
                current_context = self._get_block_context(block, self.stacks)
                if target_context == current_context and target_context != (None, None):
                    step_reward += 0.2  # 正确位置奖励
                
            except ValueError:
                step_reward = -0.3  # 块不存在
            
            total_reward += step_reward
            self.step_count += 1
        
        # 检查是否完成
        if self._is_goal_reached():
            total_reward += 2.0  # 成功完成大奖励
            # 效率奖励
            if self.step_count <= len(self.all_blocks) * 2:
                total_reward += 0.5  # 高效完成奖励
            self.done = True
        
        self.reward = total_reward
        self.current_stacks = [list(s) for s in self.stacks]  # 更新current_stacks属性
        self.observation = self.text_observation()
    
    def to_dict_compact(self) -> dict:
        """Return compact dictionary representation for logging"""
        base_dict = super().to_dict_compact()
        similarity = self._calculate_goal_similarity()
        base_dict.update({
            "total_blocks": len(self.all_blocks),
            "stacks_count": len([s for s in self.stacks if s]),  # 非空栈数量
            "goal_similarity": round(similarity, 3),
            "done": self.done,
            "steps": self.step_count,
            "reward": round(self.reward, 3)
        })
        return base_dict


# =========================================================
# 3) Sudoku 4x4 EnvState
# =========================================================

@dataclass
class Sudoku4x4EnvState(EnvStateBase):
    """Dynamic size Sudoku: Fill NxN grid with row, column, and sub-grid constraints (keeping 4x4 name for compatibility)"""
    
    puzzle: Optional[List[List[int]]] = None
    seed: Optional[int] = None
    size: int = 4  # Sudoku size, default 4x4
    config: Optional[dict] = None
    init_grid: List[List[int]] = field(default_factory=list)
    grid: List[List[int]] = field(default_factory=list)
    done: bool = False
    step_count: int = 0
    reward: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        
        # Read map_size parameter from config if available
        if self.config and hasattr(self.config, 'map_size'):
            self.size = self.config.map_size
        elif self.config and isinstance(self.config, dict) and 'map_size' in self.config:
            self.size = self.config['map_size']
       
        if self.puzzle is not None:
            assert len(self.puzzle) == self.size and all(len(row) == self.size for row in self.puzzle), f"Must be {self.size}x{self.size} grid"
            self.init_grid = [row[:] for row in self.puzzle]
        # If seed is provided, generate puzzle based on seed
        elif self.seed is not None:
            self.puzzle = self._generate_puzzle_from_seed(self.seed, self.size)
            self.init_grid = [row[:] for row in self.puzzle]
        else:
            ValueError("No puzzle provided")
        self.puzzle = [row[:] for row in self.puzzle]  # puzzle attribute needed for prompt
        self.reset()
    
    def _generate_puzzle_from_seed(self, seed: int, size: int) -> List[List[int]]:
        """Select sudoku puzzle from pre-generated JSON file based on seed"""
        import json
        import os
        
        # Get current file directory, then find datasets/sudoku_environments directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # Back to project root
        sudoku_env_dir = os.path.join(project_root, "datasets", "sudoku_environments")
        json_file = os.path.join(sudoku_env_dir, f"sudoku_{size}x{size}.json")
        
        try:
            # Try to load pre-generated environments from JSON file
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    environments = json.load(f)
                
                # Select an environment based on seed (ensure reproducibility)
                env_index = seed % len(environments)
                selected_env = environments[env_index]
                return selected_env["puzzle"]
            else:
                print(f"[WARN] Pre-generated sudoku environment file not found: {json_file}")
                return self._generate_fallback_puzzle(size, seed)
                
        except Exception as e:
            print(f"[WARN] Error loading sudoku environment: {e}")
            return self._generate_fallback_puzzle(size, seed)
    
    def _generate_fallback_puzzle(self, size: int, seed: int) -> List[List[int]]:
        """Generate simple fallback sudoku puzzle"""
        import random
        random.seed(seed)
        
        puzzle = [[0 for _ in range(size)] for _ in range(size)]
        box_size = int(size ** 0.5)
        
        # Place some numbers in each sub-grid
        for box_row in range(0, size, box_size):
            for box_col in range(0, size, box_size):
                # Randomly place 1-2 numbers in sub-grid
                positions = [(r, c) for r in range(box_row, box_row + box_size) 
                           for c in range(box_col, box_col + box_size)]
                random.shuffle(positions)
                
                fill_count = random.randint(1, min(2, len(positions)))
                for i in range(fill_count):
                    r, c = positions[i]
                    # Try to place a safe number
                    for num in random.sample(range(1, size + 1), size):
                        if self._is_safe_fallback_placement(puzzle, r, c, num, size):
                            puzzle[r][c] = num
                            break
        
        return puzzle
    
    def _is_safe_fallback_placement(self, grid: List[List[int]], row: int, col: int, num: int, size: int) -> bool:
        """Simple safety check for fallback puzzle generation"""
        box_size = int(size ** 0.5)
        
        # Check row
        for c in range(size):
            if grid[row][c] == num:
                return False
        
        # Check column
        for r in range(size):
            if grid[r][col] == num:
                return False
        
        # Check sub-grid
        start_row = (row // box_size) * box_size
        start_col = (col // box_size) * box_size
        
        for r in range(start_row, start_row + box_size):
            for c in range(start_col, start_col + box_size):
                if grid[r][c] == num:
                    return False
        
        return True
    
    
    def _get_default_puzzle(self, size: int) -> List[List[int]]:
        """Get default NxN sudoku puzzle"""
        # Use different default puzzles for different sizes
        if size == 4:
            return [[1, 0, 0, 4],
                    [0, 0, 1, 0], 
                    [0, 4, 0, 0],
                    [4, 0, 0, 1]]
        elif size == 9:
            return [
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
        elif size == 16:
            # 16x16 sudoku default puzzle (simplified version)
            puzzle = [[0 for _ in range(16)] for _ in range(16)]
            # Fill some basic numbers
            puzzle[0] = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            puzzle[1] = [0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
            return puzzle
        else:
            # For other sizes, generate a basic puzzle
            return self._generate_puzzle_from_seed(42, size)

    def reset(self):
        """Reset environment"""
        self.grid = [row[:] for row in self.init_grid]
        self.puzzle = [row[:] for row in self.init_grid]  # Update puzzle attribute
        self.done = False
        self.step_count = 0
        self.reward = 0.0
        self.observation = self.text_observation()
    
    def text_observation(self) -> str:
        """Text observation"""
        obs = []
        for row in self.grid:
            obs.append(' '.join(str(x) if x != 0 else '.' for x in row))
        return '\n'.join(obs)
    
    def available_actions(self) -> List[Tuple[int, int, int]]:
        """可用动作：(row, col, value)"""
        actions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == 0:  # 只能在空格填数
                    for v in range(1, self.size + 1):
                        actions.append((r, c, v))
        return actions
    
    def _is_valid_placement(self, r: int, c: int, v: int) -> bool:
        """检查在(r,c)放置v是否合法"""
        if not (0 <= r < self.size and 0 <= c < self.size and 1 <= v <= self.size):
            return False
        
        if self.grid[r][c] != 0:  # 非空格
            return False
        
        # 检查行
        if v in self.grid[r]:
            return False
        
        # 检查列
        if any(self.grid[rr][c] == v for rr in range(self.size)):
            return False
        
        # 检查子网格
        box_size = int(self.size ** 0.5)
        box_r, box_c = (r // box_size) * box_size, (c // box_size) * box_size
        for rr in range(box_r, box_r + box_size):
            for cc in range(box_c, box_c + box_size):
                if self.grid[rr][cc] == v:
                    return False
        
        return True
    
    def _is_solved(self) -> bool:
        """检查是否解决"""
        # 检查是否填满
        for row in self.grid:
            if 0 in row:
                return False
        
        # 检查规则
        for r in range(self.size):
            for c in range(self.size):
                v = self.grid[r][c]
                # 临时清空检查唯一性
                self.grid[r][c] = 0
                valid = self._is_valid_placement(r, c, v)
                self.grid[r][c] = v
                if not valid:
                    return False
        return True
    
    def _calculate_progress(self) -> float:
        """计算解题进度 (0-1)"""
        total_cells = self.size * self.size
        filled_cells = sum(1 for r in range(self.size) for c in range(self.size) if self.grid[r][c] != 0)
        return filled_cells / total_cells
    
    def _count_constraints_satisfied(self) -> int:
        """计算满足的约束数量"""
        satisfied = 0
        box_size = int(self.size ** 0.5)
        
        # 检查行约束
        for r in range(self.size):
            values = [self.grid[r][c] for c in range(self.size) if self.grid[r][c] != 0]
            if len(values) == len(set(values)):  # 无重复
                satisfied += len(values) - 1 if len(values) > 1 else 0
        
        # 检查列约束  
        for c in range(self.size):
            values = [self.grid[r][c] for r in range(self.size) if self.grid[r][c] != 0]
            if len(values) == len(set(values)):  # 无重复
                satisfied += len(values) - 1 if len(values) > 1 else 0
        
        # 检查子网格约束
        for box_r in range(0, self.size, box_size):
            for box_c in range(0, self.size, box_size):
                values = []
                for r in range(box_r, box_r + box_size):
                    for c in range(box_c, box_c + box_size):
                        if self.grid[r][c] != 0:
                            values.append(self.grid[r][c])
                if len(values) == len(set(values)):  # 无重复
                    satisfied += len(values) - 1 if len(values) > 1 else 0
        
        return satisfied
    
    def _get_possible_values(self, r: int, c: int) -> Set[int]:
        """获取位置(r,c)的可能取值"""
        if self.grid[r][c] != 0:
            return set()
        
        possible = set(range(1, self.size + 1))
        box_size = int(self.size ** 0.5)
        
        # 排除同行的值
        for cc in range(self.size):
            if self.grid[r][cc] in possible:
                possible.remove(self.grid[r][cc])
        
        # 排除同列的值
        for rr in range(self.size):
            if self.grid[rr][c] in possible:
                possible.remove(self.grid[rr][c])
        
        # 排除同子网格的值
        box_r, box_c = (r // box_size) * box_size, (c // box_size) * box_size
        for rr in range(box_r, box_r + box_size):
            for cc in range(box_c, box_c + box_size):
                if self.grid[rr][cc] in possible:
                    possible.remove(self.grid[rr][cc])
        
        return possible

    def step(self, action):
        """执行动作，更新环境状态。动作格式：完整NxN网格 或 填入步骤列表[[r,c,v],...]"""
        if self.done:
            self.reward = 0.0
            return
        
        # 检查动作格式
        if isinstance(action, list) and len(action) == self.size and all(isinstance(row, list) and len(row) == self.size for row in action):
            # 格式1：完整NxN网格 [[1,2,3,4],[3,4,1,2],...]
            new_grid = action
            
            # 验证网格格式
            if not all(isinstance(val, int) and 1 <= val <= self.size for row in new_grid for val in row):
                self.reward = -1.0  # 无效网格值惩罚
                return
            
            # 记录填入前的状态
            prev_progress = self._calculate_progress()
            prev_constraints = self._count_constraints_satisfied()
            
            # 更新网格
            self.grid = [row[:] for row in new_grid]
            self.puzzle = [row[:] for row in self.grid]  # 更新puzzle属性
            
            # 计算奖励
            self.reward = -0.01  # 基础步数惩罚
            
            # 验证解决方案的正确性
            if self._is_solved():
                self.reward += 2.0  # 成功完成大奖励
                # 效率奖励
                empty_count = sum(1 for r in range(self.size) for c in range(self.size) if self.init_grid[r][c] == 0)
                if self.step_count <= empty_count:  # 最优步数
                    self.reward += 0.5
                self.done = True
            else:
                # 部分正确的奖励
                current_progress = self._calculate_progress()
                current_constraints = self._count_constraints_satisfied()
                
                progress_reward = (current_progress - prev_progress) * 0.5
                constraint_improvement = current_constraints - prev_constraints
                self.reward += progress_reward + constraint_improvement * 0.02
                
                # 如果有错误，给予惩罚
                if not self._is_grid_valid():
                    self.reward -= 0.5
        
        elif isinstance(action, list) and all(isinstance(step, list) and len(step) == 3 for step in action):
            # 格式2：填入步骤列表 [[r,c,v], [r,c,v], ...]
            total_reward = 0.0
            
            for step in action:
                r, c, v = step
                step_reward = -0.01  # 基础步数惩罚
                
                # 记录填入前的状态
                prev_progress = self._calculate_progress()
                prev_constraints = self._count_constraints_satisfied()
                
                if self._is_valid_placement(r, c, v):
                    # 智能填入奖励
                    possible_values = self._get_possible_values(r, c)
                    if len(possible_values) == 1:
                        step_reward += 0.2  # 唯一解奖励
                    elif len(possible_values) <= 2:
                        step_reward += 0.1  # 选择少奖励
                    
                    self.grid[r][c] = v
                    
                    # 计算填入后的状态
                    current_progress = self._calculate_progress()
                    current_constraints = self._count_constraints_satisfied()
                    
                    # 稠密奖励设计
                    progress_reward = (current_progress - prev_progress) * 0.5
                    constraint_improvement = current_constraints - prev_constraints
                    step_reward += progress_reward + constraint_improvement * 0.02 + current_progress * 0.1
                    
                    # 策略奖励
                    empty_cells = [(rr, cc) for rr in range(self.size) for cc in range(self.size) if self.grid[rr][cc] == 0]
                    if empty_cells:
                        current_constraints_count = len(self._get_possible_values(r, c))
                        avg_constraints = sum(len(self._get_possible_values(rr, cc)) for rr, cc in empty_cells) / len(empty_cells)
                        if current_constraints_count <= avg_constraints:
                            step_reward += 0.05
                else:
                    step_reward = -0.3  # 无效动作惩罚
                
                total_reward += step_reward
                self.step_count += 1
            
            # 检查是否完成
            if self._is_solved():
                total_reward += 2.0  # 成功完成大奖励
                empty_count = sum(1 for r in range(self.size) for c in range(self.size) if self.init_grid[r][c] == 0)
                if self.step_count <= empty_count:
                    total_reward += 0.5
                self.done = True
            
            self.reward = total_reward
        else:
            self.reward = -1.0  # 无效动作格式惩罚
            return
        
        self.step_count += 1
        self.puzzle = [row[:] for row in self.grid]  # 更新puzzle属性
        self.observation = self.text_observation()
    
    def _is_grid_valid(self) -> bool:
        """检查当前网格是否违反数独规则"""
        box_size = int(self.size ** 0.5)
        
        # 检查行
        for r in range(self.size):
            values = [self.grid[r][c] for c in range(self.size) if self.grid[r][c] != 0]
            if len(values) != len(set(values)):
                return False
        
        # 检查列
        for c in range(self.size):
            values = [self.grid[r][c] for r in range(self.size) if self.grid[r][c] != 0]
            if len(values) != len(set(values)):
                return False
        
        # 检查子网格
        for box_r in range(0, self.size, box_size):
            for box_c in range(0, self.size, box_size):
                values = []
                for r in range(box_r, box_r + box_size):
                    for c in range(box_c, box_c + box_size):
                        if self.grid[r][c] != 0:
                            values.append(self.grid[r][c])
                if len(values) != len(set(values)):
                    return False
        
        return True

    def to_dict_compact(self) -> dict:
        """Return compact dictionary representation for logging"""
        base_dict = super().to_dict_compact()
        base_dict.update({
            "sudoku_size": f"{self.size}x{self.size}",
            "filled_cells": sum(1 for r in range(self.size) for c in range(self.size) if self.grid[r][c] != 0),
            "total_cells": self.size * self.size,
            "done": self.done,
            "steps": self.step_count,
            "reward": round(self.reward, 3)
        })
        return base_dict


@dataclass
class PlanPathGridEnvState(EnvStateBase):
    """
    2D 网格路径规划 worker（BFS 基准） + 动作/奖励接口。
    - 网格: '.' 可通行, '#' 不可通行
    - 动作: U/D/L/R（4-邻域）
    - 用法：逐步交互: reset_agent() -> step(action_list) ... -> done
    - 动作格式：动作序列 ["R", "R", "D", "D"]
    """
    
    seed: int
    grid_h: int = 10
    grid_w: int = grid_h
    block_ratio: float = 0.22
    r_step: Optional[float] = None
    r_invalid: Optional[float] = None
    r_goal: Optional[float] = None
    r_opt: Optional[float] = None
    r_fail: Optional[float] = None
    gamma: Optional[float] = None
    lambda_pot: Optional[float] = None
    max_steps: Optional[int] = None
    config: Optional[dict] = None
    
    # 环境状态属性
    grid: str = ""
    grid_list: List[str] = field(default_factory=list)
    h: int = 0
    w: int = 0
    start: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    goal: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    _shortest_path_cache: Optional[List[Tuple[int, int]]] = None
    
    # 逐步交互状态
    pos: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    done: bool = False
    steps: int = 0
    step_count: int = 0
    invalid_count: int = 0
    total_reward: float = 0.0
    reward: float = 0.0
    _last_phi: float = 0.0

    # ====== 默认奖励系数（可按需改/在 __init__ 里覆写）======
    DEFAULT_R_STEP: ClassVar[float] = -0.01   # 每步轻微惩罚
    DEFAULT_R_INVALID: ClassVar[float] = -0.10   # 非法动作惩罚（越界/撞墙/非邻接）
    DEFAULT_R_GOAL: ClassVar[float] = +1.00   # 抵达终点奖励
    DEFAULT_R_OPT: ClassVar[float] = +0.50   # 最短路加成（若最短）
    DEFAULT_R_FAIL: ClassVar[float] = -1.00   # 失败（终止但未达）/不可行
    DEFAULT_GAMMA: ClassVar[float] = 0.99    # 折扣仅用于 shaping
    DEFAULT_LAMBDA_POT: ClassVar[float] = 1.00    # shaping 系数
    DEFAULT_MAX_STEPS: ClassVar[int] = 10_000  # 上限（防死循环）

    ACTIONS: ClassVar[Dict[str, Tuple[int,int]]] = {
        "U": (-1, 0),
        "D": (+1, 0),
        "L": ( 0,-1),
        "R": ( 0,+1),
    }

    def __post_init__(self):
        super().__post_init__()
        # Read map_size parameter from config if available
        if self.config and hasattr(self.config, 'map_size'):
            self.grid_h = self.config.map_size
            self.grid_w = self.config.map_size
        elif self.config and isinstance(self.config, dict) and 'map_size' in self.config:
            self.grid_h = self.config['map_size']
            self.grid_w = self.config['map_size']
        
        # 根据seed生成随机环境
        grid, start, goal = self._generate_random_environment(self.seed, self.grid_h, self.grid_w, self.block_ratio)
        
        # 地图/基础
        self.grid = '\n'.join(grid)  # prompt需要的字符串格式
        self.grid_list = grid  # 保留列表格式供内部使用
        self.h = len(grid)
        self.w = len(grid[0]) if self.h > 0 else 0
        self.start = tuple(start)
        self.goal = tuple(goal)
        self._shortest_path_cache = None

        # 奖励参数
        self.r_step     = self.DEFAULT_R_STEP     if self.r_step     is None else self.r_step
        self.r_invalid  = self.DEFAULT_R_INVALID  if self.r_invalid  is None else self.r_invalid
        self.r_goal     = self.DEFAULT_R_GOAL     if self.r_goal     is None else self.r_goal
        self.r_opt      = self.DEFAULT_R_OPT      if self.r_opt      is None else self.r_opt
        self.r_fail     = self.DEFAULT_R_FAIL     if self.r_fail     is None else self.r_fail
        self.gamma      = self.DEFAULT_GAMMA      if self.gamma      is None else self.gamma
        self.lambda_pot = self.DEFAULT_LAMBDA_POT if self.lambda_pot is None else self.lambda_pot
        self.max_steps  = self.DEFAULT_MAX_STEPS  if self.max_steps  is None else self.max_steps

        # 逐步交互状态
        self.reset_agent()
        
        # 为新step方法添加的属性
        self.reward = 0.0
        self.done = False
        self.step_count = 0
        self.observation = self.text_observation()
    
    def _generate_random_environment(self, seed: int, grid_h: int, grid_w: int, block_ratio: float) -> Tuple[List[str], Tuple[int, int], Tuple[int, int]]:
        """根据seed生成随机的grid、起始点和终点"""
        # 设置随机种子确保可重现性
        rng = random.Random(seed)
        np.random.seed(seed)
        
        max_trials = max(2000, 50)  # 最大尝试次数
        for _ in range(max_trials):
            # 生成随机网格 (0=可通行, 1=障碍物)
            grid_array = (np.random.rand(grid_h, grid_w) < block_ratio).astype(int)
            
            # 找到所有可通行的位置
            free_positions = [(r, c) for r in range(grid_h) for c in range(grid_w) if grid_array[r, c] == 0]
            
            if len(free_positions) < 2:  # 至少需要两个可通行位置
                continue
                
            # 随机选择起始点和终点
            start = rng.choice(free_positions)
            goal = rng.choice(free_positions)
            while goal == start:  # 确保起始点和终点不同
                goal = rng.choice(free_positions)
            
            # 检查是否存在从起始点到终点的路径
            if self._bfs_check_reachable(grid_array, start, goal):
                # 将numpy数组转换为字符串列表格式
                grid_str = []
                for row in grid_array:
                    row_str = ''.join('.' if cell == 0 else '#' for cell in row)
                    grid_str.append(row_str)
                
                return grid_str, start, goal
        
        # 如果无法生成有效环境，创建一个简单的默认环境
        print(f"[WARN] 无法为seed {seed}生成有效环境，使用默认环境")
        return self._create_default_environment(grid_h, grid_w)
    
    def _bfs_check_reachable(self, grid_array: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """使用BFS检查从起始点到终点是否可达"""
        h, w = grid_array.shape
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
                
            # 检查四个方向的邻居
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and 
                    grid_array[nr, nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return False
    
    def _create_default_environment(self, grid_h: int, grid_w: int) -> Tuple[List[str], Tuple[int, int], Tuple[int, int]]:
        """创建一个简单的默认环境（全部可通行）"""
        grid = ['.' * grid_w for _ in range(grid_h)]
        start = (0, 0)
        goal = (grid_h - 1, grid_w - 1)
        return grid, start, goal
    
    def text_observation(self) -> str:
        """Text observation"""
        obs_lines = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                if (r, c) == self.pos:
                    row += "S"  # 当前位置
                elif (r, c) == self.goal:
                    row += "G"  # 目标位置
                else:
                    row += self.grid_list[r][c]
            obs_lines.append(row)
        return "\n".join(obs_lines)

    # ============== 几何/图搜索 ==============
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def passable(self, r: int, c: int) -> bool:
        return self.grid_list[r][c] != '#'

    def neighbors(self, r: int, c: int):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.passable(nr, nc):
                yield (nr, nc)

    def shortest_path(self) -> Optional[List[Tuple[int, int]]]:
        """BFS 求最短路（含起终点）；不可达返回 None"""
        if self._shortest_path_cache is not None:
            return self._shortest_path_cache
        from collections import deque
        q = deque([self.start])
        prev: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {self.start: None}
        while q:
            cur = q.popleft()
            if cur == self.goal:
                path = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = prev[node]
                path.reverse()
                self._shortest_path_cache = path
                return path
            for nxt in self.neighbors(*cur):
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        return None


    # ============== 表示/描述 ==============

    def describe(self) -> str:
        return (
            "PlanPathGridWorker: 2D grid shortest-path (BFS). "
            "'.' passable, '#' blocked; moves: U/D/L/R (4-neighborhood)."
        )

    # ============== 动作接口（逐步交互）===============
    def reset_agent(self):
        """重置逐步交互状态"""
        self.pos: Tuple[int,int] = self.start
        self.done: bool = False
        self.steps: int = 0
        self.step_count: int = 0
        self.invalid_count: int = 0
        self.total_reward: float = 0.0
        self.reward: float = 0.0
        self._last_phi: float = self._potential(self.pos)
        self.observation = self.text_observation()

    def get_valid_actions(self, pos: Optional[Tuple[int,int]] = None) -> List[str]:
        """返回当前位置可行动作集合（不含越界/撞墙）"""
        if pos is None: pos = self.pos
        valid = []
        for a, (dr, dc) in self.ACTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if self.in_bounds(nr, nc) and self.passable(nr, nc):
                valid.append(a)
        return valid

    def _potential(self, pos: Tuple[int,int]) -> float:
        """势能: 负的 Manhattan 距离（越接近目标值越大）"""
        return - (abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1]))

    def _apply_action(self, pos: Tuple[int,int], action: str) -> Tuple[Tuple[int,int], bool]:
        """尝试应用动作；返回 (next_pos, is_valid)"""
        if action not in self.ACTIONS:
            return pos, False
        dr, dc = self.ACTIONS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not self.in_bounds(nr, nc) or not self.passable(nr, nc):
            return pos, False
        return (nr, nc), True

    def step(self, action):
        """
        执行动作，更新环境状态。动作格式：
        动作序列: ["R", "R", "D", "D"]
        """
        if self.done:
            self.reward = 0.0
            return
        
        # 检查动作格式
        if isinstance(action, list) and all(isinstance(item, str) for item in action):
            # 动作序列 ["R", "R", "D", "D"]
            self._execute_action_sequence(action)
        else:
            self.reward = -1.0  # 无效动作格式
    
    def _execute_action_sequence(self, actions: List[str]):
        """执行动作序列"""
        total_reward = 0.0
        for action in actions:
            pos, reward, done, _ = self.step_single(action)
            total_reward += reward
            if done:
                break
        self.reward = total_reward
    
    
    def step_single(self, action: str) -> Tuple[Tuple[int,int], float, bool, Dict[str,Any]]:
        """
        执行单个动作一步，并返回:
          next_pos, reward, done, info
        奖励 = 基础步惩罚/非法惩罚 + potential-based shaping (+ 终止奖励/最优加成)
        """
        if self.done:
            return self.pos, 0.0, True, {"msg": "episode already done"}
        prev_pos = self.pos
        next_pos, valid = self._apply_action(prev_pos, action)

        # 基础奖励
        reward = 0.0
        if valid:
            reward += self.r_step
        else:
            reward += self.r_invalid
            self.invalid_count += 1

        # Shaping（不改变最优策略）
        cur_phi = self._last_phi
        nxt_phi = self._potential(next_pos)
        shaping = self.lambda_pot * (self.gamma * nxt_phi - cur_phi)
        reward += shaping

        # 状态更新
        self.pos = next_pos if valid else prev_pos
        self._last_phi = self._potential(self.pos)
        self.steps += 1

        # 终止判定
        if self.pos == self.goal:
            # 抵达终点
            reward += self.r_goal
            # 最优加成（如果可达）
            sp = self.shortest_path()
            if sp is not None:
                if self.steps == len(sp) - 1:  # 注意：steps 是“动作步数”，sp 是“节点数”
                    reward += self.r_opt
            self.done = True
        elif self.steps >= self.max_steps:
            # 超步失败
            reward += self.r_fail
            self.done = True

        self.total_reward += reward
        info = {
            "valid": valid,
            "steps": self.steps,
            "invalid_count": self.invalid_count,
            "shaping": shaping,
            "pos": self.pos,
            "goal": self.goal,
            "done": self.done,
        }
        return self.pos, reward, self.done, info

    def to_dict_compact(self) -> dict:
        """Return compact dictionary representation for logging"""
        base_dict = super().to_dict_compact()
        base_dict.update({
            "grid_size": f"{self.h}x{self.w}",
            "start": self.start,
            "goal": self.goal,
            "current_pos": self.pos,
            "distance_to_goal": abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1]),
            "done": self.done,
            "steps": self.steps,
            "reward": round(self.reward, 3)
        })
        return base_dict

    



# =========================================================
# Benchmark Registration System
# =========================================================

# 注册所有可用的状态类
STATE_REGISTRY = {
    "EightQueens": EightQueensEnvState,
    "Blocksworld": BlocksworldEnvState, 
    "sudoku4x4": Sudoku4x4EnvState,
    "PlanPath": PlanPathGridEnvState,
    # 可以根据需要添加更多benchmark
}


def get_state_class_by_benchmark(benchmark_name: str):


    if benchmark_name not in STATE_REGISTRY:
        available_benchmarks = list(STATE_REGISTRY.keys())
        raise ValueError(f"未知的benchmark名称: {benchmark_name}. 可用的benchmark有: {available_benchmarks}")
    
    return STATE_REGISTRY[benchmark_name]


def register_state_class(benchmark_name: str, state_class):
    STATE_REGISTRY[benchmark_name] = state_class


def list_available_benchmarks() -> List[str]:
    return list(STATE_REGISTRY.keys())

