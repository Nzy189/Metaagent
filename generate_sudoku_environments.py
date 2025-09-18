#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数独环境生成器
为不同尺寸的数独生成预定义的环境并保存为JSON文件
支持：4x4, 6x6, 8x8, 10x10, 16x16
每个尺寸生成400个不同的环境
"""

import json
import random
import numpy as np
from typing import List, Dict, Any
import os

def generate_simple_sudoku_template(size: int, seed: int) -> List[List[int]]:
    """
    生成简单的数独模板，使用更直接的方法避免回溯算法的性能问题
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建空网格
    grid = [[0 for _ in range(size)] for _ in range(size)]
    box_size = int(size ** 0.5)
    
    # 为每个子网格填充一些基础数字，确保不违反规则
    for box_row in range(0, size, box_size):
        for box_col in range(0, size, box_size):
            # 在每个子网格中随机放置几个数字
            available_numbers = list(range(1, size + 1))
            random.shuffle(available_numbers)
            
            # 在子网格中随机选择一些位置放置数字
            positions = [(r, c) for r in range(box_row, box_row + box_size) 
                        for c in range(box_col, box_col + box_size)]
            random.shuffle(positions)
            
            # 根据难度决定填充多少个数字
            fill_count = random.randint(1, min(3, len(available_numbers)))
            
            for i in range(fill_count):
                if i >= len(positions) or i >= len(available_numbers):
                    break
                    
                r, c = positions[i]
                num = available_numbers[i]
                
                # 简单检查是否可以放置（避免明显的冲突）
                if is_safe_placement(grid, r, c, num, size):
                    grid[r][c] = num
    
    # 在网格的其他位置随机添加一些数字
    empty_positions = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == 0]
    random.shuffle(empty_positions)
    
    # 添加一些随机数字，但要确保不违反基本规则
    additional_count = random.randint(size // 4, size // 2)
    for i, (r, c) in enumerate(empty_positions[:additional_count]):
        for num in random.sample(range(1, size + 1), size):
            if is_safe_placement(grid, r, c, num, size):
                grid[r][c] = num
                break
    
    return grid

def is_safe_placement(grid: List[List[int]], row: int, col: int, num: int, size: int) -> bool:
    """检查在指定位置放置数字是否安全（不违反数独规则）"""
    box_size = int(size ** 0.5)
    
    # 检查行
    for c in range(size):
        if grid[row][c] == num:
            return False
    
    # 检查列
    for r in range(size):
        if grid[r][col] == num:
            return False
    
    # 检查子网格
    start_row = (row // box_size) * box_size
    start_col = (col // box_size) * box_size
    
    for r in range(start_row, start_row + box_size):
        for c in range(start_col, start_col + box_size):
            if grid[r][c] == num:
                return False
    
    return True

def generate_sudoku_environments_for_size(size: int, count: int = 400) -> List[Dict[str, Any]]:
    """为指定尺寸生成指定数量的数独环境"""
    print(f"正在生成 {size}x{size} 数独环境，共 {count} 个...")
    
    environments = []
    
    for i in range(count):
        # 使用不同的seed确保多样性
        seed = i * 1000 + size * 100
        
        try:
            puzzle = generate_simple_sudoku_template(size, seed)
            
            # 计算一些统计信息
            filled_cells = sum(1 for row in puzzle for cell in row if cell != 0)
            total_cells = size * size
            fill_ratio = filled_cells / total_cells
            
            env_data = {
                "id": i,
                "size": size,
                "seed": seed,
                "puzzle": puzzle,
                "filled_cells": filled_cells,
                "total_cells": total_cells,
                "fill_ratio": round(fill_ratio, 3),
                "difficulty": "easy" if fill_ratio > 0.6 else "medium" if fill_ratio > 0.4 else "hard"
            }
            
            environments.append(env_data)
            
            # 每100个输出一次进度
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{count} 个 {size}x{size} 环境")
                
        except Exception as e:
            print(f"  生成第 {i} 个 {size}x{size} 环境时出错: {e}")
            # 生成一个最小的fallback环境
            puzzle = [[0 for _ in range(size)] for _ in range(size)]
            # 在对角线上放一些数字
            for j in range(min(3, size)):
                if j < size:
                    puzzle[j][j] = (j % size) + 1
            
            env_data = {
                "id": i,
                "size": size,
                "seed": seed,
                "puzzle": puzzle,
                "filled_cells": min(3, size),
                "total_cells": size * size,
                "fill_ratio": min(3, size) / (size * size),
                "difficulty": "minimal"
            }
            environments.append(env_data)
    
    print(f"✅ 完成生成 {size}x{size} 数独环境，共 {len(environments)} 个")
    return environments

def main():
    """主函数：生成所有尺寸的数独环境"""
    print("🎯 开始生成数独环境...")
    
    # 支持的数独尺寸（都是完全平方数）
    sizes = [4, 6, 9, 12, 16]  # 简化为主要尺寸，6x6和8x8不是完全平方数
    count_per_size = 400
    
    # 创建输出目录
    output_dir = "datasets/sudoku_environments"
    os.makedirs(output_dir, exist_ok=True)
    
    all_environments = {}
    
    for size in sizes:
        print(f"\n📋 处理 {size}x{size} 数独...")
        
        # 生成环境
        environments = generate_sudoku_environments_for_size(size, count_per_size)
        
        # 保存到单独的文件
        filename = f"{output_dir}/sudoku_{size}x{size}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(environments, f, indent=2, ensure_ascii=False)
        
        print(f"💾 已保存到 {filename}")
        
        # 添加到总集合
        all_environments[f"{size}x{size}"] = environments
        
        # 显示统计信息
        fill_ratios = [env["fill_ratio"] for env in environments]
        avg_fill_ratio = sum(fill_ratios) / len(fill_ratios)
        print(f"📊 {size}x{size} 统计: 平均填充率 {avg_fill_ratio:.3f}")
    
    # 保存所有环境到一个总文件
    all_filename = f"{output_dir}/all_sudoku_environments.json"
    with open(all_filename, 'w', encoding='utf-8') as f:
        json.dump(all_environments, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 所有数独环境生成完成！")
    print(f"📁 输出目录: {output_dir}/")
    print(f"📄 总文件: {all_filename}")
    
    # 显示文件大小
    for size in sizes:
        filename = f"{output_dir}/sudoku_{size}x{size}.json"
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"   - sudoku_{size}x{size}.json: {size_mb:.2f} MB")

def test_generated_environments():
    """测试生成的环境"""
    print("\n🧪 测试生成的环境...")
    
    # 测试小尺寸
    test_env = generate_simple_sudoku_template(4, 42)
    print("4x4 测试环境:")
    for row in test_env:
        print("  " + " ".join(f"{x:2d}" if x != 0 else " ." for x in row))
    
    # 验证基本规则
    size = 4
    box_size = 2
    valid = True
    
    # 检查行
    for r in range(size):
        row_nums = [test_env[r][c] for c in range(size) if test_env[r][c] != 0]
        if len(row_nums) != len(set(row_nums)):
            valid = False
            print(f"❌ 行 {r} 有重复数字")
    
    # 检查列
    for c in range(size):
        col_nums = [test_env[r][c] for r in range(size) if test_env[r][c] != 0]
        if len(col_nums) != len(set(col_nums)):
            valid = False
            print(f"❌ 列 {c} 有重复数字")
    
    if valid:
        print("✅ 测试环境通过基本验证")
    else:
        print("❌ 测试环境未通过验证")

if __name__ == "__main__":
    # 运行测试
    test_generated_environments()
    
    # 生成所有环境
    main()
