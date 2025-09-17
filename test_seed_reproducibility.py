#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试PlanPathGridEnvState的seed可重现性
验证相同的seed会生成相同的环境配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pettingllms.multi_agent_env.plan_path.env_state import PlanPathGridEnvState

def test_seed_reproducibility():
    """测试相同seed生成相同环境"""
    print("测试seed可重现性...")
    
    # 测试多个不同的seed值
    test_seeds = [42, 123, 456, 789, 1000]
    
    for seed in test_seeds:
        print(f"\n测试seed: {seed}")
        
        # 创建两个相同seed的环境实例
        env1 = PlanPathGridEnvState(seed=seed)
        env2 = PlanPathGridEnvState(seed=seed)
        
        # 验证grid相同
        grid1 = env1.grid
        grid2 = env2.grid
        assert grid1 == grid2, f"Grid不匹配！seed={seed}"
        
        # 验证start相同
        start1 = env1.start
        start2 = env2.start
        assert start1 == start2, f"Start不匹配！seed={seed}, {start1} != {start2}"
        
        # 验证goal相同
        goal1 = env1.goal
        goal2 = env2.goal
        assert goal1 == goal2, f"Goal不匹配！seed={seed}, {goal1} != {goal2}"
        
        print(f"✓ seed {seed}: grid={env1.h}x{env1.w}, start={start1}, goal={goal1}")
        print(f"Grid:")
        print(grid1)
    
    print("\n✅ 所有测试通过！相同seed确实生成相同的环境配置。")

def test_different_seeds_generate_different_environments():
    """测试不同seed生成不同环境"""
    print("\n测试不同seed生成不同环境...")
    
    seeds = [1, 2, 3, 4, 5]
    environments = []
    
    for seed in seeds:
        env = PlanPathGridEnvState(seed=seed)
        env_config = (env.grid, env.start, env.goal)
        environments.append(env_config)
        print(f"seed {seed}: start={env.start}, goal={env.goal}")
    
    # 检查是否有不同的环境配置
    unique_configs = set(environments)
    if len(unique_configs) > 1:
        print(f"✅ 不同seed生成了{len(unique_configs)}种不同的环境配置")
    else:
        print("⚠️  所有seed生成了相同的环境配置")

def test_custom_parameters():
    """测试自定义参数"""
    print("\n测试自定义参数...")
    
    # 测试不同的网格大小和障碍物比例
    configs = [
        {"grid_h": 3, "grid_w": 3, "block_ratio": 0.1},
        {"grid_h": 6, "grid_w": 4, "block_ratio": 0.3},
        {"grid_h": 4, "grid_w": 4, "block_ratio": 0.15},
    ]
    
    for i, config in enumerate(configs):
        seed = 100 + i
        env = PlanPathGridEnvState(seed=seed, **config)
        print(f"配置 {i+1}: {config['grid_h']}x{config['grid_w']}, 障碍物比例={config['block_ratio']}")
        print(f"  实际大小: {env.h}x{env.w}, start={env.start}, goal={env.goal}")
        print(f"  Grid:")
        print(env.grid)
        print()

if __name__ == "__main__":
    test_seed_reproducibility()
    test_different_seeds_generate_different_environments()
    test_custom_parameters()
    print("\n🎉 所有测试完成！")
