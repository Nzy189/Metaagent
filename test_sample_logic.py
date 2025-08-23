#!/usr/bin/env python3
"""
测试新的agent sample笛卡尔积逻辑
"""

import itertools
from typing import Dict, List

def test_agent_sample_combinations():
    """测试agent sample组合生成逻辑"""
    
    # 模拟配置
    turn_order = ["code_generator", "test_generator", "reviewer"]
    agent_sample_nums = {
        "code_generator": 2,
        "test_generator": 3, 
        "reviewer": 2
    }
    
    print("Testing Agent Sample Combinations")
    print("=" * 50)
    
    # 计算总sample数
    total_samples = 1
    for sample_num in agent_sample_nums.values():
        total_samples *= sample_num
    
    print(f"Agent configurations:")
    for agent_name in turn_order:
        sample_num = agent_sample_nums[agent_name]
        print(f"  {agent_name}: {sample_num} samples")
    
    print(f"\nTotal combinations: {total_samples}")
    
    # 生成笛卡尔积组合
    agent_sample_ranges = []
    agent_names_order = []
    
    for agent_name in turn_order:
        sample_num = agent_sample_nums[agent_name]
        agent_sample_ranges.append(list(range(sample_num)))
        agent_names_order.append(agent_name)
    
    # 生成笛卡尔积
    combinations = []
    for combination in itertools.product(*agent_sample_ranges):
        combo_dict = {}
        for i, agent_name in enumerate(agent_names_order):
            combo_dict[agent_name] = combination[i]
        combinations.append(combo_dict)
    
    print(f"\nGenerated {len(combinations)} combinations:")
    for i, combo in enumerate(combinations):
        combo_str = ", ".join(f"{name}={idx}" for name, idx in combo.items())
        print(f"  Rollout {i}: {combo_str}")
    
    # 验证每个agent的每个sample索引都被使用了
    print(f"\nVerification:")
    for agent_name in turn_order:
        used_indices = set()
        for combo in combinations:
            used_indices.add(combo[agent_name])
        expected_indices = set(range(agent_sample_nums[agent_name]))
        print(f"  {agent_name}: used {sorted(used_indices)}, expected {sorted(expected_indices)}")
        assert used_indices == expected_indices, f"Missing indices for {agent_name}"
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_agent_sample_combinations()
