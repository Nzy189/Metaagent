#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 Sudoku4x4EnvState 的 map_size 配置参数功能
验证从 config.map_size 参数中读取数独大小的功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pettingllms.multi_agent_env.plan_path.env_state import Sudoku4x4EnvState

def test_sudoku_config_map_size():
    """测试从config中读取map_size参数"""
    print("测试Sudoku4x4EnvState的config.map_size参数功能...")
    
    # 测试1: 使用字典形式的config，4x4数独
    print("\n测试1: 4x4数独（字典形式config）")
    config_dict = {'map_size': 4}
    env1 = Sudoku4x4EnvState(seed=42, config=config_dict)
    print(f"✓ 使用config={'map_size': 4}, size={env1.size}")
    print(f"  生成的puzzle:")
    for row in env1.puzzle:
        print(f"  {row}")
    assert env1.size == 4, f"期望 size=4, 实际 size={env1.size}"
    assert len(env1.puzzle) == 4, f"期望 puzzle高度=4, 实际={len(env1.puzzle)}"
    assert all(len(row) == 4 for row in env1.puzzle), "期望所有行长度为4"
    
    # 测试2: 使用对象形式的config，9x9数独
    print("\n测试2: 9x9数独（对象形式config）")
    class MockConfig:
        def __init__(self, map_size):
            self.map_size = map_size
    
    config_obj = MockConfig(map_size=9)
    env2 = Sudoku4x4EnvState(seed=123, config=config_obj)
    print(f"✓ 使用config.map_size=9, size={env2.size}")
    print(f"  生成的puzzle前3行:")
    for i, row in enumerate(env2.puzzle[:3]):
        print(f"  {row}")
    assert env2.size == 9, f"期望 size=9, 实际 size={env2.size}"
    assert len(env2.puzzle) == 9, f"期望 puzzle高度=9, 实际={len(env2.puzzle)}"
    assert all(len(row) == 9 for row in env2.puzzle), "期望所有行长度为9"
    
    # 测试3: 没有config时使用默认值
    print("\n测试3: 没有config时使用默认值")
    env3 = Sudoku4x4EnvState(seed=456)
    print(f"✓ 没有config, size={env3.size}")
    print(f"  生成的puzzle:")
    for row in env3.puzzle:
        print(f"  {row}")
    assert env3.size == 4, f"期望 size=4, 实际 size={env3.size}"
    
    # 测试4: 非完全平方数会被调整
    print("\n测试4: 非完全平方数会被调整")
    config_invalid = {'map_size': 5}  # 5不是完全平方数
    env4 = Sudoku4x4EnvState(seed=789, config=config_invalid)
    print(f"✓ 使用config={'map_size': 5}, 调整后size={env4.size}")
    assert env4.size == 9, f"期望调整后 size=9, 实际 size={env4.size}"
    
    # 测试5: 验证生成的数独有效性
    print("\n测试5: 验证生成的数独有效性")
    config_16 = {'map_size': 16}
    env5 = Sudoku4x4EnvState(seed=111, config=config_16)
    print(f"✓ 使用config={'map_size': 16}, size={env5.size}")
    print(f"  16x16数独的前2行:")
    for i, row in enumerate(env5.puzzle[:2]):
        print(f"  {row}")
    
    # 验证网格有效性
    assert env5.size == 16, f"期望 size=16, 实际 size={env5.size}"
    assert len(env5.puzzle) == 16, f"期望 puzzle高度=16, 实际={len(env5.puzzle)}"
    assert all(len(row) == 16 for row in env5.puzzle), "期望所有行长度为16"
    
    # 验证数字范围
    for row in env5.puzzle:
        for val in row:
            assert 0 <= val <= 16, f"数独中的值应该在0-16范围内，实际值: {val}"
    
    # 测试6: 验证seed的可重现性
    print("\n测试6: 验证seed的可重现性")
    config_same = {'map_size': 9}
    env6a = Sudoku4x4EnvState(seed=999, config=config_same)
    env6b = Sudoku4x4EnvState(seed=999, config=config_same)
    
    print(f"✓ 相同seed生成的两个9x9数独应该相同")
    assert env6a.puzzle == env6b.puzzle, "相同seed应该生成相同的puzzle"
    print(f"  验证通过：相同seed生成相同puzzle")
    
    print("\n✅ 所有测试通过！Sudoku4x4EnvState的config.map_size参数功能正常工作。")
    print("现在数独环境支持4x4、9x9、16x16等不同大小，通过config.map_size参数控制。")

def test_sudoku_gameplay():
    """测试数独游戏功能"""
    print("\n\n测试数独游戏功能...")
    
    # 创建一个4x4数独
    config = {'map_size': 4}
    env = Sudoku4x4EnvState(seed=42, config=config)
    
    print(f"初始4x4数独puzzle:")
    print(env.text_observation())
    
    # 测试available_actions
    actions = env.available_actions()
    print(f"\n可用动作数量: {len(actions)}")
    print(f"前5个可用动作: {actions[:5]}")
    
    # 验证动作范围
    for r, c, v in actions:
        assert 0 <= r < 4, f"行索引应该在0-3范围内: {r}"
        assert 0 <= c < 4, f"列索引应该在0-3范围内: {c}"  
        assert 1 <= v <= 4, f"值应该在1-4范围内: {v}"
    
    print("✅ 数独游戏功能测试通过！")

if __name__ == "__main__":
    test_sudoku_config_map_size()
    test_sudoku_gameplay()
