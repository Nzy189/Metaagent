#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试新的数独环境路径是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pettingllms.multi_agent_env.plan_path.env_state import Sudoku4x4EnvState

def test_new_path():
    """测试新路径下的数独环境加载"""
    print("🧪 测试新路径下的数独环境加载...")
    
    # 测试不同尺寸
    sizes = [4, 9, 16]
    
    for size in sizes:
        print(f"\n📋 测试 {size}x{size} 数独...")
        
        config = {"map_size": size}
        
        try:
            # 创建环境
            env = Sudoku4x4EnvState(seed=42, config=config)
            
            print(f"✅ 成功从新路径加载 {size}x{size} 数独环境")
            print(f"   实际大小: {env.size}x{env.size}")
            
            # 显示一些基本信息
            filled_cells = sum(1 for row in env.puzzle for cell in row if cell != 0)
            total_cells = size * size
            print(f"   填充率: {filled_cells}/{total_cells} ({filled_cells/total_cells:.2%})")
            
        except Exception as e:
            print(f"❌ 从新路径加载 {size}x{size} 数独环境失败: {e}")
    
    print("\n🎉 新路径测试完成！")

if __name__ == "__main__":
    test_new_path()
