#!/usr/bin/env python3
"""
测试稠密奖励系统的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pettingllms.multi_agent_env.plan_path.env_state import PlanPathGridEnvState

def test_dense_rewards():
    """测试稠密奖励系统"""
    print("=== 测试稠密奖励系统 ===")
    
    # 创建一个简单的环境
    env = PlanPathGridEnvState(
        seed=42,
        grid_h=5,
        grid_w=5,
        block_ratio=0.1,  # 较少障碍物
        r_step=-0.1,
        r_on_path=0.1,
        r_progress=0.2,
        r_direction=0.05
    )
    
    print(f"环境网格:")
    print(env.text_observation())
    print(f"起始位置: {env.start}")
    print(f"目标位置: {env.goal}")
    
    # 获取最优路径
    optimal_path = env.shortest_path()
    if optimal_path:
        print(f"最优路径: {optimal_path}")
        print(f"最优路径长度: {len(optimal_path)} 个位置")
    else:
        print("无法找到最优路径")
        return
    
    print("\n=== 测试移动奖励 ===")
    
    # 重置环境
    env.reset_agent()
    total_reward = 0.0
    
    # 尝试几个动作并观察奖励
    test_actions = ["R", "D", "R", "D"]  # 简单的测试动作序列
    
    for i, action in enumerate(test_actions):
        if env.done:
            break
            
        print(f"\n步骤 {i+1}: 执行动作 '{action}'")
        print(f"当前位置: {env.pos}")
        
        pos, reward, done, info = env.step_single(action)
        total_reward += reward
        
        print(f"新位置: {pos}")
        print(f"奖励: {reward:.3f}")
        print(f"稠密奖励详情: {info.get('dense_rewards', {})}")
        print(f"在最优路径上: {env._is_on_optimal_path(pos)}")
        print(f"累计奖励: {total_reward:.3f}")
        
        if done:
            print(f"回合结束: {'成功到达目标' if pos == env.goal else '失败'}")
            break
    
    print(f"\n最终累计奖励: {total_reward:.3f}")
    
def test_optimal_path_following():
    """测试沿着最优路径移动的奖励"""
    print("\n\n=== 测试最优路径跟随 ===")
    
    env = PlanPathGridEnvState(
        seed=123,
        grid_h=4,
        grid_w=4,
        block_ratio=0.0,  # 无障碍物
        r_step=-0.1,
        r_on_path=0.15,
        r_progress=0.25,
        r_direction=0.1
    )
    
    print(f"环境网格:")
    print(env.text_observation())
    
    optimal_path = env.shortest_path()
    if not optimal_path:
        print("无法找到最优路径")
        return
        
    print(f"最优路径: {optimal_path}")
    
    # 计算从路径推导出的动作序列
    actions = []
    for i in range(len(optimal_path) - 1):
        curr = optimal_path[i]
        next_pos = optimal_path[i + 1]
        
        # 计算动作
        dr = next_pos[0] - curr[0]
        dc = next_pos[1] - curr[1]
        
        if dr == -1: actions.append("U")
        elif dr == 1: actions.append("D")
        elif dc == -1: actions.append("L")
        elif dc == 1: actions.append("R")
    
    print(f"最优动作序列: {actions}")
    
    # 执行最优路径
    env.reset_agent()
    total_reward = 0.0
    
    for i, action in enumerate(actions):
        print(f"\n最优步骤 {i+1}: 执行动作 '{action}'")
        pos, reward, done, info = env.step_single(action)
        total_reward += reward
        
        print(f"位置: {pos}, 奖励: {reward:.3f}")
        print(f"稠密奖励: {info.get('dense_rewards', {})}")
        
        if done:
            break
    
    print(f"\n最优路径总奖励: {total_reward:.3f}")

if __name__ == "__main__":
    test_dense_rewards()
    test_optimal_path_following()
