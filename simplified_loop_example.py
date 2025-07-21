#!/usr/bin/env python3
"""
示例：使用简化的multi-agent loop函数

这个示例展示了新的设计：
1. loop()函数只处理单步交互
2. 返回最基本的数据：prompt, response, action, reward
3. 循环控制在外部处理（如在engine中）
"""

import asyncio
import tempfile
from typing import Dict, Any

from pettingllms.agentgpraphs.design_human_interact.agent_collaboration_graph import FrontendDesignAgentGraph
from pettingllms.agentgpraphs.design_human_interact.websight_env import WebEnv


async def demonstrate_simplified_loop():
    """演示简化的loop函数使用"""
    print("🚀 演示简化的Multi-Agent Loop设计")
    print("=" * 50)
    
    # 创建agent graph
    agent_graph = FrontendDesignAgentGraph(
        hostname="localhost",
        code_port=8000,
        visual_port=8001,
        max_iterations=3,
        temp_path=tempfile.gettempdir()
    )
    
    # 创建mock环境和观察
    mock_sample = {
        "task_id": "demo_task",
        "problem_description": "Create a navigation bar",
        "ground_truth": "<html><nav>Navigation</nav></html>"
    }
    
    env = WebEnv(task=mock_sample, max_turns=6, temp_path=tempfile.gettempdir())
    
    # 重置agents
    agents_info = agent_graph._get_agents_list()
    print(f"🔍 检测到 {len(agents_info)} 个agents: {[name for name, _, _ in agents_info]}")
    
    for _, agent_instance, _ in agents_info:
        agent_instance.reset()
    
    # 获取初始观察
    obs, _ = env.reset()
    
    print(f"\n📝 任务: {mock_sample['problem_description']}")
    
    # 模拟engine中的多步循环
    for step_idx in range(3):
        print(f"\n🔄 Step {step_idx + 1}")
        print("-" * 30)
        
        # ============ 调用简化的loop函数 ============
        print("📞 调用 agent_graph.loop()...")
        step_data = await agent_graph.loop(obs, step_idx)
        
        # 显示loop函数的输出
        print("📊 Loop函数输出:")
        for agent_name, agent_data in step_data.items():
            print(f"  {agent_name} ({agent_data['original_name']}):")
            print(f"    ↳ Action Type: {agent_data['action_type']}")
            print(f"    ↳ Action: {agent_data['action'][:50]}...")
            print(f"    ↳ Response: {agent_data['response'][:50]}...")
        
        # ============ 环境交互 (在engine中处理) ============
        print("\n🌍 环境交互:")
        
        for agent_name, agent_data in step_data.items():
            action_type = agent_data["action_type"]
            action = agent_data["action"]
            
            print(f"  {agent_name}: {action_type} action")
            
            # 与环境交互
            obs, reward, done, info = env.step(action_type, action)
            
            print(f"    ↳ Reward: {reward}")
            
            # 更新奖励
            env_results = {"default_reward": reward}
            step_data = await agent_graph.update_rewards(step_data, env_results)
        
        # ============ 显示更新后的数据 ============
        print("\n✅ 更新后的数据:")
        for agent_name, agent_data in step_data.items():
            print(f"  {agent_name}: reward = {agent_data['reward']}")
        
        # 检查终止条件
        if done or not obs.get("current_image"):
            print("🏁 达到终止条件")
            break
    
    print("\n✨ 演示完成!")
    env.cleanup()


def show_data_structure():
    """展示简化的数据结构"""
    print("\n📋 简化的数据结构:")
    print("=" * 40)
    
    example_loop_output = {
        "agent1": {
            "original_name": "code_agent",
            "prompt": [{"role": "user", "content": "Generate HTML..."}],
            "response": "Generated HTML response...",
            "action": "<html><body>Generated content</body></html>",
            "action_type": "code",
            "reward": 0.8
        },
        "agent2": {
            "original_name": "visual_agent", 
            "prompt": [{"role": "user", "content": "Analyze design..."}],
            "response": "Visual analysis response...",
            "action": "Add more padding and improve colors",
            "action_type": "visual",
            "reward": 0.6
        }
    }
    
    import json
    print(json.dumps(example_loop_output, indent=2, ensure_ascii=False))


def compare_old_vs_new():
    """对比旧设计与新设计"""
    print("\n📊 设计对比:")
    print("=" * 40)
    
    comparison = """
    旧设计 (multi_agent_loop):
    ❌ 处理完整的多步循环
    ❌ 复杂的数据结构 (trajectory_steps, execution_time等)
    ❌ 包含时间统计、终止判断等逻辑
    ❌ 难以与现有engine集成
    
    新设计 (loop):
    ✅ 只处理单步交互
    ✅ 简化的数据结构 (prompt, response, action, reward)
    ✅ 专注于最核心的数据更新
    ✅ 易于集成到现有engine中
    
    职责分离:
    📍 loop()函数: 处理agent与模型的交互
    📍 engine: 处理循环控制、轨迹管理、性能统计
    📍 环境: 处理环境交互和奖励计算
    """
    
    print(comparison)


def usage_in_engine_example():
    """展示在engine中的使用方式"""
    print("\n🔧 在Engine中的使用示例:")
    print("=" * 40)
    
    engine_usage = '''
    # 在Agent Execution Engine中的使用方式
    
    class MultiAgentExecutionEngine(AgentExecutionEngine):
        def __init__(self, agent_graph, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.agent_graph = agent_graph
        
        async def run_multi_agent_trajectory(self, env, max_steps=5):
            """运行多agent轨迹"""
            obs, _ = env.reset()
            
            trajectory_data = {
                "steps": [],
                "rewards": [],
                "total_reward": 0.0
            }
            
            for step_idx in range(max_steps):
                # 调用简化的loop函数
                step_data = await self.agent_graph.loop(obs, step_idx)
                
                # 处理环境交互
                step_rewards = []
                for agent_name, agent_data in step_data.items():
                    obs, reward, done, info = env.step(
                        agent_data["action_type"], 
                        agent_data["action"]
                    )
                    step_rewards.append(reward)
                
                # 更新奖励
                env_results = {"default_reward": sum(step_rewards)}
                step_data = await self.agent_graph.update_rewards(
                    step_data, env_results
                )
                
                # 存储轨迹数据
                trajectory_data["steps"].append(step_data)
                trajectory_data["rewards"].extend(step_rewards)
                trajectory_data["total_reward"] += sum(step_rewards)
                
                if done:
                    break
            
            return trajectory_data
    '''
    
    print(engine_usage)


if __name__ == "__main__":
    print("🎯 简化Multi-Agent Loop设计演示")
    print("📘 新设计专注于最核心的功能，便于与engine集成")
    print()
    
    # 展示数据结构
    show_data_structure()
    
    # 对比设计
    compare_old_vs_new()
    
    # Engine使用示例
    usage_in_engine_example()
    
    # 运行演示
    print("\n🚀 运行实际演示...")
    asyncio.run(demonstrate_simplified_loop()) 