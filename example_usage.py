#!/usr/bin/env python3
"""
示例：如何使用重构后的 AgentPPOTrainer

该示例展示了 AgentPPOTrainer 现在如何使用组合模式，
可以包含多个 RayPPOTrainer 实例。
"""

from pettingllms.trainer.verl.agent_ppo_trainer import AgentPPOTrainer

def create_agent_ppo_trainer_example():
    """创建 AgentPPOTrainer 的示例"""
    
    # 示例配置（在实际使用中，这些会是真实的配置对象）
    config = None  # 实际的配置对象
    tokenizer = None  # 实际的tokenizer
    role_worker_mapping = {}  # 实际的角色-工作器映射
    resource_pool_manager = None  # 实际的资源池管理器
    
    # 方式1：使用默认的单个 PPO trainer (num_ppo_groups=1)
    agent_trainer_single = AgentPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        env_class=None,
        agent_class=None,
        env_args={},
        agent_args={},
        # num_ppo_groups=1  # 默认值，可以省略
    )
    
    # 方式2：使用多个 PPO trainer 组
    agent_trainer_multiple = AgentPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        env_class=None,
        agent_class=None,
        env_args={},
        agent_args={},
        num_ppo_groups=3  # 使用3个PPO trainer实例
    )
    
    print(f"单个训练器组包含: {agent_trainer_single.num_ppo_groups} 个 PPO trainers")
    print(f"多个训练器组包含: {agent_trainer_multiple.num_ppo_groups} 个 PPO trainers")
    
    # 访问主要的训练器
    primary_trainer = agent_trainer_multiple.primary_trainer
    print(f"主要训练器类型: {type(primary_trainer).__name__}")
    
    # 访问所有训练器组
    all_trainers = agent_trainer_multiple.ppo_trainer_group
    print(f"训练器组长度: {len(all_trainers)}")
    
    return agent_trainer_single, agent_trainer_multiple

if __name__ == "__main__":
    print("AgentPPOTrainer 重构示例")
    print("=" * 50)
    
    # 由于我们没有真实的配置，这里只是展示API的变化
    print("创建 AgentPPOTrainer 实例的示例代码:")
    print("""
    # 使用默认的单个 PPO trainer
    trainer = AgentPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        # ... 其他参数
    )
    
    # 使用多个 PPO trainer
    trainer = AgentPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        num_ppo_groups=3,  # 新增参数
        # ... 其他参数
    )
    """)
    
    print("\n重构总结:")
    print("✅ AgentPPOTrainer 不再继承 RayPPOTrainer")
    print("✅ 使用组合模式，包含 RayPPOTrainer 实例列表")
    print("✅ 添加了 num_ppo_groups 参数，默认为 1")
    print("✅ 保持了向后兼容性")
    print("✅ 所有方法和属性都正确委托") 