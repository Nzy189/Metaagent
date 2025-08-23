#!/usr/bin/env python3
"""
测试Ray状态日志功能的示例脚本

这个脚本演示了如何使用更新后的日志系统来检查Ray的启动和状态。
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pettingllms.utils.logger_config import get_multi_logger

def test_ray_status_logging():
    """测试Ray状态日志功能"""
    
    # 获取多日志器实例
    multi_logger = get_multi_logger()
    
    print("=== 测试Ray状态日志功能 ===")
    
    # 1. 在Ray初始化之前检查状态
    print("\n1. 检查Ray初始化前状态...")
    multi_logger.log_ray_status(context="测试脚本_初始化前")
    
    # 2. 尝试初始化Ray
    print("\n2. 尝试初始化Ray...")
    try:
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
            print("Ray初始化成功!")
        else:
            print("Ray已经初始化!")
    except Exception as e:
        print(f"Ray初始化失败: {e}")
    
    # 3. 在Ray初始化后检查状态
    print("\n3. 检查Ray初始化后状态...")
    multi_logger.log_ray_status(context="测试脚本_初始化后")
    
    # 4. 模拟一个rollout结束的情况
    print("\n4. 模拟rollout结束情况...")
    multi_logger.log_rollout_summary(
        rollout_idx=999,
        agent_rewards={"test_agent": 0.85, "test_agent_2": 0.92},
        termination_reason="测试完成",
        extra_data={"test_mode": True, "ray_status_test": True}
    )
    
    # 5. 测试错误情况下的Ray状态检查
    print("\n5. 测试错误情况...")
    multi_logger.log_ray_status(context="模拟错误情况")
    
    # 6. 清理Ray
    print("\n6. 清理Ray...")
    try:
        if 'ray' in locals() and ray.is_initialized():
            ray.shutdown()
            print("Ray已关闭!")
    except Exception as e:
        print(f"Ray关闭时出错: {e}")
    
    # 7. 在Ray关闭后检查状态
    print("\n7. 检查Ray关闭后状态...")
    multi_logger.log_ray_status(context="测试脚本_关闭后")
    
    print("\n=== 测试完成 ===")
    print("请查看logs目录下的日志文件以查看Ray状态信息。")

if __name__ == "__main__":
    test_ray_status_logging()
