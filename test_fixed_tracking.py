#!/usr/bin/env python3
"""
测试修复后的Tracking类
"""
import sys
import os
sys.path.insert(0, '/home/lah003/workspace/PettingLLMs')

def test_pettingllms_tracking():
    """测试pettingllms的Tracking类"""
    print("🔍 测试pettingllms的Tracking类...")
    
    try:
        from pettingllms.utils.tracking import Tracking
        
        # 测试wandb后端
        print("\n📍 测试wandb后端...")
        tracker = Tracking(
            project_name="test-fixed-tracking",
            experiment_name="test-run",
            default_backend=["console", "wandb"],
            config={"test": "value"}
        )
        
        print("✅ Tracking类初始化成功")
        
        # 测试日志记录
        test_data = {"step": 1, "loss": 0.5, "accuracy": 0.8}
        tracker.log(test_data, step=1)
        print("✅ 日志记录成功")
        
        # 清理
        if "wandb" in tracker.logger:
            tracker.logger["wandb"].finish()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_verl_tracking():
    """测试verl的Tracking类"""
    print("\n🔍 测试verl的Tracking类...")
    
    try:
        sys.path.insert(0, '/home/lah003/workspace/PettingLLMs/verl')
        from verl.utils.tracking import Tracking
        
        # 测试wandb后端
        print("\n📍 测试verl wandb后端...")
        tracker = Tracking(
            project_name="test-verl-tracking",
            experiment_name="test-verl-run",
            default_backend=["console", "wandb"],
            config={"test": "verl_value"}
        )
        
        print("✅ verl Tracking类初始化成功")
        
        # 测试日志记录
        test_data = {"step": 1, "loss": 0.3, "reward": 1.2}
        tracker.log(test_data, step=1)
        print("✅ verl 日志记录成功")
        
        # 清理
        if "wandb" in tracker.logger:
            tracker.logger["wandb"].finish()
        
        return True
        
    except Exception as e:
        print(f"❌ verl测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 测试修复后的Tracking类")
    print("=" * 50)
    
    success1 = test_pettingllms_tracking()
    success2 = test_verl_tracking()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ 所有测试通过！wandb超时问题已修复")
    else:
        print("❌ 部分测试失败，需要进一步调试")
