#!/usr/bin/env python3
"""
测试wandb连接的脚本
"""
import os
import sys
import time
import traceback
from pathlib import Path

def test_wandb_connection():
    """测试wandb连接"""
    print("🔍 开始测试wandb连接...")
    
    try:
        import wandb
        print("✅ wandb包导入成功")
        
        # 测试离线模式
        print("\n📍 测试离线模式...")
        try:
            wandb.init(
                project="test-connection-offline",
                name="offline-test",
                mode="offline",
                reinit=True
            )
            print("✅ wandb离线模式初始化成功")
            wandb.finish()
        except Exception as e:
            print(f"❌ wandb离线模式失败: {e}")
            
        # 测试在线模式（带超时）
        print("\n📍 测试在线模式（30秒超时）...")
        try:
            # 设置超时环境变量
            os.environ['WANDB_INIT_TIMEOUT'] = '30'
            
            start_time = time.time()
            wandb.init(
                project="test-connection-online",
                name="online-test",
                mode="online",
                reinit=True,
                settings=wandb.Settings(
                    init_timeout=30,
                    start_method="thread"
                )
            )
            elapsed = time.time() - start_time
            print(f"✅ wandb在线模式初始化成功 (用时: {elapsed:.2f}秒)")
            wandb.finish()
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ wandb在线模式失败 (用时: {elapsed:.2f}秒): {e}")
            print(f"错误类型: {type(e).__name__}")
            
        # 测试网络连接
        print("\n📍 测试网络连接...")
        try:
            import requests
            response = requests.get("https://api.wandb.ai/", timeout=10)
            print(f"✅ wandb API可达 (状态码: {response.status_code})")
        except Exception as e:
            print(f"❌ wandb API不可达: {e}")
            
        # 检查环境变量
        print("\n📍 检查wandb相关环境变量...")
        wandb_vars = [
            'WANDB_API_KEY', 'WANDB_PROJECT', 'WANDB_ENTITY', 
            'WANDB_MODE', 'WANDB_DIR', 'WANDB_CONFIG_DIR',
            'WANDB_INIT_TIMEOUT', 'WANDB_SILENT'
        ]
        
        for var in wandb_vars:
            value = os.environ.get(var)
            if value:
                if 'API_KEY' in var:
                    print(f"  {var}: {'*' * len(value[:4]) + value[4:8] if len(value) > 8 else '***'}")
                else:
                    print(f"  {var}: {value}")
            else:
                print(f"  {var}: 未设置")
                
    except ImportError:
        print("❌ wandb包未安装")
        return False
    except Exception as e:
        print(f"❌ wandb测试失败: {e}")
        traceback.print_exc()
        return False
    
    print("\n🎯 wandb连接测试完成")
    return True

def test_wandb_with_settings():
    """测试不同的wandb设置"""
    print("\n🔧 测试不同的wandb设置...")
    
    try:
        import wandb
        
        # 测试1: 使用自定义设置
        print("\n📍 测试自定义设置...")
        settings = wandb.Settings(
            init_timeout=15,
            start_method="thread",
            console="off",
            silent=True
        )
        
        wandb.init(
            project="test-settings",
            name="custom-settings-test",
            mode="offline",
            settings=settings,
            reinit=True
        )
        print("✅ 自定义设置测试成功")
        wandb.finish()
        
        # 测试2: 最小化设置
        print("\n📍 测试最小化设置...")
        wandb.init(
            project="test-minimal",
            mode="disabled",
            reinit=True
        )
        print("✅ 最小化设置测试成功")
        wandb.finish()
        
    except Exception as e:
        print(f"❌ 设置测试失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 wandb连接测试开始")
    print("=" * 50)
    
    success = test_wandb_connection()
    test_wandb_with_settings()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 测试完成，请查看上述结果")
    else:
        print("❌ 测试过程中发现问题，请查看上述错误信息")
