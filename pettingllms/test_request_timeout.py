#!/usr/bin/env python3
"""
测试脚本：验证AsyncLLMServerManager的请求超时清理机制
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingllms.trainer.utils import AsyncLLMServerManager, RequestState
from omegaconf import DictConfig
import ray


class MockServer:
    """模拟vLLM服务器"""
    def __init__(self):
        self.requests = {}
        self.abort_called = []
    
    async def generate(self, prompt_ids, sampling_params, request_id):
        """模拟生成响应"""
        self.requests[request_id] = {
            'prompt_ids': prompt_ids,
            'sampling_params': sampling_params,
            'timestamp': datetime.now()
        }
        
        # 模拟生成一些token ids
        await asyncio.sleep(0.1)  # 模拟处理时间
        return [1, 2, 3, 4, 5]  # 模拟响应token
    
    async def abort_request(self, request_id):
        """模拟中止请求"""
        self.abort_called.append(request_id)
        if request_id in self.requests:
            del self.requests[request_id]
        print(f"✅ 模拟服务器已中止请求: {request_id}")


async def test_request_timeout_mechanism():
    """测试请求超时机制"""
    print("🚀 开始测试请求超时清理机制...")
    
    # 创建模拟配置和服务器
    config = DictConfig({
        "actor_rollout_ref": {
            "rollout": {
                "prompt_length": 512,
                "response_length": 256
            }
        }
    })
    
    # 创建模拟服务器
    mock_server = MockServer()
    
    # 创建AsyncLLMServerManager，设置3秒超时
    manager = AsyncLLMServerManager(
        config=config,
        server_handles=[mock_server],
        request_timeout_seconds=3.0  # 3秒超时用于快速测试
    )
    
    print(f"📊 管理器创建完成，超时设置: {manager.request_timeout_seconds}秒")
    
    try:
        # 模拟发送请求
        from verl.protocol import DataProto
        from transformers import AutoTokenizer
        import torch
        
        # 创建模拟tokenizer
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="测试响应")
        
        # 创建模拟DataProto
        mock_dpr = Mock()
        mock_dpr.batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        }
        
        print("📤 发送测试请求...")
        
        # 发送请求但不立即获取响应，模拟"添加请求后不获取响应"的情况
        response_dpr, response_str = await manager.generate(
            dpr_prompt=mock_dpr,
            tokenizer=tokenizer,
            application_id="test_app",
            rollout_idx=0,
            policy_name="test_policy"
        )
        
        print(f"✅ 请求已发送，当前活跃请求数: {manager.get_active_requests_count()}")
        
        # 等待超时时间 + 一些缓冲时间
        print("⏰ 等待超时清理机制触发...")
        await asyncio.sleep(8)  # 等待超过3秒超时 + 5秒清理周期
        
        # 检查清理结果
        active_count = manager.get_active_requests_count()
        pending_count = manager.get_pending_cleanup_count()
        
        print(f"📊 超时后状态:")
        print(f"   - 活跃请求数: {active_count}")
        print(f"   - 等待清理数: {pending_count}")
        print(f"   - 服务器中止调用次数: {len(mock_server.abort_called)}")
        
        if active_count == 0:
            print("✅ 超时清理机制工作正常！请求已被自动清理")
        else:
            print("❌ 超时清理机制可能存在问题")
            
        # 测试手动清理
        print("\n🧹 测试手动清理机制...")
        
        # 再发送一个请求
        response_dpr2, response_str2 = await manager.generate(
            dpr_prompt=mock_dpr,
            tokenizer=tokenizer,
            application_id="test_app_2",
            rollout_idx=1,
            policy_name="test_policy"
        )
        
        print(f"📤 发送第二个请求，活跃请求数: {manager.get_active_requests_count()}")
        
        # 手动清理这个请求
        request_ids = list(manager.active_requests.keys())
        if request_ids:
            manager.manually_cleanup_request(request_ids[0])
            print(f"🧹 手动清理请求: {request_ids[0]}")
            print(f"📊 清理后活跃请求数: {manager.get_active_requests_count()}")
        
    finally:
        # 停止清理任务
        manager.stop_cleanup_task()
        print("🛑 已停止清理任务")


async def test_multiple_requests_timeout():
    """测试多个请求的超时处理"""
    print("\n🔄 测试多个请求的超时处理...")
    
    config = DictConfig({
        "actor_rollout_ref": {
            "rollout": {
                "prompt_length": 512,
                "response_length": 256
            }
        }
    })
    
    mock_server = MockServer()
    manager = AsyncLLMServerManager(
        config=config,
        server_handles=[mock_server],
        request_timeout_seconds=2.0  # 2秒超时
    )
    
    try:
        # 创建模拟数据
        from transformers import AutoTokenizer
        import torch
        
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="测试响应")
        
        mock_dpr = Mock()
        mock_dpr.batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]])
        }
        
        # 快速发送多个请求
        print("📤 发送5个测试请求...")
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                manager.generate(
                    dpr_prompt=mock_dpr,
                    tokenizer=tokenizer,
                    application_id=f"multi_test_app_{i}",
                    rollout_idx=i,
                    policy_name="test_policy"
                )
            )
            tasks.append(task)
        
        # 等待所有请求完成
        await asyncio.gather(*tasks)
        
        print(f"✅ 5个请求已发送，活跃请求数: {manager.get_active_requests_count()}")
        
        # 等待超时清理
        print("⏰ 等待批量超时清理...")
        await asyncio.sleep(7)  # 等待超过2秒超时 + 5秒清理周期
        
        print(f"📊 批量清理后活跃请求数: {manager.get_active_requests_count()}")
        print(f"📊 服务器中止调用次数: {len(mock_server.abort_called)}")
        
        if manager.get_active_requests_count() == 0:
            print("✅ 批量超时清理机制工作正常！")
        else:
            print("❌ 批量超时清理可能存在问题")
            
    finally:
        manager.stop_cleanup_task()


if __name__ == "__main__":
    print("🧪 AsyncLLMServerManager 请求超时机制测试")
    print("=" * 50)
    
    # 由于我们没有真实的Ray环境，这里只能做模拟测试
    # 实际使用时需要真实的Ray和vLLM环境
    
    try:
        # 测试单个请求超时
        asyncio.run(test_request_timeout_mechanism())
        
        # 测试多个请求超时
        asyncio.run(test_multiple_requests_timeout())
        
        print("\n✅ 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
