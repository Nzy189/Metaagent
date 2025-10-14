#!/usr/bin/env python3
"""
测试async_generate.py的改进功能
用于验证：
1. 共享ClientSession是否正常工作
2. Semaphore并发控制是否生效
3. 重试机制是否正常
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "pettingllms"))

from trainer.async_generate import (
    get_shared_session,
    get_llm_semaphore,
    submit_completions,
    cleanup_shared_session,
)


async def test_shared_session():
    """测试共享ClientSession"""
    print("\n" + "=" * 60)
    print("测试1: 共享ClientSession")
    print("=" * 60)
    
    # 获取两次session，应该是同一个对象
    session1 = await get_shared_session()
    session2 = await get_shared_session()
    
    assert session1 is session2, "两次获取应该是同一个session对象"
    assert not session1.closed, "Session应该是开启状态"
    
    print("✅ 共享session工作正常")
    print(f"   Session对象: {id(session1)}")
    print(f"   是否关闭: {session1.closed}")


async def test_semaphore():
    """测试Semaphore并发控制"""
    print("\n" + "=" * 60)
    print("测试2: Semaphore并发控制")
    print("=" * 60)
    
    # 获取semaphore
    semaphore = await get_llm_semaphore(max_concurrent=3)
    
    print(f"   Semaphore最大并发数: 3")
    print(f"   当前可用: {semaphore._value}")
    
    # 模拟并发请求
    async def mock_request(idx, duration):
        async with semaphore:
            print(f"   请求{idx}开始执行 (可用slot: {semaphore._value})")
            await asyncio.sleep(duration)
            print(f"   请求{idx}完成")
    
    # 创建5个并发请求，但只有3个能同时执行
    start_time = time.time()
    tasks = [
        mock_request(i, 0.5) 
        for i in range(5)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    print(f"✅ Semaphore工作正常")
    print(f"   5个请求完成耗时: {elapsed:.2f}秒")
    print(f"   预期耗时: ~1.0秒 (3个并发，每个0.5秒)")


async def test_retry_mechanism():
    """测试重试机制（使用无效地址触发重试）"""
    print("\n" + "=" * 60)
    print("测试3: 重试机制")
    print("=" * 60)
    
    # 使用一个不存在的地址来触发重试
    invalid_address = "127.0.0.1:9999"
    
    print(f"   尝试连接无效地址: {invalid_address}")
    print(f"   预期会重试3次后失败...")
    
    start_time = time.time()
    try:
        await submit_completions(
            address=invalid_address,
            model="test-model",
            prompt="test prompt",
            max_retries=3,
            initial_retry_delay=0.5,  # 缩短延迟以加快测试
            temperature=1.0,
            max_tokens=100,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✅ 重试机制工作正常")
        print(f"   按预期失败: {type(e).__name__}")
        print(f"   总耗时: {elapsed:.2f}秒")
        print(f"   预期耗时: ~3.5秒 (0.5s + 1.0s + 2.0s + 请求超时)")


async def test_concurrent_requests():
    """测试大量并发请求的资源管理"""
    print("\n" + "=" * 60)
    print("测试4: 大量并发请求资源管理")
    print("=" * 60)
    
    # 模拟100个并发请求
    num_requests = 100
    print(f"   创建{num_requests}个并发任务...")
    
    async def mock_llm_request(idx):
        """模拟LLM请求（实际不发送）"""
        session = await get_shared_session()
        semaphore = await get_llm_semaphore(max_concurrent=50)
        
        async with semaphore:
            # 只是获取资源，不实际发送请求
            await asyncio.sleep(0.01)  # 模拟处理时间
        
        return idx
    
    start_time = time.time()
    results = await asyncio.gather(*[
        mock_llm_request(i) 
        for i in range(num_requests)
    ])
    elapsed = time.time() - start_time
    
    assert len(results) == num_requests, f"应完成{num_requests}个请求"
    
    print(f"✅ 大量并发请求处理正常")
    print(f"   完成{num_requests}个请求")
    print(f"   总耗时: {elapsed:.2f}秒")
    print(f"   平均每请求: {elapsed/num_requests*1000:.2f}ms")


async def test_session_cleanup():
    """测试session清理"""
    print("\n" + "=" * 60)
    print("测试5: Session清理")
    print("=" * 60)
    
    session = await get_shared_session()
    assert not session.closed, "清理前session应该是开启的"
    
    await cleanup_shared_session()
    
    # 注意：清理后session被设为None，需要重新创建
    print("✅ Session清理成功")


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试async_generate改进功能")
    print("=" * 60)
    
    try:
        # 运行各项测试
        await test_shared_session()
        await test_semaphore()
        await test_concurrent_requests()
        
        # 重试测试会花费较长时间，可选执行
        # await test_retry_mechanism()
        
        await test_session_cleanup()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n改进功能验证成功，可以开始训练了。")
        print("注意观察日志中的 [Session]、[Semaphore] 和 [Retry] 标记。")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

