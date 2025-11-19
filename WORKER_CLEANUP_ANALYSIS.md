# Worker 清理机制分析报告

## 检查日期
2025-11-18

## 问题分析

根据错误日志：
```
(WorkerDict pid=2087134) INFO 11-18 01:58:00 [gpu_worker.py:116] Sleep mode freed 118.30 GiB memory, 15.95 GiB memory is still in use.
(raylet) file_system_monitor.cc:116: /tmp/verl_spill_2059298 is over 95% full, available space: 5.79204 GB; capacity: 1758.73 GB
```

主要问题：
1. Worker 数量过多导致内存和磁盘空间压力大
2. `/tmp` 目录磁盘空间不足（95%已满）
3. GPU worker 即使在 sleep 模式下仍占用大量内存（15.95 GiB）

## 当前清理机制状态

### ✅ 已实现的清理机制

#### 1. **全局清理 Hook** (`pettingllms/utils/clean_up.py`)
- ✅ 使用 `atexit` 和信号处理器在程序退出时清理
- ✅ 清理 Ray actors
- ✅ 清理临时目录 (`/tmp/verl_ray_*`, `/tmp/verl_spill_*`)
- ✅ 调用 `ray.shutdown()`

#### 2. **训练主函数清理** (`pettingllms/trainer/train.py`)
- ✅ 在 `finally` 块中调用 `trainer.cleanup()`
- ✅ 杀死远程训练引擎
- ✅ 注册临时目录到清理系统

#### 3. **MultiAgentsPPOTrainer 清理** (`pettingllms/trainer/multi_agents_ppo_trainer.py`)
- ✅ 实现了 `cleanup()` 方法
- ✅ 清理 execution engine
- ✅ 清理 LLM servers
- ✅ 清理所有 PPO trainers
- ✅ 清理 resource pool managers

#### 4. **RayPPOTrainer 清理** (`pettingllms/verl/ray_trainer.py`)
- ✅ 实现了 `cleanup()` 方法
- ✅ 清理 async rollout manager (调用 `sleep()`)
- ✅ 清理所有 worker groups (actor, rollout, critic, ref, rm)
- ✅ 使用 `ray.kill()` 杀死 workers
- ✅ 清理 resource pool manager

#### 5. **ResourcePoolManager 清理** (`pettingllms/verl/ray_trainer.py`)
- ✅ 实现了 `cleanup()` 方法
- ✅ 杀死所有 resource pool 中的 actors

### ❌ 缺失的清理机制

#### 1. **AsyncLLMServerManager 缺少显式清理方法**
```python
class AsyncLLMServerManager:
    # ❌ 没有 cleanup() 方法
    # ❌ async_llm_servers 没有被清理
    # ❌ chat_scheduler_loop 没有被停止
    # ❌ chat_scheduler_thread 没有被 join
```

**问题**：
- `async_llm_servers` 是 Ray actors，需要显式 `ray.kill()`
- `chat_scheduler_loop` 事件循环在后台线程中持续运行
- 线程可能成为僵尸线程

#### 2. **Agent Loop Workers 没有清理**
从配置中看到 `agent.num_workers=7`，但没有找到这些 workers 的清理代码。

#### 3. **VLLM Engine 资源没有完全释放**
- GPU memory 在 sleep 后仍占用 15.95 GiB
- KV cache 可能没有完全释放

## 修复方案

### 1. 立即修复：减少并行数量（已完成）
在 `code_L1_model_1_7B.sh` 中：
```bash
# ✅ 已减少
training.train_batch_size=16              # 从 32 减少
training.train_sample_num=4               # 从 8 减少
rollout.max_num_seqs=128                  # 从 256 减少
rollout.max_num_batched_tokens=4096       # 从 6144 减少
rollout.gpu_memory_utilization=0.6        # 从 0.7 减少
rollout.agent.num_workers=3               # 从 7 减少 ✨ 关键
```

### 2. 高优先级：添加 AsyncLLMServerManager 清理

需要在 `pettingllms/verl/async_server.py` 添加：

```python
def cleanup(self):
    """Clean up all async LLM servers and scheduler resources"""
    print("Cleaning up AsyncLLMServerManager...")
    
    # 1. Stop chat scheduler loop
    if self.chat_scheduler_loop is not None:
        self.chat_scheduler_loop.call_soon_threadsafe(self.chat_scheduler_loop.stop)
    
    # 2. Wait for scheduler thread to complete
    if self.chat_scheduler_thread is not None and self.chat_scheduler_thread.is_alive():
        self.chat_scheduler_thread.join(timeout=5)
    
    # 3. Kill all async LLM server actors
    if self.async_llm_servers:
        for i, server in enumerate(self.async_llm_servers):
            if server is not None:
                try:
                    ray.kill(server)
                    print(f"  Killed async_llm_server {i}")
                except Exception as e:
                    print(f"  Warning: Failed to kill server {i}: {e}")
        self.async_llm_servers.clear()
    
    print("AsyncLLMServerManager cleanup completed")
```

### 3. 中优先级：改进 RayPPOTrainer 清理

在 `cleanup()` 方法中添加对 `async_rollout_manager` 的完整清理：

```python
# 当前只调用 sleep()，应该添加：
if hasattr(self, 'async_rollout_manager') and self.async_rollout_manager is not None:
    try:
        self.async_rollout_manager.sleep()
        # ✨ 添加完整清理
        if hasattr(self.async_rollout_manager, 'cleanup'):
            self.async_rollout_manager.cleanup()
    except Exception as e:
        print(f"Warning: Error cleaning up async_rollout_manager: {e}")
```

### 4. 建议：改进磁盘空间管理

在 `train.py` 中：
```python
# 定期清理旧的 spill 文件
import shutil
spill_dir = f"/tmp/verl_spill_{pid}"

# 添加磁盘空间检查
import shutil
disk_usage = shutil.disk_usage(spill_dir)
if disk_usage.free / disk_usage.total < 0.1:  # 少于 10% 可用
    print(f"WARNING: Low disk space on {spill_dir}")
    # 可以考虑提前清理或调整策略
```

## 测试建议

### 1. 验证清理是否生效
```bash
# 运行前检查
ps aux | grep -E "(python|ray|vllm)" | wc -l
du -sh /tmp/verl_*

# 运行训练
bash scripts/train/code/code_L1_model_1_7B.sh

# 训练结束或中断后检查
ps aux | grep -E "(python|ray|vllm)" | wc -l  # 应该显著减少
du -sh /tmp/verl_*  # 目录应该被清理

# 检查 Ray 状态
ray status  # 应该显示没有活动的 workers
```

### 2. 强制中断测试
```bash
# 发送 Ctrl+C 信号测试清理
# 或
kill -TERM <pid>

# 然后检查进程和临时文件是否被清理
```

## 总结

| 清理项 | 状态 | 优先级 |
|--------|------|--------|
| 全局清理 Hook | ✅ 已实现 | - |
| Trainer 清理 | ✅ 已实现 | - |
| Worker Groups 清理 | ✅ 已实现 | - |
| Resource Pool 清理 | ✅ 已实现 | - |
| AsyncLLMServerManager 清理 | ❌ 缺失 | 🔴 高 |
| Agent Loop Workers 清理 | ❓ 未知 | 🟡 中 |
| VLLM Engine 资源释放 | ⚠️ 不完整 | 🟡 中 |
| 磁盘空间监控 | ❌ 缺失 | 🟢 低 |

## 下一步行动

1. ✅ **已完成**: 减少 worker 并行数量（立即缓解问题）
2. **建议**: 添加 `AsyncLLMServerManager.cleanup()` 方法
3. **建议**: 改进 `async_rollout_manager` 的清理逻辑
4. **建议**: 添加磁盘空间监控和预警机制
5. **测试**: 验证所有清理机制是否正常工作

## 参考文件
- `/home/lah003/workspace/verl_efficient/pettingllms/utils/clean_up.py`
- `/home/lah003/workspace/verl_efficient/pettingllms/trainer/train.py`
- `/home/lah003/workspace/verl_efficient/pettingllms/trainer/multi_agents_ppo_trainer.py`
- `/home/lah003/workspace/verl_efficient/pettingllms/verl/ray_trainer.py`
- `/home/lah003/workspace/verl_efficient/pettingllms/verl/async_server.py`
