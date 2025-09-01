#!/usr/bin/env python3
"""
演示多个独立Ray实例的脚本

这个脚本展示了如何同时运行多个独立的Ray实例，每个实例都有自己的命名空间。
"""

import os
import sys
import time
import subprocess
import multiprocessing

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pettingllms.trainer.utils import list_all_ray_processes

def simulate_training_process(process_id):
    """模拟一个训练进程"""
    import ray
    import uuid
    
    # 创建独立的命名空间
    namespace = f"demo_process_{process_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        print(f"进程 {process_id}: 初始化Ray，命名空间: {namespace}")
        ray.init(
            namespace=namespace,
            ignore_reinit_error=True,
            include_dashboard=False,
            logging_level="ERROR"
        )
        
        print(f"进程 {process_id}: Ray初始化成功")
        
        # 模拟一些工作
        @ray.remote
        def dummy_task(x):
            time.sleep(1)
            return x * 2
        
        # 运行一些任务
        results = []
        for i in range(3):
            result = dummy_task.remote(i)
            results.append(result)
        
        final_results = ray.get(results)
        print(f"进程 {process_id}: 任务完成，结果: {final_results}")
        
        # 保持运行一段时间
        time.sleep(10)
        
        print(f"进程 {process_id}: 关闭Ray")
        ray.shutdown()
        
    except Exception as e:
        print(f"进程 {process_id}: 发生错误: {e}")
        try:
            ray.shutdown()
        except:
            pass

def main():
    """主函数"""
    print("=== 多Ray实例演示 ===")
    print("这个演示将启动3个独立的进程，每个都有自己的Ray实例")
    
    # 首先显示当前的Ray进程
    print("\n启动前的Ray进程:")
    list_all_ray_processes()
    
    # 启动多个进程
    processes = []
    num_processes = 3
    
    print(f"\n启动 {num_processes} 个独立的训练进程...")
    
    for i in range(num_processes):
        p = multiprocessing.Process(target=simulate_training_process, args=(i+1,))
        p.start()
        processes.append(p)
        time.sleep(2)  # 错开启动时间
    
    # 等待一段时间，然后显示Ray进程
    time.sleep(5)
    print("\n运行中的Ray进程:")
    list_all_ray_processes()
    
    print(f"\n等待所有进程完成...")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("\n所有进程完成后的Ray进程:")
    list_all_ray_processes()
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    main()
