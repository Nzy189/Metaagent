#!/usr/bin/env python3
"""
测试本地数据集加载功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pettingllms.multi_agent_env.code.code_utils import load_problem_batch

def test_local_dataset_loading():
    """测试从本地加载数据集"""
    print("🧪 测试本地数据集加载功能...")
    
    # 测试加载 CodeContests_train 数据集
    try:
        print("\n📊 尝试加载 CodeContests_train 数据集...")
        problems = load_problem_batch(
            dataset_name="CodeContests_train",
            batch_size=2,
            split="train"
        )
        
        if problems:
            print(f"✅ 成功加载了 {len(problems)} 个问题")
            for i, problem in enumerate(problems):
                print(f"\n问题 {i+1}:")
                print(f"  问题描述: {problem.get('question', 'N/A')[:100]}...")
                print(f"  示例输入数量: {len(problem.get('example_input', []))}")
                print(f"  测试用例数量: {len(problem.get('test_input', []))}")
        else:
            print("❌ 没有加载到任何问题")
            
    except Exception as e:
        print(f"❌ 加载数据集时出错: {e}")
        import traceback
        traceback.print_exc()

def test_dataset_structure():
    """检查数据集目录结构"""
    print("\n📁 检查数据集目录结构...")
    
    current_dir = Path(__file__).parent
    datasets_dir = current_dir / "datasets"
    
    if datasets_dir.exists():
        print(f"✅ 找到数据集目录: {datasets_dir}")
        
        for item in datasets_dir.iterdir():
            if item.is_dir():
                print(f"  📂 {item.name}/")
                for subitem in item.iterdir():
                    if subitem.is_file():
                        size_mb = subitem.stat().st_size / (1024 * 1024)
                        print(f"    📄 {subitem.name} ({size_mb:.2f} MB)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  📄 {item.name} ({size_mb:.2f} MB)")
    else:
        print(f"❌ 数据集目录不存在: {datasets_dir}")
        print("💡 请先运行 download_codecontests.py 下载数据集")

if __name__ == '__main__':
    print("🚀 开始测试本地数据集加载功能")
    print("=" * 50)
    
    test_dataset_structure()
    test_local_dataset_loading()
    
    print("\n" + "=" * 50)
    print("🏁 测试完成")


