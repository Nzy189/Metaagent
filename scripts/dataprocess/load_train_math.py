# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess math datasets (GSM8k, MATH, etc.) to parquet format for training
"""

import argparse
import os
import re
import datasets
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """提取数学题答案中的最终数值结果"""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is not None:
        final_solution = solution.group(0)
        final_solution = final_solution.split("#### ")[1].replace(",", "")
        return final_solution
    # 如果没有找到 #### 格式，尝试其他格式或返回原字符串
    return solution_str.strip()

def main():
    """主函数：下载并处理数学数据集，保存到统一目录结构"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="openai/gsm8k", 
                       help="数据源，如 openai/gsm8k, hendrycks/competition_math 等")
    parser.add_argument("--subset", default="main", 
                       help="数据集子集，GSM8k使用'main'，MATH使用'all'")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 统一目录结构：datasets/math/train/{train.parquet,test.parquet}
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "math" / "train"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")
    
    data_source = args.data_source
    
    # 加载数据集
    print(f"🔄 从 Hugging Face 加载 {data_source} (subset={args.subset})...")
    dataset = datasets.load_dataset(data_source, args.subset)
    
    train_dataset = dataset["train"]
    test_dataset = dataset.get("test", None)
    
    # 数据处理函数
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.get("question", example.get("problem", ""))
            
            # 构建完整的问题
            question = question_raw.strip()
            answer_raw = example.get("answer", example.get("solution", ""))
            solution = extract_solution(answer_raw)
            
            data = {
                "question": question,
                "solution": solution
            }
            return data
        
        return process_fn
    
    # 处理训练集
    print(f"🔄 处理训练集...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_path = out_dir / "train.parquet"
    train_dataset.to_parquet(str(train_path))
    print(f"💾 训练集已保存到: {train_path} ({len(train_dataset)} 条)")
    
    # 处理测试集（如果存在）
    if test_dataset is not None:
        print(f"🔄 处理测试集...")
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        test_path = out_dir / "test.parquet"
        test_dataset.to_parquet(str(test_path))
        print(f"💾 测试集已保存到: {test_path} ({len(test_dataset)} 条)")
    else:
        print("⚠️ 数据集没有测试集，仅保存训练集")
    
    # 如果指定了HDFS目录，复制到HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=str(out_dir), dst=args.hdfs_dir)
        print(f"📤 数据已复制到HDFS: {args.hdfs_dir}")
    
    # 打印统计信息
    print("\n=== 数据集处理完成 ===")
    print(f"数据源: {data_source}")
    print(f"本地保存路径: {out_dir}")
    if args.hdfs_dir:
        print(f"HDFS路径: {args.hdfs_dir}")
    
    # 显示第一个样本作为示例
    print("\n=== 样本示例 ===")
    first_sample = train_dataset[0]
    print(f"问题: {first_sample['question'][:100]}...")
    print(f"答案: {first_sample['solution']}")


if __name__ == "__main__":
    main()
