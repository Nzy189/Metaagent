"""
Utility functions for mathematical problem solving and evaluation.

This module contains utilities for loading math datasets, evaluating solutions,
and computing metrics for mathematical problem solving tasks.
"""

import os
import json
import random
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import re





try:
    from datasets import load_dataset as hf_load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️ The 'datasets' library is unavailable; some features are limited")
    DATASETS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ The 'pandas' library is unavailable; some features are limited")
    PANDAS_AVAILABLE = False


@dataclass
class MathEvaluationResult:
    """
    Dataclass for math evaluation results
    """
    problem: str
    generated_solution: str
    extracted_answer: str
    ground_truth_answer: str
    is_correct: bool
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict format for compatibility"""
        return {
            "problem": self.problem,
            "generated_solution": self.generated_solution,
            "extracted_answer": self.extracted_answer,
            "ground_truth_answer": self.ground_truth_answer,
            "is_correct": self.is_correct,
            "error_type": self.error_type
        }


def load_math_problem_batch(
    env_indices: List[int],
    batch_size: int = 10,
    dataset_name: str = "train",
    split: str = "train",
    mode: str = "train",
    config: dict = None
) -> List[Dict[str, Any]]:
    """
    Load a batch of mathematical problems.
    
    Args:
        batch_size: Batch size
        dataset_name: Dataset name (统一使用 "train")
        split: Dataset split (保留兼容性，但实际不使用)
        mode: "train" or "validate"
        config: Configuration dict
        
    Returns:
        A list of dicts with keys question/solution
    """
    if not DATASETS_AVAILABLE:
        print("❌ datasets library unavailable")
        return []
    
    # 期望的目录结构：datasets/math/train/{train.parquet,test.parquet}
    current_dir = Path(__file__).parent.parent.parent.parent  # 回到 pettingllms 根目录
    local_datasets_dir = current_dir / "datasets" / "math" / dataset_name.lower().replace("/", "_")
    split_name = "train" if mode == "train" else "test"
    parquet_file = local_datasets_dir / f"{split_name}.parquet"
    print(f"📄 目标文件: {parquet_file}")
    
    if mode == "train":
        if not parquet_file.exists():
            raise FileNotFoundError(f"❌ Train mode requires local dataset at {parquet_file}, but file not found!")
        
        print(f"📁 从本地加载数学训练集: {local_datasets_dir}")
        try:
            # parquet 单文件默认 split 名称为 "train"
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
            print(f"✅ 数学训练集加载成功，共 {len(ds)} 条")
        except Exception as e:
            raise Exception(f"❌ Failed to load local dataset: {e}")
        
        # 随机选择batch_size个样本
        if len(ds) < batch_size:
            raise Exception(f"❌ Local dataset only has {len(ds)} samples, but batch_size is {batch_size}")
        
        batch_results = []
        for i, idx in enumerate(env_indices):
            example = ds[idx]
            problem_dict = _format_math_problem(example, idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
                print(f"✅ Loaded math train problem {i+1}/{batch_size} (index={idx})")
        
        print(f"✅ 成功返回 {len(batch_results)} 条数学训练样本")
        return batch_results
    
    # validation mode: 加载测试集
    else:
        if not parquet_file.exists():
            raise FileNotFoundError(
                f"❌ 验证模式需要本地数学测试集 {parquet_file}，未找到！请先运行 scripts/dataprocess/load_train_math.py 生成数据。"
            )
        print(f"📁 从本地加载数学测试集: {local_datasets_dir}")
        try:
            # parquet 单文件默认 split 名称为 "train"
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
            print(f"✅ 数学测试集加载成功，共 {len(ds)} 条")
        except Exception as e:
            raise Exception(f"❌ Failed to load local dataset: {e}")
        
        # 加载所有验证数据
        batch_results = []
        for i, example in enumerate(ds):
            problem_dict = _format_math_problem(example, i, mode="validate")
            if problem_dict:
                batch_results.append(problem_dict)
                if i % 100 == 0:  # 每100个打印一次进度
                    print(f"🔄 Loaded math validation problem {i+1}/{len(ds)}")
        
        print(f"✅ 成功返回 {len(batch_results)} 条数学验证样本")
        return batch_results



def _format_math_problem(example: Dict, index: int, mode: str = "train") -> Optional[Dict]:
    """
    Format a math problem example into a standardized dictionary.
    
    Args:
        example: Raw example from dataset (期望格式: question/solution)
        index: Index of the example
        mode: "train" or "validate"
        
    Returns:
        Formatted problem dictionary or None if invalid
    """
    try:
        # 从parquet文件中读取的标准格式
        question = example.get("question", "")
        solution = example.get("solution", "")
        
        # 根据mode处理solution字段
        if mode == "train":
            # 训练模式：保留solution作为答案
            answer = solution
        else:  # validation mode
            # 验证模式：solution设为空（因为是测试）
            answer = ""
        
        # 验证必要字段
        if not question:
            print(f"⚠️ Skipping example {index}: missing question field")
            return None
        
        return {
            "question": question,
            "solution": answer  # 统一使用solution字段
        }
        
    except Exception as e:
        print(f"⚠️ Error formatting example {index}: {e}")
        return None



async def evaluate_math_solution(
    solution: str,
    ground_truth_answer: str
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate a mathematical solution against the ground truth answer.
    
    Args:
        solution: Generated solution string
        ground_truth_answer: Ground truth answer
        
    Returns:
        (is_correct, extracted_answer)
    """
    is_correct = solution == ground_truth_answer
    return is_correct, solution

def extract_answer(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

# Test function
def test_load_math_problems(batch_size: int = 5):
    """Test loading math problems"""
    results = load_math_problem_batch(batch_size=batch_size)
    for i, result in enumerate(results):
        print(f"\n--- Problem {i+1} ---")
        print(f"Problem: {result['problem'][:200]}...")
        print(f"Answer: {result['answer']}")
        print(f"Difficulty: {result.get('difficulty', 'N/A')}")
        print(f"Type: {result.get('type', 'N/A')}")


if __name__ == "__main__":
    print("Testing math problem loading...")
    test_load_math_problems(3)
