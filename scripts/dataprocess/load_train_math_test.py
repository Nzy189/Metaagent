# build_testset.py
import argparse
import os
import re
from pathlib import Path
import pandas as pd
from pathlib import Path
import datasets
from verl.utils.hdfs_io import copy, makedirs
from huggingface_hub import hf_hub_download

# ---------- 答案抽取：覆盖常见数学基准的标注习惯 ----------
def extract_solution(answer_text: str) -> str:
    if answer_text is None:
        return ""
    s = str(answer_text).strip()

    # GSM8K 常见形式：#### xxx
    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        return m.group(1).strip().replace(",", "")

    # MATH 常见：\boxed{...}
    m = re.search(r"\\boxed\{([^{}]+)\}", s)
    if m:
        return m.group(1).strip()

    # 末尾显式标签：Final answer: xxx / Answer: xxx
    m = re.search(r"(?:final answer|answer)\s*[:：]\s*([^\n\r]+)", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # AIME 之类短答案：取最后一行的短 token
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        if re.fullmatch(r"[A-Za-z0-9\-\+\*/\.^,_]+", tail) and len(tail) <= 20:
            return tail.replace(",", "")

    return s


# ---------- 尝试多个候选字段名 ----------
def get_first_key(d: dict, candidates):
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return ""


# ---------- 五个基准的数据集配置（公开可用路径） ----------
DATASETS = {
    # AIME 2024
    # https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
    "AIME24": {
        "path": "Maxwell-Jia/AIME_2024",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
    # AIME 2025
    # https://huggingface.co/datasets/yentinglin/aime_2025
    "AIME25": {
        "path": "yentinglin/aime_2025",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["question", "Problem", "problem", "prompt"],
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
    # MATH-500
    # https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    "MATH500": {
        "path": "HuggingFaceH4/MATH-500",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt"],
        "a_keys": ["solution", "answer", "final_answer"],
    },
    # Hendrycks MATH
    # https://huggingface.co/datasets/hendrycks/competition_math
    "MATH": {
        "path": "hendrycks/competition_math",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt"],
        "a_keys": ["solution", "answer", "final_answer", "Solution", "Answer"],
    },
    # GSM8K
    # https://huggingface.co/datasets/openai/gsm8k
    "GSM8K": {
        "path": "openai/gsm8k",
        "subset": "main",  # 官方使用 main 子集
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["question", "Problem", "problem", "prompt"],
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
    # OlympiadBench
    # https://huggingface.co/datasets/Hothan/OlympiadBench
    "OlympiadBench": {
        "path": "Hothan/OlympiadBench",
        "subset": "OE_TO_maths_en_COMP",
        "prefer_splits": ["train", "test", "validation", "dev"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
}


def choose_available_split(ds_dict, prefer_splits):
    for sp in prefer_splits:
        if sp in ds_dict:
            return sp
    if len(ds_dict) > 0:
        return list(ds_dict.keys())[0]
    raise ValueError("No splits available in the loaded dataset.")


def download_and_process_gsm8k_test(out_dir: str):
    # 确保输出目录存在
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "math" / "train"
    os.makedirs(out_dir, exist_ok=True)
    
    # 下载gsm8k数据集
    print("📥 下载gsm8k数据集...")
    ds_dict = datasets.load_dataset("openai/gsm8k", "main")
    
    # 选择测试集split
    split = choose_available_split(ds_dict, ["test"])
    dataset = ds_dict[split]
    print(f"✅ 使用split: {split}（作为测试集）")
    
    # 提取答案
    def map_fn(example):
        q_raw = get_first_key(example, ["question", "Problem", "problem", "prompt"])
        a_raw = get_first_key(example, ["answer", "final_answer", "solution", "Solution", "Answer"])
        return {
            "question": q_raw,
            "solution": extract_solution(a_raw),
        }
    
    print("🔧 统一映射为 {solution} ...")
    dataset_std = dataset.map(map_fn, remove_columns=[c for c in dataset.column_names if c not in []])
    
    # 限制为前500条
    #dataset_std = dataset_std.select(range(500))

    # 存储结果
    test_path = Path(out_dir) / "gsm8k_test_full.parquet"
    dataset_std.to_parquet(str(test_path))
    print(f"💾 测试集已保存到: {test_path}（{len(dataset_std)} 条）")



def download_and_process_olympiadbench_test(out_dir: str):
    # 确保输出目录存在
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "math" / "train"
    os.makedirs(out_dir, exist_ok=True)
    
    # 下载OlympiadBench数据集
    print("📥 下载OlympiadBench数据集...")
    ds_dict = datasets.load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")
    
    # 选择训练集split
    split = choose_available_split(ds_dict, ["train"])
    dataset = ds_dict[split]
    print(f"✅ 使用split: {split}（作为测试集）")
    
    # 提取答案
    def map_fn(example):
        q_raw = get_first_key(example, ["problem", "question", "prompt", "Problem"])
        a_raw = get_first_key(example, ["answer", "final_answer", "solution", "Solution", "Answer"])
        return {
            "question": q_raw,
            "solution": extract_solution(a_raw),
        }
    
    print("🔧 统一映射为 {solution} ...")
    dataset_std = dataset.map(map_fn, remove_columns=[c for c in dataset.column_names if c not in []])
    
    # 存储结果
    test_path = Path(out_dir) / "OlympiadBench_test.parquet"
    dataset_std.to_parquet(str(test_path))
    print(f"💾 测试集已保存到: {test_path}（{len(dataset_std)} 条）")
    
    # 打印一个样本
    if len(dataset_std) > 0:
        ex = dataset_std[0]
        print("\n=== 样本示例 ===")
        print(f"问题: {ex['question']}")
        print(f"答案: {ex['solution']}")


def download_and_process_polaris_train(out_dir: str):
    # 确保输出目录存在
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "math" / "train"
    os.makedirs(out_dir, exist_ok=True)
    
    # 下载POLARIS数据集
    print("📥 下载POLARIS数据集...")
    ds_dict = datasets.load_dataset("POLARIS-Project/Polaris-Dataset-53K")
    
    # 选择训练集split
    split = choose_available_split(ds_dict, ["train", "test", "validation", "dev"])
    dataset = ds_dict[split]
    print(f"✅ 使用split: {split}（作为训练集）")
    
    # 提取答案 - POLARIS数据集字段为problem, answer, difficulty
    def map_fn(example):
        q_raw = get_first_key(example, ["problem", "question", "prompt", "Problem"])
        a_raw = get_first_key(example, ["answer", "final_answer", "solution", "Solution", "Answer"])
        return {
            "question": q_raw,
            "solution": extract_solution(a_raw),
        }
    
    print("🔧 统一映射为 {question, solution} ...")
    dataset_std = dataset.map(map_fn, remove_columns=[c for c in dataset.column_names if c not in []])
    
    # 存储结果
    train_path = Path(out_dir) / "train_polaris.parquet"
    dataset_std.to_parquet(str(train_path))
    print(f"💾 训练集已保存到: {train_path}（{len(dataset_std)} 条）")
    
    # 打印一个样本
    if len(dataset_std) > 0:
        ex = dataset_std[0]
        print("\n=== 样本示例 ===")
        print(f"问题: {ex['question']}")
        print(f"答案: {ex['solution']}")

def main():
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "math" / "train"
    train_path = out_dir / "train.parquet"
    os.makedirs(out_dir, exist_ok=True)
    in_path = project_root / "openthought2_mathsubset" / "extracted_answers_capped_at_2k.parquet"
    alt = Path("openthought2_mathsubset/extracted_answers_capped_at_2k.parquet")
    df_raw = pd.read_parquet(alt)

    # 只保留 question 和 answer_boxed 两列，并重命名
    df = df_raw[["question", "answer_boxed"]].rename(
        columns={"answer_boxed": "solution"}
    )

    # 保存为 train.parquet
    df.to_parquet(train_path, index=False)
    

    for benchmark in ["AIME24", "AIME25", "MATH500", "GSM8K", "MATH", "OlympiadBench"]:
        conf = DATASETS[benchmark]
        path = conf["path"]
        subset = conf.get("subset", None)

        # 工程根与默认输出目录（与你原始脚本保持一致的层级）
        
        os.makedirs(out_dir, exist_ok=True)
        print(f"📁 输出目录: {out_dir}")

        # 加载数据集
        print(f"🔄 从 Hugging Face 加载 {path}" + (f" (subset={subset})" if subset else "") + " ...")
        ds_dict = datasets.load_dataset(path, subset) if subset else datasets.load_dataset(path)

        # 选择一个可用 split 作为“测试集”
        split = choose_available_split(ds_dict, conf["prefer_splits"])
        dataset = ds_dict[split]
        print(f"✅ 使用 split: {split}（作为测试集）")

        q_keys = conf["q_keys"]
        a_keys = conf["a_keys"]

        def map_fn(example):
            q_raw = get_first_key(example, q_keys)
            a_raw = get_first_key(example, a_keys)
            return {
                "question": str(q_raw).strip(),
                "solution": extract_solution(a_raw),
            }

        print("🔧 统一映射为 {question, solution} ...")
        dataset_std = dataset.map(map_fn, remove_columns=[c for c in dataset.column_names if c not in []])
        
        test_path = out_dir / f"{benchmark}.parquet"
        dataset_std.to_parquet(str(test_path))
        print(f"💾 测试集已保存到: {test_path}（{len(dataset_std)} 条）")


        # 打印一个样本
        if len(dataset_std) > 0:
            ex = dataset_std[0]
            print("\n=== 样本示例 ===")
            print(f"问题: {ex['question']}")
            print(f"答案: {ex['solution']}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[2] / "datasets" / "math" / "train"
    download_and_process_polaris_train(out_dir)