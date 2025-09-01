# pip install datasets pandas pyarrow
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import os

PY_LANG_IDS = {1, 3}  # 1=PYTHON(2), 3=PYTHON3 per dataset schema

def pick_first_k_python(solutions_dict, k=1):
    """
    从 solutions_dict(含 'language': List[int], 'solution': List[str]) 中
    取出按出现顺序的前 k 个 Python 代码（语言在 {1,3}）。
    """
    langs = solutions_dict.get("language") or []
    codes = solutions_dict.get("solution") or []
    out = []
    for lang, code in zip(langs, codes):
        if lang in PY_LANG_IDS:
            out.append(code)
            if len(out) >= k:
                break
    return out

def has_at_least_k_python(solutions_dict, k=1):
    count = 0
    langs = solutions_dict.get("language") or []
    for lang in langs:
        if lang in PY_LANG_IDS:
            count += 1
            if count >= k:
                return True
    return False

def _process_split(split: str) -> pd.DataFrame:
    print(f"🔄 从 Hugging Face 加载 deepmind/code_contests split={split}...")
    ds = load_dataset("deepmind/code_contests", split=split)
    print(f"the length of ds: {len(ds)}")

    filtered_rows = []
    for ex in ds:
        # 注：原注释写 public_tests，这里历史实现使用 private_tests，保持一致
        if split == "test":
            tests = ex.get("public_tests") or {}
        else:

            tests = ex.get("private_tests") or {}
        test_in = tests.get("input") or []
        test_out = tests.get("output") or []

        solutions = ex.get("solutions") or {}
        incorrect = ex.get("incorrect_solutions") or {}

        ok = (
            has_at_least_k_python(solutions, k=1)
            and isinstance(test_in, list) and len(test_in) > 0
        )
        if not ok:
            continue

        problem_text = ex.get("description", "")
        correct_py_10 = pick_first_k_python(solutions, k=1)
        if len(test_in) > 0:
            row = {
                "question": problem_text,
                "solution": correct_py_10[0] if len(correct_py_10) > 0 else "",
                "test_input": test_in,
                "test_output": test_out,
                "difficulty": ex.get("difficulty"),
                "name": ex.get("name"),
            }
            filtered_rows.append(row)

    df = pd.DataFrame(filtered_rows)
    wanted_cols = ["question", "solution", "test_input", "test_output", "difficulty", "name"]
    if len(df) == 0:
        # 确保列存在
        for c in wanted_cols:
            if c not in df.columns:
                df[c] = []
    df = df[wanted_cols]
    print(f"✅ {split} 过滤后样本数: {len(df)}")
    return df


def main():
    # 目标目录：datasets/code/train/{train.parquet,test.parquet}
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "code" / "train"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    # 分别处理 train 与 test 两个 split
    for split in ["train", "test"]:
        df = _process_split(split)
        out_path = out_dir / f"{split}.parquet"
        df.to_parquet(str(out_path), index=False)
        print(f"💾 已保存 {split}.parquet -> {out_path}")

    # 简单统计（基于 test）
    try:
        from pandas import read_parquet
        df = read_parquet(out_dir / "test.parquet")
        print("=== Difficulty value counts (test, filtered) ===")
        print(df["difficulty"].value_counts().sort_index())

        test_counts = df["test_input"].map(lambda x: len(x) if isinstance(x, list) else 0)
        max_cnt = int(test_counts.max()) if len(test_counts) else 0
        min_cnt = int(test_counts.min()) if len(test_counts) else 0
        longest_names = df.loc[test_counts == max_cnt, "name"].tolist()
        shortest_names = df.loc[test_counts == min_cnt, "name"].tolist()

        print("\n=== test split: test_input stats (filtered) ===")
        print(f"Max #inputs per problem: {max_cnt} | Problems: {longest_names[:5]}{'...' if len(longest_names)>5 else ''}")
        print(f"Min #inputs per problem: {min_cnt} | Problems: {shortest_names[:5]}{'...' if len(shortest_names)>5 else ''}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
