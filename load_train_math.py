# pip install datasets pandas pyarrow
from datasets import load_dataset
import pandas as pd

PY_LANG_IDS = {1, 3}  # 1=PYTHON(2), 3=PYTHON3 per dataset schema

def pick_first_k_python(solutions_dict, k=10):
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

def has_at_least_k_python(solutions_dict, k=10):
    count = 0
    langs = solutions_dict.get("language") or []
    for lang in langs:
        if lang in PY_LANG_IDS:
            count += 1
            if count >= k:
                return True
    return False

def main():
    # 1) 加载 HF 数据集 split=train
    ds = load_dataset("deepmind/code_contests", split="train")

    # 2) 可选：把原始 split 直接转存为 parquet（原始结构较大且含嵌套，体积可能很大）
    #    如果不需要原始 parquet，可注释掉这两行。
    #raw_pd = ds.to_pandas()
    #raw_pd.to_parquet("code_contests_train_raw.parquet", index=False)

    # 3) 过滤：correct/incorrect 都至少有 10 个 Python 答案，且 public_tests.input 非空
    filtered_rows = []
    for ex in ds:
        public_tests = ex.get("private_tests") or {}
        test_in = public_tests.get("input") or []
        test_out = public_tests.get("output") or []


        solutions = ex.get("solutions") or {}
        incorrect = ex.get("incorrect_solutions") or {}

        ok = (
            has_at_least_k_python(solutions, k=10)
            and has_at_least_k_python(incorrect, k=10)
            and isinstance(test_in, list) and len(test_in) > 0
        )
        if not ok:
            continue

        # 4) 组装目标字段
        problem_text = ex.get("description", "")  # 题面
        correct_py_10 = pick_first_k_python(solutions, k=10)
        incorrect_py_10 = pick_first_k_python(incorrect, k=10)
        if len(test_in)>0:
            row = {
            
                "question": problem_text,
                # 分别从 correct/incorrect 中各取前 10 个 Python 代码
                "solution": correct_py_10[0],
                "test_input": test_in,      # 原 public_tests.input 列表
                "test_output": test_out,    # 原 public_tests.output 列表
                # 为了统计方便，保留难度与名称
                "difficulty": ex.get("difficulty"),
                "name": ex.get("name"),
            }
            filtered_rows.append(row)

    df = pd.DataFrame(filtered_rows)

    # 4) 存储为 parquet（仅包含你指定/需要的字段）
    #    其他字段一律不保留
    #    注意：Parquet 支持列表列；读取时用 pandas/pyarrow 可还原
    wanted_cols = ["question", "solution", "test_input", "test_output", "difficulty", "name"]
    df = df[wanted_cols]
    df.to_parquet("datasets/train/train.parquet", index=False)

    # 5) 统计并打印：难度分布；test_input 最长/最短个数
    # 难度分布
    print("=== Difficulty value counts (filtered) ===")
    print(df["difficulty"].value_counts().sort_index())

    # test_input 的条目数（每题 public_tests.input 的样例数量）
    test_counts = df["test_input"].map(lambda x: len(x) if isinstance(x, list) else 0)
    max_cnt = int(test_counts.max()) if len(test_counts) else 0
    min_cnt = int(test_counts.min()) if len(test_counts) else 0

    # 找到对应的题目名称（可能有多题并列）
    longest_names = df.loc[test_counts == max_cnt, "name"].tolist()
    shortest_names = df.loc[test_counts == min_cnt, "name"].tolist()

    print("\n=== Public test_input stats (filtered) ===")
    print(f"Max #inputs per problem: {max_cnt} | Problems: {longest_names[:5]}{'...' if len(longest_names)>5 else ''}")
    print(f"Min #inputs per problem: {min_cnt} | Problems: {shortest_names[:5]}{'...' if len(shortest_names)>5 else ''}")

    print("\nSaved:")
    print(" - code_contests_train_raw.parquet (raw, optional, large)")
    print(" - code_contests_train_filtered.parquet (filtered schema)")

if __name__ == "__main__":
    main()
