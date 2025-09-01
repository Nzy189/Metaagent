# scripts/dataprocess/load_train_code_livecodebench.py
# pip install -U datasets pandas pyarrow huggingface_hub

from datasets import load_dataset
import pandas as pd
from pathlib import Path
import os, json, ast
import numpy as np
from typing import List, Tuple, Union

# ---- mapping: 哪些文件属于哪个版本 ----
# 仓库: https://huggingface.co/datasets/livecodebench/code_generation_lite (含 test*.jsonl)
HF_PREFIX = "hf://datasets/livecodebench/code_generation_lite/"
V1_V5_FILES = [f"{HF_PREFIX}test{i}.jsonl" for i in ["", "2", "3", "4", "5"]]  # test.jsonl..test5.jsonl
V6_FILES   = [f"{HF_PREFIX}test6.jsonl"]  # v6
def _parse_tests(raw) -> tuple[list[str], list[str]]:
    """
    将 public/private_test_cases 解析为 (inputs, outputs)
    兼容几种情况：
    1) 字典: {"input":[...], "output":[...]} 或 {"tests":[{"input":..,"output":..}, ...]}
    2) 顶层列表: [{"input":..,"output":..}, ...]
    3) 双重编码: 字符串里还是 JSON（再解一层）
    4) 兜底: 解析失败时，返回 ([原始字符串], [])
    """
    def _loads_with_fallbacks(x):
        """
        尝试多轮反序列化：
        - 连续 json.loads，直到不是 str
        - 若 json 失败，使用 ast.literal_eval 兜底（处理单引号/类似 Python 字面量）
        """
        if not isinstance(x, str):
            return x

        s = x
        # 最多尝试 3 轮“解开洋葱”
        for _ in range(3):
            # 先试 JSON
            try:
                y = json.loads(s)
                if isinstance(y, str):
                    s = y
                    continue
                return y
            except Exception:
                # 再试 Python 字面量
                try:
                    y = ast.literal_eval(s)
                    if isinstance(y, str):
                        s = y
                        continue
                    return y
                except Exception:
                    break
        return x

    data = _loads_with_fallbacks(raw)

    def _normalize_input_value(v) -> str:
        # 将单个“测试用例的输入”统一为多行字符串：递归扁平化后逐值一行，并追加末尾换行
        if v is None:
            return ""
        flat: list[str] = []
        def _flatten(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                for xi in list(x):
                    _flatten(xi)
            else:
                flat.append(str(x))
        _flatten(v)
        if not flat and isinstance(v, str):
            # 不是可迭代结构，直接返回原字符串（若无换行则追加）
            s = v
            return s if s.endswith("\n") else s + "\n"
        s = "\n".join(flat)
        return s if s.endswith("\n") else s + "\n"

    def _normalize_output_value(v) -> str:
        # 将单个“期望输出”统一为字符串；若多行则按换行拼接，但不强制末尾换行
        if v is None:
            return ""
        if isinstance(v, (list, tuple, np.ndarray)):
            lines: list[str] = []
            for item in list(v):
                if isinstance(item, (list, tuple, np.ndarray)):
                    lines.append(" ".join(str(x) for x in list(item)))
                else:
                    lines.append(str(item))
            return "\n".join(lines)
        return str(v)

    # 情况 A: dict 结构
    if isinstance(data, dict):
        # A1: {"input":[...], "output":[...]}
        if "input" in data and "output" in data:
            ins = data.get("input") or []
            outs = data.get("output") or []
            ins = ins if isinstance(ins, list) else [ins]
            outs = outs if isinstance(outs, list) else [outs]
            norm_ins = [_normalize_input_value(x) for x in ins]
            norm_outs = [_normalize_output_value(x) for x in outs]
            return norm_ins, norm_outs

        # A2: {"tests":[{"input":...,"output":...}, ...]}
        if "tests" in data and isinstance(data["tests"], list):
            ins, outs = [], []
            for t in data["tests"]:
                if isinstance(t, dict):
                    # 兼容不同键名
                    input_val = t.get("input", t.get("inputs", ""))
                    output_val = t.get("output", t.get("expected_output", t.get("outputs", "")))
                    ins.append(_normalize_input_value(input_val))
                    outs.append(_normalize_output_value(output_val))
                else:
                    # 若元素不是 dict，就当成字符串兜底
                    ins.append(_normalize_input_value(t))
                    outs.append("")
            return ins, outs

    # 情况 B: 顶层就是 list -> 每个元素通常是 {"input":...,"output":...}
    if isinstance(data, list):
        ins, outs = [], []
        for item in data:
            if isinstance(item, dict):
                input_val = item.get("input", item.get("inputs", ""))
                output_val = item.get("output", item.get("expected_output", item.get("outputs", "")))
                ins.append(_normalize_input_value(input_val))
                outs.append(_normalize_output_value(output_val))
            else:
                ins.append(_normalize_input_value(item))
                outs.append("")
        return ins, outs

    # 情况 C: 兜底
    try:
        # 最后一次尝试：如果还是字符串，看看是否像 JSON/list 结构
        if isinstance(raw, str):
            s = raw.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                maybe = _loads_with_fallbacks(s)
                if isinstance(maybe, (list, dict)):
                    return _parse_tests(maybe)
        return ([str(raw)], [])
    except Exception:
        return [], []

def _hf_jsonl_to_df(data_files: List[str]) -> pd.DataFrame:
    # 利用通用 JSON Loader 直接读 Hub 文件
    ds = load_dataset("json", data_files=data_files, split="train")  
    rows = []
    for ex in ds:
        title   = ex.get("question_title") or ""
        content = ex.get("question_content") or ""
        question = (title + ("\n\n" if title and content else "") + content).strip()

        pub_raw = ex.get("public_test_cases") or ""
        pri_raw = ex.get("private_test_cases") or ""
        pub_in, pub_out = _parse_tests(pub_raw)
        pri_in, pri_out = _parse_tests(pri_raw)

        test_input  = pub_in if pub_in else pri_in
        test_output = pub_out if pub_out else pri_out

        rows.append({
            "question": question,
            "solution": "",  # LCB 无参考实现，这里占位以兼容下游
            "test_input": test_input,
            "test_output": test_output,
            "difficulty": ex.get("difficulty"),
            "name": ex.get("question_id") or "",
            # 这些字段你若需要可保留：
            # "platform": ex.get("platform") or "",
            # "contest_id": ex.get("contest_id") or "",
            # "contest_date": ex.get("contest_date") or "",
            # "starter_code": ex.get("starter_code") or "",
        })
    def _ensure_list_of_str(x):
        # 将 ndarray/tuple/None/标量 等统一转为 list[str]
        if x is None:
            return []
        if isinstance(x, (np.ndarray, tuple, set)):
            x = list(x)
        if isinstance(x, (str, bytes)):
            return [str(x)]
        if isinstance(x, list):
            return [str(i) for i in x]
        # 其他标量
        try:
            return [str(x)]
        except Exception:
            return []

    df = pd.DataFrame(rows)
    wanted = ["question", "solution", "test_input", "test_output", "difficulty", "name"]
    for c in wanted:
        if c not in df.columns:
            df[c] = []
    # 规范化两列：保证为 Python list[str]
    df["test_input"] = df["test_input"].apply(_ensure_list_of_str)
    df["test_output"] = df["test_output"].apply(_ensure_list_of_str)
    return df[wanted]

def main():
    # 输出到 datasets/code/livecodebench/{train.parquet,test.parquet}
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "code" / "livecodebench"
    test_dir = project_root / "datasets" / "code" /"train" 
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    print("🔄 加载 LiveCodeBench v1–v5 作为 train ...")
    train_df = _hf_jsonl_to_df(V1_V5_FILES)
    print("🔄 加载 LiveCodeBench v6 作为 test ...")
    test_df  = _hf_jsonl_to_df(V6_FILES)

    train_pq = out_dir / "train.parquet"
    test_pq  = test_dir / "test.parquet"
    train_df.to_parquet(train_pq, index=False)
    test_df.to_parquet(test_pq, index=False)
    print(f"💾 已保存 train.parquet -> {train_pq}")
    print(f"💾 已保存 test.parquet  -> {test_pq}")

    # 简单统计
    try:
        td = pd.read_parquet(test_pq)
        print("=== Difficulty value counts (test) ===")
        print(td["difficulty"].value_counts().sort_index())
        def _seq_len(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                try:
                    return len(x)
                except Exception:
                    return 0
            return 0
        tcnt = td["test_input"].map(_seq_len)
        print(f"\n=== test_input stats ===")
        print(f"Max #inputs: {int(tcnt.max()) if len(tcnt) else 0}")
        print(f"Min #inputs: {int(tcnt.min()) if len(tcnt) else 0}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
