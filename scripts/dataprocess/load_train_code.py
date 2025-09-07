# pip install datasets pandas pyarrow
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import os
import re, ast, json, zlib, pickle, base64
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# ============================================================
# 公共：把任意“参数/字面量说明”规约为 stdin 风格的多行文本
#（参考你提供的 transform_* / replace_* 思路）
# ============================================================

def _find_matching_bracket(s: str, start: int) -> Optional[int]:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '[':
            depth += 1
        elif s[i] == ']':
            depth -= 1
            if depth == 0:
                return i
    return None

def transform_tokens(s: str) -> str:
    """
    把输入串 s 规约为 stdin 多行：
      • 2D 数组 [[...], ...] → 每行 “x y …”
      • 1D 数组 [a,b,c]      → “a b c”
      • 引号字符串           → 去引号
      • 其它 token           → 原样
    输出以 '\n' 结尾。
    """
    events = []
    masked = s

    # 先抓第一段 [[...]]，避免内部再被拆
    start2 = s.find('[[')
    if start2 != -1:
        end2 = _find_matching_bracket(s, start2)
        if end2 is not None:
            arr_lit = s[start2:end2+1]
            try:
                arr2d = ast.literal_eval(arr_lit)
            except Exception:
                arr2d = []
            events.append((start2, 'array2d', arr2d))
            masked = masked[:start2] + ' '*(end2+1 - start2) + masked[end2+1:]

    token_re = re.compile(r'\[[^\]]*\]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\S+')
    for m in token_re.finditer(masked):
        tok = m.group(0); pos = m.start()
        if tok.startswith('[') and tok.endswith(']'):
            try:
                raw = ast.literal_eval(tok)
                parts = [str(x) for x in raw]
            except Exception:
                parts = [x.strip() for x in tok.strip('[]').split(',') if x.strip()]
            events.append((pos, 'array1d', parts))
        elif (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
            events.append((pos, 'scalar', tok[1:-1]))
        else:
            events.append((pos, 'scalar', tok))

    events.sort(key=lambda e: e[0])

    out = []
    for _, typ, data in events:
        if typ == 'scalar':
            out.append(str(data))
        elif typ == 'array1d':
            out.append(" ".join(data))
        else:
            for row in data:
                out.append(" ".join(map(str, row)))
    return "\n".join(out) + "\n"

def transform_input_block(spec: str) -> str:
    """
    “key = value / 裸 value”的自然语言块 → stdin 多行。
    """
    events: List[Tuple[int, str, object]] = []
    token_re = re.compile(
        r"""
        (?P<kv_array>      \b\w+\s*=\s*\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])
      | (?P<kv_scalar>     \b\w+\s*=\s*(?: "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | True | False | -?\d+(?:\.\d+)? | \w+))
      | (?P<array>         \[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])
      | (?P<scalar>        "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | True | False | -?\d+(?:\.\d+)? | \w+)
        """,
        re.X,
    )

    for m in token_re.finditer(spec):
        if m.group("kv_array"):
            lit = m.group("kv_array").split("=", 1)[1].lstrip()
            try: arr = ast.literal_eval(lit)
            except Exception: arr = []
            events.append((m.start(), "array", arr)); continue
        if m.group("kv_scalar"):
            val = m.group("kv_scalar").split("=", 1)[1].lstrip()
            if val and val[0] in "\"'":
                val = val[1:-1]
            events.append((m.start(), "scalar", val)); continue
        if m.group("array"):
            try: arr = ast.literal_eval(m.group("array"))
            except Exception: arr = []
            events.append((m.start(), "array", arr)); continue
        if m.group("scalar"):
            tok = m.group("scalar")
            if tok and tok[0] in "\"'":
                tok = tok[1:-1]
            events.append((m.start(), "scalar", tok))

    events.sort(key=lambda e: e[0])
    lines: List[str] = []
    for _, kind, val in events:
        if kind == "scalar":
            lines.append(str(val))
        else:
            if isinstance(val, list) and val and all(isinstance(r, list) for r in val):
                lines.extend(" ".join(map(str, r)) for r in val)
            else:
                lines.append(" ".join(map(str, val)))
    return "\n".join(lines) + "\n"

def replace_input_block(text: str) -> str:
    def _repl(m):
        return f"{m.group(1)}\n{transform_input_block(m.group(2))}"
    pattern = re.compile(r'(Input\s*:\s*)(.*?)(?=\s*(?:Output\s*:|$))', flags=re.I|re.S)
    return pattern.sub(_repl, text)

def replace_output_block(text: str) -> str:
    def strip_quotes(tok: str) -> str:
        return tok[1:-1] if len(tok) >= 2 and tok[0] in "\"'" and tok[-1] == tok[0] else tok
    out, last = [], 0
    for m in re.finditer(r'Output\s*:', text):
        out.append(text[last:m.end()]); i = m.end()
        while i < len(text) and text[i].isspace():
            out.append(text[i]); i += 1
        if i >= len(text): break
        if text[i] == '[':
            start, end = i, _find_matching_bracket(text, i)
            literal = text[start:end+1] if end is not None else "[]"
            try: arr = ast.literal_eval(literal)
            except Exception: arr = []
            lines = ([" ".join(map(str, r)) for r in arr]
                     if arr and all(isinstance(r, list) for r in arr)
                     else [" ".join(map(str, arr))])
            out.append("\n" + "\n".join(lines) + "\n")
            last = (end or i) + 1
        else:
            m2 = re.match(r'(-?\d+|True|False|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', text[i:])
            if m2:
                out.append(f"\n{strip_quotes(m2.group(0))}\n")
                last = i + len(m2.group(0))
            else:
                last = i
    out.append(text[last:])
    return "".join(out)

def _normalize_cell(s: str) -> str:
    s = (s or "").strip().replace("\\n", "\n")
    if not s.endswith("\n"):
        s += "\n"
    return s

# ============================================================
# 断言 → (input, output) 抽取（MBPP / HumanEval 用）
# ============================================================

_ASSERT_PATTERNS: List[re.Pattern] = [
    # assert f(a,b,...) == expected
    re.compile(r'^\s*assert\s+(?P<call>\w+\s*\(.*\))\s*==\s*(?P<exp>.+?)\s*$', re.S),
    # assert expected == f(a,b,...)
    re.compile(r'^\s*assert\s+(?P<exp>.+?)\s*==\s*(?P<call>\w+\s*\(.*\))\s*$', re.S),
    # check(f(a,b,...), expected) / check_equal( ... )
    re.compile(r'^\s*(?:assert\s+)?(?:check|check_equal|check_solution)\s*\(\s*(?P<call>\w+\s*\(.*\))\s*,\s*(?P<exp>.+?)\s*\)\s*$', re.S),
    # check(expected, f(a,b,...))
    re.compile(r'^\s*(?:assert\s+)?(?:check|check_equal|check_solution)\s*\(\s*(?P<exp>.+?)\s*,\s*(?P<call>\w+\s*\(.*\))\s*\)\s*$', re.S),
]

def _strip_trailing_comment(s: str) -> str:
    return re.split(r'#(?![^\'"]*["\'])', s, maxsplit=1)[0].strip()

def _extract_args_from_call(call: str, prefer_fn: Optional[str] = None) -> Optional[str]:
    """
    给定 'foo(1, [2,3], "x")' → 返回括号内的原始参数串。
    若 prefer_fn 提供且 call 不是该函数，仍然接受（匹配不到再退化为任何函数）。
    """
    m = re.match(r'(?P<fn>\w+)\s*\((?P<args>.*)\)\s*$', call.strip(), re.S)
    if not m:
        return None
    fn = m.group("fn")
    if prefer_fn and fn != prefer_fn:
        # 允许其它函数（有些测试包装器内部仍会传被测函数）
        pass
    return m.group("args")

def parse_asserts_to_io(lines: List[str], prefer_fn: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    从断言行里抽取 (stdin 化的 input, stdin 化的 output) 列表。
    """
    ins, outs = [], []
    for raw in lines:
        line = _strip_trailing_comment(raw)
        if not line:
            continue
        matched = None
        for pat in _ASSERT_PATTERNS:
            m = pat.match(line)
            if m:
                matched = m; break
        if not matched:
            continue
        call = matched.group("call")
        exp  = matched.group("exp")
        args = _extract_args_from_call(call, prefer_fn=prefer_fn)
        if args is None:
            continue
        # 规约：把参数串 / 期望值串转为 stdin 风格
        in_text  = _normalize_cell(transform_tokens(args))
        out_text = _normalize_cell(transform_tokens(exp))
        ins.append(in_text); outs.append(out_text)
    return ins, outs

# ============================================================
# CodeContests
# ============================================================

PY_LANG_IDS = {1, 3}  # deepmind/code_contests: 1=PYTHON(2), 3=PYTHON3

def _pick_first_py(solutions: Dict[str, Any], k: int = 1) -> List[str]:
    langs = solutions.get("language") or []
    codes = solutions.get("solution") or []
    out = []
    for lang, code in zip(langs, codes):
        if lang in PY_LANG_IDS:
            out.append(code)
            if len(out) >= k:
                break
    return out

def _has_py(solutions: Dict[str, Any]) -> bool:
    return any((l in PY_LANG_IDS) for l in (solutions.get("language") or []))

def process_code_contests(split: str) -> pd.DataFrame:
    print(f"🔄 加载 deepmind/code_contests split={split} ...")
    ds = load_dataset("deepmind/code_contests", split=split)
    rows = []
    for ex in ds:
        tests = (ex.get("public_tests") or {}) if split == "test" else (ex.get("private_tests") or {})
        test_in = tests.get("input") or []
        test_out = tests.get("output") or []
        if not (isinstance(test_in, list) and test_in):
            continue
        solutions = ex.get("solutions") or {}
        if not _has_py(solutions):
            # 即便没有 Python 参考解，也允许；solution 置空
            solution = ""
        else:
            py = _pick_first_py(solutions, k=1)
            solution = py[0] if py else ""
        rows.append({
            "question": (ex.get("description") or "").strip(),
            "solution": solution,
            "test_input": [ _normalize_cell(str(x)) for x in test_in ],
            "test_output": [ _normalize_cell(str(x)) for x in (test_out or []) ],
        })
    df = pd.DataFrame(rows, columns=["question","test_input","test_output","solution"])
    print(f"✅ code_contests/{split}: {len(df)}")
    return df

# ============================================================
# MBPP
# ============================================================

def process_mbpp() -> pd.DataFrame:
    print("🔄 加载 MBPP（优先 sanitized/test）...")
    try:
        ds = load_dataset("mbpp", name="sanitized", split="test")
    except Exception:
        ds = load_dataset("mbpp", split="test")
    rows = []
    for ex in ds:
        question = (ex.get("text") or ex.get("prompt") or ex.get("description") or "").strip()
        solution = (ex.get("code") or ex.get("solution") or "")

        # tests: list[str] 或单串
        test_list = ex.get("test_list") or ex.get("test") or []
        if isinstance(test_list, str):
            test_lines = [ln for ln in test_list.splitlines() if ln.strip()]
        elif isinstance(test_list, list):
            # 普遍为断言字符串列表
            test_lines = []
            for t in test_list:
                test_lines += [ln for ln in str(t).splitlines() if ln.strip()]
        else:
            test_lines = []

        # 解析断言 → I/O
        inputs, outputs = parse_asserts_to_io(test_lines, prefer_fn=None)
        rows.append({
            "question": question,
            "solution": solution or "",
            "test_input": inputs,
            "test_output": outputs,
        })
    df = pd.DataFrame(rows, columns=["question","test_input","test_output","solution"])
    print(f"✅ mbpp: {len(df)}")
    return df

# ============================================================
# HumanEval
# ============================================================

def process_humaneval() -> pd.DataFrame:
    print("🔄 加载 openai_humaneval/test ...")
    ds = load_dataset("openai_humaneval", split="test")
    rows = []
    for ex in ds:
        question = (ex.get("prompt") or "").strip()
        solution = (ex.get("canonical_solution") or ex.get("solution") or "")
        entry_point = ex.get("entry_point") or None
        test_str = ex.get("test") or ""
        test_lines = [ln for ln in str(test_str).splitlines() if ln.strip()]
        inputs, outputs = parse_asserts_to_io(test_lines, prefer_fn=entry_point)
        rows.append({
            "question": question,
            "solution": solution or "",
            "test_input": inputs,
            "test_output": outputs,
        })
    df = pd.DataFrame(rows, columns=["question","test_input","test_output","solution"])
    print(f"✅ human_eval: {len(df)}")
    return df

# ============================================================
# LiveCodeBench（使用你提供的 code_generation_lite + 规约）
# ============================================================

class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"

@dataclass
class _LCB_Test:
    input: str
    output: str
    testtype: TestType
    def __post_init__(self):
        self.testtype = TestType(self.testtype)

@dataclass
class _LCB_Problem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[_LCB_Test]
    private_test_cases: list[_LCB_Test]
    metadata: dict
    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)
        # public
        pts = json.loads(self.public_test_cases)
        self.public_test_cases = [_LCB_Test(**t) for t in pts]
        # private 可能是 json 或 zlib+pickle+base64
        try:
            pr = json.loads(self.private_test_cases)
        except Exception:
            pr = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))
                    )
                )
            )
        self.private_test_cases = [_LCB_Test(**t) for t in pr]
        self.metadata = json.loads(self.metadata)

def _load_lcb_lite(release_version: str = "release_v2",
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> List[_LCB_Problem]:
    raw = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=release_version,
    )
    problems = [_LCB_Problem(**p) for p in raw]
    if start_date:
        p0 = datetime.strptime(start_date, "%Y-%m-%d")
        problems = [p for p in problems if p.contest_date >= p0]
    if end_date:
        p1 = datetime.strptime(end_date, "%Y-%m-%d")
        problems = [p for p in problems if p.contest_date <= p1]
    print(f"LCB loaded: {len(problems)}")
    return problems

def process_livecodebench(release_version: str = "release_v2",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
    print(f"🔄 加载 LiveCodeBench/code_generation_lite ({release_version}) ...")
    probs = _load_lcb_lite(release_version, start_date, end_date)
    rows = []
    for p in probs:
        # 规范化题面中的 Input/Output 叙述（如果有）
        qtext = p.question_content.strip()
        qtext = replace_input_block(qtext)
        qtext = replace_output_block(qtext)

        if p.private_test_cases and p.private_test_cases[0].testtype.value == "functional":
            ins  = [_normalize_cell(transform_tokens(t.input))  for t in p.private_test_cases]
            outs = [_normalize_cell(transform_tokens(t.output)) for t in p.private_test_cases]
        else:
            ins  = [_normalize_cell(t.input)  for t in p.private_test_cases]
            outs = [_normalize_cell(t.output) for t in p.private_test_cases]

        rows.append({
            "question": qtext,
            "solution": "",  # LCB 官方通常不提供完整参考解，这里留空
            "test_input": ins,
            "test_output": outs,
        })
    df = pd.DataFrame(rows, columns=["question","test_input","test_output","solution"])
    print(f"✅ livecodebench: {len(df)}")
    return df

# ============================================================
# 主流程：写出 4 份 parquet（仅 4 列）
# ============================================================

def main():
    # 输出目录：datasets/code/train/
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "code" / "train"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    # 1) 训练集：CodeContests(train) → train.parquet
    df_train = process_code_contests(split="train")
    (out_dir / "train.parquet").unlink(missing_ok=True)
    df_train.to_parquet(out_dir / "train.parquet", index=False)
    print(f"💾 保存: {out_dir / 'train.parquet'}")

    # 2) 测试集四份：各自名字.parquet（仅含 4 列）
    # 2.1 CodeContests(test)
    df_cc_test = process_code_contests(split="test")
    (out_dir / "code_contests.parquet").unlink(missing_ok=True)
    df_cc_test.to_parquet(out_dir / "code_contests.parquet", index=False)
    print(f"💾 保存: {out_dir / 'code_contests.parquet'}")

    # 2.2 MBPP
    df_mbpp = process_mbpp()
    (out_dir / "mbpp.parquet").unlink(missing_ok=True)
    df_mbpp.to_parquet(out_dir / "mbpp.parquet", index=False)
    print(f"💾 保存: {out_dir / 'mbpp.parquet'}")

    # 2.3 HumanEval
    df_he = process_humaneval()
    (out_dir / "human_eval.parquet").unlink(missing_ok=True)
    df_he.to_parquet(out_dir / "human_eval.parquet", index=False)
    print(f"💾 保存: {out_dir / 'human_eval.parquet'}")

    # 2.4 LiveCodeBench（使用 lite）
    df_lcb = process_livecodebench(release_version="release_v2")
    (out_dir / "livecodebench.parquet").unlink(missing_ok=True)
    df_lcb.to_parquet(out_dir / "livecodebench.parquet", index=False)
    print(f"💾 保存: {out_dir / 'livecodebench.parquet'}")

if __name__ == "__main__":
    main()
