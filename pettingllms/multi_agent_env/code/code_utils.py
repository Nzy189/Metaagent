"""
Utility functions for code generation and testing.

This module contains utilities for code execution, validation, data loading,
and metric computation. It references the eval part of the CURE-main project
and supports streaming data loading.
"""

import os
import sys
import json
import io
import time
import typing
import multiprocessing as mp
import re
import random
from typing import Any, Dict, Optional, Tuple, List, Union
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

@dataclass
class evaluate_result:
    """
    Dataclass for test results
    """
    test_case_id: int
    input: str
    expected_output: str
    actual_output: str
    passed: bool
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict format to keep backward compatibility"""
        return {
            "test_case_id": self.test_case_id,
            "input": self.input,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "passed": self.passed,
            "error_type": self.error_type
        }

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


def load_problem_batch(
    dataset_name: str,
    batch_size: int,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """
    Load a batch of programming problems.
    
    Args:
        dataset_name: Dataset name (e.g., "deepmind/code_contests", "Gen-Verse/CodeContests")
        batch_size: Batch size
        split: Dataset split ("train", "test", etc.)
        
    Returns:
        A list of dicts of length batch_size with keys problem/golden_code/golden_test_input/golden_test_output
    """
    if not DATASETS_AVAILABLE:
        print("❌ datasets库不可用")
        return []
    
    print(f"🔄 Loading {batch_size} problems from dataset {dataset_name}...")
    
    try:
        # Try streaming mode first
        try:
            ds = hf_load_dataset(dataset_name, streaming=True)[split]
            
            batch_results = []
            iterator = ds.take(batch_size * 2)  # Take more to ensure enough valid problems
            
            for i, example in enumerate(iterator):
                if len(batch_results) >= batch_size:
                    break
                    
                problem, golden_code, test_input, test_output = _format_competition_problem_with_golden(example, i)
                if problem:
                    result_dict = {
                        "problem": problem,
                        "golden_code": golden_code,
                        "golden_test_input": test_input,
                        "golden_test_output": test_output
                    }
                    batch_results.append(result_dict)
                    print(f"✅ Loaded problem {len(batch_results)}/{batch_size} (index={i})")
            
            if batch_results:
                print(f"✅ Successfully loaded {len(batch_results)} problems")
                return batch_results
            else:
                raise Exception("No valid problems found in streaming mode")
            
        except Exception as e:
            print(f"⚠️ Streaming mode failed: {e}")
            print("💡 Trying traditional loading method...")
            
            # Traditional loading method
            ds = hf_load_dataset(dataset_name)[split]
            
            if len(ds) == 0:
                print("❌ Dataset is empty")
                return []
            
            batch_results = []
            max_attempts = min(batch_size * 3, len(ds))  # Try more to ensure enough valid problems
            
            for i in range(max_attempts):
                if len(batch_results) >= batch_size:
                    break
                    
                problem, golden_code, test_input, test_output = _format_competition_problem_with_golden(ds[i], i)
                if problem:
                    result_dict = {
                        "problem": problem,
                        "golden_code": golden_code,
                        "golden_test_input": test_input,
                        "golden_test_output": test_output
                    }
                    batch_results.append(result_dict)
                    print(f"✅ Loaded problem {len(batch_results)}/{batch_size} (index={i})")
            
            if batch_results:
                print(f"✅ Successfully loaded {len(batch_results)} problems")
                return batch_results
            else:
                print("❌ Failed to find valid problems")
                return []
            
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        print("💡 Trying fallback method...")
        try:
            # Fallback: traditional loading
            ds = hf_load_dataset(dataset_name)[split]
            if len(ds) == 0:
                print("❌ Dataset is empty")
                return []
            
            batch_results = []
            max_attempts = min(batch_size * 2, len(ds))
            
            for i in range(max_attempts):
                if len(batch_results) >= batch_size:
                    break
                    
                problem, golden_code, test_input, test_output = _format_competition_problem_with_golden(ds[i], i)
                if problem:
                    result_dict = {
                        "problem": problem,
                        "golden_code": golden_code,
                        "golden_test_input": test_input,
                        "golden_test_output": test_output
                    }
                    batch_results.append(result_dict)
                    print(f"✅ Loaded problem {len(batch_results)}/{batch_size} (index={i})")
            
            if batch_results:
                print(f"✅ Fallback succeeded, loaded {len(batch_results)} problems")
                return batch_results
            else:
                print("❌ Fallback did not find valid problems")
                return []
                
        except Exception as e2:
            print(f"❌ Fallback method also failed: {e2}")
            return []


def _format_competition_problem(example: Dict, index: int) -> Optional[Dict]:
    """
    Convert raw dataset example to a unified competition problem format.
    
    Args:
        example: Raw dataset sample
        index: Sample index
        
    Returns:
        Formatted problem dict or None
    """
    try:
        # Try different dataset formats
        if "description" in example:
            # Code Contests format
            return {
                "problem_id": f"code_contests_{index}",
                "question": example["description"],
                "test_time_limit": example.get("time_limit", 1.0),
                "example_input": example.get("public_tests", {}).get("input", []),
                "example_output": example.get("public_tests", {}).get("output", []),
                "test_input": example.get("private_tests", {}).get("input", []),
                "test_output": example.get("private_tests", {}).get("output", []),
                "difficulty": example.get("difficulty", "unknown"),
                "memory_limit": example.get("memory_limit_bytes", 0)
            }
        elif "question" in example:
            # CURE format
            return {
                "problem_id": f"cure_{index}",
                "question": example["question"],
                "test_time_limit": example.get("test_time_limit", 1.0),
                "example_input": example.get("example_input", []),
                "example_output": example.get("example_output", []),
                "test_input": example.get("test_input", []),
                "test_output": example.get("test_output", [])
            }
        elif "prompt" in example:
            # APPS/HumanEval format
            return {
                "problem_id": f"apps_{index}",
                "question": example["prompt"],
                "test_time_limit": 1.0,
                "example_input": [],
                "example_output": [],
                "test_input": example.get("input_output", {}).get("inputs", []),
                "test_output": example.get("input_output", {}).get("outputs", [])
            }
        else:
            print(f"⚠️ Unknown data format: {list(example.keys())[:5]}")
            return None
            
    except Exception as e:
        print(f"⚠️ Failed to format problem {index}: {e}")
        return None


def _format_competition_problem_with_golden(example: Dict, index: int) -> Tuple[Optional[Dict], Optional[str], Optional[List], Optional[List]]:
    """
    Convert raw dataset format and extract golden_code and test data.
    
    Args:
        example: Raw dataset sample
        index: Sample index
        
    Returns:
        (problem, golden_code, test_input, test_output) or (None, None, None, None)
    """
    try:
        # First, format the problem
        problem = _format_competition_problem(example, index)
        if not problem:
            return None, None, None, None
        
        # Extract golden_code
        golden_code = _extract_golden_code(example)
        
        # Extract test inputs/outputs
        test_input = problem.get("test_input", [])
        test_output = problem.get("test_output", [])
        
        return problem, golden_code, test_input, test_output
            
    except Exception as e:
        print(f"⚠️ Failed to format problem {index}: {e}")
        return None, None, None, None


def _extract_golden_code(example: Dict) -> Optional[str]:
    """
    Extract golden_code from raw dataset example.
    
    Args:
        example: Raw dataset sample
        
    Returns:
        golden_code string or None
    """
    try:
        # Try various field names
        code_fields = ["solution", "golden_code", "code", "answer", "implementation"]
        
        for field in code_fields:
            if field in example and example[field]:
                code = example[field]
                if isinstance(code, str):
                    return code
                elif isinstance(code, list) and len(code) > 0:
                    return code[0] if isinstance(code[0], str) else str(code[0])
        
        # If not found, return None
        return None
        
    except Exception as e:
        print(f"⚠️ Failed to extract golden_code: {e}")
        return None


# =================== Code execution and validation ===================

def execute_code_with_timeout(
    code: str, 
    test_input: str, 
    timeout: float = 1.0
) -> str:
    """
    Execute Python code with a timeout.
    
    Args:
        code: Python code to execute
        test_input: Input string
        timeout: Timeout in seconds
        
    Returns:
        Output string or error message
    """
    def worker_target(script, input_val, output_queue):
        input_lines = iter(input_val.splitlines())
        
        def fake_input(prompt=""):
            try:
                return next(input_lines)
            except StopIteration:
                raise EOFError("No more input")
        
        stdout_capture = io.StringIO()
        original_stdout = sys.stdout
        original_stdin = sys.stdin
        sys.stdout = stdout_capture
        sys.stdin = io.StringIO(input_val)

        context = {
            "__name__": "__main__",
            "input": fake_input,
            "List": typing.List,
            "Tuple": typing.Tuple,
            "Optional": typing.Optional,
        }

        try:
            exec(script, context)
            printed_output = stdout_capture.getvalue()
            output_queue.put(printed_output)

        except SystemExit:
            printed_output = stdout_capture.getvalue()
            output_queue.put(printed_output)

        except Exception as e:
            output_queue.put(f"error: {e}")

        finally:
            sys.stdout = original_stdout
            sys.stdin = original_stdin

    # Use multiprocessing to enforce timeout
    output_queue = mp.Queue()
    process = mp.Process(target=worker_target, args=(code, test_input, output_queue))
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return "Timeout Error"
    
    try:
        return output_queue.get_nowait()
    except:
        return "Execution Error"


def test_code_equality(output1: str, output2: str) -> bool:
    """
    Test equality of two outputs ignoring whitespace differences.
    
    Args:
        output1: First output
        output2: Second output
        
    Returns:
        Boolean indicating equality
    """
    return " ".join(output1.split()) == " ".join(output2.split())


def detailed_test_comparison(
    actual_output: str, 
    expected_output: str, 
    test_case_info: Dict
) -> evaluate_result:
    """
    Compare outputs and return a detailed result.
    
    Args:
        actual_output: Actual output
        expected_output: Expected output
        test_case_info: Test case information
        
    Returns:
        evaluate_result object
    """
    is_correct = test_code_equality(actual_output, expected_output)
    
    return evaluate_result(
        test_case_id=test_case_info.get("test_case", 0),
        input=test_case_info.get("input", ""),
        expected_output=expected_output,
        actual_output=actual_output,
        passed=is_correct,
        error_type=None if is_correct else (
            "timeout" if "Timeout" in actual_output else 
            "execution_error" if "error:" in actual_output else "output_mismatch"
        )
    )


def evaluate_code_against_tests(
    code: str, 
    test_inputs: List[str], 
    test_outputs: List[str],
    timeout: float = 1.0
) -> Tuple[float, Dict]:
    """
    Evaluate code against test cases and return detailed results.
    
    Args:
        code: Code to evaluate
        test_inputs: List of test inputs
        test_outputs: List of expected outputs
        timeout: Execution timeout
        
    Returns:
        (reward_score, detailed_info_dict)
    """
    if not test_inputs or not test_outputs:
        return 0.0, {"error": "No available test cases"}
    
    if len(test_inputs) != len(test_outputs):
        return 0.0, {"error": "Mismatched number of inputs and outputs"}
    
    passed_tests = 0
    total_tests = len(test_inputs)
    execution_results = []
    passed_cases = []
    failed_cases = []
    
    for i, (test_input, expected_output) in enumerate(zip(test_inputs, test_outputs)):
        test_case_info = {"test_case": i, "input": test_input}
        
        try:
            actual_output = execute_code_with_timeout(code, str(test_input), timeout)
            detailed_result = detailed_test_comparison(
                actual_output, str(expected_output), test_case_info
            )
            
            if detailed_result.passed:
                passed_tests += 1
                passed_cases.append(detailed_result)
            else:
                failed_cases.append(detailed_result)
            
            execution_results.append(detailed_result)
            
        except Exception as e:
            error_result = evaluate_result(
                test_case_id=i,
                input=test_input,
                expected_output=expected_output,
                actual_output=f"Exception: {str(e)}",
                passed=False,
                error_type="exception"
            )
            failed_cases.append(error_result)
            execution_results.append(error_result)
    
    reward = passed_tests / total_tests if total_tests > 0 else 0.0
    
    # Build detailed evaluation result
    detailed_info = {
        "overall_result": {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_rate": reward,
            "all_passed": passed_tests == total_tests,
            "summary": f"{passed_tests}/{total_tests} tests passed"
        },
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "execution_results": execution_results,
        "statistics": {
            "timeout_errors": len([r for r in execution_results if r.get("error_type") == "timeout"]),
            "execution_errors": len([r for r in execution_results if r.get("error_type") == "execution_error"]),
            "output_mismatches": len([r for r in execution_results if r.get("error_type") == "output_mismatch"]),
            "exceptions": len([r for r in execution_results if r.get("error_type") == "exception"])
        }
    }
    
    return reward, detailed_info


def evaluate_tests_against_golden_code(
    test_cases: List[Dict], 
    golden_code: str,
    timeout: float = 1.0
) -> Tuple[float, Dict]:
    """
    Evaluate generated test cases against golden code.
    
    Args:
        test_cases: List of test cases, each containing 'input' and 'output'
        golden_code: Golden standard code
        timeout: Execution timeout
        
    Returns:
        (reward_score, detailed_info_dict)
    """
    if not test_cases:
        return 0.0, {"error": "No test cases provided"}
    
    if not golden_code:
        return 0.0, {"error": "No golden code available"}
    
    passed_tests = 0
    total_tests = len(test_cases)
    execution_results = []
    passed_cases = []
    failed_cases = []
    
    for i, test_case in enumerate(test_cases):
        test_case_info = {"test_case": i, "input": test_case.get("input", "")}
        
        try:
            test_input = test_case.get("input", "")
            expected_output = test_case.get("output", "")
            
            actual_output = execute_code_with_timeout(golden_code, str(test_input), timeout)
            detailed_result = detailed_test_comparison(
                actual_output, str(expected_output), test_case_info
            )
            
            if detailed_result.passed:
                passed_tests += 1
                passed_cases.append(detailed_result)
            else:
                failed_cases.append(detailed_result)
            
            execution_results.append(detailed_result)
            
        except Exception as e:
            error_result = evaluate_result(
                test_case_id=i,
                input=test_case.get("input", ""),
                expected_output=test_case.get("output", ""),
                actual_output=f"Exception: {str(e)}",
                passed=False,
                error_type="exception"
            )
            failed_cases.append(error_result)
            execution_results.append(error_result)
    
    reward = passed_tests / total_tests if total_tests > 0 else 0.0
    
    # Build detailed evaluation result
    detailed_info = {
        "overall_result": {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_rate": reward,
            "all_passed": passed_tests == total_tests,
            "summary": f"{passed_tests}/{total_tests} test cases are valid"
        },
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "execution_results": execution_results,
        "statistics": {
            "timeout_errors": len([r for r in execution_results if r.get("error_type") == "timeout"]),
            "execution_errors": len([r for r in execution_results if r.get("error_type") == "execution_error"]),
            "output_mismatches": len([r for r in execution_results if r.get("error_type") == "output_mismatch"]),
            "exceptions": len([r for r in execution_results if r.get("error_type") == "exception"])
        },
        "generated_test_cases": test_cases
    }
    
    return reward, detailed_info


# =================== Test case parsing ===================

def parse_test_cases_from_response(response: str) -> List[Dict]:
    """
    Parse test cases from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        List of parsed test cases
    """
    test_cases = []
    
    # Try extracting input/output pairs
    pattern_input = r'\*\*Test Input:\*\*\s*```(.*?)```'
    pattern_output = r'\*\*Test Output:\*\*\s*```(.*?)```'
    
    inputs = re.findall(pattern_input, response, re.DOTALL)
    outputs = re.findall(pattern_output, response, re.DOTALL)
    
    # Fallback patterns (without backticks)
    if not inputs:
        pattern_input_plain = r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)'
        inputs = re.findall(pattern_input_plain, response, re.DOTALL)
    
    if not outputs:
        pattern_output_plain = r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Test Input:|$)'
        outputs = re.findall(pattern_output_plain, response, re.DOTALL)
    
    # Try other common formats
    if not inputs or not outputs:
        # Format: Input: ... Output: ...
        pattern_alt = r'Input:\s*(.*?)\s*Output:\s*(.*?)(?=Input:|$)'
        matches = re.findall(pattern_alt, response, re.DOTALL)
        if matches:
            inputs = [m[0].strip() for m in matches]
            outputs = [m[1].strip() for m in matches]
    
    for i in range(min(len(inputs), len(outputs))):
        test_cases.append({
            "input": inputs[i].strip(),
            "output": outputs[i].strip()
        })
    
    return test_cases


def extract_code_from_response(response: str) -> str:
    """
    Extract code from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        Extracted code string
    """
    # Look for Python code block
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()  # Return the last code block
    
    # Look for generic code block
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    # If no code block found, return entire response
    return response.strip()


# =================== Metric computation ===================

def compute_pass_at_k_metrics(
    results: List[Dict], 
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Pass@K metrics.
    
    Args:
        results: Evaluation results list
        k_values: List of K values
        
    Returns:
        Dict of Pass@K metrics
    """
    if not results:
        return {f"pass@{k}": 0.0 for k in k_values}
    
    # Compute pass status for each problem
    problem_results = {}
    for result in results:
        problem_id = result.get("task_id", result.get("problem_id", "unknown"))
        if problem_id not in problem_results:
            problem_results[problem_id] = []
        
        passed = result.get("success", False) or result.get("all_passed", False)
        problem_results[problem_id].append(passed)
    
    metrics = {}
    total_problems = len(problem_results)
    
    for k in k_values:
        passed_problems = 0
        for problem_id, passes in problem_results.items():
            # Take top-k results
            k_results = passes[:k]
            if any(k_results):  # At least one pass
                passed_problems += 1
        
        pass_rate = passed_problems / total_problems if total_problems > 0 else 0.0
        metrics[f"pass@{k}"] = pass_rate
    
    return metrics


def compute_basic_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute basic evaluation metrics.
    
    Args:
        results: Evaluation results list
        
    Returns:
        Dict of basic metrics
    """
    if not results:
        return {
            "total_tasks": 0,
            "success_rate": 0.0,
            "average_iterations": 0.0,
            "average_test_pass_rate": 0.0
        }
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get("success", False))
    
    # Compute average iterations
    iterations = [r.get("total_iterations", 0) for r in results]
    avg_iterations = sum(iterations) / len(iterations) if iterations else 0.0
    
    # Compute average test pass rate
    test_pass_rates = []
    for r in results:
        if "final_test_results" in r and "pass_rate" in r["final_test_results"]:
            test_pass_rates.append(r["final_test_results"]["pass_rate"])
        elif "code_evaluation" in r and "pass_rate" in r["code_evaluation"]:
            test_pass_rates.append(r["code_evaluation"]["pass_rate"])
    
    avg_test_pass_rate = sum(test_pass_rates) / len(test_pass_rates) if test_pass_rates else 0.0
    
    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": successful_tasks / total_tasks,
        "average_iterations": avg_iterations,
        "average_test_pass_rate": avg_test_pass_rate,
        "total_errors": total_tasks - successful_tasks
    }


def compute_error_analysis(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute error analysis metrics.
    
    Args:
        results: Evaluation results list
        
    Returns:
        Error analysis dict
    """
    error_types = {
        "timeout_errors": 0,
        "execution_errors": 0,
        "output_mismatches": 0,
        "exceptions": 0,
        "no_solution": 0
    }
    
    termination_reasons = {}
    
    for result in results:
        # Analyze termination reasons
        reason = result.get("termination_reason", "unknown")
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        
        # Analyze error types
        if "iterations" in result:
            for iteration in result["iterations"]:
                if "code_execution_result" in iteration:
                    exec_result = iteration["code_execution_result"]
                    stats = exec_result.get("statistics", {})
                    
                    for error_type in error_types:
                        error_types[error_type] += stats.get(error_type, 0)
    
    return {
        "error_statistics": error_types,
        "termination_reasons": termination_reasons
    }


def compute_comprehensive_metrics(
    results: List[Dict], 
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        results: Evaluation results list
        k_values: List of K values for Pass@K
        
    Returns:
        Comprehensive metrics dict
    """
    basic_metrics = compute_basic_metrics(results)
    pass_at_k_metrics = compute_pass_at_k_metrics(results, k_values)
    error_analysis = compute_error_analysis(results)
    
    return {
        **basic_metrics,
        **pass_at_k_metrics,
        **error_analysis,
        "evaluation_timestamp": time.time(),
        "num_evaluated_tasks": len(results)
    }


# =================== Helper functions ===================

def save_evaluation_results(
    results: Dict[str, Any], 
    output_path: str,
    pretty_print: bool = True
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results dict
        output_path: Output file path
        pretty_print: Whether to pretty print JSON
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)
                
        print(f"💾 Evaluation results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """
    Print evaluation summary.
    
    Args:
        metrics: Evaluation metrics dict
    """
    print(f"\n🎯 Evaluation Summary:")
    print(f"  📊 Total tasks: {metrics.get('total_tasks', 0)}")
    print(f"  ✅ Successful: {metrics.get('successful_tasks', 0)}")
    print(f"  📈 Success rate: {metrics.get('success_rate', 0):.2%}")
    print(f"  🔄 Avg iterations: {metrics.get('average_iterations', 0):.1f}")
    print(f"  🧪 Avg test pass rate: {metrics.get('average_test_pass_rate', 0):.2%}")
    
    # Print Pass@K metrics
    for k in [1, 5, 10]:
        if f"pass@{k}" in metrics:
            print(f"  📊 Pass@{k}: {metrics[f'pass@{k}']:.2%}")
    
    # Print error statistics
    if "error_statistics" in metrics:
        print(f"\n❌ 错误统计:")
        for error_type, count in metrics["error_statistics"].items():
            if count > 0:
                print(f"  {error_type}: {count}")


# =================== 主要评估函数 ===================

def evaluate_code_generation_task(
    code: str,
    problem: Dict,
    timeout: float = 1.0
) -> Dict[str, Any]:
    """
    评估单个代码生成任务
    
    Args:
        code: 生成的代码
        problem: 问题字典
        timeout: 执行超时时间
        
    Returns:
        评估结果字典
    """
    # 获取测试用例
    test_inputs = problem.get("test_input", [])
    test_outputs = problem.get("test_output", [])
    
    # 如果没有私有测试用例，使用示例测试用例
    if not test_inputs or not test_outputs:
        test_inputs = problem.get("example_input", [])
        test_outputs = problem.get("example_output", [])
    
    if not test_inputs or not test_outputs:
        return {
            "success": False,
            "error": "没有可用的测试用例",
            "pass_rate": 0.0
        }
    
    # 执行评估
    reward, detailed_info = evaluate_code_against_tests(
        code, test_inputs, test_outputs, timeout
    )
    
    overall_result = detailed_info.get("overall_result", {})
    
    return {
        "success": overall_result.get("all_passed", False),
        "pass_rate": overall_result.get("pass_rate", 0.0),
        "passed_tests": overall_result.get("passed_tests", 0),
        "total_tests": overall_result.get("total_tests", 0),
        "reward": reward,
        "detailed_results": detailed_info,
        "execution_statistics": detailed_info.get("statistics", {})
    }


def evaluate_test_generation_task(
    test_cases: List[Dict],
    golden_code: str,
    timeout: float = 1.0
) -> Dict[str, Any]:
    """
    评估单个测试生成任务
    
    Args:
        test_cases: 生成的测试用例列表
        golden_code: 黄金标准代码
        timeout: 执行超时时间
        
    Returns:
        评估结果字典
    """
    if not test_cases:
        return {
            "success": False,
            "error": "没有生成测试用例",
            "pass_rate": 0.0
        }
    
    if not golden_code:
        return {
            "success": False,
            "error": "没有黄金代码可用",
            "pass_rate": 0.0
        }
    
    # 执行评估
    reward, detailed_info = evaluate_tests_against_golden_code(
        test_cases, golden_code, timeout
    )
    
    overall_result = detailed_info.get("overall_result", {})
    
    return {
        "success": overall_result.get("all_passed", False),
        "pass_rate": overall_result.get("pass_rate", 0.0),
        "valid_tests": overall_result.get("passed_tests", 0),
        "total_tests": overall_result.get("total_tests", 0),
        "reward": reward,
        "detailed_results": detailed_info,
        "generated_test_cases": test_cases
    }


def test_load_problem(benchmark: str, batch_size: int):
    # 获取问题
    results= load_problem_batch(
        dataset_name=benchmark,
        batch_size=batch_size
    )

    for problem in results:
        print(f"问题描述: {problem['problem']}")
        print(f"Golden Code: {problem['golden_code']}")
        print(f"测试用例数量: {len(problem['golden_test_input'])}")
        print(f"测试用例: {problem['golden_test_input']}")
        print(f"测试用例输出: {problem['golden_test_output']}")
    print(f"问题数量: {len(results)}")

if __name__ == "__main__":
    test_load_problem("deepmind/code_contests", 10)