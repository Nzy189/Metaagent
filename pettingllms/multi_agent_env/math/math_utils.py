"""
Utility functions for mathematical problem solving and evaluation.

This module contains utilities for loading math datasets, evaluating solutions,
and computing metrics for mathematical problem solving tasks.
"""

import os
import json
import random
import asyncio
import re
import subprocess
import signal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import ray
import shutil
import tempfile
import time
import contextlib

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

def extract_answer(solution_str):
    """
    Extract answer from solution string using \\boxed{} format.
    
    Args:
        solution_str: Solution text containing \\boxed{answer}
        
    Returns:
        Extracted answer string or None if not found
    """
    # Look for \\boxed{...} pattern
    boxed_pattern = r"\\boxed\s*\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, solution_str)
    
    if matches:
        # Return the last boxed answer found
        return matches[-1].strip()
    
    # Fallback: look for the old #### format for backward compatibility
    solution = re.findall(r"####\s*(.+?)(?:\n|$)", solution_str)
    if solution:
        return solution[-1].strip()
    
    return None

def extract_reasoning_steps(response: str):
    """
    Extract reasoning steps from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        Extracted reasoning steps
    """
    # Use regex to match Reasoning Steps part in ```
    match = re.search(r"\*\*Reasoning Steps:\*\*\s*```(.*?)```", response, re.DOTALL)
    if not match:
        return []
    
    steps_block = match.group(1).strip()
    
    # 按行分割并去除空行
    steps = [line.strip() for line in steps_block.split("\n") if line.strip()]
    return steps

def extract_code(response: str) -> str:
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


def extract_code(response: str) -> str:
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


async def _await_ray_object_ref(obj_ref, timeout_seconds: float = 10.0):
    import ray
    import time
    
    start_time = time.time()
    while True:
        ready, _ = ray.wait([obj_ref], timeout=0.1)
        if ready:
            return ray.get(obj_ref)
        
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise asyncio.TimeoutError(f"Ray task timed out after {timeout_seconds}s")
        

        await asyncio.sleep(0.01)


async def test_if_eq(x, y):
    """
    Test equality of two outputs ignoring whitespace differences.
    Based on the reference test_if_eq function provided.
    """
    return " ".join(x.split()) == " ".join(y.split())





async def evaluate_code_against_tests(
    code: str, 
    test_inputs: List[str], 
    test_outputs: List[str],
    timeout: float = 40.0,
    *,
    image: str = "python:3.11-slim",
    ray_actor: Any | None = None,
    rollout_idx: int | None = None,
) -> Tuple[float, List, List]:
    """
    Evaluate code against test cases and return detailed results.
    Uses async execution for improved performance.
    
    Args:
        code: Code to evaluate
        test_inputs: List of test inputs
        test_outputs: List of expected outputs
        timeout: Execution timeout
        
    Returns:
        (passed_ratio, passed_cases, failed_cases)
    """
    if not test_inputs or not test_outputs:
        return 0.0, [], []
    
    
    total_tests = len(test_inputs)
    results: List[Dict[str, Any]] = []
    tasks = [
                asyncio.create_task(
                    _worker_docker(code, test_inputs[i], test_outputs[i], timeout, image)
                ) for i in range(total_tests)
            ]
    results = await asyncio.gather(*tasks)
  
    passed_tests = 0
    passed_cases: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    for i, result in enumerate(results):
        actual_output = result.get("code_execution_output")
        expected_output = result.get("test_output")
        if_passed = result.get("passed", False)
        test_case_info = {
            "test_input": test_inputs[i],
            "code_execution_output": actual_output,
            "generated_test_output": expected_output,
            "passed": if_passed,
        }

        if actual_output is None:
            if_passed = False
        elif isinstance(actual_output, str) and actual_output.startswith("error:"):
            if_passed = False
        else:
            if_passed = await test_if_eq(actual_output, str(expected_output))

        if if_passed:
            passed_tests += 1
            passed_cases.append(test_case_info)
        else:
            failed_cases.append(test_case_info)

    passed_ratio = passed_tests / total_tests if total_tests > 0 else 0.0
    return passed_ratio, passed_cases, failed_cases



def _ensure_ray_initialized() -> bool:
    from pettingllms.utils.logger_config import get_multi_logger
    multi_logger = get_multi_logger()
    import ray  

    if not ray.is_initialized():
        multi_logger.log_ray_status(mode="train", context="test_ray_log_function ")
       
        
        try:
            num_cpus_env = os.getenv("RAY_NUM_CPUS")
            multi_logger.log_ray_status(mode="train", context="before_code_utils_ray_init")
            init_kwargs = dict(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
            )
            if num_cpus_env:
                try:
                    num_cpus = float(num_cpus_env)
                    if num_cpus > 0:
                        init_kwargs["num_cpus"] = num_cpus
                    else:
                        print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
                except (ValueError, TypeError):
                    print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

            # Ensure Ray temp and spill directories
            try:
                # Use /tmp for shorter socket paths (Unix socket path limit is 107 bytes)
                ray_tmp_dir = "/tmp/verl_ray"
                ray_spill_dir = "/tmp/verl_spill"
                os.makedirs(ray_tmp_dir, exist_ok=True)
                os.makedirs(ray_spill_dir, exist_ok=True)

                init_kwargs["_temp_dir"] = ray_tmp_dir
                spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
                init_kwargs["_system_config"] = {
                    "object_spilling_config": json.dumps(spilling_conf)
                }
            except Exception as _e:
                print(f"Warning: failed to prepare Ray temp/spill dirs: {_e}")

            ray.init(**init_kwargs)

            try:
                cluster = ray.cluster_resources()
                avail = ray.available_resources()
                multi_logger.log_ray_status(
                    mode="train", context="after_code_utils_ray_init"
                )
            except Exception as e:
                print(f"Warning: failed to get ray cluster info: {e}")
                pass
        except Exception as e:
            print(f"Failed to initialize ray: {e}")
            multi_logger.log_ray_status(mode="train", context="code_utils_ray_init_failed")
            return False
    else:
        try:
            import ray  
            from pettingllms.utils.logger_config import get_multi_logger
            multi_logger = get_multi_logger()
            cluster = ray.cluster_resources()
            avail = ray.available_resources()
            
        except Exception as e:
            print(f"Warning: failed to get ray cluster info: {e}")
            pass

    return True







async def _await_ray_object_ref(obj_ref, timeout_seconds: float = 10.0):
    import ray
    import time
    
    start_time = time.time()
    while True:
        ready, _ = ray.wait([obj_ref], timeout=0.1)
        if ready:
            return ray.get(obj_ref)
        
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise asyncio.TimeoutError(f"Ray task timed out after {timeout_seconds}s")
        

        await asyncio.sleep(0.01)


async def test_if_eq(x, y):
    """
    Test equality of two outputs ignoring whitespace differences.
    Based on the reference test_if_eq function provided.
    """
    return " ".join(x.split()) == " ".join(y.split())





def _ensure_ray_initialized() -> bool:
    from pettingllms.utils.logger_config import get_multi_logger
    multi_logger = get_multi_logger()
    import ray  

    if not ray.is_initialized():
        multi_logger.log_ray_status(mode="train", context="test_ray_log_function ")
       
        
        try:
            num_cpus_env = os.getenv("RAY_NUM_CPUS")
            multi_logger.log_ray_status(mode="train", context="before_code_utils_ray_init")
            init_kwargs = dict(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
            )
            if num_cpus_env:
                try:
                    num_cpus = float(num_cpus_env)
                    if num_cpus > 0:
                        init_kwargs["num_cpus"] = num_cpus
                    else:
                        print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
                except (ValueError, TypeError):
                    print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

            # Ensure Ray temp and spill directories
            try:
                # Use /tmp for shorter socket paths (Unix socket path limit is 107 bytes)
                ray_tmp_dir = "/tmp/verl_ray"
                ray_spill_dir = "/tmp/verl_spill"
                os.makedirs(ray_tmp_dir, exist_ok=True)
                os.makedirs(ray_spill_dir, exist_ok=True)

                init_kwargs["_temp_dir"] = ray_tmp_dir
                spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
                init_kwargs["_system_config"] = {
                    "object_spilling_config": json.dumps(spilling_conf)
                }
            except Exception as _e:
                print(f"Warning: failed to prepare Ray temp/spill dirs: {_e}")

            ray.init(**init_kwargs)

            try:
                cluster = ray.cluster_resources()
                avail = ray.available_resources()
                multi_logger.log_ray_status(
                    mode="train", context="after_code_utils_ray_init"
                )
            except Exception as e:
                print(f"Warning: failed to get ray cluster info: {e}")
                pass
        except Exception as e:
            print(f"Failed to initialize ray: {e}")
            multi_logger.log_ray_status(mode="train", context="code_utils_ray_init_failed")
            return False
    else:
        try:
            import ray  
            from pettingllms.utils.logger_config import get_multi_logger
            multi_logger = get_multi_logger()
            cluster = ray.cluster_resources()
            avail = ray.available_resources()
            
        except Exception as e:
            print(f"Warning: failed to get ray cluster info: {e}")
            pass

    return True












async def _worker_docker(
    script: str,
    timeout: float = 40.0,
    image: str = "python:3.11-slim"
) -> str:
    # Ensure base tmp directory exists
    try:
        os.makedirs("tmp", exist_ok=True)
    except Exception:
        pass
    tmpdir = tempfile.mkdtemp(prefix="pllm_exec_", dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    stdout_path = os.path.join(tmpdir, "stdout.txt")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    stdout_file = open(stdout_path, "wb")
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            script_path,
            stdout=stdout_file,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=tmpdir,
            start_new_session=True,
        )

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                # 强制终止进程及其子进程
                if proc.pid:
                    # 终止整个进程组
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    
                    # 强制终止主进程
                    proc.kill()
                    
                    # 等待进程确实结束，但设置短超时
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # 如果还没结束，再次尝试强制终止
                        try:
                            proc.terminate()
                            await asyncio.wait_for(proc.wait(), timeout=1.0)
                        except:
                            pass
            except Exception:
                pass
            finally:
                # 强制清理临时文件，即使进程可能还在运行
                try:
                    if not stdout_file.closed:
                        stdout_file.close()
                    if os.path.exists(tmpdir):
                        try:
                            shutil.rmtree(tmpdir)
                        except Exception:
                            try:
                                subprocess.run(['rm', '-rf', tmpdir], timeout=5, capture_output=True)
                            except Exception:
                                pass
                except Exception:
                    pass
                
            return "timeout"
    finally:
        # 确保文件句柄被关闭
        if not stdout_file.closed:
            stdout_file.close()

    try:
        with open(stdout_path, "rb") as f_out:
            out_bytes = f_out.read()
        result = out_bytes.decode(errors="replace")
    finally:
        # 正常执行完成后强制清理临时文件
        try:
            if os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    try:
                        subprocess.run(['rm', '-rf', tmpdir], timeout=5, capture_output=True)
                    except Exception:
                        pass
        except Exception:
            pass
    
    return result


_RAY_TASK_HANDLE = None  # 缓存 Ray 远程函数句柄



async def test_if_eq(x, y):
    """
    Test equality of two outputs ignoring whitespace differences.
    Based on the reference test_if_eq function provided.
    """
    return " ".join(x.split()) == " ".join(y.split())





async def get_code_execution_output(
    code: str, 
    timeout: float = 40.0,
    ray_actor: Any | None = None,
) -> str:
    """
    Execute Python code and return the output.
    Uses Ray worker for execution with proper timeout handling for concurrent rollouts.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout
        ray_actor: Ray actor for code execution
        
    Returns:
        Code execution output as string
    """
    try:
        if ray_actor is None:
            raise ValueError("ray_actor is required")
        
        # 为大规模并发增加超时缓冲时间
        # 对于500个rollout，Ray调度和执行需要更多时间
        timeout_buffer = max(timeout * 2.0, 30.0)  # 至少30秒缓冲
        total_timeout = timeout + timeout_buffer
        
        #print(f"🔧 执行代码，超时设置: {total_timeout}s (原始: {timeout}s + 缓冲: {timeout_buffer}s)")
        
        # 使用 Ray actor 执行代码，并用 _await_ray_object_ref 处理超时
        obj_ref = ray_actor.run.remote(code, timeout)
        result = await _await_ray_object_ref(obj_ref, total_timeout)
        
        if isinstance(result, str) and result.startswith("error:"):
            print(f"⚠️ Ray执行返回错误: {result}")
        else:
            print(f"✅ Ray执行成功，输出长度: {len(str(result))} 字符")
            
        return result
        
    except asyncio.TimeoutError as e:
        error_msg = f"Ray execution timed out after {total_timeout}s"
        print(f"❌ {error_msg}")
        return f"error: {error_msg}"
    except Exception as e:
        error_msg = f"Ray execution failed: {e}"
        print(f"❌ {error_msg}")
        return f"error: {error_msg}"



def _ensure_ray_initialized() -> bool:
    from pettingllms.utils.logger_config import get_multi_logger
    multi_logger = get_multi_logger()
    import ray  

    if not ray.is_initialized():
        multi_logger.log_ray_status(mode="train", context="test_ray_log_function ")
       
        
        try:
            num_cpus_env = os.getenv("RAY_NUM_CPUS")
            multi_logger.log_ray_status(mode="train", context="before_code_utils_ray_init")
            init_kwargs = dict(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
            )
            if num_cpus_env:
                try:
                    num_cpus = float(num_cpus_env)
                    if num_cpus > 0:
                        init_kwargs["num_cpus"] = num_cpus
                    else:
                        print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
                except (ValueError, TypeError):
                    print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

            # Ensure Ray temp and spill directories
            try:
                # Use /tmp for shorter socket paths (Unix socket path limit is 107 bytes)
                ray_tmp_dir = "/tmp/verl_ray"
                ray_spill_dir = "/tmp/verl_spill"
                os.makedirs(ray_tmp_dir, exist_ok=True)
                os.makedirs(ray_spill_dir, exist_ok=True)

                init_kwargs["_temp_dir"] = ray_tmp_dir
                spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
                init_kwargs["_system_config"] = {
                    "object_spilling_config": json.dumps(spilling_conf)
                }
            except Exception as _e:
                print(f"Warning: failed to prepare Ray temp/spill dirs: {_e}")

            ray.init(**init_kwargs)

            try:
                cluster = ray.cluster_resources()
                avail = ray.available_resources()
                multi_logger.log_ray_status(
                    mode="train", context="after_code_utils_ray_init"
                )
            except Exception as e:
                print(f"Warning: failed to get ray cluster info: {e}")
                pass
        except Exception as e:
            print(f"Failed to initialize ray: {e}")
            multi_logger.log_ray_status(mode="train", context="code_utils_ray_init_failed")
            return False
    else:
        try:
            import ray  
            from pettingllms.utils.logger_config import get_multi_logger
            multi_logger = get_multi_logger()
            cluster = ray.cluster_resources()
            avail = ray.available_resources()
            
        except Exception as e:
            print(f"Warning: failed to get ray cluster info: {e}")
            pass

    return True




def get_ray_docker_worker_cls():
    try:
        import ray  # type: ignore
    except Exception as e:
        print(f"Failed to import ray: {e}")
        return None

    try:
        _ensure_ray_initialized()
    except Exception as e:
        print(f"Failed to ensure ray initialized: {e}")
        return None

    if hasattr(get_ray_docker_worker_cls, "_cls"):
        return getattr(get_ray_docker_worker_cls, "_cls")

    try:
        _max_conc_env = os.getenv("RAY_ACTOR_MAX_CONCURRENCY")
        try:
            _max_conc = int(_max_conc_env) if _max_conc_env else 20
        except (ValueError, TypeError):
            print(f"Warning: invalid RAY_ACTOR_MAX_CONCURRENCY value: {_max_conc_env}, using default 20")
            _max_conc = 20

        # 优化配置：支持500个rollout，每个rollout可能有多个测试用例
        # 使用极少的CPU资源但支持大量并发
        @ray.remote(num_cpus=0.001, max_concurrency=2000)
        class _RayDockerWorker:
            def __init__(self, idx):
                if not isinstance(idx, (int, float)):
                    print(f"Warning: idx parameter is not numeric: {type(idx)}, converting to int")
                    try:
                        self.idx = int(idx) if idx is not None else 0
                    except (ValueError, TypeError):
                        self.idx = 0
                else:
                    self.idx = int(idx)

            def get_idx(self):
                """获取 actor 的索引"""
                return self.idx

            async def run(
                self,
                script: str,
                timeout: float = 40.0,
                image: str = "python:3.11-slim",
            ) -> str:
                """
                Execute Python script and return output.
                
                Args:
                    script: Python script to execute
                    timeout: Execution timeout
                    image: Docker image to use (not used in current implementation)
                    
                Returns:
                    Script execution output as string
                """
                try:
                    return await _worker_docker(
                        script=script,
                        timeout=timeout,
                        image=image,
                    )
                except Exception as e:
                    print(f"RayDockerWorker.run failed: {e}")
                    return f"error: {e}"

        RayDockerWorker = _RayDockerWorker
        setattr(get_ray_docker_worker_cls, "_cls", RayDockerWorker)
        return RayDockerWorker
        
    except Exception as e:
        print(f"Failed to create RayDockerWorker class: {e}")
        return None




# ============ RayDockerWorker 池管理 ============
_RAY_DOCKER_ACTOR_POOL: List[Any] | None = None




def modify(c):
    c = c.replace("plaintext\n", "")
    c = c.replace("\\n", "\n")
    if not c.endswith("\n"):
        c += "\n"
    return c
# ===================TODO: Test case parsing ===================
def extract_test_cases(text: str):
    """
    从包含多组 **Test Input:** / **Test Output:** 代码块的字符串中提取内容。
    返回形如 {"input": [..], "output": [..]} 的字典。
    """
    # 统一换行
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # 支持 ``` 或 ```txt / ```python 等形式的代码块
    input_blocks = re.findall(
        r"\*\*Test Input:\*\*\s*```(?:[a-zA-Z0-9_+\-]*\n)?(.*?)```",
        s, flags=re.DOTALL
    )
    output_blocks = re.findall(
        r"\*\*Test Output:\*\*\s*```(?:[a-zA-Z0-9_+\-]*\n)?(.*?)```",
        s, flags=re.DOTALL
    )

    # 去掉首尾空白，但保留内容中的换行
    test_input = [blk.strip() for blk in input_blocks]
    test_output = [blk.strip() for blk in output_blocks]

    # 对齐长度（防止不等长）
    n = min(len(test_input), len(test_output))
    test_input = test_input[:n]
    test_output = test_output[:n]

    test_action = {"input": test_input, "output": test_output}
    return test_action




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






















def load_math_problem_batch(
    env_indices: List[int],
    dataset_name: str = "train",
    split: str = "train",
    mode: str = "train",
    config: dict = None,
    difficulty: str = "difficult",
    benchmark_name: str = "MATH500",
    validate_samples: int = 8
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
    if mode == "train":
        # 检查config.difficulty是否为train_polaris
        config_difficulty = getattr(config, "difficulty", None) if config else None
        if difficulty == "train_polaris" or config_difficulty == "train_polaris":
            parquet_file = local_datasets_dir / f"train_polaris.parquet"
        else:
            parquet_file = local_datasets_dir / f"train.parquet"
    else:
        parquet_file = local_datasets_dir / f"{benchmark_name}.parquet"
    print(f"📄 目标文件: {parquet_file}")
    
    if mode == "train":
        if not parquet_file.exists():
            raise FileNotFoundError(f"❌ Train mode requires local dataset at {parquet_file}, but file not found!")
        
        print(f"📁 从本地加载数学训练集: {local_datasets_dir}")
        try:
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
            print(f"✅ 数学训练集加载成功，共 {len(ds)} 条")
        except Exception as e:
            raise Exception(f"❌ Failed to load local dataset: {e}")
        
        if len(ds) < len(env_indices):
            raise Exception(f"❌ Local dataset only has {len(ds)} samples, but batch_size is {len(env_indices)}")
        
        indices = random.sample(range(len(ds)), len(env_indices))
        batch_results = []
        
        for i, idx in enumerate(indices):
            example = ds[idx]
            problem_dict = _format_math_problem(example, idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
                print(f"✅ Loaded math train problem {i+1}/{len(env_indices)} (index={idx})")
        
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
        if benchmark_name == "AIME24" or benchmark_name == "AIME25":
            validate_samples = 2
        else:
            validate_samples = 1
        for i, example in enumerate(ds):
            problem_dict = _format_math_problem(example, i, mode="validate")
            if problem_dict:
                for _ in range(validate_samples):
                    batch_results.append(problem_dict)
                    if i % 100 == 0:  # 每100个打印一次进度
                        print(f"🔄 Loaded math validation problem {i+1}*{validate_samples}")
            
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
        question = example.get("question", "")
        solution = example.get("solution", "")
        answer = solution
        
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



"""
Math answer matcher:
- extract_answer: 从文本中抽取候选答案（优先 boxed）
- float_close: 浮点数近似判断（相对/绝对误差）
- symbolic_equal: 符号表达式等价（化简、equals、数值采样兜底）
- math_equal: 统一入口（先解析，再根据类型选择比较策略）
"""

import re
import math
from typing import Optional, Tuple, Union, Iterable
import random

import sympy as sp


# ---------------------------
# 1) 简易“答案抽取器”
# ---------------------------

_BOXED_RE = re.compile(r"\\boxed\s*\{(?P<inner>[^{}]+|\{[^{}]*\})+\}", re.S)

def extract_answer_eval(text: str) -> str:
    """
    从自由文本中抽取一个"最可能"的答案字符串。
    规则：
      - 若存在 \boxed{...}，取最后一个 boxed 内的内容（支持嵌套）
      - 寻找 "答案是"、"答案："、"答案为" 等标记词
      - 寻找数学表达式模式（分数、根号、等式等）
      - 否则取最后一行的最后一个数学片段（简单启发式）
    """
    if not text:
        return ""

    # 1) 优先 \boxed{...} - 取最后一个出现的
    matches = list(_BOXED_RE.finditer(text))
    if matches:
        m = matches[-1]  # 取最后一个匹配
        boxed = m.group(0)
        # 去掉 \boxed{ ... }
        inner = boxed[boxed.find("{")+1: boxed.rfind("}")]
        return inner.strip()

    # 2) 寻找答案标记词
    answer_patterns = [
        r"答案是[:：]\s*([^\n。．.!]+)",
        r"答案为[:：]\s*([^\n。．.!]+)", 
        r"答案[:：]\s*([^\n。．.!]+)",
        r"最终答案[:：]\s*([^\n。．.!]+)",
        r"因此[:：]\s*([^\n。．.!]+)",
        r"所以[:：]\s*([^\n。．.!]+)",
        r"answer\s*[:=]\s*([^\n。．.!]+)",
        r"solution\s*[:=]\s*([^\n。．.!]+)",
        r"result\s*[:=]\s*([^\n。．.!]+)"
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 取最后一个匹配，并清理
            answer = matches[-1].strip()
            # 移除常见的结束词
            answer = re.sub(r"[。．.!\s]+$", "", answer)
            if answer:
                return answer

    # 3) 寻找数学表达式模式
    math_patterns = [
        r"([+-]?\d*\.?\d+/\d+)",  # 分数
        r"([+-]?\d+\.?\d*)",      # 数字
        r"(\\sqrt\{[^}]+\})",     # 根号
        r"(\\frac\{[^}]+\}\{[^}]+\})",  # LaTeX分数
        r"(\([^)]+\))",           # 括号内容
        r"(\[[^\]]+\])"           # 方括号内容
    ]
    
    # 从后往前搜索，优先找到的数学表达式
    for pattern in math_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

    # 4) 回退：取末行的末个"数学片段"
    # 简单策略：最后一行去掉多余空白，取最后一个空格分割的片段
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else text.strip()
    # 去掉句尾标点
    last = re.sub(r"[。．.!\s]+$", "", last)
    # 取最后一个 token
    tokens = last.split()
    return tokens[-1].strip() if tokens else last


# ---------------------------
# 2) 浮点数近似判断
# ---------------------------

def float_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """
    使用 Python 文档定义的近似等式：
    abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    参考: Python docs (math.isclose) 与 PEP 485
    """
    # 与 math.isclose 保持一致的判据
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# ---------------------------
# 3) 符号表达式等价判断
# ---------------------------

_SYMPY_LOCALS = {
    # 允许的一些符号与常量
    "pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "log": sp.log, "ln": sp.log, "exp": sp.exp,
    "sqrt": sp.sqrt, "abs": sp.Abs
}

def _sympify_safe(s: str) -> sp.Expr:
    """
    尝试将字符串解析为 SymPy 表达式。
    做一些轻度规范化：去逗号、中文逗号、前后空白。
    """
    try:
        s = (s or "").strip()
        if not s:  # 空字符串直接返回 None
            return None
        s = s.replace(",", "")  # 千分位逗号
        s = s.replace("，", "")
        # 把形如 "答案: 2/3" 的前缀去掉（非常启发式）
        s = re.sub(r"^[^0-9\-\+\(\[]*:", "", s).strip()
        if not s:  # 处理后变成空字符串
            return None
        return sp.sympify(s, locals=_SYMPY_LOCALS)
    except Exception:
        return None

def _both_numbers(a, b) -> bool:
    """
    检查两个对象是否都是 SymPy 数值类型。
    处理可能的列表、None 或其他非 SymPy 表达式类型。
    """
    try:
        # 检查是否为 None
        if a is None or b is None:
            return False
        
        # 检查是否为列表或其他非 SymPy 表达式类型
        if not hasattr(a, 'is_Number') or not hasattr(b, 'is_Number'):
            return False
            
        return a.is_Number and b.is_Number
    except (AttributeError, TypeError):
        return False

def _num_equal(a, b, rel_tol=1e-9, abs_tol=0.0) -> bool:
    """
    比较两个 SymPy 数值表达式是否相等。
    增加了类型检查以防止非 SymPy 表达式导致的错误。
    """
    try:
        # 确保输入是有效的 SymPy 表达式且有 evalf 方法
        if not hasattr(a, 'evalf') or not hasattr(b, 'evalf'):
            return False
            
        fa = float(a.evalf())  # evalf 以支持如 pi/3 等
        fb = float(b.evalf())
        return float_close(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)
    except (AttributeError, TypeError, ValueError):
        return False

def _simplify_equal(a, b) -> bool:
    """
    先做代数化简：simplify(a-b)==0
    SymPy 官方建议：用 simplify/expand 等看 a-b 是否能化为 0
    增加了类型检查以防止非 SymPy 表达式导致的错误。
    """
    try:
        # 检查输入是否为有效的 SymPy 表达式
        if a is None or b is None:
            return False
        if not hasattr(a, '__sub__') or not hasattr(b, '__sub__'):
            return False
            
        diff = sp.simplify(a - b)
        return diff == 0
    except Exception:
        return False

def _equals_random_samples(a, b, trials: int = 8, domain: Tuple[int, int] = (-7, 7)) -> bool:
    """
    对含符号的表达式，用随机点数值检验作为兜底。
    采用 SymPy 的 equals 思路：在多个点上代入并比较数值是否近似。
    - 避免在可能导致除零的点采样。
    增加了类型检查以防止非 SymPy 表达式导致的错误。
    """
    try:
        # 检查输入是否为有效的 SymPy 表达式
        if a is None or b is None:
            return False
        if not hasattr(a, 'free_symbols') or not hasattr(b, 'free_symbols'):
            return False
        if not hasattr(a, 'subs') or not hasattr(b, 'subs'):
            return False
            
        # 找到自由符号
        free_syms = sorted(list(a.free_symbols.union(b.free_symbols)), key=lambda x: x.name)
    except (AttributeError, TypeError):
        return False
    if not free_syms:
        # 无符号时不该来到这里
        return False

    for _ in range(trials):
        subs_map = {}
        for sym in free_syms:
            # 避免 0/除零等，采样非零整数
            val = 0
            while val == 0:
                val = random.randint(domain[0], domain[1])
            subs_map[sym] = val
        try:
            av = sp.N(a.subs(subs_map))
            bv = sp.N(b.subs(subs_map))
            if not float_close(float(av), float(bv), rel_tol=1e-8, abs_tol=1e-10):
                return False
        except Exception:
            # 遇到奇异点就重试一次
            continue
    return True


def _is_percentage_equivalent(a: sp.Expr, b: sp.Expr, rel_tol: float = 1e-9) -> bool:
    """
    检查两个表达式是否在百分比意义下等价
    例如: 0.5 == 50% == 1/2
    """
    try:
        # 尝试将两个表达式都转换为数值
        val_a = float(a.evalf())
        val_b = float(b.evalf())
        
        # 检查 a 是否等于 b*100 或 b/100
        if float_close(val_a, val_b * 100, rel_tol=rel_tol):
            return True
        if float_close(val_a * 100, val_b, rel_tol=rel_tol):
            return True
            
        return False
    except Exception:
        return False


def _is_scientific_equivalent(a: sp.Expr, b: sp.Expr, rel_tol: float = 1e-9) -> bool:
    """
    检查科学计数法表示是否等价
    例如: 1.5e3 == 1500 == 15*10^2
    """
    try:
        # 直接数值比较
        val_a = float(a.evalf())
        val_b = float(b.evalf())
        return float_close(val_a, val_b, rel_tol=rel_tol)
    except Exception:
        return False


def symbolic_equal(a_expr: Union[str, sp.Expr],
                   b_expr: Union[str, sp.Expr],
                   rel_tol: float = 1e-9,
                   abs_tol: float = 0.0) -> bool:
    """
    符号表达式等价判断：
      1) 解析为 SymPy 表达式
      2) 若都是数值 -> 浮点近似
      3) 尝试 simplify(a-b) == 0
      4) 特殊形式处理（分数、百分比、科学计数法等）
      5) 兜底：随机数值采样 equals（多点）
    """
    try:
        a = _sympify_safe(a_expr) if isinstance(a_expr, str) else a_expr
        b = _sympify_safe(b_expr) if isinstance(b_expr, str) else b_expr
    except Exception:
        return False

    # 检查解析结果是否有效
    if a is None or b is None:
        return False

    # 都是数字 -> 用浮点近似
    if _both_numbers(a, b):
        return _num_equal(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # 特殊形式处理：百分比比较
    if _is_percentage_equivalent(a, b, rel_tol):
        return True

    # 特殊形式处理：科学计数法
    if _is_scientific_equivalent(a, b, rel_tol):
        return True

    # 尝试代数化简
    if _simplify_equal(a, b):
        return True

    # 兜底：数值采样 equals
    return _equals_random_samples(a, b)


# ---------------------------
# 4) 统一入口
# ---------------------------
def normalize_math(expr: str) -> str:
    """
    数学表达式标准化处理：
    - 移除LaTeX格式和环境
    - 标准化常见数学符号
    - 处理单位和百分比
    - 统一空白字符处理
    """
    if not expr:
        return expr
        
    # 去除 $$…$$
    expr = re.sub(r'(\$\$)(?:(?!\1)[\s\S])*\1',
                  lambda m: m.group(0)[2:-2], expr)
    # 去除 $…$
    expr = re.sub(r'(\$)(?:(?!\1)[\s\S])*\1',
                  lambda m: m.group(0)[1:-1], expr)
    
    # 清理 LaTeX 环境及定界符
    expr = re.sub(r'\\begin\{.*?\}|\\end\{.*?\}', '', expr)
    expr = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', expr)
    expr = expr.replace("\\\\", "")  # 去除换行命令 \\
    
    # 标准化常见符号
    expr = expr.replace("×", "*")
    expr = expr.replace("÷", "/") 
    expr = expr.replace("·", "*")
    expr = expr.replace("∙", "*")
    expr = expr.replace("−", "-")  # 数学减号转为ASCII减号
    
    # 处理百分比符号
    expr = re.sub(r'(\d+(?:\.\d+)?)%', r'\1/100', expr)  # 50% -> 50/100
    expr = re.sub(r'(\d+(?:\.\d+)?)\\%', r'\1/100', expr)  # 50\% -> 50/100
    
    # 处理常见单位（移除）
    units = ['cm', 'mm', 'm', 'km', 'kg', 'g', 'mg', 's', 'min', 'h', 'hour', 'day', 
             'degree', 'degrees', '°', '℃', '℉', 'inch', 'ft', 'feet', 'yard', 'mile']
    for unit in units:
        expr = re.sub(rf'\b{re.escape(unit)}s?\b', '', expr, flags=re.IGNORECASE)
    
    # 处理科学计数法：1.5e3 -> 1.5*10^3
    expr = re.sub(r'(\d+(?:\.\d+)?)e([+-]?\d+)', r'\1*10^(\2)', expr, flags=re.IGNORECASE)
    
    # 标准化根号：√ -> sqrt
    expr = expr.replace("√", "sqrt")
    
    # 处理分数线：确保分数格式正确
    expr = re.sub(r'(\d+)/(\d+)', r'(\1)/(\2)', expr)  # 3/4 -> (3)/(4)
    
    # **关键新增**：去除全部空白字符
    expr = re.sub(r"\s+", "", expr)  # 包括空格、制表符、换行等
    return expr

def evaluate_math_solution(pred_text: str,
               gold_text: str,
               rel_tol: float = 1e-9,
               abs_tol: float = 0.0) -> bool:
    """
    统一入口：给定原始文本（模型输出 / 参考答案），
    - 先抽取候选答案
    - 再做符号等价判断（内部会处理数值/符号两种情况）
    """
    pred = extract_answer_eval(pred_text)
    pred = normalize_math(pred)
    gold = extract_answer_eval(gold_text)
    gold = normalize_math(gold)
    return symbolic_equal(pred, gold, rel_tol=rel_tol, abs_tol=abs_tol)

# Test function
def test_load_math_problems(batch_size: int = 5):
    """Test loading math problems"""
    results = load_math_problem_batch(env_indices=list(range(batch_size)), mode="train",difficulty="train_polaris")
    for i, result in enumerate(results):
        print(f"\n--- Problem {i+1} ---")
        print(f"Problem: {result['question']}")
        print(f"Answer: {result['solution']}")


if __name__ == "__main__":
    print("Testing math problem loading...")
    test_load_math_problems(3)
