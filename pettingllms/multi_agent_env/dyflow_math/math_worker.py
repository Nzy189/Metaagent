"""
Worker utilities for code execution in mathematical problem solving.

This module contains Ray and Docker worker implementations for safe
code execution with proper timeout handling and resource management.
"""

import os
import json
import asyncio
import signal
import subprocess
import shutil
import tempfile
import time
import re
from typing import Any, List

import ray


_RAY_TASK_HANDLE = None
_RAY_DOCKER_ACTOR_POOL: List[Any] | None = None


async def _await_ray_object_ref(obj_ref, timeout_seconds: float = 10.0):
    """
    Await a Ray object reference with timeout.
    
    Args:
        obj_ref: Ray object reference to await
        timeout_seconds: Maximum time to wait
        
    Returns:
        Result from Ray task
        
    Raises:
        asyncio.TimeoutError: If task exceeds timeout
    """
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


def _ensure_ray_initialized() -> None:
    """
    Ensure Ray is initialized with proper configuration.
    
    Initializes Ray if not already running, with custom temp directories
    and resource configurations from environment variables.
    """
    import ray  

    if ray.is_initialized():
        return
    

    init_kwargs = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "logging_level": "ERROR",
    }

    num_cpus_env = os.getenv("RAY_NUM_CPUS")
    if num_cpus_env:
        try:
            num_cpus = float(num_cpus_env)
            if num_cpus > 0:
                init_kwargs["num_cpus"] = num_cpus
            else:
                print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
        except (ValueError, TypeError):
            print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

    ray_tmp_dir = "/tmp/verl_ray"
    ray_spill_dir = "/tmp/verl_spill"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)
    
    init_kwargs["_temp_dir"] = ray_tmp_dir
    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    init_kwargs["_system_config"] = {
        "object_spilling_config": json.dumps(spilling_conf)
    }
    
    ray.init(**init_kwargs)


async def _worker_docker(
    script: str,
    timeout: float = 40.0,
    image: str = "python:3.11-slim"
) -> str:
    """
    Execute Python script in isolated environment with timeout.
    
    Creates a temporary directory, writes the script, executes it in a subprocess,
    and returns the output. Handles timeout and cleanup properly.
    
    Args:
        script: Python script content to execute
        timeout: Maximum execution time in seconds
        image: Docker image name (currently unused, kept for compatibility)
        
    Returns:
        Script output as string, or "timeout" if execution exceeds timeout
    """
    os.makedirs("tmp", exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="pllm_exec_", dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    stdout_path = os.path.join(tmpdir, "stdout.txt")
    
    proc = None
    stdout_file = None
    result = "timeout"
    
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        
        stdout_file = open(stdout_path, "wb")
        proc = await asyncio.create_subprocess_exec(
            "python",
            script_path,
            stdout=stdout_file,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=tmpdir,
            start_new_session=True,
        )
        
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        
        stdout_file.close()
        stdout_file = None
        
        with open(stdout_path, "rb") as f_out:
            out_bytes = f_out.read()
        result = out_bytes.decode(errors="replace")
        
    except asyncio.TimeoutError:
        if proc and proc.pid:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        result = "timeout"
        
    finally:
        if stdout_file and not stdout_file.closed:
            stdout_file.close()
        
        if proc and proc.returncode is None:
            if proc.pid:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            if os.path.exists(tmpdir):
                subprocess.run(['rm', '-rf', tmpdir], timeout=5, capture_output=True)
    
    return result


def get_ray_docker_worker_cls():
    """
    Get or create the Ray Docker worker class.
    
    Returns a Ray remote actor class that can execute Python scripts
    with timeout handling. Uses caching to avoid recreating the class.
    
    Returns:
        Ray remote actor class for code execution
    """
    _ensure_ray_initialized()

    if hasattr(get_ray_docker_worker_cls, "_cls"):
        return getattr(get_ray_docker_worker_cls, "_cls")

    @ray.remote(num_cpus=0.001, max_concurrency=2000)
    class _RayDockerWorker:
        def __init__(self, idx):
            if isinstance(idx, (int, float)):
                self.idx = int(idx)
            elif isinstance(idx, str) and re.fullmatch(r"\s*-?\d+\s*", idx):
                self.idx = int(idx)
            else:
                self.idx = 0

        def get_idx(self):
            """Get the actor's index"""
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
            return await _worker_docker(
                script=script,
                timeout=timeout,
                image=image,
            )

    RayDockerWorker = _RayDockerWorker
    setattr(get_ray_docker_worker_cls, "_cls", RayDockerWorker)
    return RayDockerWorker


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
        timeout: Execution timeout in seconds
        ray_actor: Ray actor for code execution
        
    Returns:
        Code execution output as string, or error message if execution fails
        
    Raises:
        ValueError: If ray_actor is None
    """
    try:
        if ray_actor is None:
            raise ValueError("ray_actor is required")
        
        timeout_buffer = max(timeout * 2.0, 30.0)
        total_timeout = timeout + timeout_buffer
        
        obj_ref = ray_actor.run.remote(code, timeout)
        result = await _await_ray_object_ref(obj_ref, total_timeout)
        
        if isinstance(result, str) and result.startswith("error:"):
            print(f"Warning: Ray execution returned error: {result}")
        else:
            print(f"Success: Ray execution completed, output length: {len(str(result))} characters")
            
        return result
        
    except asyncio.TimeoutError as e:
        error_msg = f"Ray execution timed out after {total_timeout}s"
        print(f"Error: {error_msg}")
        return f"error: {error_msg}"
    except Exception as e:
        error_msg = f"Ray execution failed: {e}"
        print(f"Error: {error_msg}")
        return f"error: {error_msg}"

