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

import datetime
import inspect
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from codetiming import Timer



def log_print(ctn: Any):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split("/")[-1]
    print(f"[{current_time}-{file_name}:{line_number}:{function_name}]: {ctn}")


def _timer(name: str, timing_raw: dict[str, float]):
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def simple_timer(name: str, timing_raw: dict[str, float]):
    """Context manager for basic timing without NVTX markers.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with platform markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds platform markers for profiling.
    This function is a default implementation when hardware profiler is not available.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the marker. Defaults to None.
        domain (Optional[str]): Domain for the marker. Defaults to None.
        category (Optional[str]): Category for the marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


def reduce_timing(timing_raw: dict[str, float]) -> dict[str, float]:
    """Reduce timing information across all processes.

    This function uses distributed communication to gather and sum the timing
    information from all processes in a distributed environment.

    Args:
        timing_raw (Dict[str, float]): Dictionary containing timing information.

    Returns:
        Dict[str, float]: Reduced timing information.
    """
    if not dist.is_initialized():
        return timing_raw

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])
    
    # Get the current device
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    timing_list = torch.tensor(timing_list, dtype=torch.float32, device=device)
    torch.distributed.all_reduce(timing_list, op=torch.distributed.ReduceOp.AVG)
    timing_list = [tensor.item() for tensor in timing_list.to("cpu")]
    timing_generate = {key_list[i]: timing_list[i] for i in range(len(key_list))}
    return timing_generate

def colorful_print(text, color="white"):
    """Simple colorful print function for debugging"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")
class SimplerTimer:
    """Simple timer for debugging with minimal overhead"""
    
    def __init__(self, name: str = "Timer", enable: bool = True):
        self.name = name
        self.enable = enable
        self.start_time = None
        self.last_time = None
        self.checkpoints: Dict[str, float] = {}
    
    def start(self, msg: str = "Started") -> None:
        """Start timing"""
        if not self.enable:
            return
        self.start_time = time.time()
        self.last_time = self.start_time
        print(f"[{self.name}] {msg} at {time.strftime('%H:%M:%S', time.localtime(self.start_time))}")
    
    def checkpoint(self, msg: str, reset_last: bool = True) -> float:
        """Print checkpoint time"""
        if not self.enable:
            return 0.0
        
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            self.last_time = current_time
        
        elapsed_total = current_time - self.start_time
        elapsed_since_last = current_time - (self.last_time or self.start_time)
        
        self.checkpoints[msg] = current_time
        
        print(f"[{self.name}] {msg} | +{elapsed_since_last:.2f}s | Total: {elapsed_total:.2f}s")
        
        if reset_last:
            self.last_time = current_time
        
        return elapsed_since_last
    
    def end(self, msg: str = "Completed") -> float:
        """End timing and print total"""
        if not self.enable:
            return 0.0
        
        if self.start_time is None:
            print(f"[{self.name}] Timer was never started")
            return 0.0
        
        current_time = time.time()
        total_elapsed = current_time - self.start_time
        
        print(f"[{self.name}] {msg} | Total: {total_elapsed:.2f}s")
        return total_elapsed
    
    def reset(self) -> None:
        """Reset timer"""
        self.start_time = None
        self.last_time = None
        self.checkpoints.clear()


def create_timer(name: str, enable: bool = True) -> SimplerTimer:
    """Create a new timer instance"""
    return SimplerTimer(name, enable)