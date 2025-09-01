"""
Simple timer utility for debugging and performance monitoring.
"""
import time
from typing import Dict, Optional

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

# Global timer instance for easy access
_global_timer = SimplerTimer("Global")

def timer_start(msg: str = "Started") -> None:
    """Start global timer"""
    _global_timer.start(msg)

def timer_checkpoint(msg: str) -> float:
    """Global timer checkpoint"""
    return _global_timer.checkpoint(msg)

def timer_end(msg: str = "Completed") -> float:
    """End global timer"""
    return _global_timer.end(msg)

def create_timer(name: str, enable: bool = True) -> SimplerTimer:
    """Create a new timer instance"""
    return SimplerTimer(name, enable)
