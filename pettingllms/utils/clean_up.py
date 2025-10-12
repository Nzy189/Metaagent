import aiohttp
import atexit
import signal
import sys
import shutil
import subprocess
import os
import time
from pathlib import Path

import ray


_CLEANED = False


def force_kill_ray_processes():
    """Force kill all remaining Ray processes to free GPU memory"""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        ray_pids = []
        for line in result.stdout.splitlines():
            if "ray::" in line or "ray_" in line or "/ray/" in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        ray_pids.append(pid)
                    except (ValueError, IndexError):
                        continue
        
        if ray_pids:
            print(f"Found {len(ray_pids)} remaining Ray processes, cleaning up...")
            
            # Try graceful shutdown first (SIGTERM)
            for pid in ray_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, Exception):
                    pass
            
            time.sleep(2)
            
            # Force kill remaining processes (SIGKILL)
            for pid in ray_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, Exception):
                    pass
            
            print(f"Cleaned up {len(ray_pids)} Ray processes")
    except Exception:
        pass


def cleanup_ray_and_tmp_dirs():
    """Clean up Ray and temporary directories"""
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True
    
    print("\nCleaning up Ray resources and temporary files...")
    
    # Graceful Ray shutdown
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    
    # Force kill remaining Ray processes
    force_kill_ray_processes()
    
    # Clean temporary directories
    for tmp_path in ["/tmp/verl_ray", "/tmp/verl_spill"]:
        try:
            tmp_dir = Path(tmp_path)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
    
    print("Cleanup completed\n")


def install_cleanup_hooks():
    """Install cleanup hooks for normal exit, Ctrl+C, and exceptions"""
    atexit.register(cleanup_ray_and_tmp_dirs)

    def _signal_handler(signum, frame):
        print("\nReceived interrupt signal, cleaning up...")
        try:
            cleanup_ray_and_tmp_dirs()
        finally:
            sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _signal_handler)
        except Exception:
            pass

    _orig_excepthook = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        try:
            cleanup_ray_and_tmp_dirs()
        finally:
            _orig_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

