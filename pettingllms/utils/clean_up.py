import atexit
import signal
import sys
import shutil
import subprocess
from pathlib import Path
import ray


_CLEANED = False
_TEMP_DIRS = []


def register_temp_dirs(*dirs):
    """Register temporary directories to be cleaned up"""
    _TEMP_DIRS.extend(dirs)


def kill_ray_processes():
    """Kill all Ray-related processes"""
    patterns = [
        "ray::",
        "raylet",
        "gcs_server",
        "plasma_store",
        "default_worker.py",
        "worker.py",
    ]
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass


def cleanup_ray():
    """Clean up Ray resources and temporary directories for current session"""
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True
    
    print("\nCleaning up Ray session...")
    
    if ray.is_initialized():
        ray.shutdown()
    
    for temp_dir in _TEMP_DIRS:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Removed temporary directory: {temp_dir}")
    
    print("Cleanup completed\n")


def cleanup_ray_runtime():
    """Complete Ray cleanup including process termination"""
    cleanup_ray()
    kill_ray_processes()


def install_cleanup_hooks():
    """Install cleanup hooks for normal exit and signals"""
    atexit.register(cleanup_ray)
    
    def signal_handler(signum, frame):
        cleanup_ray()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

