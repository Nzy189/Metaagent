import atexit
import signal
import sys
import shutil
from pathlib import Path

import ray


_CLEANED = False


def cleanup_ray_and_tmp_dirs():
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    for tmp_path in ["/tmp/verl_ray", "/tmp/verl_spill"]:
        try:
            tmp_dir = Path(tmp_path)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
    print("Cleaned up temporary files")


def install_cleanup_hooks():
    atexit.register(cleanup_ray_and_tmp_dirs)

    def _signal_handler(signum, frame):
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


