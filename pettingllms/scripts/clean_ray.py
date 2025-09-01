import os
import time
import subprocess
import tempfile
import shutil

# 设置临时目录环境变量，确保 Python 可以找到可用的临时目录
def setup_temp_dir():
    """设置临时目录环境变量"""
    temp_dirs = ['tmp']
    if not os.path.exists('tmp'):
        try:
            os.makedirs('tmp', mode=0o777, exist_ok=True)
            print("Created tmp directory")
        except Exception as e:
            print(f"Failed to create tmp: {e}")
    
    # 设置权限
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                os.chmod(temp_dir, 0o777)
                os.environ['TMPDIR'] = temp_dir
                os.environ['TMP'] = temp_dir
                os.environ['TEMP'] = temp_dir
                print(f"Set temporary directory to: {temp_dir}")
                break
            except Exception as e:
                print(f"Failed to set permissions for {temp_dir}: {e}")
                continue


setup_temp_dir()

# 现在可以安全地导入其他模块
try:
    import ray
except ImportError:
    print("Ray is not installed, skipping ray import")
    ray = None

def force_kill_ray_processes():
    """强制杀死所有 Ray 进程"""
    try:
        print("Force killing Ray processes...")
        # 杀死所有 Ray 相关进程
        commands = [
            ['pkill', '-9', '-f', 'ray'],
            ['pkill', '-9', '-f', 'raylet'],
            ['pkill', '-9', '-f', 'python.*ray'],
            ['pkill', '-9', '-f', 'gcs_server'],
            ['pkill', '-9', '-f', 'dashboard'],
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=5)
            except:
                pass
        
        print("Force killed all Ray processes")
    except Exception as e:
        print(f"Error force killing Ray processes: {e}")


def force_kill_pllm_processes():
    """强制杀死所有与 /tmp/pllm 相关的进程"""
    try:
        print("Force killing /tmp/pllm related processes...")
        # 使用 pkill -f 匹配命令行中包含 /tmp/pllm 的进程
        try:
            subprocess.run(['pkill', '-9', '-f', '/tmp/pllm'], capture_output=True, timeout=5)
        except Exception:
            pass
        # 使用 lsof 查找占用 /tmp/pllm 的进程并强制结束（best-effort）
        try:
            subprocess.run(['bash', '-lc', "lsof +D /tmp/pllm 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r -n1 kill -9"], capture_output=True, timeout=10)
        except Exception:
            pass
        print("Force killed /tmp/pllm related processes")
    except Exception as e:
        print(f"Error force killing /tmp/pllm processes: {e}")


def cleanup_ray_directory():
    """清理 /tmp/ray/ 目录下的所有文件和子目录"""
    ray_tmp_dir = "/tmp/ray"
    
    try:
        print(f"Cleaning up Ray directory: {ray_tmp_dir}")
        
        if not os.path.exists(ray_tmp_dir):
            print(f"Ray directory {ray_tmp_dir} does not exist, nothing to clean")
            return
        
        # 首先尝试使用 lsof 查找并杀死占用 /tmp/ray 的进程
        try:
            print("Killing processes using /tmp/ray...")
            subprocess.run(['bash', '-c', f"lsof +D {ray_tmp_dir} 2>/dev/null | awk 'NR>1 {{print $2}}' | sort -u | xargs -r -n1 kill -9"], 
                         capture_output=True, timeout=10)
            time.sleep(1)  # 等待进程完全退出
        except Exception:
            pass
        
        # 递归删除所有文件和目录
        try:
            # 使用 shutil.rmtree 删除整个目录树
            if os.path.exists(ray_tmp_dir):
                shutil.rmtree(ray_tmp_dir, ignore_errors=True)
                print(f"✓ Removed {ray_tmp_dir} directory tree")
            
            # 如果 shutil.rmtree 失败，使用系统命令强制删除
            if os.path.exists(ray_tmp_dir):
                subprocess.run(['rm', '-rf', ray_tmp_dir], capture_output=True, timeout=30)
                print(f"✓ Force removed {ray_tmp_dir} using rm -rf")
            
        except Exception as e:
            print(f"✗ Failed to remove {ray_tmp_dir}: {e}")
            # 最后尝试：逐个删除文件
            try:
                for root, dirs, files in os.walk(ray_tmp_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                        except:
                            pass
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except:
                            pass
                try:
                    os.rmdir(ray_tmp_dir)
                except:
                    pass
                print(f"✓ Cleaned up {ray_tmp_dir} file by file")
            except Exception as e2:
                print(f"✗ Final cleanup attempt failed: {e2}")
        
        # 验证清理结果
        if not os.path.exists(ray_tmp_dir):
            print(f"✓ Successfully cleaned {ray_tmp_dir}")
        else:
            remaining_items = []
            try:
                for item in os.listdir(ray_tmp_dir):
                    remaining_items.append(item)
                if remaining_items:
                    print(f"⚠ Some items remain in {ray_tmp_dir}: {remaining_items[:5]}...")
                else:
                    print(f"✓ {ray_tmp_dir} is empty")
            except Exception:
                print(f"⚠ Cannot access {ray_tmp_dir} after cleanup")
                
    except Exception as e:
        print(f"Error cleaning Ray directory: {e}")


def cleanup_ray():
    """清理 Ray 资源 - 强制版本"""
    print("\n" + "="*50)
    print("STARTING RAY CLEANUP...")
    print("="*50)
    
    try:
        # 方法1: 正常关闭
        if ray and ray.is_initialized():
            print("Step 1: Attempting normal Ray shutdown...")
            try:
                ray.shutdown()
                print("✓ Normal Ray shutdown completed.")
                time.sleep(2)  # 等待进程完全关闭
            except Exception as e:
                print(f"✗ Normal Ray shutdown failed: {e}")
        else:
            print("Ray is not initialized or not available, but will force cleanup anyway...")
    except Exception as e:
        print(f"Error checking Ray status: {e}")
    
    # 方法2: 强制杀死进程
    try:
        print("Step 2: Force killing Ray processes...")
        force_kill_ray_processes()
        time.sleep(1)
    except Exception as e:
        print(f"Error in force kill: {e}")
    
    # 方法2b: 强制杀死 /tmp/pllm 相关进程
    try:
        print("Step 2b: Force killing /tmp/pllm related processes...")
        force_kill_pllm_processes()
        time.sleep(1)
    except Exception as e:
        print(f"Error in force kill /tmp/pllm: {e}")
    
    # 方法3: 清理 /tmp/ray/ 目录
    try:
        print("Step 3: Cleaning /tmp/ray/ directory...")
        cleanup_ray_directory()
        time.sleep(1)
    except Exception as e:
        print(f"Error cleaning Ray directory: {e}")
    
    # 方法4: 清理环境变量
    try:
        print("Step 4: Cleaning Ray environment variables...")
        ray_env_vars = [key for key in os.environ.keys() if key.startswith('RAY_')]
        for var in ray_env_vars:
            del os.environ[var]
        print(f"Cleared {len(ray_env_vars)} Ray environment variables")
    except Exception as e:
        print(f"Error cleaning environment: {e}")
    
    print("="*50)
    print("RAY CLEANUP COMPLETED")
    print("="*50)


if __name__ == "__main__":
    cleanup_ray()