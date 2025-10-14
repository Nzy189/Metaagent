import os
import time
import subprocess
import tempfile
import shutil

# 设置临时目录环境变量，确保 Python 可以找到可用的临时目录
def setup_temp_dir():
    """设置临时目录环境变量"""
    temp_dirs = ['./tmp']
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
    """强制杀死所有 Ray 及相关训练进程（多轮次、尽可能覆盖）。"""
    try:
        print("Force killing Ray processes...")
        # 更全面的匹配模式
        patterns = [
            'ray', 'raylet', 'python.*ray', 'gcs_server', 'dashboard', 'plasma_store',
            'WorkerDict', 'ActorRolloutRefWorker', 'AsyncActorRolloutRefWorker',
            'pettingllms.trainer.train', 'pettingllms/trainer/train.py', 'train.py',
            'default_worker.py', 'worker.py', 'ray::'
        ]
        # 多轮尝试，避免竞争条件
        for _ in range(3):
            for pat in patterns:
                try:
                    subprocess.run(['pkill', '-9', '-f', pat], capture_output=True, timeout=5)
                except Exception:
                    pass
            # 同时杀掉占用项目 tmp 的进程
            try:
                proj_tmp = os.path.join(os.getcwd(), 'tmp')
                cmd = f"lsof +D {proj_tmp} 2>/dev/null | awk 'NR>1 {{print $2}}' | sort -u"
                res = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=10)
                pids = [p for p in res.stdout.strip().split('\n') if p]
                if pids:
                    subprocess.run(['bash', '-lc', f"kill -9 {' '.join(pids)}"], timeout=10)
            except Exception:
                pass
            time.sleep(1)
        print("Force killed all Ray-related processes (best-effort)")
    except Exception as e:
        print(f"Error force killing Ray processes: {e}")


def force_kill_high_memory_processes():
    """强制杀死高内存占用的Python进程"""
    try:
        print("Checking for high memory Python processes...")
        # 获取所有Python进程的内存使用情况
        result = subprocess.run(['ps', 'aux', '--sort=-rss'], 
                               capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
            killed_count = 0
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    memory_kb = parts[5]
                    command = ' '.join(parts[10:])
                    
                    # 检查是否是高内存的Python进程（超过1GB）
                    try:
                        memory_mb = int(memory_kb) / 1024
                        if (memory_mb > 1024 and 
                            ('python' in command.lower() or 'ray::' in command) and
                            ('pettingllms' in command or 'ray::' in command)):
                            
                            print(f"Killing high memory process (PID: {pid}, Memory: {memory_mb:.1f}MB): {command[:100]}...")
                            try:
                                subprocess.run(['kill', '-9', pid], timeout=2)
                                killed_count += 1
                            except:
                                pass
                    except (ValueError, IndexError):
                        continue
            
            print(f"Killed {killed_count} high memory processes")
        
    except Exception as e:
        print(f"Error killing high memory processes: {e}")


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
    """强制清理项目 tmp 下所有 ray 相关与执行残留目录，并兜底清理 /tmp/ray。"""
    project_root = os.getcwd()
    project_tmp = os.path.join(project_root, 'tmp')
    ray_tmp = os.path.join(project_root, 'tmp', 'ray_tmp')
    ray_spill = os.path.join(project_root, 'tmp', 'ray_spill')
    sys_tmp_ray = '/tmp/ray'

    dirs_to_clean = [project_tmp, ray_tmp, ray_spill, sys_tmp_ray]

    def print_dir_size(path: str):
        try:
            if os.path.exists(path):
                result = subprocess.run(['du', '-sh', path], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    size = result.stdout.strip().split('\t')[0]
                    print(f" - {path} 大小: {size}")
        except Exception:
            pass

    print("即将清理以下目录（存在的才会处理）：")
    for d in dirs_to_clean:
        print_dir_size(d)

    # 第一步：尝试释放目录权限并杀掉占用句柄的进程
    for d in dirs_to_clean:
        if not os.path.exists(d):
            continue
        try:
            try:
                subprocess.run(['chmod', '-R', 'u+w', d], timeout=10, capture_output=True)
            except Exception:
                pass
            cmd = f"lsof +D {d} 2>/dev/null | awk 'NR>1 {{print $2}}' | sort -u"
            res = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=10)
            pids = [p for p in res.stdout.strip().split('\n') if p]
            if pids:
                print(f" - 发现占用 {d} 的进程: {', '.join(pids)}, 执行 kill -9...")
                subprocess.run(['bash', '-lc', f"kill -9 {' '.join(pids)}"], timeout=10)
        except Exception as e:
            print(f" - 终止占用 {d} 的进程时出错: {e}")

    # 第二步：逐子项删除，确保 tmp 下所有内容被删（包含隐藏项）
    def wipe_children(dir_path: str):
        if not os.path.isdir(dir_path):
            return
        try:
            for name in os.listdir(dir_path):
                child = os.path.join(dir_path, name)
                try:
                    if os.path.isdir(child) and not os.path.islink(child):
                        try:
                            shutil.rmtree(child, ignore_errors=False)
                        except Exception:
                            subprocess.run(['rm', '-rf', child], timeout=60, capture_output=True)
                    else:
                        try:
                            os.unlink(child)
                        except Exception:
                            subprocess.run(['rm', '-f', child], timeout=30, capture_output=True)
                except Exception:
                    pass
        except Exception:
            pass

    # 对项目 tmp 目录做逐子项删除（保留 tmp 自身）
    if os.path.isdir(project_tmp):
        wipe_children(project_tmp)
    else:
        try:
            os.makedirs(project_tmp, exist_ok=True)
        except Exception:
            pass

    # 对 ray_tmp 与 ray_spill 做彻底删除并重建
    for d in [ray_tmp, ray_spill]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d, ignore_errors=False)
            except Exception:
                try:
                    subprocess.run(['rm', '-rf', d], timeout=60, capture_output=True)
                except Exception:
                    pass
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    # 对 /tmp/ray 做彻底删除并重建
    if os.path.exists(sys_tmp_ray):
        try:
            shutil.rmtree(sys_tmp_ray, ignore_errors=False)
        except Exception:
            try:
                subprocess.run(['rm', '-rf', sys_tmp_ray], timeout=60, capture_output=True)
            except Exception:
                pass
    try:
        os.makedirs(sys_tmp_ray, exist_ok=True)
    except Exception:
        pass

    # 额外清理 tmp 下的 pllm_exec_* 目录（再次兜底）
    try:
        exec_glob_cmd = "find ./tmp -mindepth 1 -maxdepth 1 -type d -name 'pllm_exec_*' -print0 | xargs -0 -r rm -rf"
        subprocess.run(['bash', '-lc', exec_glob_cmd], timeout=30, capture_output=True)
    except Exception:
        pass

    # 最后兜底：使用 dotglob 清空 tmp 下所有子项
    try:
        bash_clear = "shopt -s dotglob nullglob; rm -rf ./tmp/*"
        subprocess.run(['bash', '-lc', bash_clear], timeout=30, capture_output=True)
    except Exception:
        pass

    print("清理后目录大小：")
    for d in dirs_to_clean:
        print_dir_size(d)


def check_system_memory():
    """检查系统内存使用情况"""
    try:
        print("检查系统内存状况...")
        result = subprocess.run(['free', '-h'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("系统内存状况:")
            print(result.stdout)
        
        # 检查最占内存的进程
        result = subprocess.run(['ps', 'aux', '--sort=-rss', '--no-headers'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[:10]  # 前10个最占内存的进程
            print("内存占用最高的10个进程:")
            for line in lines:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    memory_kb = parts[5]
                    memory_mb = int(memory_kb) / 1024
                    command = ' '.join(parts[10:])[:50]
                    print(f"  PID: {pid}, Memory: {memory_mb:.1f}MB, Command: {command}")
    except Exception as e:
        print(f"检查内存状况失败: {e}")
    
def cleanup_ray():
    """清理 Ray 资源 - 增强版本"""
    print("\n" + "="*50)
    print("STARTING ENHANCED RAY CLEANUP...")
    print("="*50)
    
    # 步骤0: 检查系统内存状况
    try:
        print("Step 0: Checking system memory...")
        check_system_memory()
        time.sleep(1)
    except Exception as e:
        print(f"Error checking memory: {e}")
    
    # 步骤1: 清理高内存进程
    try:
        print("Step 1: Killing high memory processes...")
        force_kill_high_memory_processes()
        time.sleep(2)
    except Exception as e:
        print(f"Error killing high memory processes: {e}")
    
    # 步骤2: 正常关闭Ray
    try:
        if ray and ray.is_initialized():
            print("Step 2: Attempting normal Ray shutdown...")
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
    
    # 步骤3: 强制杀死Ray进程
    try:
        print("Step 3: Force killing Ray processes...")
        force_kill_ray_processes()
        time.sleep(2)
    except Exception as e:
        print(f"Error in force kill: {e}")
    
    # 步骤4: 清理pllm相关进程
    try:
        print("Step 4: Killing pllm related processes...")
        force_kill_pllm_processes()
        time.sleep(1)
    except Exception as e:
        print(f"Error killing pllm processes: {e}")
    
    # 步骤5: 清理临时目录
    try:
        print("Step 5: Cleaning temporary directories...")
        cleanup_ray_directory()
        time.sleep(1)
    except Exception as e:
        print(f"Error cleaning Ray directory: {e}")
    
    # 步骤6: 清理环境变量
    try:
        print("Step 6: Cleaning Ray environment variables...")
        ray_env_vars = [key for key in os.environ.keys() if key.startswith('RAY_')]
        for var in ray_env_vars:
            del os.environ[var]
        print(f"Cleared {len(ray_env_vars)} Ray environment variables")
    except Exception as e:
        print(f"Error cleaning environment: {e}")
    
    # 步骤7: 最终内存检查
    try:
        print("Step 7: Final memory check...")
        check_system_memory()
    except Exception as e:
        print(f"Error in final memory check: {e}")
    
    print("="*50)
    print("ENHANCED RAY CLEANUP COMPLETED")
    print("="*50)


if __name__ == "__main__":
    cleanup_ray()