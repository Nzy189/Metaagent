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
            ['pkill', '-9', '-f', 'WorkerDict'],
            ['pkill', '-9', '-f', 'train_multi_agents'],
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                if result.returncode == 0:
                    print(f"✓ Killed processes matching: {' '.join(cmd[3:])}")
            except:
                pass
        
        print("Force killed all Ray processes")
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
    # 自动检测项目根目录中的tmp文件夹
    # 当前工作目录应该是项目根目录
    ray_tmp_dir = os.path.join(os.getcwd(), "tmp")
    
    # 检查目录大小
    try:
        if os.path.exists(ray_tmp_dir):
            result = subprocess.run(['du', '-sh', ray_tmp_dir], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                size = result.stdout.strip().split('\t')[0]
                print(f"临时目录大小: {size}")
    except:
        pass
    
    try:
        # 对于大量文件，最快的方法是重新创建目录
        print(f"开始超快速清理: {ray_tmp_dir}")
        
        # 方法1: 重新创建目录（最快）
        if os.path.exists(ray_tmp_dir):
            # 创建临时目录名
            temp_name = ray_tmp_dir + "_temp_" + str(int(time.time()))
            
            # 重命名原目录（瞬间完成）
            os.rename(ray_tmp_dir, temp_name)
            
            # 立即创建新的空目录
            os.makedirs(ray_tmp_dir, exist_ok=True)
            
            # 在后台异步删除旧目录，使用多个进程加速
            subprocess.Popen(['rm', '-rf', temp_name], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # 额外的清理进程，确保删除完成
            subprocess.Popen(['bash', '-c', f'sleep 5 && rm -rf {temp_name}_*'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            print(f"超快速清理完成: {ray_tmp_dir} (旧文件在后台删除中)")
        else:
            # 如果目录不存在，直接创建
            os.makedirs(ray_tmp_dir, exist_ok=True)
            print(f"目录不存在，已创建: {ray_tmp_dir}")
            
    except Exception as e:
        print(f"超快速清理失败，使用备用方法: {e}")
        try:
            # 备用方法1: 使用系统命令
            subprocess.run(f'rm -rf {ray_tmp_dir}/*', shell=True, capture_output=True, timeout=30)
            subprocess.run(f'rm -rf {ray_tmp_dir}/.*', shell=True, capture_output=True, timeout=30)
            print(f"备用方法1完成: {ray_tmp_dir}")
        except Exception as e2:
            print(f"备用方法1失败: {e2}")
            # 备用方法2: 使用shutil
            shutil.rmtree(ray_tmp_dir, ignore_errors=True)
            os.makedirs(ray_tmp_dir, exist_ok=True)
            print(f"备用方法2完成: {ray_tmp_dir}")


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