# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
import atexit
import signal
import sys
import os
import subprocess
import time
from omegaconf import OmegaConf, DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
# Local application imports
from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


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


def cleanup_ray():
    """清理 Ray 资源 - 强制版本"""
    print("\n" + "="*50)
    print("STARTING RAY CLEANUP...")
    print("="*50)
    
    try:
        # 方法1: 正常关闭
        if ray.is_initialized():
            print("Step 1: Attempting normal Ray shutdown...")
            try:
                ray.shutdown()
                print("✓ Normal Ray shutdown completed.")
                time.sleep(2)  # 等待进程完全关闭
            except Exception as e:
                print(f"✗ Normal Ray shutdown failed: {e}")
        else:
            print("Ray is not initialized, but will force cleanup anyway...")
    except Exception as e:
        print(f"Error checking Ray status: {e}")
    
    # 方法2: 强制杀死进程
    try:
        print("Step 2: Force killing Ray processes...")
        force_kill_ray_processes()
        time.sleep(1)
    except Exception as e:
        print(f"Error in force kill: {e}")
    
    # 方法3: 清理环境变量
    try:
        print("Step 3: Cleaning Ray environment...")
        ray_env_vars = [key for key in os.environ.keys() if key.startswith('RAY_')]
        for var in ray_env_vars:
            del os.environ[var]
        print(f"Cleared {len(ray_env_vars)} Ray environment variables")
    except Exception as e:
        print(f"Error cleaning environment: {e}")
    
    print("="*50)
    print("RAY CLEANUP COMPLETED")
    print("="*50)


def signal_handler(signum, frame):
    """信号处理器，在接收到终止信号时清理资源"""
    print(f"Received signal {signum}, cleaning up...")
    cleanup_ray()
    sys.exit(1)


# 多重清理机制
def emergency_cleanup():
    """紧急清理 - 确保无论如何都要关闭 Ray"""
    try:
        cleanup_ray()
    except:
        # 如果正常清理失败，直接强制杀死进程
        try:
            force_kill_ray_processes()
        except:
            pass


# 注册多个清理函数确保一定执行
atexit.register(emergency_cleanup)
atexit.register(cleanup_ray)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

# 尝试注册更多信号处理器（如果系统支持）
try:
    signal.signal(signal.SIGQUIT, signal_handler)  # Quit
    signal.signal(signal.SIGHUP, signal_handler)   # Hangup
except (AttributeError, OSError):
    pass  # 某些信号在某些系统上不可用


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    try:
        OmegaConf.to_yaml(config)
        #print(config.models.model_0.ppo_trainer_config.actor_rollout_ref.model.path)
        #print(config.models.model_1.ppo_trainer_config.actor_rollout_ref.model)
        #print(config.models.model_0.ppo_trainer_config)

        run_ppo(config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C)")
        emergency_cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with unexpected error: {e}")
        emergency_cleanup()
        raise e
    finally:
        # 最终保障 - 无论如何都要清理 Ray
        print("Executing final cleanup in main...")
        try:
            emergency_cleanup()
        except:
            pass


def run_ppo(config):
    try:
        if not ray.is_initialized():
            # this is for local ray cluster
            ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

        ray.get(train_multi_agents.remote(config))
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Cleaning up Ray cluster due to error...")
        emergency_cleanup()
        raise e
    finally:
        # 确保在函数退出时清理 Ray
        print("Executing cleanup in run_ppo...")
        try:
            emergency_cleanup()
        except:
            pass


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def train_multi_agents(config):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs
   
    

    
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    # Safely read the multi_modal flag
    multi_modal = getattr(config, 'multi_modal', False)

    # Initialize tokenizer dictionary for multiple models
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer_dict = {}
    processor_dict = {}
    ppo_trainer_config_dict = {}
    
    # Check if we have models configuration for multi-model training
    if hasattr(config, 'models') and config.models is not None:
        print("Multi-model training mode detected")
        
        # Process each model in the models configuration
        for model_key, model_config in config.models.items():
            model_path = model_config.path
            model_name = model_config.name
            
            print(f"Processing model: {model_name} at path: {model_path}")
            
            # Download the model checkpoint from hdfs
            local_path = copy_local_path_from_hdfs(model_path)
            
            # Get trust_remote_code setting from model config or use default
            trust_remote_code = getattr(model_config, 'trust_remote_code', False)
            if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
                trust_remote_code = config.resource.trust_remote_code
            
            # Create tokenizer for this model
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
            tokenizer_dict[model_name] = tokenizer
            if multi_modal:
                processor_dict[model_name] = processor
            ppo_trainer_config = model_config.ppo_trainer_config
            ppo_trainer_config_dict[model_name] = ppo_trainer_config
        
            
            

    from pettingllms.verl.ray_trainer import ResourcePoolManager, Role
    ray_worker_group_cls = RayWorkerGroup

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    
    # Access resource configuration safely
    n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if hasattr(config, 'resource') else 1
    nnodes = getattr(config.resource, 'nnodes', 1) if hasattr(config, 'resource') else 1
    
    managers = []
    for id in range(2):
        global_pool_id = f"global_pool_{id}"
        resource_pool_spec = {
           global_pool_id: [n_gpus_per_node] * nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        managers.append(resource_pool_manager)


    trainer = None
    try:
        trainer = MultiAgentsPPOTrainer(
            config=config,
            tokenizer_dict=tokenizer_dict,
            processor_dict=processor_dict,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=managers,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        trainer.init_workers()
        trainer.init_multi_agent_sys_execution_engine()
        trainer.fit()
    except Exception as e:
        print(f"Training failed in train_multi_agents: {e}")
        if trainer is not None:
            try:
                # 如果有清理方法，调用它
                if hasattr(trainer, 'cleanup'):
                    trainer.cleanup()
            except Exception as cleanup_error:
                print(f"Error during trainer cleanup: {cleanup_error}")
        raise e
    finally:
        # 最终清理
        print("Executing final cleanup in train_multi_agents...")
        if trainer is not None:
            try:
                if hasattr(trainer, 'cleanup'):
                    trainer.cleanup()
            except:
                pass


if __name__ == "__main__":
    main()
