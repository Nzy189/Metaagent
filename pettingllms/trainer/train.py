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
from pettingllms.utils.simpler_timer import create_timer, timer_checkpoint

def force_kill_ray_processes():
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
    print("\n" + "="*50)
    print("STARTING RAY CLEANUP...")
    print("="*50)
    
    try:
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
    

def signal_handler(signum, frame):
    print(f"Received signal {signum}, cleaning up...")
    cleanup_ray()
    sys.exit(1)



atexit.register(cleanup_ray)
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Terminate signal

try:
    signal.signal(signal.SIGQUIT, signal_handler)  # Quit
    signal.signal(signal.SIGHUP, signal_handler)   # Hangup
except (AttributeError, OSError):
    pass  # Some signals are not available on some systems


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    main_timer = create_timer("MainTraining")
    main_timer.start("Starting main training function")
    
    try:
        main_timer.checkpoint("Loading config")
        OmegaConf.to_yaml(config)
        main_timer.checkpoint("Config loaded, starting PPO training")
        run_ppo(config)
        main_timer.checkpoint("PPO training completed")
    except KeyboardInterrupt:
        main_timer.checkpoint("Training interrupted by user (Ctrl+C)")
        print("\nTraining interrupted by user (Ctrl+C)")
        cleanup_ray()
        sys.exit(1)
    except Exception as e:
        main_timer.checkpoint("Training failed with unexpected error")
        print(f"Training failed with unexpected error: {e}")
        cleanup_ray()
        raise e
    finally:
        main_timer.checkpoint("Executing final cleanup in main")
        print("Executing final cleanup in main...")
        cleanup_ray()
        main_timer.end("Main training function completed")


def run_ppo(config):
    ppo_timer = create_timer("PPORunner")
    ppo_timer.start("Starting run_ppo function")
    
    try:
        if not ray.is_initialized():
            ppo_timer.checkpoint("Initializing Ray cluster")
            # Prepare Ray temp and spill directories under project root
            try:
                from pathlib import Path as _Path
                _project_root = _Path(__file__).resolve().parents[2]
                _ray_tmp_dir = os.path.join(str(_project_root), "tmp", "ray_tmp")
                _ray_spill_dir = os.path.join(str(_project_root), "tmp", "ray_spill")
                os.makedirs(_ray_tmp_dir, exist_ok=True)
                os.makedirs(_ray_spill_dir, exist_ok=True)
                _spilling_conf = {"type": "filesystem", "params": {"directory_path": [_ray_spill_dir]}}
                _system_config = {"object_spilling_config": json.dumps(_spilling_conf)}
            except Exception as _e:
                print(f"Warning: failed to prepare Ray temp/spill dirs in trainer: {_e}")
                _ray_tmp_dir = None
                _system_config = None

            # this is for local ray cluster
            ray.init(
                runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
                **({"_temp_dir": _ray_tmp_dir} if _ray_tmp_dir else {}),
                **({"_system_config": _system_config} if _system_config else {})
            )
            ppo_timer.checkpoint("Ray cluster initialized")

        ppo_timer.checkpoint("Starting remote multi-agent training")
        ray.get(train_multi_agents.remote(config))
        ppo_timer.checkpoint("Remote multi-agent training completed")
    except Exception as e:
        ppo_timer.checkpoint("Training failed with error")
        print(f"Training failed with error: {e}")
        print("Cleaning up Ray cluster due to error...")
        cleanup_ray()
        raise e
    finally:
        # Ensure cleanup when function exits
        ppo_timer.checkpoint("Executing cleanup in run_ppo")
        print("Executing cleanup in run_ppo...")
        cleanup_ray()
        ppo_timer.end("run_ppo function completed")


@ray.remote(num_cpus=224)  # please make sure main_task is not scheduled on head
def train_multi_agents(config):
    train_timer = create_timer("MultiAgentsTraining")
    train_timer.start("Starting train_multi_agents function")
    n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1)
    
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
    
    train_timer.checkpoint("Starting model processing")
    model_num = 0
    # Check if we have models configuration for multi-model training
    if hasattr(config, 'models') and config.models is not None:
        print("Multi-model training mode detected")
        
        # Process each model in the models configuration
        for model_key, model_config in config.models.items():
            model_num += 1
            model_path = model_config.path
            model_name = model_config.name
            
            print(f"Processing model: {model_name} at path: {model_path}")
            
            train_timer.checkpoint(f"Downloading model {model_name}")
            # Download the model checkpoint from hdfs
            local_path = copy_local_path_from_hdfs(model_path)
            
            # Get trust_remote_code setting from model config or use default
            trust_remote_code = getattr(model_config, 'trust_remote_code', False)
            if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
                trust_remote_code = config.resource.trust_remote_code
            
            train_timer.checkpoint(f"Creating tokenizer for {model_name}")
            # Create tokenizer for this model
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
            tokenizer_dict[model_name] = tokenizer
            if multi_modal:
                processor_dict[model_name] = processor
            ppo_trainer_config = model_config.ppo_trainer_config
            ppo_trainer_config_dict[model_name] = ppo_trainer_config
        
    n_gpus_per_model = n_gpus_per_node // model_num
    print(f"n_gpus_per_model: {n_gpus_per_model}")
            
    train_timer.checkpoint("Setting up resource pools and worker mappings")
    
    from pettingllms.verl.ray_trainer import ResourcePoolManager, Role
    ray_worker_group_cls = RayWorkerGroup

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    
    # Access resource configuration safely
    #n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if hasattr(config, 'resource') else 1
    nnodes = getattr(config.resource, 'nnodes', 1) if hasattr(config, 'resource') else 1
    
    managers = []
    for model_key, model_config in config.models.items():
        global_pool_id = f"global_pool_{model_key}"
        resource_pool_spec = {
           global_pool_id: [n_gpus_per_model] * nnodes,
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
        train_timer.checkpoint("Creating MultiAgentsPPOTrainer")
        trainer = MultiAgentsPPOTrainer(
            config=config,
            tokenizer_dict=tokenizer_dict,
            processor_dict=processor_dict,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=managers,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        train_timer.checkpoint("Initializing workers")
        trainer.init_workers()
        train_timer.checkpoint("Initializing multi-agent execution engine")
        trainer.init_multi_agent_sys_execution_engine()
        train_timer.checkpoint("Starting training (fit)")
        trainer.fit()
        train_timer.checkpoint("Training completed successfully")
    except Exception as e:
        train_timer.checkpoint("Training failed with exception")
        print(f"Training failed in train_multi_agents: {e}")
        if trainer is not None:
            try:
                if hasattr(trainer, 'cleanup'):
                    trainer.cleanup()
            except Exception as cleanup_error:
                print(f"Error during trainer cleanup: {cleanup_error}")
        raise e
    finally:
        train_timer.checkpoint("Executing final cleanup in train_multi_agents")
        print("Executing final cleanup in train_multi_agents...")
        if trainer is not None:
            try:
                if hasattr(trainer, 'cleanup'):
                    trainer.cleanup()
            except:
                pass
        train_timer.end("train_multi_agents function completed")


if __name__ == "__main__":
    main()
