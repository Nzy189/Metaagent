# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
import atexit
import os
import json
import shutil
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer


def cleanup_ray():
  
    if ray.is_initialized():
        ray.shutdown()
    
    # clean up tmp directories
    for tmp_path in ["/tmp/verl_ray", "/tmp/verl_spill"]:
        tmp_dir = Path(tmp_path)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Cleaned up temporary files")


atexit.register(cleanup_ray)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    OmegaConf.to_yaml(config)
    run_ppo(config)


def run_ppo(config):
    if not ray.is_initialized():
        # Prepare Ray temp and spill directories
        # Use /tmp for shorter socket paths (Unix socket path limit is 107 bytes)
        ray_tmp_dir = "/tmp/verl_ray"
        ray_spill_dir = "/tmp/verl_spill"
        os.makedirs(ray_tmp_dir, exist_ok=True)
        os.makedirs(ray_spill_dir, exist_ok=True)
        
        spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
        system_config = {"object_spilling_config": json.dumps(spilling_conf)}

        # Get GPU count from config
        n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if hasattr(config, 'resource') else 1
        
        # Validate GPU availability
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible_devices:
            available_gpu_count = len(cuda_visible_devices.split(','))
            n_gpus_per_node = min(n_gpus_per_node, available_gpu_count)
        
        print(f"Initializing Ray with {n_gpus_per_node} GPUs")
        ray.init(
            num_gpus=n_gpus_per_node,
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            _temp_dir=ray_tmp_dir,
            _system_config=system_config
        )


    def make_trainer_remote():
        num_cpus = int(ray.cluster_resources()["CPU"])
        return ray.remote(num_cpus=num_cpus)(train_multi_agents)

    multiagent_training_engine = make_trainer_remote()
    ray.get(multiagent_training_engine.remote(config))

def train_multi_agents(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer, hf_processor
    from pettingllms.verl.ray_trainer import ResourcePoolManager, Role
    
    n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1)
    nnodes = getattr(config.resource, 'nnodes', 1)
    
    # 打印配置
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    multi_modal = getattr(config, 'multi_modal', False)
    
    # 初始化tokenizer
    tokenizer_dict = {}
    processor_dict = {}
    ppo_trainer_config_dict = {}
    model_num = 0
    
    if hasattr(config, 'models') and config.models is not None:
        print("Multi-model training mode detected")
        
        for model_key, model_config in config.models.items():
            model_num += 1
            model_path = model_config.path
            model_name = model_config.name
            
            print(f"Processing model: {model_name} at path: {model_path}")
            
            local_path = copy_local_path_from_hdfs(model_path)
            
            trust_remote_code = getattr(model_config, 'trust_remote_code', False)
            if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
                trust_remote_code = config.resource.trust_remote_code
            
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
            tokenizer_dict[model_name] = tokenizer
            if multi_modal:
                processor_dict[model_name] = processor
            ppo_trainer_config_dict[model_name] = model_config.ppo_trainer_config
    
    n_gpus_per_model = n_gpus_per_node // model_num
    print(f"n_gpus_per_model: {n_gpus_per_model}")
    
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
    }
    
    managers = []
    for model_key, model_config in config.models.items():
        global_pool_id = f"global_pool_{model_key}"
        resource_pool_spec = {global_pool_id: [n_gpus_per_model] * nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        
        print(f"Creating resource pool for {model_key}: {resource_pool_spec}")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        resource_pool_manager.create_resource_pool()
        managers.append(resource_pool_manager)
    
    trainer = MultiAgentsPPOTrainer(
        config=config,
        tokenizer_dict=tokenizer_dict,
        processor_dict=processor_dict,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=managers,
        ray_worker_group_cls=RayWorkerGroup,
    )
    
    trainer.init_workers()
    trainer.init_multi_agent_sys_execution_engine()
    trainer.fit()


if __name__ == "__main__":
    main()