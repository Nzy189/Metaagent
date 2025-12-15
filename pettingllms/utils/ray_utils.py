"""Ray initialization utilities"""
import os
import json
import ray


def init_ray_with_temp_dirs(config=None, n_gpus_per_node=None):
    """
    Initialize Ray with temporary directories and spilling configuration
    
    Args:
        config: Optional config object with resource settings
        n_gpus_per_node: Number of GPUs per node (overrides config if provided)
    
    Returns:
        Tuple of (ray_tmp_dir, ray_spill_dir)
    """
    from .clean_up import register_temp_dirs
    
    if ray.is_initialized():
        print("Ray is already initialized")
        return None, None
    
    # Create experiment-specific temporary directories using process ID
    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)
    
    # Register directories for cleanup
    register_temp_dirs(ray_tmp_dir, ray_spill_dir)
    
    # Configure spilling
    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    system_config = {"object_spilling_config": json.dumps(spilling_conf)}
    
    # Determine GPU count
    if n_gpus_per_node is None:
        n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if config and hasattr(config, 'resource') else 1
    
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
    
    return ray_tmp_dir, ray_spill_dir
