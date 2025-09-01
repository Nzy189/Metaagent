import argparse
import json
import os
import signal
import sys
import time
import uuid

import ray
from omegaconf import OmegaConf, DictConfig


def _get_ray_address_from_internal() -> str | None:
    try:
        # Try Ray internal node info (works for local head started via ray.init())
        from ray._private import worker as ray_worker

        node = getattr(ray_worker, "_global_node", None)
        if node is None:
            return None
        # Prefer address_info['address'] if available; fallback to gcs_address
        try:
            address = node.address_info.get("address")
            if address:
                return address
        except Exception:
            pass
        try:
            address = getattr(node, "gcs_address", None)
            return address
        except Exception:
            return None
    except Exception:
        return None


def main():
    """Main function with original argparse, but enhanced config processing like train.py"""
    parser = argparse.ArgumentParser(description="Launch a detached Async vLLM server on a fresh local Ray and write registry.")
    parser.add_argument("--trainer-config", type=str, default="pettingllms/config/code/ppo_trainer/eval.yaml")
    parser.add_argument("--actor-name", type=str, default="async_llm_server")
    parser.add_argument("--registry-path", type=str, default="logs/ray_vllm_registry.json")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=100)
    parser.add_argument("--namespace", type=str, default="pettingllms", help="Namespace prefix. A unique suffix will be appended to avoid conflicts.")
    args = parser.parse_args()

    # Load and process config similar to train.py
    trainer_config = OmegaConf.load(args.trainer_config)
    
    # Process config similar to train.py - resolve any interpolations
    OmegaConf.resolve(trainer_config)
    
    # Ensure logs dir exists
    os.makedirs(os.path.dirname(args.registry_path), exist_ok=True)
    
    # Call the main logic with processed config
    launch_vllm_server(trainer_config, args.actor_name, args.registry_path, args.num_gpus, args.num_cpus, args.namespace)


def launch_vllm_server(trainer_config, actor_name, registry_path, num_gpus, num_cpus, namespace):
    """Main logic for launching vLLM server"""
    # Ensure we always start a brand-new local Ray and never attach to external cluster
    ray.shutdown()
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_NAMESPACE", None)

    # Env vars for vLLM
    os.environ["VERL_VLLM_DISTRIBUTED_BACKEND"] = "none"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.2")

    # Start a fresh local Ray with a unique namespace and random dashboard port to avoid conflicts
    unique_ns = f"{namespace}-{uuid.uuid4().hex[:8]}"
    ray.init(namespace=unique_ns, include_dashboard=False, num_gpus=num_gpus, num_cpus=num_cpus)
    ray_address = _get_ray_address_from_internal()
    print(f"Started fresh local Ray; address={ray_address} (namespace={unique_ns})")

    # Lazy imports to avoid heavy deps at module import time
    from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, RayResourcePool, create_colocated_worker_cls
    import torch

    # 1. First create the AsyncActorRolloutRefWorker actors that AsyncvLLMServer needs
    try:
        tp_size = int(getattr(trainer_config.actor_rollout_ref.rollout, "tensor_model_parallel_size", 1))
    except Exception:
        tp_size = 1
    
    # Ensure tp_size doesn't exceed available GPUs
    available_gpus = num_gpus
    if tp_size > available_gpus:
        print(f"Warning: tp_size ({tp_size}) exceeds available GPUs ({available_gpus}), reducing to {available_gpus}")
        tp_size = available_gpus
    
    dp_size = 1  # For simplicity, use dp_size=1
    world_size = dp_size * tp_size
    
    print(f"Using tp_size={tp_size}, dp_size={dp_size}, world_size={world_size}")
    
    # Create resource pool for workers
    # For single node, each process gets 1 GPU, so process_on_nodes should be [tp_size] not [world_size]
    resource_pool = RayResourcePool(
        process_on_nodes=[tp_size],  # Each process gets 1 GPU
        use_gpu=True,
        name_prefix="actor_rollout_pool",
        max_colocate_count=1,
    )
    
    # Create worker class with init args - each worker only needs 1 GPU
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ray.remote(num_gpus=1)(AsyncActorRolloutRefWorker),  # Each worker gets 1 GPU
        config=trainer_config.actor_rollout_ref,
        role="actor_rollout"
    )
    
    # Create colocated worker class
    class_dict = {"actor_rollout": actor_rollout_cls}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    
    # Create worker group and spawn workers
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    spawn_wg = wg.spawn(prefix_set=class_dict.keys())
    actor_rollout_wg = spawn_wg["actor_rollout"]
    
    # Initialize the workers (分阶段初始化以避免内存峰值)
    print(f"Initializing {world_size} AsyncActorRolloutRefWorker actors...")
    actor_rollout_wg.init_model()
    
    print(f"Successfully created and initialized {world_size} AsyncActorRolloutRefWorker actors")

    # 2. Now create the AsyncvLLMServer actor
    runtime_env = {"env": {}}
    for key in [
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_USE_V1",
        "CUDA_LAUNCH_BLOCKING",
        "VERL_VLLM_DISTRIBUTED_BACKEND",
    ]:
        if key in os.environ:
            runtime_env["env"][key] = os.environ[key]

    # AsyncvLLMServer doesn't need GPUs directly, it delegates to worker actors
    server_gpus = 0  # Server itself doesn't need GPU resources

    # Fix: Pass the whole trainer_config, not just actor_rollout_ref
    # AsyncvLLMServer.__init__ expects config.actor_rollout_ref, so it needs the full config
    server = AsyncvLLMServer.options(
        name=actor_name,
        lifetime="detached",
        num_gpus=server_gpus,
        runtime_env=runtime_env if runtime_env["env"] else None,
    ).remote(trainer_config, dp_size, 0, "actor_rollout")

    # Get address and initialize engine
    address = ray.get(server.get_server_address.remote())
    ray.get(server.init_engine.remote())

    # Write registry
    registry = {
        "ray_address": ray_address,
        "namespace": unique_ns,
        "actor_names": [actor_name],
        "addresses": [address],
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f)
    print(f"Wrote vLLM server registry to {registry_path}: {registry}")

    # Keep process alive so that the local Ray (and detached actor) persists
    print("Keeping launcher process alive. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down launcher...")


if __name__ == "__main__":
    main()

