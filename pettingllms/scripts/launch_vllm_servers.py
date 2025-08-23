import argparse
import json
import os
import signal
import sys
import time

import ray
from omegaconf import OmegaConf


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
    parser = argparse.ArgumentParser(description="Launch detached Async vLLM server(s) and write registry.")
    parser.add_argument("--trainer-config", type=str, default="pettingllms/config/code/ppo_trainer/eval.yaml")
    parser.add_argument("--actor-name", type=str, default="async_llm_server")
    parser.add_argument("--registry-path", type=str, default="logs/ray_vllm_registry.json")
    parser.add_argument("--ray-address", type=str, default="", help="Existing Ray cluster address to connect, otherwise start local.")
    parser.add_argument("--namespace", type=str, default="pettingllms", help="Ray namespace for named actors.")
    parser.add_argument("--num-cpus", type=int, default=224)
    args = parser.parse_args()

    # Ensure logs dir exists
    os.makedirs(os.path.dirname(args.registry_path), exist_ok=True)

    # Env vars consistent with test
    os.environ["VERL_VLLM_DISTRIBUTED_BACKEND"] = "none"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Lower GPU memory utilization threshold to mitigate startup errors on busy GPUs
    os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.2")

    # Connect or start Ray
    if args.ray_address:
        ray.init(address=args.ray_address, namespace=args.namespace)
        ray_address = args.ray_address
        print(f"Connected to existing Ray cluster at {ray_address} (namespace={args.namespace})")
    else:
        ray.init(num_cpus=args.num_cpus, namespace=args.namespace)
        ray_address = _get_ray_address_from_internal()
        print(f"Started local Ray; address={ray_address} (namespace={args.namespace})")

    # Lazy imports to avoid heavy deps at module import time
    from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
    from pettingllms.trainer.utils import initialize_llm_servers

    trainer_config = OmegaConf.load(args.trainer_config)

    # Start as detached named actor if not exists; else reuse
    servers, addresses = initialize_llm_servers(
        None,
        AsyncvLLMServer,
        trainer_config,
        reuse_existing=True,
        lifetime_detached=True,
        actor_name=args.actor_name,
        write_registry_path=args.registry_path,
        strict_reuse=False,
    )

    # Write registry
    registry = {
        "ray_address": ray_address,
        "namespace": args.namespace,
        "actor_names": [args.actor_name for _ in servers],
        "addresses": addresses,
    }
    with open(args.registry_path, "w") as f:
        json.dump(registry, f)
    print(f"Wrote vLLM server registry to {args.registry_path}: {registry}")

    # Keep process alive if we started local Ray, so that detached actors persist
    if not args.ray_address:
        print("Keeping launcher process alive. Press Ctrl+C to stop.")
        try:
            # Sleep loop, interruptible by SIGINT
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("Shutting down launcher...")
    else:
        print("Launcher connected to external Ray cluster; exiting without stopping cluster.")


if __name__ == "__main__":
    main()


