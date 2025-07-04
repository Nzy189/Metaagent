"""
Example usage of the Rollout Serving system.
This script demonstrates how to use the asynchronous rollout generation system
with multiple producers and consumers.
"""
import asyncio
import logging
import time
from typing import List, Dict
import hydra
from transformers import AutoTokenizer
from verl import DataProto

from .rollout_serving import RolloutServingManager
from .async_batch_manager import AsyncBatchManager
from .rollout_producer import RolloutProducer
from .rollout_consumer import RolloutConsumer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def example_basic_usage(config):
    """Basic example of using the rollout serving system."""
    print("=== Basic Usage Example ===")
    
    # Initialize serving manager
    serving_manager = RolloutServingManager(
        config=config,
        num_producers=2,
        num_consumers=3,
        max_queue_size=50,
        batch_timeout=30.0
    )
    
    try:
        # Initialize and start the serving system
        print("Initializing serving manager...")
        await serving_manager.initialize()
        await serving_manager.start()
        
        # Check system health
        health = await serving_manager.health_check()
        print(f"System health: {health['status']}")
        print(f"Producers: {health['num_producers']}, Consumers: {health['num_consumers']}")
        
        # Create sample environment outputs
        sample_env_outputs = [
            {
                "env_id": 0,
                "history": [
                    {"state": "Game state 1", "actions_left": 3}
                ],
                "group_id": 0,
                "tag": "test_env",
                "metrics": {"test_env/success_rate": 0.5}
            },
            {
                "env_id": 1, 
                "history": [
                    {"state": "Game state 2", "actions_left": 3}
                ],
                "group_id": 0,
                "tag": "test_env", 
                "metrics": {"test_env/success_rate": 0.3}
            }
        ]
        
        # Generate a single rollout batch
        print("Generating rollout batch...")
        start_time = time.time()
        result = await serving_manager.generate_rollout_batch(
            sample_env_outputs, 
            mode="validation",
            timeout=30.0
        )
        processing_time = time.time() - start_time
        
        print(f"Batch processed in {processing_time:.2f} seconds")
        print(f"Result metadata: {result.meta_info}")
        
        if result.batch:
            print(f"Batch keys: {list(result.batch.keys())}")
            if 'input_ids' in result.batch:
                print(f"Input IDs shape: {result.batch['input_ids'].shape}")
            if 'rm_scores' in result.batch:
                print(f"RM scores shape: {result.batch['rm_scores'].shape}")
        
        if result.non_tensor_batch:
            print(f"Non-tensor batch keys: {list(result.non_tensor_batch.keys())}")
        
    finally:
        await serving_manager.shutdown()
        print("Basic usage example completed")


async def example_multi_turn_rollout(config):
    """Example of multi-turn rollout generation."""
    print("\n=== Multi-turn Rollout Example ===")
    
    serving_manager = RolloutServingManager(
        config=config,
        num_producers=1,
        num_consumers=2
    )
    
    try:
        await serving_manager.initialize()
        await serving_manager.start()
        
        # Create initial environment outputs for multi-turn
        initial_env_outputs = [
            {
                "env_id": 0,
                "history": [
                    {"state": "Initial state", "actions_left": 5}
                ],
                "group_id": 0,
                "tag": "multi_turn_env",
                "metrics": {"multi_turn_env/episode_length": 0}
            }
        ]
        
        print("Starting multi-turn rollout...")
        start_time = time.time()
        
        result = await serving_manager.generate_multi_turn_rollout(
            initial_env_outputs,
            mode="validation",
            max_turns=3,
            timeout_per_turn=15.0
        )
        
        total_time = time.time() - start_time
        print(f"Multi-turn rollout completed in {total_time:.2f} seconds")
        print(f"Final result metadata: {result.meta_info}")
        
    finally:
        await serving_manager.shutdown()
        print("Multi-turn example completed")


async def example_concurrent_batches(config):
    """Example of processing multiple batches concurrently.""" 
    print("\n=== Concurrent Batches Example ===")
    
    serving_manager = RolloutServingManager(
        config=config,
        num_producers=2,
        num_consumers=4
    )
    
    try:
        await serving_manager.initialize()
        await serving_manager.start()
        
        # Create multiple different batch requests
        batch_requests = []
        for i in range(5):
            env_outputs = [
                {
                    "env_id": i * 2,
                    "history": [
                        {"state": f"Concurrent state {i}a", "actions_left": 2}
                    ],
                    "group_id": i,
                    "tag": "concurrent_env",
                    "metrics": {"concurrent_env/batch_id": i}
                },
                {
                    "env_id": i * 2 + 1,
                    "history": [
                        {"state": f"Concurrent state {i}b", "actions_left": 2}
                    ],
                    "group_id": i,
                    "tag": "concurrent_env", 
                    "metrics": {"concurrent_env/batch_id": i}
                }
            ]
            batch_requests.append(env_outputs)
        
        print(f"Processing {len(batch_requests)} batches concurrently...")
        start_time = time.time()
        
        # Submit all batches concurrently
        tasks = [
            serving_manager.generate_rollout_batch(batch, mode="validation")
            for batch in batch_requests
        ]
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        print(f"All {len(results)} batches completed in {total_time:.2f} seconds")
        print(f"Average time per batch: {total_time / len(results):.2f} seconds")
        
        # Print results summary
        for i, result in enumerate(results):
            print(f"Batch {i}: {result.meta_info.get('batch_id', 'unknown')}")
        
    finally:
        await serving_manager.shutdown()
        print("Concurrent batches example completed")


async def example_performance_monitoring(config):
    """Example of monitoring system performance."""
    print("\n=== Performance Monitoring Example ===")
    
    serving_manager = RolloutServingManager(
        config=config,
        num_producers=2,
        num_consumers=3
    )
    
    try:
        await serving_manager.initialize()
        await serving_manager.start()
        
        # Run benchmark
        print("Running performance benchmark...")
        benchmark_results = await serving_manager.benchmark(
            num_test_batches=10,
            batch_size=3,
            mode="validation"
        )
        
        print("Benchmark Results:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value}")
        
        # Monitor system during operation
        print("\nSystem monitoring during operation:")
        
        # Submit some work
        sample_env_outputs = [
            {
                "env_id": 0,
                "history": [{"state": "Monitoring test", "actions_left": 1}],
                "group_id": 0,
                "tag": "monitor_env",
                "metrics": {}
            }
        ]
        
        # Monitor queue status before and after
        queue_stats_before = serving_manager.get_queue_status()
        print(f"Queue stats before: {queue_stats_before}")
        
        # Submit batch
        result = await serving_manager.generate_rollout_batch(sample_env_outputs)
        
        queue_stats_after = serving_manager.get_queue_status()
        print(f"Queue stats after: {queue_stats_after}")
        
        # Final health check
        final_health = await serving_manager.health_check()
        print(f"Final health check:")
        print(f"  Uptime: {final_health['uptime_seconds']:.2f}s")
        print(f"  Total batches: {final_health['total_batches_processed']}")
        print(f"  Throughput: {final_health['avg_throughput']:.2f} batches/sec")
        
    finally:
        await serving_manager.shutdown()
        print("Performance monitoring example completed")


async def run_all_examples(config):
    """Run all example scenarios."""
    print("Starting Rollout Serving Examples...")
    
    await example_basic_usage(config)
    await asyncio.sleep(1)  # Brief pause between examples
    
    await example_multi_turn_rollout(config) 
    await asyncio.sleep(1)
    
    await example_concurrent_batches(config)
    await asyncio.sleep(1)
    
    await example_performance_monitoring(config)
    
    print("\nAll examples completed successfully!")


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """Main function to run examples."""
    asyncio.run(run_all_examples(config))


if __name__ == "__main__":
    main() 