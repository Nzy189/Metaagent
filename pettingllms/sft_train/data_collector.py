"""
Generalized Data Collection Script for Multi-Agent SFT Training

This script collects training data from various multi-agent environments for SFT.
- Supports multiple agent types and environments
- Configurable filtering (e.g., only keep data where env.success is True)
- Flexible data collection from any multi-agent environment
- Saves in JSONL format for SFT training
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SFTDataPoint:
    """Single SFT training data point"""
    agent_name: str  # Name of the agent (e.g., "code_agent", "search_agent")
    policy_name: str  # Name of the policy model
    prompt: str
    response: str
    reward: float
    env_success: bool  # Whether the environment marked this as successful
    timestamp: str
    metadata: Dict[str, Any]


class SFTDataCollector:
    """Generalized collector for SFT training data from multi-agent environments"""

    def __init__(
        self,
        output_dir: str = "./sft_data",
        collect_mode: str = "env",  # "env" or "agent" - determines success criteria
        agent_names: Optional[List[str]] = None,  # List of agent names to collect data for
        agent_ratios: Optional[Dict[str, float]] = None,  # Sampling ratios for each agent
    ):
        """
        Initialize SFT data collector

        Args:
            output_dir: Output directory for collected data
            collect_mode: Collection mode
                - "env": Only collect data when env.success is True (all agents must succeed)
                - "agent": Collect data when agent.success is True (per-agent success)
            agent_names: List of agent names to collect data for
            agent_ratios: Sampling ratios for each agent (optional)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.collect_mode = collect_mode
        if collect_mode not in ["env", "agent"]:
            raise ValueError(f"collect_mode must be 'env' or 'agent', got '{collect_mode}'")

        self.agent_names = agent_names or []
        self.agent_ratios = agent_ratios or {}  # If empty, no ratio balancing

        # Data storage: separate list for each agent
        self.agent_data: Dict[str, List[SFTDataPoint]] = {
            agent_name: [] for agent_name in self.agent_names
        }

        # Statistics
        self.stats = {
            "total_collected": 0,
            "total_successful": 0,
            "total_episodes": 0,
            "successful_episodes": 0,
        }
        # Per-agent statistics
        for agent_name in self.agent_names:
            self.stats[f"{agent_name}_collected"] = 0
            self.stats[f"{agent_name}_passed"] = 0

        logger.info(f"SFTDataCollector initialized: collect_mode={collect_mode}, "
                   f"agent_names={self.agent_names}, agent_ratios={self.agent_ratios}")

    def add_data(
        self,
        agent_name: str,
        policy_name: str,
        prompt: str,
        response: str,
        reward: float,
        env_success: bool,
        agent_success: bool,
        metadata: Dict[str, Any] = None
    ):
        """
        Add data point for any agent

        Args:
            agent_name: Name of the agent
            policy_name: Name of the policy model
            prompt: Input prompt to the agent
            response: Agent's response
            reward: Reward received
            env_success: Whether the environment marked this as successful
            agent_success: Whether this specific agent was successful
            metadata: Additional metadata
        """
        # Initialize storage for this agent if not exists
        if agent_name not in self.agent_data:
            self.agent_data[agent_name] = []
            self.stats[f"{agent_name}_collected"] = 0
            self.stats[f"{agent_name}_passed"] = 0

        self.stats[f"{agent_name}_collected"] += 1
        self.stats["total_collected"] += 1

        # Filter based on collect_mode
        should_collect = False
        if self.collect_mode == "env":
            # In env mode, only collect when environment succeeds
            should_collect = env_success
            if not env_success:
                logger.debug(f"Skipping {agent_name} data (env mode, env_success={env_success})")
        elif self.collect_mode == "agent":
            # In agent mode, collect when individual agent succeeds
            should_collect = agent_success
            if not agent_success:
                logger.debug(f"Skipping {agent_name} data (agent mode, agent_success={agent_success})")

        if not should_collect:
            return

        self.stats[f"{agent_name}_passed"] += 1
        self.stats["total_successful"] += 1

        data_point = SFTDataPoint(
            agent_name=agent_name,
            policy_name=policy_name,
            prompt=prompt,
            response=response,
            reward=reward,
            env_success=env_success,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        self.agent_data[agent_name].append(data_point)
        logger.debug(f"Added {agent_name} data: reward={reward}, env_success={env_success}, "
                    f"metadata={metadata}")

    def save_data(self, output_file: str = None):
        """
        Balance data according to ratios (if specified) and save to JSONL

        Args:
            output_file: Output file path (optional)

        Returns:
            Tuple of (output_file, stats_file)
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"sft_data_{timestamp}.jsonl")

        # Log data counts
        data_counts = {agent_name: len(data) for agent_name, data in self.agent_data.items()}
        logger.info(f"Collected data counts: {data_counts}")

        # Check if we have any data
        total_data = sum(data_counts.values())
        if total_data == 0:
            logger.warning("No data collected for any agent!")
            return None, None

        # Balance data according to ratios if specified
        all_data = []
        if self.agent_ratios:
            # Apply ratio balancing
            sampled_data = self._balance_by_ratio()
            all_data = sampled_data
        else:
            # Use all collected data
            for agent_name, data_list in self.agent_data.items():
                all_data.extend(data_list)

        # Shuffle data
        random.shuffle(all_data)

        # Convert to chat format for SFT
        sft_examples = []
        for data_point in all_data:
            # Standard chat format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": data_point.prompt
                    },
                    {
                        "role": "assistant",
                        "content": data_point.response
                    }
                ],
                "metadata": {
                    "agent_name": data_point.agent_name,
                    "policy_name": data_point.policy_name,
                    "reward": data_point.reward,
                    "env_success": data_point.env_success,
                    "timestamp": data_point.timestamp,
                    **data_point.metadata
                }
            }
            sft_examples.append(example)

        # Save to JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in sft_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(sft_examples)} SFT examples to {output_file}")

        # Log per-agent counts
        if self.agent_ratios:
            sampled_counts = {}
            for agent_name in self.agent_data.keys():
                count = sum(1 for dp in all_data if dp.agent_name == agent_name)
                sampled_counts[agent_name] = count
                logger.info(f"  - {agent_name}: {count} samples")
        else:
            for agent_name, count in data_counts.items():
                logger.info(f"  - {agent_name}: {count} samples")

        # Save statistics
        stats_file = output_file.replace('.jsonl', '_stats.json')
        final_stats = {
            **self.stats,
            "total_sft_examples": len(sft_examples),
            "data_counts": data_counts,
        }

        if self.agent_ratios:
            final_stats["agent_ratios"] = self.agent_ratios
            final_stats["sampled_counts"] = sampled_counts

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved statistics to {stats_file}")

        return output_file, stats_file

    def _balance_by_ratio(self) -> List[SFTDataPoint]:
        """
        Balance data according to agent_ratios

        Returns:
            List of balanced SFT data points
        """
        # Normalize ratios to sum to 1.0
        total_ratio = sum(self.agent_ratios.values())
        normalized_ratios = {
            agent: ratio / total_ratio
            for agent, ratio in self.agent_ratios.items()
        }

        # Find the limiting agent (one with smallest available samples relative to its ratio)
        min_samples_per_ratio = float('inf')
        for agent_name, ratio in normalized_ratios.items():
            if agent_name in self.agent_data and ratio > 0:
                available = len(self.agent_data[agent_name])
                samples_per_ratio = available / ratio
                min_samples_per_ratio = min(min_samples_per_ratio, samples_per_ratio)

        # Sample according to ratios
        sampled_data = []
        for agent_name, ratio in normalized_ratios.items():
            if agent_name not in self.agent_data or ratio == 0:
                continue

            target_count = int(min_samples_per_ratio * ratio)
            available_data = self.agent_data[agent_name]

            if target_count <= len(available_data):
                sampled = random.sample(available_data, target_count)
            else:
                # If we don't have enough, use all available
                sampled = available_data
                logger.warning(f"Not enough data for {agent_name}: requested {target_count}, "
                             f"available {len(available_data)}")

            sampled_data.extend(sampled)

        return sampled_data

    def print_stats(self):
        """Print collection statistics"""
        logger.info("=" * 80)
        logger.info("SFT Data Collection Statistics")
        logger.info("=" * 80)

        # Overall statistics
        logger.info(f"Overall:")
        logger.info(f"  - Total collected: {self.stats['total_collected']}")
        logger.info(f"  - Total successful: {self.stats['total_successful']}")
        logger.info(f"  - Success rate: {self.stats['total_successful']/max(1, self.stats['total_collected'])*100:.1f}%")

        # Per-agent statistics
        for agent_name in self.agent_data.keys():
            collected_key = f"{agent_name}_collected"
            passed_key = f"{agent_name}_passed"

            if collected_key in self.stats and passed_key in self.stats:
                collected = self.stats[collected_key]
                passed = self.stats[passed_key]
                pass_rate = passed / max(1, collected) * 100

                logger.info(f"{agent_name}:")
                logger.info(f"  - Collected: {collected}")
                logger.info(f"  - Passed: {passed}")
                logger.info(f"  - Pass rate: {pass_rate:.1f}%")

        # Episode statistics
        if 'total_episodes' in self.stats:
            logger.info(f"Episodes:")
            logger.info(f"  - Total: {self.stats['total_episodes']}")
            logger.info(f"  - Successful: {self.stats['successful_episodes']}")

        logger.info("=" * 80)


def create_qwen_sft_config(
    model_name: str = "Qwen/Qwen2.5-8B",
    output_dir: str = "./qwen_sft_output",
    train_data_path: str = "./sft_data/sft_data.jsonl",
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    """
    Create training configuration for Qwen3-8B SFT

    Returns a dict that can be used with HuggingFace Trainer or similar
    """
    config = {
        "model_name_or_path": model_name,
        "output_dir": output_dir,
        "train_file": train_data_path,

        # Training hyperparameters
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",

        # Optimization
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,

        # Logging and saving
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 3,

        # Memory optimization
        "fp16": True,
        "gradient_checkpointing": True,

        # Chat template for Qwen
        "chat_template": "qwen",

        # LoRA configuration (optional, for efficient training)
        "use_lora": True,
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    return config


if __name__ == "__main__":
    # Example usage
    # Example 1: Env mode - only collect when env.success is True
    collector_env = SFTDataCollector(
        output_dir="./sft_data_env",
        collect_mode="env",
        agent_names=["code_agent", "test_agent"],
    )

    # Simulate data collection in env mode
    # Both agents' data will only be collected if env_success=True
    collector_env.add_data(
        agent_name="code_agent",
        policy_name="shared_model",
        prompt="Write a function to solve this problem...",
        response="def solve():\n    ...",
        reward=1.0,
        env_success=True,
        agent_success=True,
        metadata={"env_name": "code_env", "problem_id": "prob_001"}
    )

    collector_env.add_data(
        agent_name="test_agent",
        policy_name="shared_model",
        prompt="Write test cases for this function...",
        response="def test_solve():\n    ...",
        reward=1.0,
        env_success=True,
        agent_success=True,
        metadata={"env_name": "code_env", "problem_id": "prob_001"}
    )

    # Save collected data
    output_file, stats_file = collector_env.save_data()
    collector_env.print_stats()

    logger.info(f"[Env Mode] Saved SFT data to {output_file}")
    logger.info(f"[Env Mode] Saved statistics to {stats_file}")

    # Example 2: Agent mode - collect when agent.success is True
    collector_agent = SFTDataCollector(
        output_dir="./sft_data_agent",
        collect_mode="agent",
        agent_names=["code_agent", "test_agent"],
    )

    # In agent mode, each agent's data is collected independently
    # code_agent succeeds, will be collected
    collector_agent.add_data(
        agent_name="code_agent",
        policy_name="shared_model",
        prompt="Write a function...",
        response="def solve():\n    ...",
        reward=0.8,
        env_success=False,  # env failed
        agent_success=True,  # but this agent succeeded
        metadata={"env_name": "code_env", "problem_id": "prob_002"}
    )

    # test_agent fails, will NOT be collected
    collector_agent.add_data(
        agent_name="test_agent",
        policy_name="shared_model",
        prompt="Write tests...",
        response="def test():\n    ...",
        reward=0.2,
        env_success=False,
        agent_success=False,
        metadata={"env_name": "code_env", "problem_id": "prob_002"}
    )

    output_file_agent, stats_file_agent = collector_agent.save_data()
    collector_agent.print_stats()

    logger.info(f"[Agent Mode] Saved SFT data to {output_file_agent}")
    logger.info(f"[Agent Mode] Saved statistics to {stats_file_agent}")

    # Example 3: With ratio balancing (e.g., 1:3 ratio)
    collector_with_ratio = SFTDataCollector(
        output_dir="./sft_data",
        collect_mode="env",
        agent_names=["agent_1", "agent_2"],
        agent_ratios={"agent_1": 0.25, "agent_2": 0.75}  # 1:3 ratio
    )

    logger.info("Example with ratio balancing created")
