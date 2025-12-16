"""
Generalized SFT Training Script for Multi-Agent Environments

This script collects SFT data from various multi-agent environments and fine-tunes models.
Supports:
- Multiple multi-agent environments (code, math, search, stateful, etc.)
- Configurable via Hydra config files
- Only collects data where env.success is True
- LoRA and full fine-tuning support
"""

import hydra
import ray
import os
import json
import asyncio
import logging
from omegaconf import OmegaConf, DictConfig
from typing import Dict, List
from pathlib import Path
from pettingllms.sft_train.data_collector import SFTDataCollector
from pettingllms.trainer.mas_turn_order_register import ENV_CLASS_MAPPING, AGENT_CLASS_MAPPING, ENV_BATCH_CLASS_MAPPING
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_local_path_from_hdfs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SFTDataGenerator:
    """Generate SFT training data from multi-agent environments"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.env_name = config.env.name
        self.max_turns = getattr(config.env, 'max_turns', 8)

        # Get agent configurations
        self.agent_configs = config.agent_policy_configs.agent_configs
        self.turn_order = config.multi_agent_interaction.turn_order

        # Initialize environment and agent classes
        self.env_class = ENV_CLASS_MAPPING[self.env_name]
        self.env_batch_class = ENV_BATCH_CLASS_MAPPING[self.env_name]
        self.agent_class_list = [AGENT_CLASS_MAPPING[agent_name] for agent_name in self.turn_order]

        # Load tokenizers and processors for LLM APIs
        self.tokenizer_dict = {}
        self.processor_dict = {}
        self._load_models()

        # Initialize API client if API mode is enabled
        self.use_api = getattr(config.training, 'use_api', False)
        self.api_client = None
        if self.use_api:
            self._init_api_client()

        # Initialize data collector
        collect_mode = getattr(config.training, 'collect_mode', 'env')
        self.data_collector = SFTDataCollector(
            output_dir=getattr(config.training, 'sft_data_dir', './sft_data'),
            collect_mode=collect_mode,
            agent_names=self.turn_order
        )

        logger.info(f"Data collection mode: {collect_mode}")
        if collect_mode == "env":
            logger.info("  - Will collect data only when env.success is True")
        else:
            logger.info("  - Will collect data when individual agent.success is True")

        if self.use_api:
            logger.info(f"API mode enabled: {self.config.training.api_type}")

    def _load_models(self):
        """Load tokenizers and processors for each model"""
        if hasattr(self.config, 'models') and self.config.models is not None:
            for model_key, model_config in self.config.models.items():
                model_path = model_config.path
                model_name = model_config.name

                logger.info(f"Loading tokenizer for model: {model_name} at path: {model_path}")

                local_path = copy_local_path_from_hdfs(model_path)
                trust_remote_code = getattr(model_config, 'trust_remote_code', False)

                tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
                processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

                self.tokenizer_dict[model_name] = tokenizer
                if getattr(self.config, 'multi_modal', False):
                    self.processor_dict[model_name] = processor

    def _init_api_client(self):
        """Initialize API client for external LLM APIs"""
        from pettingllms.utils.api_client import create_api_client

        api_type = getattr(self.config.training, 'api_type', 'openai')
        api_model = getattr(self.config.training, 'api_model', None)
        api_base_url = getattr(self.config.training, 'api_base_url', None)
        api_temperature = getattr(self.config.training, 'api_temperature', 0.7)
        api_max_tokens = getattr(self.config.training, 'api_max_tokens', 2048)
        api_timeout = getattr(self.config.training, 'api_timeout', 60.0)

        logger.info(f"Initializing API client: {api_type}")
        logger.info(f"  Model: {api_model}")
        logger.info(f"  Temperature: {api_temperature}")
        logger.info(f"  Max tokens: {api_max_tokens}")
        if api_base_url:
            logger.info(f"  Base URL: {api_base_url}")

        self.api_client = create_api_client(
            api_type=api_type,
            model=api_model,
            base_url=api_base_url if api_base_url else None,
            temperature=api_temperature,
            max_tokens=api_max_tokens,
            timeout=api_timeout
        )

        logger.info(f"API client initialized successfully")

    async def generate_rollout(self, env_idx: int, rollout_idx: int):
        """
        Generate a single rollout and collect SFT data

        Args:
            env_idx: Environment index
            rollout_idx: Rollout index

        Returns:
            True if successful (env.success == True), False otherwise
        """
        # Create environment
        env = self.env_class(
            env_idx=env_idx,
            rollout_idx=rollout_idx,
            max_turns=self.max_turns,
            config=self.config
        )

        # Create agents
        agents = []
        for agent_idx, agent_name in enumerate(self.turn_order):
            agent_class = self.agent_class_list[agent_idx]
            agent_config = self.agent_configs[f"agent_{agent_idx}"]

            agent = agent_class(
                env_idx=env_idx,
                agent_sample_idx=rollout_idx,
                rollout_idx=rollout_idx,
                benchmark=getattr(self.config.env, 'benchmark', 'default')
            )
            agents.append((agent_name, agent, agent_config))

        # Run episode
        for turn_idx in range(self.max_turns):
            if env.done:
                break

            for agent_name, agent, agent_config in agents:
                # Update agent from environment
                agent.update_from_env(turn_idx, env)

                # Get prompt
                prompt = agent.current_prompt
                if isinstance(prompt, dict):
                    prompt_text = prompt.get('text', '')
                else:
                    prompt_text = str(prompt)

                # Get LLM response (using API client if enabled)
                response = await self._get_llm_response(prompt_text, agent_config)

                # Update agent from model response
                agent.update_from_model(response)

                # Step environment
                await agent.step(env)

                # Collect data for this agent
                policy_name = agent_config.policy_name
                reward = agent.agent_reward if hasattr(agent, 'agent_reward') else 0.0
                agent_success = agent.success if hasattr(agent, 'success') else False

                self.data_collector.add_data(
                    agent_name=agent_name,
                    policy_name=policy_name,
                    prompt=prompt_text,
                    response=response,
                    reward=reward,
                    env_success=env.success,
                    agent_success=agent_success,
                    metadata={
                        'env_name': self.env_name,
                        'env_idx': env_idx,
                        'rollout_idx': rollout_idx,
                        'turn_idx': turn_idx,
                        'agent_idx': agents.index((agent_name, agent, agent_config))
                    }
                )

        return env.success

    async def _get_llm_response(self, prompt: str, agent_config) -> str:
        """
        Get LLM response for the given prompt

        Uses API client if API mode is enabled, otherwise returns empty string

        Args:
            prompt: User prompt text
            agent_config: Agent configuration

        Returns:
            Generated response text
        """
        if self.use_api and self.api_client:
            try:
                # Prepare messages for API
                messages = []

                # Add system prompt if available in agent config
                if hasattr(agent_config, 'system_prompt'):
                    system_prompt = agent_config.system_prompt
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})

                # Add user prompt
                messages.append({"role": "user", "content": prompt})

                # Generate response using API
                response = await self.api_client.generate(messages)

                logger.debug(f"API response received (length: {len(response)})")
                return response

            except Exception as e:
                logger.error(f"Error getting API response: {e}")
                return ""
        else:
            # Placeholder for local model inference
            # This should call the same LLM API used in training
            logger.warning("Local model inference not implemented, returning empty string")
            return ""

    async def collect_data(self, num_episodes: int):
        """
        Collect SFT data from multiple episodes

        Args:
            num_episodes: Number of episodes to collect
        """
        logger.info(f"Collecting SFT data from {num_episodes} episodes")

        success_count = 0
        tasks = []

        for episode_idx in range(num_episodes):
            task = self.generate_rollout(env_idx=episode_idx, rollout_idx=episode_idx)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Episode failed with error: {result}")
            elif result:
                success_count += 1

        logger.info(f"Successfully collected data from {success_count}/{num_episodes} episodes")

        # Save collected data
        output_file, stats_file = self.data_collector.save_data()
        logger.info(f"Saved SFT data to {output_file}")
        logger.info(f"Saved statistics to {stats_file}")

        return output_file, stats_file


async def collect_sft_data(config: DictConfig):
    """
    Main function to collect SFT data

    Args:
        config: Hydra config
    """
    generator = SFTDataGenerator(config)

    num_episodes = getattr(config.training, 'sft_num_episodes', 1000)
    output_file, stats_file = await generator.collect_data(num_episodes)

    return output_file, stats_file


def train_sft_model(config: DictConfig, train_data_path: str):
    """
    Train SFT model on collected data

    Args:
        config: Hydra config
        train_data_path: Path to training data JSONL file
    """
    from pettingllms.sft_train.train_sft import train_qwen_sft, QwenSFTConfig

    # Extract SFT training config
    sft_config_dict = getattr(config.training, 'sft_config', {})

    # Get model path from config
    model_path = None
    if hasattr(config, 'models') and config.models is not None:
        first_model = next(iter(config.models.values()))
        model_path = first_model.path

    # Create SFT config
    sft_config = QwenSFTConfig(
        model_name_or_path=sft_config_dict.get('model_name_or_path', model_path),
        train_file=train_data_path,
        val_file=sft_config_dict.get('val_file', None),
        max_seq_length=sft_config_dict.get('max_seq_length', 4096),
        output_dir=sft_config_dict.get('output_dir', './sft_output'),
        num_train_epochs=sft_config_dict.get('num_train_epochs', 3),
        per_device_train_batch_size=sft_config_dict.get('per_device_train_batch_size', 2),
        gradient_accumulation_steps=sft_config_dict.get('gradient_accumulation_steps', 8),
        learning_rate=sft_config_dict.get('learning_rate', 5e-5),
        use_lora=sft_config_dict.get('use_lora', True),
        lora_r=sft_config_dict.get('lora_r', getattr(config, 'lora_rank', 64)),
        lora_alpha=sft_config_dict.get('lora_alpha', getattr(config, 'lora_alpha', 16)),
    )

    logger.info("Starting SFT training...")
    train_qwen_sft(sft_config)
    logger.info("SFT training completed!")


@hydra.main(config_path="../config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    """
    Main entry point for SFT data collection and training

    Args:
        config: Hydra config (uses same config structure as PPO training)
    """
    # Set default values for lora_rank and lora_alpha if not defined
    if 'lora_rank' not in config or config.lora_rank is None:
        OmegaConf.set_struct(config, False)
        config.lora_rank = 0
        OmegaConf.set_struct(config, True)

    if 'lora_alpha' not in config or config.lora_alpha is None:
        OmegaConf.set_struct(config, False)
        config.lora_alpha = 0
        OmegaConf.set_struct(config, True)

    logger.info("=" * 80)
    logger.info("SFT Data Collection and Training")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(config))

    # Step 1: Collect SFT data
    logger.info("Step 1: Collecting SFT data from multi-agent environments")
    train_data_path, stats_file = asyncio.run(collect_sft_data(config))

    # Step 2: Train SFT model (optional, can be disabled)
    if getattr(config.training, 'run_sft_training', True):
        logger.info("Step 2: Training SFT model on collected data")
        train_sft_model(config, train_data_path)
    else:
        logger.info("Step 2: Skipping SFT training (run_sft_training=False)")

    logger.info("=" * 80)
    logger.info("SFT pipeline completed!")
    logger.info(f"Training data: {train_data_path}")
    logger.info(f"Statistics: {stats_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
