import hydra

from pettingllms.agents import ToolAgent
from pettingllms.data.dataset import DatasetRegistry
from pettingllms.environments.tools.tool_env import ToolEnvironment
from pettingllms.rewards.reward_fn import math_reward_fn
from pettingllms.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://pettingllms.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    agent_args = {"tools": ["python"], "parser_name": "qwen", "system_prompt": "You are a math assistant that can write python to solve math problems."}
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
