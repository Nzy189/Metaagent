import hydra

from pettingllms.agents.math_agent import MathAgent
from pettingllms.data.dataset import DatasetRegistry
from pettingllms.environments.base.single_turn_env import SingleTurnEnvironment
from pettingllms.rewards.reward_fn import math_reward_fn
from pettingllms.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://pettingllms.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    env_args = {"reward_fn": math_reward_fn}

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
