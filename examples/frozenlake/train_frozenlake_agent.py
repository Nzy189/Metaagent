import hydra

from pettingllms.agents.frozenlake_agent import FrozenLakeAgent
from pettingllms.data import DatasetRegistry
from pettingllms.environments.frozenlake.frozenlake import FrozenLakeEnv
from pettingllms.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://pettingllms.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("frozenlake", "train")
    val_dataset = DatasetRegistry.load_dataset("frozenlake", "test")

    trainer = AgentTrainer(
        agent_class=FrozenLakeAgent,
        env_class=FrozenLakeEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
