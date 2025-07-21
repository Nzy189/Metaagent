import hydra
from omegaconf import DictConfig

from pettingllms.trainer.agent_sft_trainer import AgentSFTTrainer


@hydra.main(config_path="pkg://pettingllms.trainer.config", config_name="sft_trainer", version_base=None)
def main(config: DictConfig):
    # initialize trainer
    trainer = AgentSFTTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
