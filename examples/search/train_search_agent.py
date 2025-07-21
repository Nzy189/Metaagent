import hydra

from pettingllms.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from pettingllms.agents.tool_agent import ToolAgent
from pettingllms.data import DatasetRegistry
from pettingllms.environments.tools.tool_env import ToolEnvironment
from pettingllms.rewards.reward_fn import search_reward_fn
from pettingllms.trainer.agent_trainer import AgentTrainer

from .local_retrieval_tool import LocalRetrievalTool


@hydra.main(config_path="pkg://pettingllms.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")

    tool_map = {"local_search": LocalRetrievalTool}

    env_args = {
        "max_steps": 20,
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
    }

    agent_args = {"system_prompt": SEARCH_SYSTEM_PROMPT, "tool_map": tool_map, "parser_name": "qwen"}

    # Use the registry-based approach (comment out the other approach)
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
