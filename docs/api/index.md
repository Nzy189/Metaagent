# API Reference

Complete API documentation for PettingLLMs.

## Overview

PettingLLMs provides APIs for:

- **Training**: Multi-agent RL training with AT-GRPO
- **Environments**: Task-specific multi-agent environments
- **Agents**: LLM agent implementations
- **Configuration**: Training and environment configuration
- **Evaluation**: Model evaluation utilities

## Core Modules

### Training

- `pettingllms.trainer.train` - Main training loop
- `pettingllms.trainer.multi_agents_ppo_trainer` - AT-GRPO trainer
- `pettingllms.trainer.multi_agents_execution_engine` - Rollout engine

### Environments

- `pettingllms.multi_agent_env.base` - Base environment interface
- `pettingllms.multi_agent_env.code` - Code task environments
- `pettingllms.multi_agent_env.math` - Math task environments
- `pettingllms.multi_agent_env.stateful` - Game/planning environments

### Configuration

- `pettingllms.config.code` - Code task configs
- `pettingllms.config.math` - Math task configs
- `pettingllms.config.stateful` - Game/planning configs
- `pettingllms.config.ppo_trainer` - Training configs

### Evaluation

- `pettingllms.evaluate.evaluate` - Evaluation utilities

### Utilities

- `pettingllms.utils` - Helper functions
- `pettingllms.utils.cleanup` - Resource cleanup

## Quick Examples

### Training

```python
from pettingllms.trainer import train

# Run training with config
train(
    config_name="code_single_policy",
    num_iterations=2000,
    checkpoint_dir="checkpoints/",
)
```

### Environment

```python
from pettingllms.multi_agent_env.code import CodeEnv

# Create code environment
env = CodeEnv(config)
obs = env.reset()

# Run episode
done = False
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
```

### Evaluation

```python
from pettingllms.evaluate import evaluate

# Evaluate model
results = evaluate(
    model_path="checkpoints/model_2000",
    config_name="code_single_policy",
    num_episodes=200,
)

print(f"Success Rate: {results['success_rate']:.2%}")
```

## Detailed Documentation

For detailed documentation of each module, see the source code and docstrings.

## Configuration Files

Configuration examples are in `pettingllms/config/`:

```python
# Example: Code task config
from pettingllms.config.code import single_policy

config = single_policy.get_config()
config["learning_rate"] = 1e-5
config["batch_size"] = 128
```

## Next Steps

- Review [Training Guides](../training/overview.md)
- Check [Core Concepts](../core-concepts/overview.md)
- See [Examples](../training/overview.md)

