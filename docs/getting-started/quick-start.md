# Quick Start

This guide will walk you through running your first PettingLLMs training session.

## Dataset Preparation

Before training, you need to prepare the datasets for your chosen task.

### Code Tasks (APPS, CodeContests, LiveCodeBench)

```bash
python scripts/dataprocess/load_code.py
```

### Math Tasks (AIME24/25, OlympiadBench)

```bash
python scripts/dataprocess/load_math.py
```

### Game/Planning Tasks (Sokoban, Sudoku)

```bash
python scripts/dataprocess/load_sokoban.py
```

Datasets will be saved to:
- `datasets/code/` - Code datasets
- `datasets/math/` - Math datasets
- `datasets/sudoku_environments/` - Game datasets

## Your First Training Run

Let's train a multi-agent system on math tasks:

```bash
bash scripts/train/math.sh
```

This will:

1. Initialize the training environment
2. Load the base model (Qwen3-1.7B by default)
3. Start rollout workers for trajectory collection
4. Begin RL training with AT-GRPO
5. Save checkpoints to the logs directory

## Monitor Training

Training logs are saved to:
```
logs/<task_name>/<date>/<time>/
├── summary.log          # Training summary
├── train.log           # Detailed training logs
└── validate/           # Validation results
```

You can monitor training progress:

```bash
# View training summary
tail -f logs/math_single_policy/*/summary.log

# View detailed logs
tail -f logs/math_single_policy/*/train.log
```

## Available Training Scripts

PettingLLMs provides pre-configured training scripts for different tasks:

### Game Domain
```bash
# Sokoban with two specialized policies
bash scripts/train/sokoban_two_policy.sh

# Sudoku with single shared policy
bash scripts/train/sokodu_single.sh
```

### Planning Domain
```bash
# Plan-Path with single policy
bash scripts/train/plan_path_single.sh

# Plan-Path with two specialized policies
bash scripts/train/plan_path_two_policy.sh
```

### Code Domain
```bash
# Code tasks with single policy
bash scripts/train/code_single_policy.sh

# Code tasks with two specialized policies
bash scripts/train/code_two_policy.sh
```

### Math Domain
```bash
# Math tasks
bash scripts/train/math.sh
```

## Evaluation

After training, evaluate your model:

1. Edit `scripts/evaluate/evaluate.sh`:
```bash
MODEL_PATHS=("/path/to/your/checkpoint")
CONFIG_NAME="math_single_policy"  # Match your training config
```

2. Run evaluation:
```bash
bash scripts/evaluate/evaluate.sh
```

## Configuration

Training configurations are stored in `pettingllms/config/`:

```
pettingllms/config/
├── code/               # Code task configs
├── math/               # Math task configs
├── stateful/           # Game/planning configs
└── ppo_trainer/        # Trainer configs
```

You can modify these configs to:
- Change model architectures
- Adjust hyperparameters
- Modify reward structures
- Customize agent workflows

## Example: Customizing Training

To train with a different base model:

```bash
# Edit the training script
export MODEL_PATH="/path/to/your/model"
bash scripts/train/math.sh
```

To change hyperparameters, modify the config file:

```python
# pettingllms/config/math/single_policy.py
config = {
    "learning_rate": 1e-5,
    "batch_size": 128,
    "num_epochs": 3,
    # ... other parameters
}
```

## Next Steps

Now that you've run your first training session:

- Prepare more datasets: [Dataset Guide](datasets.md)
- Run model evaluation: [Evaluation Guide](evaluation.md)
- Learn detailed training configurations: [Training Guide](training.md)

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
- Reduce batch size in config
- Use gradient accumulation
- Reduce the number of rollout workers

### Slow Training

To speed up training:
- Increase the number of GPUs
- Adjust rollout worker count
- Enable mixed precision training

### Connection Errors

If Ray workers fail to connect:
- Check firewall settings
- Verify Ray cluster is properly initialized
- Review logs in the `logs/` directory

