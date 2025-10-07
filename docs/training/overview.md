# Training Overview

This guide provides an overview of training with PettingLLMs.

## Training Pipeline

The complete training pipeline consists of:

1. **Dataset Preparation** - Load and process task datasets
2. **Configuration** - Set up model, environment, and training configs
3. **Training** - Run AT-GRPO training loop
4. **Evaluation** - Test trained models on held-out data

## Quick Start

### 1. Prepare Data

```bash
# Choose your task domain
python scripts/dataprocess/load_code.py   # For code tasks
python scripts/dataprocess/load_math.py   # For math tasks
python scripts/dataprocess/load_sokoban.py # For games
```

### 2. Run Training

```bash
# Use pre-configured training scripts
bash scripts/train/code_single_policy.sh    # Code with shared policy
bash scripts/train/math.sh                  # Math tasks
bash scripts/train/plan_path_single.sh      # Planning tasks
```

### 3. Monitor Progress

```bash
# View training logs
tail -f logs/<task_name>/*/train.log

# Check summary metrics
tail -f logs/<task_name>/*/summary.log
```

### 4. Evaluate

```bash
# Edit evaluation script
vim scripts/evaluate/evaluate.sh

# Run evaluation
bash scripts/evaluate/evaluate.sh
```

## Training Scripts

PettingLLMs provides pre-configured scripts for different tasks:

### Game Tasks

```bash
# Sudoku (shared policy)
bash scripts/train/sokodu_single.sh

# Sokoban (per-role policies)
bash scripts/train/sokoban_two_policy.sh
```

### Planning Tasks

```bash
# Plan-Path (shared policy)
bash scripts/train/plan_path_single.sh

# Plan-Path (per-role policies)
bash scripts/train/plan_path_two_policy.sh
```

### Code Tasks

```bash
# Code (shared policy)
bash scripts/train/code_single_policy.sh

# Code (per-role policies)
bash scripts/train/code_two_policy.sh
```

### Math Tasks

```bash
# Math tasks
bash scripts/train/math.sh
```

## Configuration

### Config Structure

```
pettingllms/config/
├── code/
│   ├── single_policy.py     # Shared policy config
│   └── two_policy.py        # Per-role policies config
├── math/
│   └── single_policy.py
├── stateful/
│   ├── sokoban_single.py
│   └── planpath_two_policy.py
└── ppo_trainer/
    └── trainer_config.py    # Training hyperparameters
```

### Key Configuration Options

#### Model Config

```python
config = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
}
```

#### Agent Config

```python
agent_config = {
    "agents": [
        {
            "name": "planner",
            "role": "planning",
            "system_prompt": "You are a planning agent...",
        },
        {
            "name": "executor",
            "role": "execution",
            "system_prompt": "You are an execution agent...",
        }
    ],
    "policy_mapping": "shared",  # or "per_role"
}
```

#### Training Config

```python
training_config = {
    "num_iterations": 2000,
    "batch_size": 128,
    "learning_rate": 1e-5,
    "ppo_epochs": 3,
    "gamma": 0.99,
    "clip_epsilon": 0.2,
    "alpha": 0.7,  # Global/local reward mixing
}
```

#### Environment Config

```python
env_config = {
    "max_turns": 10,
    "num_envs": 1000,
    "timeout": 300,  # seconds
    "seed": 42,
}
```

### Customizing Training

To customize training:

1. **Copy existing config**:
```bash
cp pettingllms/config/code/single_policy.py pettingllms/config/code/my_config.py
```

2. **Modify parameters**:
```python
# pettingllms/config/code/my_config.py
config = {
    "learning_rate": 5e-6,  # Lower learning rate
    "batch_size": 256,      # Larger batch
    # ... other changes
}
```

3. **Update training script**:
```bash
# scripts/train/my_training.sh
CONFIG_NAME="code/my_config"
bash scripts/train/code_single_policy.sh
```

## Resource Requirements

### Minimal Setup

For small-scale experiments:

- **GPUs**: 2× (1 rollout, 1 update)
- **RAM**: 32 GB
- **CPUs**: 16 cores
- **Storage**: 50 GB

### Recommended Setup

For full training runs:

- **GPUs**: 8× (4 rollout, 4 update)
- **RAM**: 128 GB
- **CPUs**: 64 cores
- **Storage**: 200 GB

### Estimating Requirements

```python
# GPU memory per model
model_size_gb = {
    "1.7B": 7,   # FP16
    "8B": 16,    # FP16
    "32B": 64,   # FP16
}

# Total GPU memory needed
total_gpu_memory = (
    model_size_gb * num_rollout_workers +
    model_size_gb * num_update_workers
)
```

## Training Time

Expected training times:

| Task | Model | GPUs | Time |
|------|-------|------|------|
| Plan-Path | 1.7B | 8 | 8 hours |
| Code | 1.7B | 8 | 24 hours |
| Math | 1.7B | 8 | 16 hours |
| Plan-Path | 8B | 16 | 16 hours |
| Code | 8B | 16 | 48 hours |
| Math | 8B | 16 | 32 hours |

## Monitoring Training

### Logs

Training logs are saved to:
```
logs/<task_name>/<date>/<time>/
├── train.log           # Detailed logs
├── summary.log         # Metrics summary
├── checkpoints/        # Model checkpoints
└── validate/           # Validation results
```

### Key Metrics

Monitor these metrics during training:

- **Reward**: Average episode reward
- **Success Rate**: Percentage of successful episodes
- **Episode Length**: Average turns per episode
- **Policy Loss**: PPO policy loss
- **Value Loss**: Value function loss (if used)

### TensorBoard (Optional)

```bash
# Start TensorBoard
tensorboard --logdir logs/<task_name>

# View at http://localhost:6006
```

### Weights & Biases (Optional)

```python
# Add to config
config["wandb"] = {
    "project": "pettingllms",
    "entity": "your_username",
    "name": "experiment_name",
}
```

## Checkpointing

### Automatic Checkpoints

Checkpoints are saved every N iterations:

```python
checkpoint_config = {
    "save_interval": 100,  # Save every 100 iterations
    "keep_last_n": 5,      # Keep last 5 checkpoints
    "save_best": True,     # Save best checkpoint
}
```

### Manual Checkpoints

Save checkpoint manually:

```bash
# During training, send signal
kill -USR1 <trainer_pid>
```

### Loading Checkpoints

Resume from checkpoint:

```bash
# Set checkpoint path in config
export CHECKPOINT_PATH="/path/to/checkpoint"
bash scripts/train/code_single_policy.sh
```

## Troubleshooting

### Out of Memory

**Symptoms**: CUDA OOM errors

**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (FP16)
- Reduce number of rollout workers

### Slow Training

**Symptoms**: Low throughput

**Solutions**:
- Increase number of GPUs
- Enable mixed precision
- Optimize dataloader
- Check for CPU bottlenecks

### Poor Performance

**Symptoms**: Low reward, no improvement

**Solutions**:
- Check reward structure
- Verify environment logic
- Adjust learning rate
- Increase training iterations

### Ray Errors

**Symptoms**: Worker connection failures

**Solutions**:
- Check firewall settings
- Increase timeout values
- Restart Ray cluster
- Check GPU availability

## Next Steps

- [Game Training](games.md) - Train on Sudoku/Sokoban
- [Planning Training](planning.md) - Train on Plan-Path
- [Code Training](code.md) - Train on coding tasks
- [Math Training](math.md) - Train on math tasks

