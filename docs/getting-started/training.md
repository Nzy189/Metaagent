# Training Guide

Guide for training models with PettingLLMs using AT-GRPO.

## Quick Start

### 1. Prepare Data

```bash
# Choose your task domain
python scripts/dataprocess/load_code.py   # For code tasks
python scripts/dataprocess/load_math.py   # For math tasks
python scripts/dataprocess/load_sokoban.py # For games
```

### 2. Edit Training Script

Modify the training script in `scripts/train/` directory:

```bash
# Example configuration
export CUDA_VISIBLE_DEVICES=0
GPU_num=1

# Model Configuration
base_models.policy_0.path="meta-llama/Llama-3.1-8B-Instruct"

# Training Configuration
training.experiment_name=math_training
training.total_training_steps=200
training.epoch_size=20
training.train_batch_size=32

# Generation Limits
training.max_prompt_length=8192
training.max_response_length=8192

# Dataset Configuration
env.dataset=polaris
env.benchmark=AIME24
```

### 3. Run Training

```bash
bash scripts/train/math/math_L1_prompt.sh
```

## Configuration Parameters

| Parameter | Type | Default | Description | Where to Modify |
|-----------|------|---------|-------------|-----------------|
| `CUDA_VISIBLE_DEVICES` | String | - | GPU device IDs to use | Training script |
| `GPU_num` | Integer | 1 | Number of GPUs per model | Training script |
| `base_models.policy_0.path` | String | - | Base model path | Training script |
| `training.experiment_name` | String | - | Experiment identifier | Training script |
| `training.total_training_steps` | Integer | 200 | Total training iterations | Training script |
| `training.epoch_size` | Integer | 20 | Episodes per epoch | Training script |
| `training.train_batch_size` | Integer | 32 | Batch size for training | Training script |
| `training.max_prompt_length` | Integer | 8192 | Maximum input tokens | Training script |
| `training.max_response_length` | Integer | 8192 | Maximum output tokens | Training script |
| `training.val_freq` | Integer | 10 | Validation frequency (steps) | Training script |
| `env.dataset` | String | - | Dataset name | Training script |
| `env.benchmark` | String | - | Specific benchmark/subset | Training script |

### Key Parameters Explained

#### Model Configuration
- **base_models.policy_0.path**: HuggingFace model name or local checkpoint path
- **training.experiment_name**: Names the training run (logs saved to `logs/{experiment_name}/`)

#### GPU Configuration
- **CUDA_VISIBLE_DEVICES**: Which GPUs to use (e.g., "0,1,2,3")
- **GPU_num**: Tensor parallelism size (1 for small models, 2-4 for large models)

#### Training Iteration Parameters
- **training.total_training_steps**: Number of training iterations (200-2000)
- **training.epoch_size**: Episodes per iteration (10-50)
- **training.train_batch_size**: Batch size (16-128, depends on GPU memory)

#### Generation Limits
- **training.max_prompt_length**: Maximum input tokens
- **training.max_response_length**: Maximum output tokens per turn

#### Dataset Configuration
- **env.dataset**: Dataset name (`"polaris"`, `"gsm8k"`, `"code_contests"`, etc.)
- **env.benchmark**: Specific subset (`"AIME24"`, `"interview"`, etc.)

## Training Scripts

Pre-configured scripts for different tasks:

```bash
# Math tasks
bash scripts/train/math/math_L1_prompt.sh

# Code tasks
bash scripts/train/code/code_L1_prompt.sh

# Game tasks
bash scripts/train/game/sudoku_single.sh
bash scripts/train/game/sokoban_two_policy.sh
```

## Monitoring Training

### Logs Location

Training logs are saved to:
```
logs/<experiment_name>/<date>/<time>/
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

### Monitor Progress

```bash
# View training logs
tail -f logs/<experiment_name>/*/train.log

# Check summary metrics
tail -f logs/<experiment_name>/*/summary.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Next Steps

After training is complete:

- Evaluate your trained models: [Evaluation Guide](evaluation.md)
- Learn about dataset formats: [Dataset Guide](datasets.md)

