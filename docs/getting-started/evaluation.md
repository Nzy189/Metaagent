# Evaluation Guide

Guide for evaluating trained models with PettingLLMs.

## Quick Start

### 1. Edit Evaluation Script

Modify the evaluation script in `scripts/evaluate/` directory:

```bash
# Example configuration
MODEL_PATHS=("/path/to/your/checkpoint")
EXPERIMENT_NAME="my_evaluation"
CONFIG_NAME="code_L1_prompt"  # Match your training config
BENCHMARK="code_contests"

# GPU Configuration
GPU_START_ID=0
TP_SIZE=1
GPU_MEM=0.8

# Generation Limits
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=8192
MAX_TURNS=5
```

### 2. Run Evaluation

```bash
bash scripts/evaluate/code/evaluate_L1.sh
```

## Configuration Parameters

| Parameter | Type | Default | Description | Where to Modify |
|-----------|------|---------|-------------|-----------------|
| `MODEL_PATHS` | Array | - | Paths to model checkpoints | Evaluation script |
| `EXPERIMENT_NAME` | String | - | Name for the evaluation run | Evaluation script |
| `CONFIG_NAME` | String | - | Configuration file name | Evaluation script |
| `BENCHMARK` | String | - | Dataset to evaluate on | Evaluation script |
| `GPU_START_ID` | Integer | 0 | First GPU to use | Evaluation script |
| `TP_SIZE` | Integer | 1 | Tensor parallelism size | Evaluation script |
| `GPU_MEM` | Float | 0.8 | GPU memory utilization | Evaluation script |
| `MAX_PROMPT_LENGTH` | Integer | 8192 | Maximum input tokens | Evaluation script |
| `MAX_RESPONSE_LENGTH` | Integer | 8192 | Maximum output tokens | Evaluation script |
| `MAX_TURNS` | Integer | 5 | Maximum conversation turns | Evaluation script |

### Key Parameters Explained

#### Model Configuration
- **MODEL_PATHS**: Array of checkpoint paths to evaluate
- **CONFIG_NAME**: Must match the config used during training (found in `pettingllms/config/`)
- **BENCHMARK**: Dataset name (`"code_contests"`, `"gsm8k"`, `"sudoku"`, etc.)

#### GPU Configuration
- **TP_SIZE**: Number of GPUs per model (1 for small models, 2-4 for large models)
- **GPU_MEM**: Memory fraction to use (0.8 = 80% of GPU memory)
- **GPU_START_ID**: Starting GPU ID (0-indexed)

#### Generation Limits
- **MAX_PROMPT_LENGTH**: Maximum input tokens
- **MAX_RESPONSE_LENGTH**: Maximum output tokens per turn
- **MAX_TURNS**: Maximum agent interaction rounds

## Output

Results are saved to:
```
logs/<config_name>/<date>/<time>/validate/
├── summary.log          # Overall metrics
├── <episode_id>/        # Episode logs
└── metrics.json         # Detailed metrics
```

Key metrics include:
- **Success Rate**: Percentage of successful episodes
- **Average Turns**: Mean turns per episode
- **Average Reward**: Mean episode reward

## Next Steps

After evaluation:

- Adjust training parameters and retrain: [Training Guide](training.md)
- Prepare additional test datasets: [Dataset Guide](datasets.md)

