# Evaluation Guide

Guide for evaluating trained models.

## Quick Start

### 1. Edit Evaluation Script

```bash
vim scripts/evaluate/evaluate.sh
```

Set your parameters:
```bash
MODEL_PATHS=("/path/to/checkpoint1" "/path/to/checkpoint2")
CONFIG_NAME="math_single_policy"
EVAL_SPLIT="test"
NUM_EPISODES=200
```

### 2. Run Evaluation

```bash
bash scripts/evaluate/evaluate.sh
```

## Evaluation Script

The main evaluation script supports:

- Multiple model checkpoints
- Different task configurations
- Train/test split selection
- Customizable episode count

## Configuration

### Model Paths

Specify one or more checkpoints:
```bash
MODEL_PATHS=(
    "/path/to/checkpoint_1000"
    "/path/to/checkpoint_2000"
)
```

### Config Name

Match your training config:
```bash
# For code tasks
CONFIG_NAME="code_single_policy"  # or "code_two_policy"

# For math tasks
CONFIG_NAME="math_single_policy"

# For games
CONFIG_NAME="stateful/sudoku_single"
CONFIG_NAME="stateful/sokoban_two_policy"

# For planning
CONFIG_NAME="stateful/planpath_single"
```

### Evaluation Split

Choose data split:
```bash
EVAL_SPLIT="test"   # Test set (default)
# or
EVAL_SPLIT="train"  # Training set (for debugging)
```

### Number of Episodes

```bash
NUM_EPISODES=200  # Evaluate on 200 episodes
```

## Output

### Logs Directory

```
logs/<config_name>/<date>/<time>/validate/
├── summary.log          # Overall metrics
├── <episode_id>/
│   ├── env_agent.log   # Episode transcript
│   └── async.log       # System logs
└── metrics.json         # Detailed metrics
```

### Metrics

Key evaluation metrics:

#### All Tasks
- **Success Rate**: Percentage of successful episodes
- **Average Turns**: Mean turns per episode
- **Average Reward**: Mean episode reward

#### Code Tasks
- **Test Pass Rate**: Percentage of tests passed
- **Code Quality**: Code correctness score
- **Test Quality**: Test coverage score

#### Math Tasks
- **Exact Match**: Exact answer correctness
- **Partial Credit**: Partial correctness (if applicable)

#### Game/Planning Tasks
- **Solution Length**: Moves to solution
- **Optimal Solutions**: Percentage of optimal solutions
- **Invalid Actions**: Rate of invalid moves

## Evaluation Modes

### Single-Agent Evaluation

Evaluate model in single-agent mode:
```bash
EVAL_MODE="single_agent"
bash scripts/evaluate/evaluate.sh
```

### Multi-Agent Evaluation

Evaluate in multi-agent mode (default):
```bash
EVAL_MODE="multi_agent"
bash scripts/evaluate/evaluate.sh
```

## Advanced Options

### Custom Test Set

Evaluate on custom data:
```bash
export CUSTOM_DATA_PATH="/path/to/custom/data"
bash scripts/evaluate/evaluate.sh
```

### Detailed Logging

Enable verbose logging:
```bash
export VERBOSE_LOGGING=1
bash scripts/evaluate/evaluate.sh
```

### Visualization

Save episode visualizations:
```bash
export SAVE_VISUALIZATION=1
bash scripts/evaluate/evaluate.sh
```

## Comparing Models

### Multiple Checkpoints

Compare different training iterations:
```bash
MODEL_PATHS=(
    "logs/math/checkpoint_500"
    "logs/math/checkpoint_1000"
    "logs/math/checkpoint_2000"
)
```

### Different Configs

Compare shared vs per-role policies:
```bash
# First evaluation
CONFIG_NAME="code_single_policy"
bash scripts/evaluate/evaluate.sh

# Second evaluation
CONFIG_NAME="code_two_policy"
bash scripts/evaluate/evaluate.sh
```

## Analysis Scripts

### Parse Results

Extract metrics from logs:
```python
from pettingllms.evaluate import parse_results

results = parse_results("logs/math/*/validate/")
print(f"Success Rate: {results['success_rate']:.2%}")
print(f"Average Reward: {results['avg_reward']:.3f}")
```

### Compare Runs

Compare multiple evaluation runs:
```python
from pettingllms.evaluate import compare_results

compare_results([
    "logs/run1/validate/",
    "logs/run2/validate/",
])
```

## Troubleshooting

### Evaluation Hangs

**Symptoms**: Evaluation doesn't progress

**Solutions**:
- Check for environment deadlocks
- Reduce timeout values
- Check GPU availability

### Low Success Rate

**Symptoms**: Model performs poorly

**Solutions**:
- Verify correct checkpoint loaded
- Check evaluation config matches training
- Review episode logs for failure patterns

### Inconsistent Results

**Symptoms**: High variance in metrics

**Solutions**:
- Increase number of evaluation episodes
- Set deterministic seeds
- Check for environment randomness

## Best Practices

### 1. Evaluate Regularly

Monitor training progress:
```bash
# Evaluate every 500 iterations
for iter in 500 1000 1500 2000; do
    MODEL_PATHS=("logs/training/checkpoint_$iter")
    bash scripts/evaluate/evaluate.sh
done
```

### 2. Use Held-Out Data

Always evaluate on test set:
```bash
EVAL_SPLIT="test"  # Never "train"
```

### 3. Multiple Seeds

Run with different seeds for robustness:
```bash
for seed in 42 123 456; do
    export EVAL_SEED=$seed
    bash scripts/evaluate/evaluate.sh
done
```

### 4. Save Episode Logs

Keep logs for analysis:
```bash
export SAVE_LOGS=1
bash scripts/evaluate/evaluate.sh
```

## Next Steps

- Review [Benchmark Results](../results/benchmarks.md)
- Check [Ablation Studies](../results/ablations.md)
- Explore [Training Guides](../training/overview.md)

