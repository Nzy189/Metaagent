# Training on Game Tasks

This guide covers training on Sudoku and Sokoban tasks.

## Overview

Game tasks in PettingLLMs:

- **Sudoku 4×4**: Number placement puzzle
- **Sokoban 6×6**: Box-pushing puzzle

Both use a Planner-Executor workflow.

## Dataset Preparation

```bash
python scripts/dataprocess/load_sokoban.py
```

This creates:
```
datasets/sudoku_environments/
├── sokoban/
│   ├── train/ (1000 instances)
│   └── test/ (200 instances)
└── sudoku/
    ├── train/ (1000 instances)
    └── test/ (200 instances)
```

## Training Scripts

### Sudoku (Shared Policy)

```bash
bash scripts/train/sokodu_single.sh
```

**Configuration**: `pettingllms/config/stateful/sudoku_single.py`

### Sokoban (Per-Role Policies)

```bash
bash scripts/train/sokoban_two_policy.sh
```

**Configuration**: `pettingllms/config/stateful/sokoban_two_policy.py`

## Configuration Details

### Sudoku Config Example

```python
config = {
    # Model
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_length": 1024,
    
    # Agents (shared policy)
    "agents": [
        {
            "name": "player",
            "role": "playing",
            "system_prompt": "You play Sudoku...",
        }
    ],
    "policy_mapping": "shared",
    
    # Environment
    "env": {
        "name": "sudoku",
        "grid_size": 4,
        "max_turns": 20,
    },
    
    # Training
    "num_iterations": 2000,
    "batch_size": 128,
    "learning_rate": 1e-5,
    
    # Reward
    "reward_shaping": {
        "step_penalty": -0.01,
        "success_reward": 1.0,
        "failure_reward": -0.5,
    },
}
```

### Sokoban Config Example

```python
config = {
    # Model
    "model_name": "Qwen/Qwen2.5-8B-Instruct",
    "max_length": 2048,
    
    # Agents (per-role policies)
    "agents": [
        {
            "name": "planner",
            "role": "planning",
            "policy_id": "policy_planner",
            "system_prompt": "You plan Sokoban moves...",
        },
        {
            "name": "executor",
            "role": "execution",
            "policy_id": "policy_executor",
            "system_prompt": "You execute Sokoban moves...",
        }
    ],
    "policy_mapping": "per_role",
    
    # Environment
    "env": {
        "name": "sokoban",
        "grid_size": 6,
        "max_turns": 50,
        "num_boxes": 4,
    },
    
    # Training
    "num_iterations": 2000,
    "batch_size": 256,
    "learning_rate": 5e-6,
    
    # Reward
    "reward_shaping": {
        "step_penalty": -0.01,
        "box_on_target": 0.25,
        "success_reward": 1.0,
        "failure_reward": -1.0,
    },
}
```

## Workflow

### Sudoku Workflow

```
1. Agent observes grid state
2. Agent chooses cell and number
3. Environment validates and updates grid
4. Repeat until solved or max turns
```

### Sokoban Workflow

```
1. Planner analyzes board state
2. Planner proposes next move
3. Executor executes move
4. Environment updates board
5. Repeat until all boxes on targets
```

## Reward Structure

### Sudoku Rewards

```python
# Per step
step_reward = -0.01

# Terminal
if solved:
    terminal_reward = 1.0
elif invalid_move:
    terminal_reward = -0.5
elif max_turns:
    terminal_reward = -0.5
```

### Sokoban Rewards

```python
# Per step
step_reward = -0.01

# Progress (boxes on targets)
boxes_on_target = count_boxes_on_targets()
progress_reward = 0.25 * boxes_on_target

# Terminal
if all_boxes_on_targets:
    terminal_reward = 1.0
else:
    terminal_reward = -1.0

# Total
reward = step_reward + progress_reward + terminal_reward
```

## Expected Results

### Sudoku 4×4

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 7% |
| 1.7B | AT-GRPO | **99%** |
| 8B | Baseline | 48% |
| 8B | AT-GRPO | **99.5%** |

### Sokoban 6×6

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 0% |
| 1.7B | AT-GRPO | **11.5%** |
| 8B | Baseline | 9% |
| 8B | AT-GRPO | **98%** |

## Training Time

| Task | Model | GPUs | Time |
|------|-------|------|------|
| Sudoku | 1.7B | 4 | 6 hours |
| Sudoku | 8B | 8 | 12 hours |
| Sokoban | 1.7B | 4 | 8 hours |
| Sokoban | 8B | 8 | 16 hours |

## Monitoring

### Key Metrics

Monitor during training:

- **Success Rate**: Percentage of solved puzzles
- **Average Turns**: Moves to solution
- **Invalid Actions**: Rate of invalid moves
- **Reward**: Average episode reward

### Example Logs

```
Iteration 500:
  Success Rate: 45.2%
  Avg Turns: 12.3
  Invalid Actions: 2.1%
  Avg Reward: 0.23

Iteration 1000:
  Success Rate: 78.5%
  Avg Turns: 9.8
  Invalid Actions: 0.5%
  Avg Reward: 0.64

Iteration 2000:
  Success Rate: 99.0%
  Avg Turns: 7.2
  Invalid Actions: 0.1%
  Avg Reward: 0.92
```

## Evaluation

After training, evaluate on test set:

```bash
# Edit evaluation script
vim scripts/evaluate/evaluate.sh

# Set paths
MODEL_PATHS=("/path/to/checkpoint")
CONFIG_NAME="stateful/sudoku_single"  # or sokoban_two_policy

# Run evaluation
bash scripts/evaluate/evaluate.sh
```

## Troubleshooting

### Low Success Rate

**Symptoms**: Model not learning to solve puzzles

**Solutions**:
- Increase reward for progress
- Reduce step penalty
- Increase training iterations
- Check environment logic

### Too Many Invalid Actions

**Symptoms**: Model proposes invalid moves

**Solutions**:
- Add penalty for invalid actions
- Improve action space description in prompts
- Use constrained decoding (if available)

### Slow Convergence

**Symptoms**: Learning is slow

**Solutions**:
- Increase learning rate
- Increase batch size
- Use larger model
- Simplify reward structure

## Advanced Topics

### Custom Reward Shaping

Add intermediate rewards:

```python
# Reward for making progress
progress_rewards = {
    "filled_cells": 0.05,  # Per cell filled (Sudoku)
    "box_moved_closer": 0.1,  # Box closer to target (Sokoban)
}
```

### Curriculum Learning

Start with easier puzzles:

```python
curriculum = {
    "iterations_0_500": {"difficulty": "easy"},
    "iterations_500_1500": {"difficulty": "medium"},
    "iterations_1500_2000": {"difficulty": "hard"},
}
```

### Visualization

Visualize agent behavior:

```python
# In eval script
config["visualize"] = True
config["save_trajectories"] = True
```

## Next Steps

- [Planning Training](planning.md) - Similar workflow for Plan-Path
- [Core Concepts](../core-concepts/workflows.md) - Understand the workflow
- [Benchmark Results](../results/benchmarks.md) - See full results

