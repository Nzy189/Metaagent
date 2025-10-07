# Training on Planning Tasks

Train agents to solve grid-based planning problems.

## Plan-Path Task

Navigate from start to goal in a 10×10 grid with obstacles.

## Dataset Preparation

```bash
python scripts/dataprocess/load_sokoban.py
```

Creates:
```
datasets/sudoku_environments/planpath/
├── train/ (1000 instances)
└── test/ (200 instances)
```

## Training

### Shared Policy

```bash
bash scripts/train/plan_path_single.sh
```

### Per-Role Policies

```bash
bash scripts/train/plan_path_two_policy.sh
```

## Configuration

```python
config = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    
    "agents": [
        {
            "name": "planner",
            "system_prompt": "Plan optimal path...",
        },
        {
            "name": "executor",
            "system_prompt": "Execute movement actions...",
        }
    ],
    
    "env": {
        "grid_size": 10,
        "max_turns": 50,
        "obstacles": "random",
    },
    
    "reward": {
        "step_penalty": -0.01,
        "goal_reward": 1.0,
        "closer_to_goal": 0.05,
    },
}
```

## Results

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 5% |
| 1.7B | AT-GRPO (shared) | **96%** |
| 1.7B | AT-GRPO (per-role) | **97%** |
| 8B | Baseline | 12% |
| 8B | AT-GRPO | **96%** |

## Training Time

- **1.7B**: 8 hours (8 GPUs)
- **8B**: 16 hours (16 GPUs)

## Next Steps

- [Code Training](code.md)
- [Math Training](math.md)

