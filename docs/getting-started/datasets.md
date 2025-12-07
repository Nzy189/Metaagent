# Dataset Preparation

PettingLLMs supports multiple task domains, each requiring specific dataset preparation steps.

## Overview

The framework includes three data processing scripts:

- `scripts/dataprocess/load_code.py` - For coding tasks
- `scripts/dataprocess/load_math.py` - For mathematical reasoning
- `scripts/dataprocess/load_sokoban.py` - For games and planning

## Code Datasets

### Supported Datasets

- **APPS**: Competitive programming problems
- **CodeContests**: Programming contest problems
- **LiveCodeBench**: Live coding benchmarks

### Preparation

```bash
python scripts/dataprocess/load_code.py
```

This script will:

1. Download code datasets from HuggingFace
2. Process and format problems
3. Extract test cases
4. Save to `data/code/train/` and `data/code/test/`

### Dataset Structure

```
data/code/
├── train/
│   ├── apps/
│   ├── codecontests/
│   └── livecodebench/
└── test/
    ├── apps/
    ├── codecontests/
    └── livecodebench/
```

Each problem includes:
- Problem description
- Input/output examples
- Test cases
- Solution templates (if available)

## Math Datasets

### Supported Datasets

- **AIME 2024**: American Invitational Mathematics Examination 2024
- **AIME 2025**: American Invitational Mathematics Examination 2025
- **OlympiadBench**: International math olympiad problems

### Preparation

```bash
python scripts/dataprocess/load_math.py
```

This script will:

1. Download math problem datasets
2. Parse problem statements
3. Extract ground truth answers
4. Format for RL training
5. Save to `data/math/train/` and `data/math/test/`

### Dataset Structure

```
data/math/
├── train/
│   ├── aime24/
│   ├── aime25/
│   └── olympiad/
└── test/
    ├── aime24/
    ├── aime25/
    └── olympiad/
```

Each problem includes:
- Problem statement
- Ground truth answer
- Difficulty level
- Subject area

## Game & Planning Datasets

### Supported Environments

- **Sokoban**: 6×6 box-pushing puzzles
- **Sudoku**: 4×4 number placement puzzles
- **Plan-Path**: 10×10 grid navigation

### Preparation

```bash
python scripts/dataprocess/load_sokoban.py
```

This script will:

1. Generate game instances
2. Create initial states
3. Verify solvability
4. Save to `data/sudoku_environments/`

### Dataset Structure

```
data/sudoku_environments/
├── sokoban/
│   ├── train/
│   └── test/
├── sudoku/
│   ├── train/
│   └── test/
└── planpath/
    ├── train/
    └── test/
```

Each instance includes:
- Initial state
- Goal state
- Action space
- Optimal solution length (if known)

## Custom Datasets

To add your own datasets:

### 1. Create Dataset Processor

```python
# scripts/dataprocess/load_custom.py
def load_custom_dataset():
    # Load your data
    data = load_your_data()
    
    # Process and format
    processed = process_data(data)
    
    # Save in PettingLLMs format
    save_dataset(processed, "data/custom/")
```

### 2. Create Environment Config

```python
# pettingllms/config/custom/config.py
class CustomConfig:
    dataset_path = "data/custom/"
    task_type = "your_task_type"
    # ... other config
```

### 3. Implement Environment

```python
# pettingllms/multi_agent_env/custom/custom_env.py
class CustomEnv:
    def __init__(self, config):
        # Initialize environment
        pass
    
    def reset(self):
        # Reset to initial state
        pass
    
    def step(self, action):
        # Execute action
        pass
```

## Data Statistics

### Code Datasets

| Dataset | Train | Test | Avg. Length | Difficulty |
|---------|-------|------|-------------|------------|
| APPS | 5000 | 1000 | 150 lines | Easy-Hard |
| CodeContests | 10000 | 165 | 100 lines | Medium-Hard |
| LiveCodeBench | - | 400 | 120 lines | Medium |

### Math Datasets

| Dataset | Train | Test | Avg. Steps | Domain |
|---------|-------|------|------------|--------|
| AIME24 | - | 30 | 5-10 | Competition |
| AIME25 | - | 30 | 5-10 | Competition |
| OlympiadBench | 200 | 100 | 8-15 | Olympiad |

### Game/Planning Datasets

| Environment | Train | Test | Grid Size | Complexity |
|-------------|-------|------|-----------|------------|
| Sokoban | 1000 | 200 | 6×6 | Medium |
| Sudoku | 1000 | 200 | 4×4 | Easy |
| Plan-Path | 1000 | 200 | 10×10 | Medium |

## Verification

After dataset preparation, verify the data:

```bash
# Check dataset structure
ls -R data/

# Verify data format
python -c "
from pettingllms.utils import load_dataset
data = load_dataset('data/math/train/aime24/')
print(f'Loaded {len(data)} problems')
"
```

## Next Steps

After preparing datasets:

- Run your first training: [Quick Start Guide](quick-start.md)
- Configure training parameters: [Training Guide](training.md)
- Set up evaluation: [Evaluation Guide](evaluation.md)

