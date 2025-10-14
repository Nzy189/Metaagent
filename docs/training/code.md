# Training on Code Tasks

Train agents for competitive programming tasks.

## Supported Datasets

- **APPS**: Competitive programming problems
- **CodeContests**: Programming contest problems
- **LiveCodeBench**: Live coding benchmarks

## Dataset Preparation

```bash
python scripts/dataprocess/load_code.py
```

Creates:
```
datasets/code/
├── train/ (APPS, CodeContests)
└── test/ (All three datasets)
```

## Training

### Shared Policy

```bash
bash scripts/train/code_single_policy.sh
```

### Per-Role Policies (Recommended)

```bash
bash scripts/train/code_two_policy.sh
```

## Configuration

```python
config = {
    "model_name": "Qwen/Qwen2.5-8B-Instruct",
    
    "agents": [
        {
            "name": "tester",
            "policy_id": "policy_tester",
            "system_prompt": "Write comprehensive unit tests...",
        },
        {
            "name": "coder",
            "policy_id": "policy_coder",
            "system_prompt": "Implement correct solution...",
        }
    ],
    
    "env": {
        "max_turns": 10,
        "timeout": 10,  # Test execution timeout
        "language": "python3",
    },
    
    "reward": {
        "alpha": 0.7,  # Global weight
        "test_pass_rate": "global",
        "test_quality": "tester_local",
        "code_quality": "coder_local",
    },
}
```

## Workflow

```
1. Tester writes/refines unit tests
2. Coder implements/refines code
3. Environment runs tests
4. If tests pass → Success
5. Otherwise → Refine and repeat
```

## Results

### LiveCodeBench

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 11.6% |
| 1.7B | AT-GRPO (shared) | 20.9% |
| 1.7B | AT-GRPO (per-role) | **24.0%** |
| 8B | Baseline | 22.8% |
| 8B | AT-GRPO (per-role) | **33.1%** |

### APPS

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 16.2% |
| 1.7B | AT-GRPO | **18.6%** |
| 8B | Baseline | 30.2% |
| 8B | AT-GRPO | **46.5%** |

### CodeContests

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 3.6% |
| 1.7B | AT-GRPO | **7.8%** |
| 8B | Baseline | 15.75% |
| 8B | AT-GRPO | **18.1%** |

## Training Time

- **1.7B**: 24 hours (8 GPUs)
- **8B**: 48 hours (16 GPUs)

## Tips

### For Better Test Generation

- Train tester on diverse test cases
- Reward edge case coverage
- Penalize trivial tests

### For Better Code Quality

- Reward code correctness
- Consider time/space complexity
- Encourage readable code

## Next Steps

- [Math Training](math.md)
- [Benchmark Results](../results/benchmarks.md)

