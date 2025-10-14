# Training on Math Tasks

Train agents for mathematical reasoning.

## Supported Datasets

- **AIME 2024/2025**: Competition math
- **OlympiadBench**: Olympiad-level problems

## Dataset Preparation

```bash
python scripts/dataprocess/load_math.py
```

Creates:
```
datasets/math/
├── train/
│   ├── aime24/
│   ├── aime25/
│   └── olympiad/
└── test/
    └── (same structure)
```

## Training

```bash
bash scripts/train/math.sh
```

## Configuration

```python
config = {
    "model_name": "Qwen/Qwen2.5-8B-Instruct",
    
    "agents": [
        {
            "name": "tool_agent",
            "system_prompt": "Execute Python/calculator...",
        },
        {
            "name": "reasoner",
            "system_prompt": "Reason mathematically...",
        }
    ],
    
    "env": {
        "max_turns": 10,
        "tools": ["python", "calculator"],
        "verification": "exact_match",
    },
    
    "reward": {
        "progress_weight": 0.3,
        "answer_weight": 0.7,
        "partial_credit": True,
    },
}
```

## Workflow

```
1. Tool agent performs calculations
2. Reasoner interprets results
3. Reasoner plans next steps
4. Repeat until answer produced
5. Verify answer correctness
```

## Results

### AIME 2024

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 13.4% |
| 1.7B | AT-GRPO | **16.7%** |
| 8B | Baseline | 18.3% |
| 8B | AT-GRPO | **57.0%** |

### AIME 2025

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 9.8% |
| 1.7B | AT-GRPO | **18.3%** |
| 8B | Baseline | 20.0% |
| 8B | AT-GRPO | **40.0%** |

### OlympiadBench

| Model | Method | Accuracy |
|-------|--------|----------|
| 1.7B | Baseline | 22.2% |
| 1.7B | AT-GRPO | **39.6%** |
| 8B | Baseline | 55.0% |
| 8B | AT-GRPO | **56.8%** |

## Training Time

- **1.7B**: 16 hours (8 GPUs)
- **8B**: 32 hours (16 GPUs)

## Tips

### For Better Reasoning

- Reward intermediate steps
- Encourage tool usage
- Verify calculation accuracy

### For Better Tool Use

- Train on diverse problems
- Reward correct tool choice
- Handle tool errors gracefully

## Next Steps

- [Evaluation Guide](../evaluation/guide.md)
- [Benchmark Results](../results/benchmarks.md)

