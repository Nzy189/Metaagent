# **Data Preparation**

Before running any training or evaluation, prepare the task-specific datasets.

---

## **Quick Start**

```bash
# Code generation tasks (APPS, CodeContests, LiveCodeBench)
python scripts/dataprocess/load_code.py

# Math reasoning tasks (AIME24/25, OlympiadBench)
python scripts/dataprocess/load_math.py

# Planning/game tasks (Sokoban, Sudoku)
python scripts/dataprocess/load_sokoban.py
```

---

## **Output Structure**

All datasets are saved in the `datasets/` directory:

```
datasets/
├── code/
│   ├── train/
│   │   ├── apps_train.parquet
│   │   └── code_contests_train.parquet
│   └── test/
│       ├── apps_test.parquet
│       ├── code_contests_test.parquet
│       └── livecodebench_test.parquet
├── math/
│   ├── train/
│   └── test/
└── ...
```

---

## **Data Format**

For each task, all datasets share the same schema. Example for **Code Generation**:

- `question`: Problem description
- `test_input`: List of test case inputs
- `test_output`: List of expected outputs
- `golden_code`: Reference solution (optional)

---

## **Custom Dataset**

To add a custom dataset:

1. Create a script in `scripts/dataprocess/load_custom.py`
2. Load, process, and save as Parquet
3. Follow the same schema as existing datasets

Example:

```python
# scripts/dataprocess/load_custom.py
import pandas as pd
from datasets import load_dataset

def load_custom_data():
    # Load from HuggingFace
    dataset = load_dataset("your-dataset-name")
    
    # Process to match schema
    processed = []
    for item in dataset:
        processed.append({
            "question": item["problem"],
            "test_input": item["inputs"],
            "test_output": item["outputs"]
        })
    
    # Save as Parquet
    df = pd.DataFrame(processed)
    df.to_parquet("datasets/custom/train/custom_train.parquet")

if __name__ == "__main__":
    load_custom_data()
```

---

## **Next Steps**

Continue with environment setup:

- Understand the framework architecture: [Core Architecture](core-architecture.md)
- Learn about configuration system: [Configuration](configuration.md)
- Set up registrations: [Registration](registration.md)
