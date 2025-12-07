# **Environment State**

The **Environment State** (`env_data.state`) is the central storage and communication hub for multi-agent coordination.

---

## **What Environment State Contains**

Environment state consists of two main categories of information:

### **1. Shared Agent Information**

Information that agents write and share with each other through the environment:

- **Agent outputs**: Generated code, test cases, reasoning steps, plans, etc.
- **Interaction history**: Previous attempts, actions, and results from all agents
- **Evaluation results**: Test pass ratios, correctness scores, match/mismatch cases
- **Coordination data**: Feedback between agents for iterative improvement

**Example (Code Environment)**:
```python
state.generated_code              # Written by CodeGenerationAgent
state.generated_test_input        # Written by UnitTestGenerationAgent
state.generated_code_history      # Accumulated by multiple agents
state.generated_test_vs_generated_code_mismatch_cases  # Shared feedback
```

### **2. Current Environment Information**

Task-specific data that defines the current problem and environment status:

- **Task definition**: Problem description, question, constraints
- **Ground truth**: Expected outputs, golden solutions, correct answers
- **Environment status**: Current turn, completion status, success flags
- **Task-specific data**: Domain-specific information (e.g., game state, web search results)

**Example (Code Environment)**:
```python
state.problem                     # Task: programming problem description
state.ground_truth_test_input     # Task: test inputs
state.ground_truth_test_output    # Task: expected outputs
state.golden_code                 # Task: reference solution (if available)
```

---

## **Design Principles**

1. **Single Source of Truth**: All shared information flows through environment state
2. **No Direct Agent Communication**: Agents only interact via `env_data.state`
3. **Persistent History**: State preserves complete interaction history
4. **Task-Specific Structure**: Each environment defines its own state dataclass

---

## **Environment Batch (EnvBatch)**

### **Purpose**

`EnvBatch` manages multiple environment instances for **parallel execution** during training and evaluation.

### **What It Does**

```python
# From: pettingllms/multi_agent_env/code/code_env.py
class CodeEnvBatch:
    def __init__(self, env_idx_list, env_indices, rollout_idx_list, samples, ...):
        self.env_list = []  # List of independent Env instances
        
        # Load multiple problems from dataset
        self.problem_list = load_problem_batch(env_indices, ...)
        
        # Create one environment per problem (with multiple samples)
        for i, problem in enumerate(self.problem_list):
            for s in range(samples):
                env = CodeEnv(env_idx=i, rollout_idx=..., ...)
                env.state = CodeEnvState(problem=problem["question"], ...)
                self.env_list.append(env)
```

### **Key Features**

1. **Parallel Execution**: Multiple environments run simultaneously for efficient training
2. **Independent States**: Each `Env` in `env_list` has its own isolated `state`
3. **Batch Processing**: Process multiple problems/tasks at once
4. **Sampling Support**: Create multiple rollouts per problem for exploration

### **Usage in Training**

- **Training**: Batch processes many problems simultaneously for faster learning
- **Evaluation**: Runs entire test sets in parallel
- **Each environment**: Has separate agents, separate state, separate rollout

**Example**:
```python
# Create batch with 100 problems, 4 samples each = 400 environments
env_batch = CodeEnvBatch(
    env_indices=range(100),      # 100 different problems
    samples=4,                    # 4 attempts per problem
    ...
)
# Result: env_batch.env_list contains 400 CodeEnv instances
# Each has independent state for parallel execution
```

---

## **Next Steps**

Continue exploring environment setup:

- Learn how agents interact with state: [Agent Functions](agent-functions.md)
- Configure environment settings: [Configuration](configuration.md)
- Understand component registration: [Registration](registration.md)
