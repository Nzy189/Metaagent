# Three-Level Agent Specialization

## What is "Specialization"?

In PettingLLMs, **"Specialization"** refers to **how agents distinguish themselves from each other**, not whether the model is trained. In all three levels (L1, L2, L3), the models are trained using reinforcement learning (AT-GRPO). The key difference is the **mechanism used to create distinct agent behaviors**:

- **L1 (Prompt-based)**: Agents share the same model parameters but are distinguished by different system prompts
- **L2 (LoRA-based)**: Agents share the base model but have separate LoRA adapters that are trained independently
- **L3 (Full Model)**: Each agent has a completely separate model with independent parameters

> **Important**: Even in L1 (Prompt-based specialization), the shared model is still trained with RL. The "prompt-based" refers to how agents are differentiated, not the training methodology.

---

PettingLLMs provides three different levels of agent specialization methods, allowing you to flexibly choose based on task complexity, computational resources, and performance requirements: **Prompt Specialization (L1)**, **LoRA Specialization (L2)**, and **Full Model Specialization (L3)**.

## Overview

| Specialization Level | Parameter Sharing | Compute Cost | Training Complexity | Use Case |
|---------------------|-------------------|--------------|---------------------|----------|
| **L1: Prompt** | Fully shared base model | Lowest | Simplest | Similar roles, limited resources |
| **L2: LoRA** | Shared base model + separate LoRA | Medium | Medium | Balance performance and efficiency |
| **L3: Full Model** | Completely independent models | Highest | Most complex | Very different roles, max performance |

---

## L0: Single-Agent Baseline

Before introducing multi-agent specialization, L0 represents the single-agent baseline configuration for comparison.

### Features

- **Single agent**: Only one agent completes the entire task
- **No collaboration**: No multi-agent interaction or coordination
- **Baseline performance**: Used to evaluate multi-agent system improvements

### Configuration Example

```yaml
# pettingllms/config/math/math_L0_single_agent.yaml

specialization: "prompt"  # Set to prompt, but only uses single agent

# Single agent configuration
agent_policy_configs:
  num_agents: 2  # Defines 2 agents but only uses one
  policy_list: ["reasoning_generator", "tool_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "shared_model"
    agent_1:
      name: "tool_generator"
      policy_name: "shared_model"

# Only one agent interacts
multi_agent_interaction:
  turn_order: ["reasoning_generator"]  # Only use one agent
  num_interacting_agents: 1

# Only one base model
base_models:
  policy_0:
    path: "your base model path"
    name: "shared_model"
```

### Use Cases

- Establish performance baseline
- Validate necessity of multi-agent collaboration
- Rapid prototyping and testing

---

## L1: Prompt-Based Specialization

Prompt-based specialization distinguishes agent roles through different **system prompts**, while all agents share the same base model parameters. **The shared model is still trained with RL**, but agents are differentiated by their prompts.

### How It Works

```
┌─────────────────────────────────────┐
│     Shared Base Model (Trained)     │
│      Same parameters for all        │
└─────────────────────────────────────┘
          ↓              ↓
  ┌──────────────┐  ┌──────────────┐
  │ Tool Agent   │  │ Plan Agent   │
  └──────────────┘  └──────────────┘
  Different         Different
  system prompts    system prompts
```

### Advantages

- ✅ **Most parameter efficient**: Only need to train and store one model
- ✅ **Simple deployment**: Only need to load one model to GPU
- ✅ **Fast training**: Gradient updates only for one set of parameters
- ✅ **High sample efficiency**: All agents' experiences contribute to shared learning

### Disadvantages

- ❌ **Limited specialization**: Relies on model's prompt understanding capability
- ❌ **Role conflict risk**: Different roles may interfere with each other
- ❌ **Lower performance ceiling**: Best for similar roles

### Configuration Details

```yaml
# pettingllms/config/math/math_L1_prompt.yaml

specialization: "prompt"  # Key: specify specialization type as prompt

resource:
  nnodes: 1
  n_gpus_per_node: 8

# Define only one base model
base_models:
  policy_0:
    path: "Qwen/Qwen2.5-Math-7B-Instruct"
    name: "shared_model"  # All agents share this model

# Agent configuration: multiple agents, same model
agent_policy_configs:
  num_agents: 2
  policy_list: ["reasoning_generator", "tool_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "shared_model"  # ← Use shared model
      enable_thinking: false
      train_temperature: 1.0
      val_temperature: 0.6
    agent_1:
      name: "tool_generator"
      policy_name: "shared_model"  # ← Use shared model
      enable_thinking: false
      train_temperature: 1.0
      val_temperature: 0.6

# Multi-agent interaction order
multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2

# Training configuration for the single model
models:
  model_0:
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}
    ppo_trainer_config:
      # ... PPO training config
```

### How Prompts Are Defined

In the code, different agent prompts are dynamically generated through the `update_from_env` method:

```python
# pettingllms/multi_agent_env/stateful/agents/tool_agent.py

class ToolAgent(Agent):
    def update_from_env(self, turn_idx: int, env_data: Env):
        formatted_prompt = (
            "You are an AI assistant specialized in solving planning problems "
            "through code generation. Instructions:\n"
            "1. Write Python code enclosed in ```python ```\n"
            "2. Your code should output an action sequence using print()\n"
            # ... more tool agent specific instructions
        )
```

```python
# pettingllms/multi_agent_env/stateful/agents/plan_agent.py

class PlanAgent(Agent):
    def update_from_env(self, turn_idx: int, env_data: Env):
        formatted_prompt = (
            "You are a planning and reasoning agent. "
            "You will receive: The original task description, "
            "The Code Agent's code, The code execution output. "
            "Your job is to reason carefully, decide the final action..."
            # ... more plan agent specific instructions
        )
```

### Running Command

```bash
python -m pettingllms.trainer.train \
    --config-name math_L1_prompt \
    base_models.policy_0.path="Qwen/Qwen2.5-Math-7B-Instruct"
```

---

## L2: LoRA-Based Specialization

LoRA (Low-Rank Adaptation) specialization trains **independent LoRA adapters** for each agent while sharing the base model. **The base model and LoRA adapters are both trained with RL**, but agents are differentiated by their separate LoRA parameters.

### How It Works

```
┌─────────────────────────────────────┐
│   Shared Base Model (Frozen)        │
│   Parameters frozen, not updated    │
└─────────────────────────────────────┘
          ↓              ↓
  ┌──────────────┐  ┌──────────────┐
  │ LoRA Adapter │  │ LoRA Adapter │
  │  (tool)      │  │  (plan)      │
  └──────────────┘  └──────────────┘
  Independently     Independently
  trained params    trained params
          ↓              ↓
  ┌──────────────┐  ┌──────────────┐
  │ Tool Agent   │  │ Plan Agent   │
  └──────────────┘  └──────────────┘
```

### Advantages

- ✅ **High parameter efficiency**: Only train small LoRA parameters (typically < 1% of model size)
- ✅ **Strong specialization**: Each agent has independent adapters
- ✅ **Easy deployment**: Shared base model, only need to switch LoRA
- ✅ **Balanced performance**: Good trade-off between efficiency and performance

### Disadvantages

- ⚠️ **Increased training complexity**: Need to manage multiple LoRA adapters
- ⚠️ **Increased memory overhead**: Need to store LoRA parameters for each agent

### Configuration Details

```yaml
# pettingllms/config/math/math_L2_lora.yaml

specialization: "lora"  # Key: specify specialization type as lora

# LoRA hyperparameters
lora_rank: 16      # LoRA rank (r), controls adapter capacity
lora_alpha: 32     # LoRA scaling factor

resource:
  nnodes: 1
  n_gpus_per_node: 8

# Still define only one base model
base_models:
  policy_0:
    path: "Qwen/Qwen2.5-Math-7B-Instruct"
    name: "shared_model"

# Agent configuration: each agent automatically gets independent LoRA adapter
agent_policy_configs:
  num_agents: 2
  policy_list: ["reasoning_generator", "tool_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "shared_model"  # Shares base model
      # System creates independent LoRA adapter for this agent
    agent_1:
      name: "tool_generator"
      policy_name: "shared_model"  # Shares base model
      # System creates independent LoRA adapter for this agent

multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2

training:
  lora_rank: ${lora_rank}       # Reference top-level LoRA parameters
  lora_alpha: ${lora_alpha}
  checkpoint_dir: checkpoints   # LoRA weights save directory
  save_freq: 40                 # Save frequency

models:
  model_0:
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}
    ppo_trainer_config:
      # ... PPO training config
```

### LoRA Parameter Explanation

- **`lora_rank`** (r): Rank of LoRA low-rank decomposition
  - Larger = stronger expressiveness, but more parameters
  - Typical values: 8, 16, 32, 64
  - Recommended: 16-32 for most tasks

- **`lora_alpha`** (α): LoRA scaling factor
  - Controls the magnitude of LoRA updates
  - Usually set to `2 * lora_rank`
  - Recommended: lora_alpha=32 when lora_rank=16

### Training Mechanism

In LoRA mode, the system will:
1. Freeze all base model parameters
2. Create independent LoRA modules for each agent
3. Only update LoRA parameters (typically just a few MB)
4. Dynamically load the corresponding agent's LoRA weights during inference

### Running Command

```bash
python -m pettingllms.trainer.train \
    --config-name math_L2_lora \
    base_models.policy_0.path="Qwen/Qwen2.5-Math-7B-Instruct" \
    lora_rank=16 \
    lora_alpha=32
```

---

## L3: Full Model Specialization

Full Model specialization trains **completely independent models** for each agent with no parameter sharing. **Each model is independently trained with RL**, and agents are differentiated by having separate full models.

### How It Works

```
  ┌──────────────┐  ┌──────────────┐
  │ Full Model 0 │  │ Full Model 1 │
  │  (tool)      │  │  (plan)      │
  │ Completely   │  │ Completely   │
  │ independent  │  │ independent  │
  └──────────────┘  └──────────────┘
          ↓              ↓
  ┌──────────────┐  ┌──────────────┐
  │ Tool Agent   │  │ Plan Agent   │
  └──────────────┘  └──────────────┘
```

### Advantages

- ✅ **Strongest specialization**: Each agent has completely independent representation space
- ✅ **No role conflicts**: Agents are completely isolated
- ✅ **Highest performance ceiling**: Theoretically can achieve best performance
- ✅ **Maximum flexibility**: Can use different base models for different agents

### Disadvantages

- ❌ **Huge parameter count**: Need to store multiple complete models
- ❌ **High compute cost**: Training and inference require multiple times the resources
- ❌ **Low sample efficiency**: Each agent learns independently

### Configuration Details

```yaml
# pettingllms/config/math/math_L3_model.yaml

specialization: "full"  # Key: specify specialization type as full model

resource:
  nnodes: 1
  n_gpus_per_node: 8  # May need more GPUs

# Define multiple independent base models
base_models:
  policy_0:
    path: "Qwen/Qwen2.5-Math-7B-Instruct"
    name: "reasoning_generator_model"  # Dedicated to reasoning agent
  policy_1:
    path: "Qwen/Qwen2.5-Coder-7B-Instruct"  # Can use different models!
    name: "tool_generator_model"  # Dedicated to tool agent

# Agent configuration: each agent maps to a different model
agent_policy_configs:
  num_agents: 2
  policy_list: ["reasoning_generator", "tool_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "reasoning_generator_model"  # ← Use independent model
    agent_1:
      name: "tool_generator"
      policy_name: "tool_generator_model"  # ← Use independent model

multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2

# Need to define training config for each model
models:
  model_0:
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}
    ppo_trainer_config:
      # ... PPO training config
  
  model_1:  # Second model configuration
    path: ${base_models.policy_1.path}
    name: ${base_models.policy_1.name}
    ppo_trainer_config:
      # ... PPO training config
```

### Heterogeneous Model Support

A powerful feature of Full Model mode is the ability to use **different base models** for different agents:

```yaml
base_models:
  policy_0:
    path: "Qwen/Qwen2.5-Math-7B-Instruct"  # Math-specialized model
    name: "reasoning_generator_model"
  policy_1:
    path: "Qwen/Qwen2.5-Coder-7B-Instruct"  # Code-specialized model
    name: "tool_generator_model"
```

### Running Command

```bash
python -m pettingllms.trainer.train \
    --config-name math_L3_model \
    base_models.policy_0.path="Qwen/Qwen2.5-Math-7B-Instruct" \
    base_models.policy_1.path="Qwen/Qwen2.5-Coder-7B-Instruct"
```

---

## Configuration Comparison Summary

### Key Configuration Fields

| Config Field | L0: Single | L1: Prompt | L2: LoRA | L3: Full Model |
|--------------|-----------|-----------|----------|----------------|
| `specialization` | `"prompt"` | `"prompt"` | `"lora"` | `"full"` |
| `base_models` count | 1 | 1 | 1 | N (num agents) |
| `models` count | 1 | 1 | 1 | N (num agents) |
| `agent_configs.policy_name` | All same | All same | All same | All different |
| `lora_rank` / `lora_alpha` | Not needed | Not needed | Required | Optional |
| `num_interacting_agents` | 1 | ≥2 | ≥2 | ≥2 |

### Resource Requirements (7B model example)

| Metric | L0 | L1: Prompt | L2: LoRA | L3: Full (2 agents) |
|--------|-----|-----------|----------|---------------------|
| **Model Parameters** | ~7B | ~7B | ~7B + 16M×2 | ~14B |
| **GPU Memory (Training)** | ~28 GB | ~28 GB | ~30 GB | ~56 GB |
| **GPU Memory (Inference)** | ~14 GB | ~14 GB | ~14 GB | ~28 GB |
| **Disk Storage** | ~14 GB | ~14 GB | ~14.1 GB | ~28 GB |
| **Training Time (Relative)** | 1x | 1x | 1.2x | 2x |

### Performance Expectations

Based on our experimental results:

| Task Type | L0 | L1: Prompt | L2: LoRA | L3: Full Model |
|-----------|-----|-----------|----------|----------------|
| **Math Reasoning (AIME)** | Baseline | +15% | +28% | +35% |
| **Code Generation (APPS)** | Baseline | +12% | +25% | +32% |
| **Planning (Sokoban)** | Baseline | +18% | +30% | +38% |

---

## Selection Guide

### When to Use L0: Single Agent?

- ✅ Establish performance baseline
- ✅ Simple tasks that don't need multi-agent collaboration
- ✅ Rapid prototyping

### When to Use L1: Prompt Specialization?

- ✅ Limited resources (single GPU or small cluster)
- ✅ Agent roles are similar (e.g., all text generation)
- ✅ Fast experimentation and iteration
- ✅ Limited sample data

**Example Scenarios**:
- QA systems: Questioner + Answerer
- Simple dialogue: User Agent + Assistant Agent

### When to Use L2: LoRA Specialization?

- ✅ Need strong agent specialization
- ✅ Have moderate computational resources
- ✅ Seek balance between performance and efficiency
- ✅ Need flexible deployment (can quickly switch LoRA)

**Example Scenarios**:
- Math problem solving: Tool Agent + Reasoning Agent
- Code development: Coder + Tester
- Search tasks: Query Agent + Reasoning Agent

**Recommended: The preferred choice for most scenarios!**

### When to Use L3: Full Model Specialization?

- ✅ Pursue ultimate performance
- ✅ Have abundant computational resources
- ✅ Agent roles are vastly different
- ✅ Can use different types of base models

**Example Scenarios**:
- Complex coding competitions: Python Coder (CodeLlama) + Verifier (GPT-4)
- Multimodal tasks: Vision Agent (CLIP) + Language Agent (LLaMA)
- Expert systems: Domain Expert (domain model) + General Reasoner (general model)

---

## Implementation Details

### Agent-Policy Mapping Mechanism

In `MultiAgentsExecutionEngine`, the system manages agent-to-policy mapping through `agent_policy_mapping`:

```python
# pettingllms/trainer/multi_agents_execution_engine.py

def __init__(
    self,
    config,
    agent_policy_mapping=None,  # Define agent → policy mapping
    lora_differ_mode=False,     # Whether to enable LoRA mode
    agent_lora_mapping=None,    # Mapping in LoRA mode
):
    self.agent_policy_mapping = agent_policy_mapping or {}
    self.lora_differ_mode = lora_differ_mode
    self.agent_lora_mapping = agent_lora_mapping or {}
```

### LoRA Dynamic Loading

In LoRA mode, the system dynamically loads the corresponding LoRA weights for each agent during execution:

```python
# Pseudo-code example
if self.lora_differ_mode:
    current_agent = turn_order[turn_idx]
    lora_id = self.agent_lora_mapping[current_agent]
    model.load_lora_adapter(lora_id)  # Load corresponding LoRA
```

### Configuration File Inheritance

PettingLLMs uses Hydra configuration system with support for inheritance:

```yaml
# Base configuration
defaults:
  - ../ppo_trainer@models.model_0.ppo_trainer_config: eval
  - _self_

# Child config can override specific parameters
specialization: "lora"
lora_rank: 16
```

---

## Frequently Asked Questions (FAQ)

### Q1: How to migrate from L1 to L2?

Just modify the configuration file:

```yaml
# From
specialization: "prompt"

# To
specialization: "lora"
lora_rank: 16
lora_alpha: 32
```

No training code changes needed.

### Q2: Can I mix LoRA and Full Model?

Yes! You can use LoRA for some agents and full models for others:

```yaml
agent_configs:
  agent_0:
    name: "reasoning_generator"
    policy_name: "shared_model"  # Use LoRA
  agent_1:
    name: "tool_generator"
    policy_name: "independent_model"  # Use independent model
```

### Q3: How to choose LoRA rank?

General guidelines:
- **Simple tasks**: rank=8
- **Medium tasks**: rank=16 (recommended)
- **Complex tasks**: rank=32 or higher

Tune based on validation set performance.

### Q4: Can Full Model mode use different model sizes?

Yes! For example:

```yaml
base_models:
  policy_0:
    path: "Qwen/Qwen2.5-7B"  # 7B model
  policy_1:
    path: "Qwen/Qwen2.5-14B"  # 14B model
```

### Q5: How to deploy trained LoRA weights?

LoRA weights are saved in the `checkpoints/` directory:

```bash
checkpoints/
├── reasoning_generator_lora/
│   ├── adapter_config.json
│   └── adapter_model.bin
└── tool_generator_lora/
    ├── adapter_config.json
    └── adapter_model.bin
```

To load:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
model = PeftModel.from_pretrained(base_model, "checkpoints/reasoning_generator_lora")
```

---

## Next Steps

Continue exploring core concepts:

- Understand the training algorithm: [AT-GRPO Algorithm](at-grpo.md)
- Learn about distributed training: [Training System](training-system.md)
- Return to concepts overview: [Core Concepts](overview.md)

---

## Related Links

- [Core Concepts](overview.md) - Core concepts overview
- [AT-GRPO Algorithm](at-grpo.md) - AT-GRPO algorithm details
- [Training System](training-system.md) - Training system architecture

---



