# Configuration Parameters Reference Guide

This comprehensive guide explains all configuration parameters used in PettingLLMs training and evaluation. Configuration files use **Hydra** format (YAML-based hierarchical configs).

---

## Table of Contents

1. [Configuration File Overview](#configuration-file-overview)
2. [Top-Level Parameters](#top-level-parameters)
3. [Model Configuration (`models.*`)](#model-configuration-models)
4. [Data Configuration (`data.*`)](#data-configuration-data)
5. [Environment Configuration (`env.*`)](#environment-configuration-env)
6. [Multi-Agent Interaction (`multi_agent_interaction.*`)](#multi-agent-interaction-multi_agent_interaction)
7. [Agent Policy Configuration (`agent_policy_configs.*`)](#agent-policy-configuration-agent_policy_configs)
8. [Trainer Configuration (`trainer.*`)](#trainer-configuration-trainer)
9. [Resource Configuration (`resource.*`)](#resource-configuration-resource)
10. [Command-Line Overrides](#command-line-overrides)
11. [Example Configurations](#example-configurations)

---

## Configuration File Overview

Configuration files are located in `pettingllms/config/{task}/`:

```
pettingllms/config/
├── math/
│   ├── math_single_policy.yaml       # Math task, single shared policy
│   ├── math_two_policies.yaml        # Math task, role-specialized policies
│   └── ...
├── code/
│   ├── code_single_policy.yaml       # Code task, single shared policy
│   └── ...
├── stateful/
│   ├── stateful_single_policy.yaml   # Planning/game tasks
│   └── ...
└── ppo_trainer/
    ├── base.yaml                      # Base PPO trainer config
    ├── eval.yaml                      # Evaluation-specific overrides
    └── ...
```

### Hydra Defaults Mechanism

Configs use Hydra's composition feature:

```yaml
defaults:
  - ../ppo_trainer@models.model_0.ppo_trainer_config: eval
  - _self_
```

This merges the `ppo_trainer/eval.yaml` config into `models.model_0.ppo_trainer_config`.

---

## Top-Level Parameters

### `mode`
**Type**: `string`  
**Options**: `"train"`, `"validate"`, `"eval"`  
**Description**: Training/evaluation mode. Controls whether the system trains, validates, or evaluates policies.

**Example**:
```yaml
mode: "validate"
```

---

### `sample_mode`
**Type**: `string`  
**Options**: `"independent"`, `"tree"`  
**Description**: Sampling strategy for multi-agent rollouts.
- `"independent"`: Each agent samples independently at each turn
- `"tree"`: Tree-structured sampling (AT-GRPO algorithm)

**Example**:
```yaml
sample_mode: "tree"  # Use tree-structured sampling for AT-GRPO
```

---

### `enable_thinking`
**Type**: `boolean`  
**Default**: `false`  
**Description**: Enables explicit "thinking" or reasoning steps in agent responses before taking actions.

**Example**:
```yaml
enable_thinking: true  # Enable explicit reasoning
```

---

### `benchmark`
**Type**: `string`  
**Options**: Task-dependent
- Math: `"AIME24"`, `"AIME25"`, `"OlympiadBench"`, `"CodeForces"`
- Code: `"APPS"`, `"CodeContests"`, `"LiveCodeBench"`
- Planning: `"PlanPath"`, `"Sokoban"`, `"sudoku4x4"`

**Description**: Specifies which benchmark dataset to use for training/evaluation.

**Example**:
```yaml
benchmark: "AIME24"  # Use AIME 2024 dataset
```

---

### `if_dapo`
**Type**: `boolean`  
**Default**: `false`  
**Description**: Enables **DAPO** (Data Augmented Policy Optimization) filtering method for experience selection.

**Example**:
```yaml
if_dapo: true  # Use DAPO filtering
```

---

### `project_name`
**Type**: `string`  
**Description**: W&B project name for logging experiments.

**Example**:
```yaml
project_name: pettingllms
```

---

### `experiment_name`
**Type**: `string`  
**Description**: Unique experiment identifier for logging and checkpointing.

**Example**:
```yaml
experiment_name: math_single_policy_1.7B
```

---

### `logger`
**Type**: `list[string]`  
**Options**: `"console"`, `"wandb"`, `"tensorboard"`  
**Description**: Logging backends to use.

**Example**:
```yaml
logger: ['console', 'wandb']  # Log to console and W&B
```

---

## Model Configuration (`models.*`)

Defines model paths and their associated training configurations.

### Structure

```yaml
models:
  model_0:
    path: "/path/to/model"
    name: "model_identifier"
    ppo_trainer_config:
      # Nested PPO trainer config for this model
      ...
```

### `models.model_0.path`
**Type**: `string`  
**Description**: Filesystem path to the pretrained model checkpoint (HuggingFace format).

**Example**:
```yaml
models:
  model_0:
    path: "/home/nvidia/data/models/Qwen3-1.7B"
```

---

### `models.model_0.name`
**Type**: `string`  
**Description**: Human-readable identifier for this model. Used for logging and debugging.

**Example**:
```yaml
models:
  model_0:
    name: "reasoning_agent_model"
```

---

### `models.model_0.ppo_trainer_config`
**Type**: `dict`  
**Description**: Nested configuration for the PPO trainer associated with this model. Includes:
- `data`: Data-specific settings (sequence lengths)
- `actor_rollout_ref`: Actor/rollout/reference model settings
- `trainer`: GPU allocation and training settings

**Example**:
```yaml
models:
  model_0:
    ppo_trainer_config:
      data:
        max_prompt_length: ${data.max_prompt_length}
        max_response_length: ${data.max_response_length}
      actor_rollout_ref:
        model:
          path: ${models.model_0.path}
        rollout:
          n: ${data.gen_n_samples}
          temperature: ${data.sample_temperature}
          prompt_length: ${data.max_prompt_length}
          response_length: ${data.max_response_length}
          tensor_model_parallel_size: ${resource.n_gpus_per_node}
        trainer:
          n_gpus_per_node: ${resource.n_gpus_per_node}
```

**Explanation of nested fields**:
- **`data.max_prompt_length`**: Maximum tokens in the input prompt
- **`data.max_response_length`**: Maximum tokens in the model response
- **`actor_rollout_ref.model.path`**: Model checkpoint path
- **`actor_rollout_ref.rollout.n`**: Number of response samples per prompt
- **`actor_rollout_ref.rollout.temperature`**: Sampling temperature
- **`actor_rollout_ref.rollout.tensor_model_parallel_size`**: Tensor parallelism degree
- **`actor_rollout_ref.trainer.n_gpus_per_node`**: GPUs allocated to this trainer

---

## Data Configuration (`data.*`)

Controls dataset loading, sampling, and training batch configuration.

### `data.filter_method`
**Type**: `string`  
**Options**: `"mean"`, `"uid"`, `"dapo"`, `"std"`  
**Description**: Method for filtering/selecting high-quality experiences during training.
- `"mean"`: Select experiences with above-average rewards
- `"std"`: Select experiences within certain standard deviations
- `"dapo"`: Data-augmented policy optimization filtering
- `"uid"`: Uniform sampling (no filtering)

**Example**:
```yaml
data:
  filter_method: mean  # Select above-average experiences
```

---

### `data.filter_ratio`
**Type**: `float`  
**Range**: `[0.0, 1.0]`  
**Description**: Proportion of experiences to filter out. `0.0` = no filtering, `0.5` = keep top 50%.

**Example**:
```yaml
data:
  filter_ratio: 0.2  # Keep top 80% of experiences
```

---

### `data.gen_batch_size`
**Type**: `int`  
**Description**: Number of environments to run in parallel during rollout.

**Example**:
```yaml
data:
  gen_batch_size: 128  # Run 128 parallel environments
```

---

### `data.gen_n_samples`
**Type**: `int`  
**Description**: Number of response samples to generate per prompt (for tree sampling or independent sampling).

**Example**:
```yaml
data:
  gen_n_samples: 4  # Generate 4 samples per prompt
```

---

### `data.sample_temperature`
**Type**: `float`  
**Default**: `1.0`  
**Description**: Sampling temperature for model generation. Higher = more random, lower = more deterministic.

**Example**:
```yaml
data:
  sample_temperature: 1.0  # Standard temperature
```

---

### `data.val_freq`
**Type**: `int`  
**Description**: Validation frequency (in training steps). Validates every N steps.

**Example**:
```yaml
data:
  val_freq: 10  # Validate every 10 steps
```

---

### `data.resample_freq`
**Type**: `int`  
**Description**: Frequency (in epochs) to resample the training dataset. Higher = less frequent resampling.

**Example**:
```yaml
data:
  resample_freq: 1  # Resample every epoch
```

---

### `data.epoch_size`
**Type**: `int`  
**Description**: Number of training batches per epoch.

**Example**:
```yaml
data:
  epoch_size: 20  # 20 batches = 1 epoch
```

---

### `data.train_batch_size`
**Type**: `int`  
**Description**: Batch size for PPO training updates (number of experiences).

**Example**:
```yaml
data:
  train_batch_size: 256  # Train with 256 experiences per batch
```

---

### `data.val_batch_size`
**Type**: `int`  
**Description**: Batch size for validation rollouts.

**Example**:
```yaml
data:
  val_batch_size: 32  # 32 environments for validation
```

---

### `data.max_prompt_length`
**Type**: `int`  
**Description**: Maximum number of tokens in the input prompt. Longer prompts are truncated.

**Example**:
```yaml
data:
  max_prompt_length: 4096  # 4K token prompt limit
```

---

### `data.max_response_length`
**Type**: `int`  
**Description**: Maximum number of tokens in model responses. Longer responses are truncated.

**Example**:
```yaml
data:
  max_response_length: 8192  # 8K token response limit
```

---

## Environment Configuration (`env.*`)

Defines the task environment and its settings.

### `env.name`
**Type**: `string`  
**Description**: Environment type identifier (must match registration in `multiagentssys_register.py`).

**Options**:
- `"code_env"`: Code generation environment
- `"math_env"`: Math reasoning environment
- `"stateful_env"`: Planning/game environments (Sokoban, Plan-Path, etc.)

**Example**:
```yaml
env:
  name: math_env  # Use math reasoning environment
```

---

### `env.benchmark`
**Type**: `string`  
**Description**: Specific benchmark within the environment type.

**Options** (depends on `env.name`):
- For `math_env`: `"AIME24"`, `"AIME25"`, `"OlympiadBench"`, `"CodeForces"`
- For `code_env`: `"APPS"`, `"CodeContests"`, `"LiveCodeBench"`
- For `stateful_env`: `"PlanPath"`, `"Sokoban"`, `"sudoku4x4"`

**Example**:
```yaml
env:
  name: stateful_env
  benchmark: "PlanPath"  # 10x10 grid path planning
```

---

### `env.max_turns`
**Type**: `int`  
**Description**: Maximum number of agent-environment interaction turns per episode. Episode terminates early if agents complete the task.

**Example**:
```yaml
env:
  max_turns: 5  # Allow up to 5 turns per episode
```

---

### `env.resolve`
**Type**: `boolean`  
**Default**: `false`  
**Description**: (Advanced) Enables special resolution logic for certain environments.

---

### `env.multi_modal`
**Type**: `boolean`  
**Default**: `false`  
**Description**: Enables multi-modal inputs (e.g., images + text). Currently used for web environments.

---

### `env.batched_init`
**Type**: `boolean`  
**Default**: `true`  
**Description**: Enables batched environment initialization for efficiency.

---

### `env.map_size` (for `stateful_env` only)
**Type**: `int`  
**Description**: Grid size for planning/game tasks (e.g., 10 for 10x10 grid).

**Example**:
```yaml
env:
  name: stateful_env
  benchmark: "PlanPath"
  map_size: 10  # 10x10 grid
```

---

## Multi-Agent Interaction (`multi_agent_interaction.*`)

Defines how multiple agents interact within an episode.

### `multi_agent_interaction.turn_order`
**Type**: `list[string]`  
**Description**: Ordered list of agent names. Agents take turns in this order during each episode. Names must match registered agent keys in `AGENT_CLASSES`.

**Example**:
```yaml
multi_agent_interaction:
  turn_order: ["reasoning_agent", "tool_agent"]
  # Turn 0: reasoning_agent acts
  # Turn 1: tool_agent acts
  # Turn 2: reasoning_agent acts (cycle repeats)
```

**Important**: Agent names in `turn_order` must be registered in `pettingllms/trainer/multiagentssys_register.py`:
```python
AGENT_CLASSES = {
    "reasoning_agent": ReasoningAgent,
    "tool_agent": ToolAgent,
    ...
}
```

---

### `multi_agent_interaction.num_interacting_agents`
**Type**: `int`  
**Description**: Total number of agents participating in each episode. Should match the length of `turn_order`.

**Example**:
```yaml
multi_agent_interaction:
  turn_order: ["code_generator", "test_generator"]
  num_interacting_agents: 2  # Two agents
```

---

### `multi_agent_interaction.shared_observation`
**Type**: `boolean`  
**Default**: `true`  
**Description**: Whether agents share observations via the environment state.
- `true`: Agents can see each other's actions/outputs via `env.state`
- `false`: Agents only see their own observations

**Example**:
```yaml
multi_agent_interaction:
  shared_observation: true  # Agents collaborate via shared state
```

---

## Agent Policy Configuration (`agent_policy_configs.*`)

Defines which agents use which policies (models).

### `agent_policy_configs.num_agents`
**Type**: `int`  
**Description**: Total number of agents in the system.

**Example**:
```yaml
agent_policy_configs:
  num_agents: 2  # Two agents
```

---

### `agent_policy_configs.policy_list`
**Type**: `list[string]`  
**Description**: List of unique policy names. 
- **Single-policy setup**: List contains one policy name (agents share the same model)
- **Multi-policy setup**: List contains multiple policy names (role-specialized models)

**Example (single policy)**:
```yaml
agent_policy_configs:
  policy_list: ["reasoning_agent_model"]  # One policy for all agents
```

**Example (two policies)**:
```yaml
agent_policy_configs:
  policy_list: ["reasoning_agent_model", "tool_agent_model"]  # Separate policies
```

---

### `agent_policy_configs.agent_configs`
**Type**: `dict`  
**Description**: Per-agent configuration. Each agent gets:
- `name`: Agent type (must match `turn_order` entry)
- `policy_name`: Which policy/model this agent uses
- `sample_num`: Number of samples this agent generates per turn

**Example (single policy)**:
```yaml
agent_policy_configs:
  policy_list: ["reasoning_agent_model"]
  agent_configs:
    agent_0:
      name: "reasoning_agent"         # First agent in turn_order
      policy_name: "reasoning_agent_model"  # Uses the single policy
      sample_num: 4                    # Generates 4 samples per turn
    agent_1:
      name: "tool_agent"               # Second agent in turn_order
      policy_name: "reasoning_agent_model"  # SAME policy (shared)
      sample_num: 4
```

**Example (two policies)**:
```yaml
agent_policy_configs:
  policy_list: ["reasoning_agent_model", "tool_agent_model"]
  agent_configs:
    agent_0:
      name: "reasoning_agent"
      policy_name: "reasoning_agent_model"  # Policy 1
      sample_num: 4
    agent_1:
      name: "tool_agent"
      policy_name: "tool_agent_model"       # Policy 2 (different!)
      sample_num: 4
```

---

## Trainer Configuration (`trainer.*`)

Controls the PPO training process.

### `trainer.device`
**Type**: `string`  
**Options**: `"cuda"`, `"cpu"`  
**Description**: Training device.

**Example**:
```yaml
trainer:
  device: cuda
```

---

### `trainer.n_gpus_per_node`
**Type**: `int`  
**Description**: Total number of GPUs available per node.

**Example**:
```yaml
trainer:
  n_gpus_per_node: 8  # 8 GPUs per node
```

---

### `trainer.nnodes`
**Type**: `int`  
**Description**: Number of compute nodes (for multi-node training).

**Example**:
```yaml
trainer:
  nnodes: 1  # Single-node training
```

---

### `trainer.balance_batch`
**Type**: `boolean`  
**Default**: `true`  
**Description**: Balances training batches across GPUs for even load distribution.

---

### `trainer.total_epochs`
**Type**: `int`  
**Description**: Total training epochs. Usually set to `1` with `total_training_steps` controlling training length.

**Example**:
```yaml
trainer:
  total_epochs: 1
```

---

### `trainer.total_training_steps`
**Type**: `int`  
**Description**: Total number of PPO update steps. Primary control for training duration.

**Example**:
```yaml
trainer:
  total_training_steps: 400  # 400 PPO updates
```

---

### `trainer.save_freq`
**Type**: `int`  
**Description**: Checkpoint saving frequency (in training steps). `-1` = only save at end.

**Example**:
```yaml
trainer:
  save_freq: 40  # Save checkpoint every 40 steps
```

---

### `trainer.resume_mode`
**Type**: `string`  
**Options**: `"auto"`, `"force"`, `"none"`  
**Description**: Checkpoint resumption behavior.
- `"auto"`: Resume if checkpoint exists
- `"force"`: Must resume (error if no checkpoint)
- `"none"`: Always start from scratch

**Example**:
```yaml
trainer:
  resume_mode: auto
```

---

### `trainer.resume_from_path`
**Type**: `string` or `null`  
**Description**: Explicit checkpoint path to resume from. `null` = auto-detect.

**Example**:
```yaml
trainer:
  resume_from_path: null  # Auto-detect
```

---

### `trainer.val_before_train`
**Type**: `boolean`  
**Default**: `true`  
**Description**: Run validation before starting training (to get baseline metrics).

---

### `trainer.test_freq`
**Type**: `int`  
**Description**: Testing frequency (in training steps). `-1` = no testing during training.

---

### `trainer.default_local_dir`
**Type**: `string`  
**Description**: Local directory for saving checkpoints.

**Example**:
```yaml
trainer:
  default_local_dir: checkpoints/pettingllms/math_single_policy
```

---

### `trainer.max_actor_ckpt_to_keep`
**Type**: `int`  
**Description**: Maximum number of actor checkpoints to retain. Older checkpoints are deleted.

**Example**:
```yaml
trainer:
  max_actor_ckpt_to_keep: 3  # Keep only last 3 checkpoints
```

---

## Resource Configuration (`resource.*`)

Hardware resource allocation settings.

### `resource.nnodes`
**Type**: `int`  
**Description**: Number of compute nodes.

**Example**:
```yaml
resource:
  nnodes: 1
```

---

### `resource.n_gpus_per_node`
**Type**: `int`  
**Description**: GPUs per node for inference and training.

**Example**:
```yaml
resource:
  n_gpus_per_node: 6  # 6 GPUs available
```

---

### `resource.trust_remote_code`
**Type**: `boolean`  
**Default**: `true`  
**Description**: Allow loading models with custom code (required for many HuggingFace models).

---

## Command-Line Overrides

Hydra allows overriding any config value from the command line:

```bash
python3 -m pettingllms.trainer.train \
    --config-path ../config/math \
    --config-name math_single_policy \
    models.model_0.path=/path/to/model \            # Override model path
    trainer.total_training_steps=400 \               # Override training steps
    data.gen_batch_size=128 \                        # Override batch size
    experiment_name=my_experiment_v2                 # Override experiment name
```

### Common Override Patterns

**Change model**:
```bash
models.model_0.path=/home/user/models/Qwen3-8B
```

**Adjust training length**:
```bash
trainer.total_training_steps=1000 \
data.epoch_size=50
```

**Increase parallelism**:
```bash
data.gen_batch_size=256 \
data.gen_n_samples=8
```

**Change GPU allocation**:
```bash
resource.n_gpus_per_node=8 \
models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=8
```

---

## Example Configurations

### Example 1: Math Single Policy (Shared Model)

**File**: `pettingllms/config/math/math_single_policy.yaml`

```yaml
# Both agents share one policy
mode: "validate"
sample_mode: "tree"
enable_thinking: false
benchmark: "AIME24"

data:
  filter_method: mean
  filter_ratio: 0.2
  gen_batch_size: 64
  gen_n_samples: 5
  max_prompt_length: 16384
  max_response_length: 8192

env:
  name: math_env
  benchmark: "CodeForces"
  max_turns: 3

multi_agent_interaction:
  turn_order: ["reasoning_agent", "tool_agent"]
  num_interacting_agents: 2
  shared_observation: true

agent_policy_configs:
  num_agents: 2
  policy_list: ["reasoning_agent"]  # Single policy
  agent_configs:
    agent_0:
      name: "reasoning_agent"
      policy_name: "reasoning_agent_model"  # Shared policy
    agent_1:
      name: "tool_agent"
      policy_name: "reasoning_agent_model"  # Same policy

models:
  model_0:
    path: "/home/models/Qwen3-4B"
    name: "reasoning_agent_model"

trainer:
  total_training_steps: 400
  save_freq: 40
  default_local_dir: checkpoints/pettingllms/math_single_policy
```

**Usage**:
```bash
python3 -m pettingllms.trainer.train \
    --config-path ../config/math \
    --config-name math_single_policy
```

---

### Example 2: Planning Two Policies (Role-Specialized)

**File**: `pettingllms/config/stateful/stateful_two_policies.yaml`

```yaml
# Each agent has its own policy
mode: "validate"
sample_mode: "tree"
benchmark: "PlanPath"

env:
  name: stateful_env
  benchmark: "PlanPath"
  max_turns: 5
  map_size: 10

multi_agent_interaction:
  turn_order: ["plan_agent", "tool_call_agent"]
  num_interacting_agents: 2
  shared_observation: true

agent_policy_configs:
  num_agents: 2
  policy_list: ["plan_agent", "tool_agent"]  # Two separate policies
  agent_configs:
    agent_0:
      name: "plan_agent"
      policy_name: "plan_agent_model"      # Policy 1
    agent_1:
      name: "tool_call_agent"
      policy_name: "tool_agent_model"      # Policy 2

models:
  model_0:
    path: "/home/models/Qwen3-4B"
    name: "plan_agent_model"
  model_1:
    path: "/home/models/Qwen3-4B"
    name: "tool_agent_model"

trainer:
  total_training_steps: 200
  save_freq: -1
```

**Usage**:
```bash
python3 -m pettingllms.trainer.train \
    --config-path ../config/stateful \
    --config-name stateful_two_policies
```

---

## Summary: Key Configuration Patterns

| Pattern | Use Case | `policy_list` | `agent_configs.policy_name` |
|---------|----------|---------------|---------------------------|
| **Shared Policy** | Agents learn from each other's experiences | `["model_A"]` | All agents use `"model_A"` |
| **Role-Specialized** | Agents learn task-specific behaviors | `["model_A", "model_B"]` | agent_0 uses `"model_A"`, agent_1 uses `"model_B"` |

### When to Use Shared vs. Role-Specialized

- **Shared Policy**: Use when agents perform similar tasks (e.g., both reason about math problems)
- **Role-Specialized**: Use when agents have distinct roles (e.g., code generator vs. test generator)

---

## Next Steps

- **Try modifying configs**: Start with provided examples and adjust parameters
- **Monitor training**: Use W&B to track metrics and rewards
- **Experiment with policies**: Compare single-policy vs. two-policy training

For more details on the training workflow, see:
- [Main Setup Guide](./ENVIRONMENT_SETUP_AND_CORE_CONCEPTS.md)
- [Training Scripts](./scripts/train/)

---

**Happy configuring! 🚀**


