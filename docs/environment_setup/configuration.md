# **Configuration**

This guide provides a comprehensive overview of the PettingLLMs configuration system. After registering your environments and agents, you need to configure training parameters, model paths, agent interactions, and other critical settings through configuration files.

---

## **Overview**

PettingLLMs uses a [Hydra](https://hydra.cc/)-based configuration system with all configuration files in YAML format. Configuration files are located in the `pettingllms/config/` directory, organized by task type:

```
pettingllms/config/
├── code/           # Code generation task configs
├── math/           # Mathematical reasoning configs
├── search/         # Web search task configs
├── stateful/       # Stateful planning configs
└── ppo_trainer/    # Default PPO trainer configs
```

**Key Design Principles**:
- **Modular**: Configurations are divided into logical sections for easy maintenance
- **Inheritable**: Reuse configurations through Hydra's `defaults` mechanism
- **Overridable**: Command-line arguments can override any config file setting
- **Type-safe**: Configurations correspond to dataclasses in code

---

## **Configuration File Structure**

Using `math_L3_model.yaml` as an example, a complete configuration file contains the following main sections:

```yaml
# 1. Hydra defaults inheritance
defaults:
  - ../ppo_trainer@models.model_0.ppo_trainer_config: eval
  - _self_

# 2. Specialization configuration
specialization: "lora"
lora_rank: 16
lora_alpha: 32

# 3. Resource configuration
resource:
  nnodes: 1
  n_gpus_per_node: 8
  trust_remote_code: true

# 4. Environment configuration
env:
  name: math_env
  dataset: "polaris"
  benchmark: "AIME24"
  max_turns: 5
  resolve: false
  multi_modal: false
  batched_init: true

# 5. Base model configuration
base_models:
  policy_0:
    path: "your base model path"
    name: "reasoning_generator_model"
  policy_1:
    path: "your base model path"
    name: "tool_generator_model"

# 6. Agent policy configuration
agent_policy_configs:
  num_agents: 2
  policy_list: ["reasoning_generator", "tool_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "reasoning_generator_model"
    agent_1:
      name: "tool_generator"
      policy_name: "reasoning_generator_model"

# 7. Multi-agent interaction configuration
multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2

# 8. Training configuration
training:
  device: cuda
  total_training_steps: 200
  project_name: pettingllms
  experiment_name: math_eval_single_policy
  logger: ['console', 'wandb']
  # ... more training parameters

# 9. Model configuration
models:
  model_0:
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}
    ppo_trainer_config:
      # ... PPO configuration
  model_1:
    path: ${base_models.policy_1.path}
    name: ${base_models.policy_1.name}
    ppo_trainer_config:
      # ... PPO configuration
```

---

## **Configuration Sections Explained**

### **1. Specialization Configuration**

Controls the model parameter update strategy, determining how model parameters are optimized during training.

```yaml
specialization: "lora"  # Parameter update method
lora_rank: 16           # LoRA rank
lora_alpha: 32          # LoRA scaling factor
```

**Parameter Details**:

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `specialization` | str | `"prompt"`, `"lora"`, `"full"` | Parameter update method:<br>• `"prompt"`: Optimize prompts only<br>• `"lora"`: Parameter-efficient fine-tuning with LoRA<br>• `"full"`: Full parameter fine-tuning |
| `lora_rank` | int | Typical: 8-64 | Rank of LoRA low-rank matrices. Higher values provide more expressiveness but require more computation |
| `lora_alpha` | int | Typical: `lora_rank * 2` | LoRA scaling hyperparameter controlling the influence of LoRA weights |

**When to Modify**:
- **Insufficient memory**: Use `"lora"` and reduce `lora_rank`
- **Poor performance**: Increase `lora_rank` or use `"full"`
- **Quick experiments**: Use `"prompt"` for prompt optimization

---

### **2. Resource Configuration**

Defines computational resources required for training and inference.

```yaml
resource:
  nnodes: 1                    # Number of nodes
  n_gpus_per_node: 8           # Number of GPUs per node
  trust_remote_code: true      # Whether to trust remote code
```

**Parameter Details**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `nnodes` | int | Number of nodes for training. Multi-node training requires distributed environment setup |
| `n_gpus_per_node` | int | Number of GPUs per node. Used for model parallelism and data parallelism |
| `trust_remote_code` | bool | Whether to trust remote code when loading HuggingFace models. Some models require `true` |

**When to Modify**:
- **Single-machine training**: Set `nnodes: 1`, adjust `n_gpus_per_node` to match available GPUs
- **Multi-node training**: Increase `nnodes` and configure distributed training environment
- **Model loading fails**: Set `trust_remote_code: true` if model contains custom code

---

### **3. Environment Configuration**

Defines core parameters for the task environment, shared by all agents.

```yaml
env:
  name: math_env              # Environment name
  dataset: "polaris"          # Dataset name
  benchmark: "AIME24"         # Evaluation benchmark
  max_turns: 5                # Maximum interaction turns
  resolve: false              # Whether to resolve environment
  multi_modal: false          # Whether to support multimodal
  batched_init: true          # Whether to batch initialize
```

**Parameter Details**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Environment type, must be registered in `multiagentssys_register.py`.<br>Options: `code_env`, `math_env`, `search_env`, `stateful_env`, etc. |
| `dataset` | str | Dataset name for training and evaluation. Different environments support different datasets |
| `benchmark` | str | Evaluation benchmark dataset for validating model performance |
| `max_turns` | int | Maximum interaction turns per task. Environment terminates after reaching this limit |
| `resolve` | bool | Whether to enable environment resolution (environment-specific) |
| `multi_modal` | bool | Whether to support multimodal input (text + images) |
| `batched_init` | bool | Whether to batch initialize environments. `true` improves efficiency |

**Common Configurations for Different Environments**:

**Code Generation (code_env)**:
```yaml
env:
  name: code_env
  dataset: "apps"
  benchmark: "LiveCodeBench"
  max_turns: 6
  resolve: true
  multi_modal: false
  batched_init: true
```

**Mathematical Reasoning (math_env)**:
```yaml
env:
  name: math_env
  dataset: "polaris"
  benchmark: "AIME24"
  max_turns: 5
  resolve: false
  multi_modal: false
  batched_init: true
```

**Web Search (search_env)**:
```yaml
env:
  name: search_env
  dataset: "hotpotqa"
  benchmark: "HotpotQA"
  max_turns: 8
  resolve: false
  multi_modal: false
  batched_init: true
```

**When to Modify**:
- **Tasks need more reasoning**: Increase `max_turns`
- **Switch datasets**: Modify `dataset` and `benchmark`
- **Multimodal tasks**: Set `multi_modal: true`

---

### **4. Base Models Configuration**

Specifies base LLM model paths and names for training.

```yaml
base_models:
  policy_0:
    path: "your base model path"          # Model file path
    name: "reasoning_generator_model"     # Model identifier
  policy_1:
    path: "your base model path"
    name: "tool_generator_model"
```

**Parameter Details**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `policy_N.path` | str | Model file path. Can be:<br>• Local path: `/path/to/model`<br>• HuggingFace path: `meta-llama/Llama-3-8B-Instruct`<br>• Model repository path |
| `policy_N.name` | str | Unique model identifier used to reference this model in the config |

**Example Configurations**:

**Using Local Models**:
```yaml
base_models:
  policy_0:
    path: "/home/user/models/llama-3-8b-instruct"
    name: "reasoning_model"
```

**Using HuggingFace Models**:
```yaml
base_models:
  policy_0:
    path: "meta-llama/Llama-3-8B-Instruct"
    name: "reasoning_model"
  policy_1:
    path: "Qwen/Qwen2.5-7B-Instruct"
    name: "tool_model"
```

**Multi-Agent Shared Model**:
```yaml
base_models:
  policy_0:
    path: "meta-llama/Llama-3-8B-Instruct"
    name: "shared_model"
  policy_1:
    path: "meta-llama/Llama-3-8B-Instruct"  # Same model
    name: "shared_model"
```

**When to Modify**:
- **Change base model**: Modify `path` to point to new model
- **Different models for agents**: Set different paths for different policies
- **Shared model for agents**: Use same path for all policies

---

### **5. Agent Policy Configuration**

Defines how many agents exist, their names, and corresponding model policies.

```yaml
agent_policy_configs:
  num_agents: 2                                    # Number of training agents
  policy_list: ["reasoning_generator", "tool_generator"]  # Policy name list
  agent_configs:
    agent_0:
      name: "reasoning_generator"                  # Agent name
      policy_name: "reasoning_generator_model"     # Corresponding model policy
    agent_1:
      name: "tool_generator"
      policy_name: "reasoning_generator_model"
```

**Parameter Details**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_agents` | int | Number of agents participating in training. Must match the count in `agent_configs` |
| `policy_list` | List[str] | List of all policy names, identifying different behavior policies |
| `agent_configs.agent_N.name` | str | Unique agent name, must be registered in `multiagentssys_register.py` |
| `agent_configs.agent_N.policy_name` | str | Model name this agent uses, corresponding to `name` defined in `base_models` |

**Important Relationships**:
```
agent_configs.agent_N.name → AGENT_CLASS_MAPPING (registration.py)
                           ↓
                      Agent Implementation

agent_configs.agent_N.policy_name → base_models.policy_N.name
                                  ↓
                              Model Path
```

**Agent Configurations for Different Tasks**:

**Mathematical Reasoning (Math)**:
```yaml
agent_policy_configs:
  num_agents: 2
  policy_list: ["reasoning_generator", "tool_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"      # Reasoning agent
      policy_name: "reasoning_model"
    agent_1:
      name: "tool_generator"           # Tool calling agent
      policy_name: "tool_model"
```

**Code Generation (Code)**:
```yaml
agent_policy_configs:
  num_agents: 2
  policy_list: ["code_generator", "test_generator"]
  agent_configs:
    agent_0:
      name: "code_generator"           # Code generation agent
      policy_name: "code_model"
    agent_1:
      name: "test_generator"           # Test generation agent
      policy_name: "test_model"
```

**Single Agent Configuration**:
```yaml
agent_policy_configs:
  num_agents: 1
  policy_list: ["single_agent"]
  agent_configs:
    agent_0:
      name: "single_agent"
      policy_name: "base_model"
```

**When to Modify**:
- **Add new agents**: Increase `num_agents`, add new entries in `agent_configs`, ensure registration in `multiagentssys_register.py`
- **Switch agent implementation**: Modify `name` field to use different agent classes
- **Change models**: Modify `policy_name` to point to different base models

---

### **6. Multi-Agent Interaction Configuration**

Defines agent execution order and collaboration patterns.

```yaml
multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]  # Agent execution order
  num_interacting_agents: 2                              # Number of agents per episode
```

**Parameter Details**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `turn_order` | List[str] | Agent execution order. Names must match `name` in `agent_configs`.<br>Agents execute sequentially in this order |
| `num_interacting_agents` | int | Number of agents interacting per task episode |

**Execution Flow Example**:

```yaml
turn_order: ["tool_generator", "reasoning_generator"]
max_turns: 4
```

Execution sequence:
```
Turn 0: tool_generator       (Generate tool calls)
Turn 1: reasoning_generator  (Reason based on tool results)
Turn 2: tool_generator       (Call tools again based on reasoning)
Turn 3: reasoning_generator  (Final reasoning and answer)
```

**Different Interaction Modes**:

**Serial Collaboration**:
```yaml
multi_agent_interaction:
  turn_order: ["code_generator", "test_generator"]
  num_interacting_agents: 2
```
- Agents execute sequentially, later agents build on earlier outputs
- Suitable for: Code generation → Test validation, Reasoning → Tool calling

**Iterative Refinement**:
```yaml
multi_agent_interaction:
  turn_order: ["generator", "critic", "generator", "critic"]
  num_interacting_agents: 2
```
- Agents alternate, forming generate-evaluate loops
- Suitable for: Generate → Critique → Improve → Re-critique

**Single Agent Mode**:
```yaml
multi_agent_interaction:
  turn_order: ["single_agent"]
  num_interacting_agents: 1
```
- Only one agent executes
- Suitable for: Single-agent baseline experiments

**When to Modify**:
- **Change collaboration pattern**: Adjust `turn_order` sequence
- **Increase collaboration rounds**: Repeat agent names in `turn_order`
- **Test different strategies**: Experiment with different agent arrangements

---

### **7. Training Configuration**

Defines PPO reinforcement learning training hyperparameters.

```yaml
training:
  # Basic settings
  device: cuda                                    # Training device
  total_training_steps: 200                       # Total training steps
  project_name: pettingllms                       # Project name
  experiment_name: math_eval_single_policy        # Experiment name
  logger: ['console', 'wandb']                    # Loggers
  
  # Checkpoints and timeouts
  model_checkpoints_dir: checkpoints              # Model save directory
  ray_wait_register_center_timeout: 300           # Ray registration timeout (seconds)
  
  # Batch and sampling configuration
  train_batch_size: 32                            # Training batch size
  train_sample_num: 8                             # Training samples per task
  validate_sample_num: 1                          # Validation samples
  sample_temperature: 1                           # Sampling temperature
  
  # Training frequency
  val_freq: 10                                    # Validation frequency (every N steps)
  resample_freq: 3                                # Resample frequency (every N steps)
  epoch_size: 20                                  # Steps per epoch
  
  # Sequence length configuration
  max_prompt_length: 4096                         # Maximum prompt length
  max_response_length: 2048                       # Maximum response length
  
  # LoRA configuration (references top-level variables)
  lora_rank: ${lora_rank}
  lora_alpha: ${lora_alpha}
```

**Parameter Details**:

#### **Basic Settings**

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | str | Training device: `"cuda"` (GPU) or `"cpu"` |
| `total_training_steps` | int | Total training steps, controls training duration |
| `project_name` | str | Project name for organizing experiments |
| `experiment_name` | str | Experiment name to distinguish different runs |
| `logger` | List[str] | Logger list: `"console"` (terminal), `"wandb"` (Weights & Biases), `"tensorboard"` |

#### **Batch and Sampling**

| Parameter | Type | Typical Values | Description |
|-----------|------|----------------|-------------|
| `train_batch_size` | int | 16-64 | Training batch size, affects memory usage and training stability |
| `train_sample_num` | int | 4-16 | Number of training samples per task for PPO advantage estimation |
| `validate_sample_num` | int | 1-4 | Samples per task during validation, typically set to 1 |
| `sample_temperature` | float | 0.7-1.5 | Sampling temperature. Higher = more random, lower = more deterministic |

**Batch Size Recommendations**:
- **8GB GPU**: `train_batch_size: 8-16`
- **16GB GPU**: `train_batch_size: 16-32`
- **24GB+ GPU**: `train_batch_size: 32-64`

#### **Training Frequency**

| Parameter | Type | Description |
|-----------|------|-------------|
| `val_freq` | int | Validation frequency: validate every N training steps |
| `resample_freq` | int | Resample frequency: resample training data every N steps |
| `epoch_size` | int | Number of training steps per epoch |

#### **Sequence Length**

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_prompt_length` | int | Maximum prompt length (tokens). Exceeding content will be truncated |
| `max_response_length` | int | Maximum model response length (tokens). Controls generation length |

**Sequence Length Recommendations for Different Tasks**:
- **Code generation**: `prompt: 2048-4096`, `response: 2048-4096`
- **Math reasoning**: `prompt: 2048-4096`, `response: 1024-2048`
- **Short Q&A**: `prompt: 512-1024`, `response: 256-512`

#### **Checkpoints and Logging**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_checkpoints_dir` | str | Directory for saving model checkpoints |
| `ray_wait_register_center_timeout` | int | Timeout for Ray distributed system registration center (seconds) |

**When to Modify**:
- **Adjust training duration**: Modify `total_training_steps`
- **Out of memory**: Reduce `train_batch_size` or sequence lengths
- **Increase sampling diversity**: Increase `sample_temperature` or `train_sample_num`
- **Faster validation**: Reduce `val_freq`
- **Use W&B tracking**: Add `"wandb"` to `logger`

---

### **8. Models Configuration**

Configures detailed model and PPO training parameters for each agent policy.

```yaml
models:
  model_0:
    path: ${base_models.policy_0.path}              # Reference base model path
    name: ${base_models.policy_0.name}              # Reference base model name
    ppo_trainer_config:
      filter_method: mean                            # Advantage filtering method
      filter_ratio: 0.5                              # Filtering ratio
      data:
        max_prompt_length: ${training.max_prompt_length}
        max_response_length: ${training.max_response_length}
      actor_rollout_ref:
        model:
          path: ${base_models.policy_0.path}
        rollout:
          temperature: ${training.sample_temperature}
          prompt_length: ${training.max_prompt_length}
          response_length: ${training.max_response_length}
          tensor_model_parallel_size: ${resource.n_gpus_per_node}
        trainer:
          n_gpus_per_node: ${resource.n_gpus_per_node}
          n_training_gpus_per_node: ${resource.n_gpus_per_node}
  
  model_1:
    # Second model configuration (similar structure)
    ...
```

**Parameter Details**:

#### **Top-Level Configuration**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | Model path, typically references path defined in `base_models` |
| `name` | str | Model name for identification and logging |

#### **PPO Trainer Configuration (ppo_trainer_config)**

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `filter_method` | str | `"mean"`, `"median"`, `"none"` | Advantage function filtering method:<br>• `"mean"`: Filter samples below mean<br>• `"median"`: Filter samples below median<br>• `"none"`: No filtering |
| `filter_ratio` | float | 0.0-1.0 | Ratio of samples to keep. E.g., `0.5` keeps top 50% of samples |

#### **Data Configuration**

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_prompt_length` | int | Maximum prompt length, typically references `training.max_prompt_length` |
| `max_response_length` | int | Maximum response length, typically references `training.max_response_length` |

#### **Actor-Rollout-Ref Configuration**

This section configures detailed parameters for model inference (rollout) and training:

**Model Configuration**:
```yaml
model:
  path: ${base_models.policy_0.path}  # Model path
```

**Rollout Configuration**:
```yaml
rollout:
  temperature: 1.0                          # Sampling temperature
  prompt_length: 4096                       # Maximum prompt length
  response_length: 2048                     # Maximum response length
  tensor_model_parallel_size: 8             # Tensor parallelism size (typically equals GPU count)
```

| Parameter | Description |
|-----------|-------------|
| `temperature` | Controls generation randomness. Higher = more random |
| `prompt_length` | Prompt length limit during rollout |
| `response_length` | Response length limit during rollout |
| `tensor_model_parallel_size` | Number of GPUs for tensor model parallelism for large model inference |

**Trainer Configuration**:
```yaml
trainer:
  n_gpus_per_node: 8                        # GPUs per node
  n_training_gpus_per_node: 8               # GPUs for training
```

#### **Hydra Variable References**

The configuration extensively uses `${...}` syntax to reference other configuration sections:

```yaml
# Reference base model configuration
path: ${base_models.policy_0.path}

# Reference training configuration
temperature: ${training.sample_temperature}

# Reference resource configuration
tensor_model_parallel_size: ${resource.n_gpus_per_node}

# Reference top-level variables
lora_rank: ${lora_rank}
```

**Benefits**:
- **Single source of truth**: Avoid duplicating parameter definitions
- **Easy maintenance**: Change once, all references update automatically
- **Reduce errors**: Ensure configuration consistency

**When to Modify**:
- **Filter training samples**: Adjust `filter_method` and `filter_ratio`
- **Change models**: Modify `path` references
- **Adjust inference parameters**: Modify `rollout` section parameters
- **Multi-GPU configuration**: Adjust `tensor_model_parallel_size` and GPU counts

---

## **How to Modify Configuration**

### **1. Prerequisites**

Before modifying configuration, ensure you have completed:

✅ **Environment Registration**: Register environment class in `multiagentssys_register.py`
```python
ENV_CLASS_MAPPING = {
    "math_env": MathEnv,
    "code_env": CodeEnv,
    # Add your environment
}
```

✅ **Agent Registration**: Register all agent classes
```python
AGENT_CLASS_MAPPING = {
    "reasoning_generator": ReasoningGeneratorAgent,
    "tool_generator": ToolGeneratorAgent,
    # Add your agents
}
```

See [Registration](registration.md) for detailed steps.

---

### **2. Creating New Configuration Files**

**Step 1**: Choose a Template

Start from an existing configuration that's most similar to your task:

```bash
# Code generation tasks
cp pettingllms/config/code/code_L3_model.yaml pettingllms/config/code/my_code_config.yaml

# Math reasoning tasks
cp pettingllms/config/math/math_L3_model.yaml pettingllms/config/math/my_math_config.yaml

# Custom tasks
cp pettingllms/config/math/math_L3_model.yaml pettingllms/config/custom/my_custom_config.yaml
```

**Step 2**: Modify Core Parameters

Open the configuration file and modify the following key sections in order:

```yaml
# 1. Set environment (REQUIRED)
env:
  name: your_env_name           # Your registered environment name
  dataset: "your_dataset"       # Your dataset
  benchmark: "your_benchmark"   # Evaluation benchmark
  max_turns: 5                  # Adjust based on task complexity

# 2. Set model paths (REQUIRED)
base_models:
  policy_0:
    path: "/path/to/your/model"  # Actual model path
    name: "your_model_name"

# 3. Configure agents (REQUIRED)
agent_policy_configs:
  num_agents: 2
  policy_list: ["agent1", "agent2"]
  agent_configs:
    agent_0:
      name: "agent1"              # Your registered agent name
      policy_name: "your_model_name"

# 4. Set interaction order (REQUIRED)
multi_agent_interaction:
  turn_order: ["agent1", "agent2"]  # Agent execution order

# 5. Adjust training parameters (OPTIONAL)
training:
  total_training_steps: 200
  train_batch_size: 32
  experiment_name: my_experiment
```

---

### **3. Common Modification Scenarios**

#### **Scenario A: Changing Base Model**

```yaml
# Switching from Llama-3-8B to Qwen2.5-7B
base_models:
  policy_0:
    path: "Qwen/Qwen2.5-7B-Instruct"  # Change this
    name: "qwen_model"                 # Change this

# Synchronize models section references
models:
  model_0:
    path: ${base_models.policy_0.path}  # Automatically updates
    name: ${base_models.policy_0.name}  # Automatically updates
```

#### **Scenario B: Adding New Agents**

```yaml
# Expanding from 2 agents to 3 agents
agent_policy_configs:
  num_agents: 3                    # Change: 2 → 3
  policy_list: ["agent1", "agent2", "agent3"]  # Add agent3
  agent_configs:
    agent_0:
      name: "agent1"
      policy_name: "model1"
    agent_1:
      name: "agent2"
      policy_name: "model2"
    agent_2:                       # NEW
      name: "agent3"               # New agent
      policy_name: "model3"        # Corresponding model

# Update interaction order
multi_agent_interaction:
  turn_order: ["agent1", "agent2", "agent3"]  # Add agent3
  num_interacting_agents: 3                   # Change: 2 → 3

# Add new model configuration
base_models:
  policy_2:                        # NEW
    path: "path/to/model3"
    name: "model3"
```

#### **Scenario C: Adjusting Resource Configuration**

```yaml
# Reducing from 8 GPUs to 4 GPUs
resource:
  n_gpus_per_node: 4  # Change: 8 → 4

# Synchronize model parallelism configuration
models:
  model_0:
    ppo_trainer_config:
      actor_rollout_ref:
        rollout:
          tensor_model_parallel_size: 4  # Change: 8 → 4
        trainer:
          n_gpus_per_node: 4              # Change: 8 → 4
          n_training_gpus_per_node: 4     # Change: 8 → 4
```

#### **Scenario D: Enabling LoRA Training**

```yaml
# Top-level configuration
specialization: "lora"  # Change: "full" → "lora"
lora_rank: 16           # Set LoRA rank
lora_alpha: 32          # Set LoRA alpha

# Training configuration automatically references
training:
  lora_rank: ${lora_rank}      # Automatically updates
  lora_alpha: ${lora_alpha}    # Automatically updates
```

#### **Scenario E: Adjusting Training Hyperparameters**

```yaml
training:
  # Extend training
  total_training_steps: 500          # Change: 200 → 500
  
  # Reduce memory usage
  train_batch_size: 16               # Change: 32 → 16
  max_prompt_length: 2048            # Change: 4096 → 2048
  max_response_length: 1024          # Change: 2048 → 1024
  
  # Increase exploration
  sample_temperature: 1.2            # Change: 1.0 → 1.2
  train_sample_num: 12               # Change: 8 → 12
  
  # More frequent validation
  val_freq: 5                        # Change: 10 → 5
```

---

### **4. Command-Line Overrides**

You can override any parameter via command line without modifying the config file:

```bash
# Override single parameter
python -m pettingllms.trainer.train \
    --config-name math_L3_model \
    training.total_training_steps=500

# Override multiple parameters
python -m pettingllms.trainer.train \
    --config-name math_L3_model \
    training.total_training_steps=500 \
    training.train_batch_size=16 \
    resource.n_gpus_per_node=4

# Override nested parameters
python -m pettingllms.trainer.train \
    --config-name math_L3_model \
    base_models.policy_0.path="/path/to/new/model" \
    multi_agent_interaction.turn_order="[agent2,agent1]"
```

**Use Cases**:
- Quick experimentation with different hyperparameters
- Running ablation studies
- Using different resource configurations on different machines

---

### **5. Configuration Validation Checklist**

Before running training, verify the following:

#### **✅ Environment and Agents**
- [ ] `env.name` is registered in `ENV_CLASS_MAPPING`
- [ ] `agent_configs.agent_N.name` is registered in `AGENT_CLASS_MAPPING`
- [ ] All names in `turn_order` exist in `agent_configs`
- [ ] `num_agents` equals the number of agents in `agent_configs` and `turn_order`

#### **✅ Model Configuration**
- [ ] `base_models.policy_N.path` points to valid model paths
- [ ] Model paths are accessible (exist locally or downloadable from HuggingFace)
- [ ] `agent_configs.agent_N.policy_name` corresponds to `name` in `base_models`
- [ ] `models.model_N.path` correctly references `base_models`

#### **✅ Resource Configuration**
- [ ] `resource.n_gpus_per_node` doesn't exceed available GPUs
- [ ] `tensor_model_parallel_size` equals or is less than `n_gpus_per_node`
- [ ] `train_batch_size` matches GPU memory capacity

#### **✅ Training Configuration**
- [ ] `max_prompt_length + max_response_length` doesn't exceed model's max length
- [ ] `val_freq` is less than `total_training_steps`
- [ ] Log directory `model_checkpoints_dir` is writable

---

## **Configuration Best Practices**

### **1. Use Modular Configuration**

**Recommended**: Leverage Hydra's composition features to separate concerns

```yaml
# config/base/resource_8gpu.yaml
resource:
  nnodes: 1
  n_gpus_per_node: 8

# config/base/training_default.yaml
training:
  total_training_steps: 200
  train_batch_size: 32

# config/math/math_experiment.yaml
defaults:
  - ../base/resource_8gpu
  - ../base/training_default
  - _self_

# Only define task-specific configuration
env:
  name: math_env
  ...
```

### **2. Version Control Configuration Files**

Keep configuration files in Git:

```bash
# Track configuration changes
git add pettingllms/config/math/my_math_config.yaml
git commit -m "Add math config with 3-agent setup"

# Tag important configurations
git tag -a exp-math-v1.0 -m "Math experiment baseline config"
```

### **3. Log Experiment Configurations**

Save complete configuration in experiment logs:

```python
# Hydra automatically saves configuration to output directory
# Output location: outputs/<date>/<time>/.hydra/config.yaml
```

Check saved configuration:
```bash
cat outputs/2024-10-15/14-30-00/.hydra/config.yaml
```

### **4. Progressive Tuning Strategy**

**Phase 1: Small-Scale Validation**
```yaml
training:
  total_training_steps: 10
  train_batch_size: 4
  train_sample_num: 2
env:
  max_turns: 2
```
→ Quickly verify code and configuration correctness

**Phase 2: Medium-Scale Debugging**
```yaml
training:
  total_training_steps: 50
  train_batch_size: 16
  train_sample_num: 4
env:
  max_turns: 3
```
→ Debug training pipeline and hyperparameters

**Phase 3: Full-Scale Training**
```yaml
training:
  total_training_steps: 500
  train_batch_size: 32
  train_sample_num: 8
env:
  max_turns: 5
```
→ Final training run

### **5. Document Custom Configurations**

Add comments at the top of configuration files:

```yaml
# ========================================
# Math Reasoning Experiment - 3 Agents
# ========================================
# Date: 2024-10-15
# Author: Your Name
# Description:
#   - 3-agent setup: reasoning + tool + verification
#   - Using Llama-3-8B-Instruct
#   - LoRA rank 16
#   - Target: AIME24 benchmark
# ========================================

specialization: "lora"
lora_rank: 16
...
```

---

## **Troubleshooting**

### **Common Errors and Solutions**

#### **Error 1: Environment Not Registered**
```
KeyError: 'my_env' not in ENV_CLASS_MAPPING
```
**Solution**:
1. Check `pettingllms/trainer/multiagentssys_register.py`
2. Ensure environment class is imported and added to `ENV_CLASS_MAPPING`
3. Restart training script

#### **Error 2: Agent Not Registered**
```
KeyError: 'my_agent' not in AGENT_CLASS_MAPPING
```
**Solution**:
1. Check `AGENT_CLASS_MAPPING` in `multiagentssys_register.py`
2. Ensure agent class is imported and registered
3. Check `agent_configs.agent_N.name` spelling in configuration

#### **Error 3: Invalid Model Path**
```
OSError: /path/to/model does not exist
```
**Solution**:
1. Verify `base_models.policy_N.path` is correct
2. Check filesystem permissions
3. If using HuggingFace models, ensure network connection and authentication

#### **Error 4: GPU Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**:
1. Reduce `train_batch_size`
2. Reduce `max_prompt_length` and `max_response_length`
3. Enable LoRA (`specialization: "lora"`)
4. Reduce `train_sample_num`

#### **Error 5: Configuration Reference Error**
```
omegaconf.errors.InterpolationKeyError: base_models.policy_0.path
```
**Solution**:
1. Ensure referenced configuration items are defined
2. Check `${...}` syntax spelling
3. Verify reference hierarchy is correct

---

## **Advanced Topics**

### **1. Multi-Configuration Composition**

Use Hydra's configuration groups:

```yaml
# config/env/math.yaml
env:
  name: math_env
  max_turns: 5

# config/model/llama3_8b.yaml
base_models:
  policy_0:
    path: "meta-llama/Llama-3-8B-Instruct"

# config/experiment.yaml
defaults:
  - env: math
  - model: llama3_8b
  - _self_
```

Select composition at runtime:
```bash
python -m pettingllms.trainer.train \
    --config-name experiment \
    env=code \
    model=qwen2_7b
```

### **2. Environment Variables and Secret Management**

Use environment variables for sensitive information:

```yaml
# Reference environment variables in configuration
base_models:
  policy_0:
    path: ${oc.env:MODEL_PATH}  # Read from environment variable

# Set environment variable
export MODEL_PATH="/path/to/secure/model"
```

### **3. Dynamic Configuration Generation**

Write Python scripts to generate configurations:

```python
from omegaconf import OmegaConf

# Dynamically generate configuration
config = {
    "env": {"name": "math_env", "max_turns": 5},
    "base_models": {
        f"policy_{i}": {
            "path": f"/models/policy_{i}",
            "name": f"model_{i}"
        }
        for i in range(num_policies)
    }
}

# Save configuration
OmegaConf.save(config, "generated_config.yaml")
```

---

## **Complete Configuration Examples**

### **Code Generation Task**

```yaml
specialization: "lora"
lora_rank: 16
lora_alpha: 32

resource:
  nnodes: 1
  n_gpus_per_node: 8
  trust_remote_code: true

env:
  name: code_env
  dataset: "apps"
  benchmark: "LiveCodeBench"
  max_turns: 6
  resolve: true
  multi_modal: false
  batched_init: true

base_models:
  policy_0:
    path: "meta-llama/Llama-3-8B-Instruct"
    name: "code_model"
  policy_1:
    path: "meta-llama/Llama-3-8B-Instruct"
    name: "test_model"

agent_policy_configs:
  num_agents: 2
  policy_list: ["code_generator", "test_generator"]
  agent_configs:
    agent_0:
      name: "code_generator"
      policy_name: "code_model"
    agent_1:
      name: "test_generator"
      policy_name: "test_model"

multi_agent_interaction:
  turn_order: ["code_generator", "test_generator"]
  num_interacting_agents: 2

training:
  device: cuda
  total_training_steps: 300
  project_name: pettingllms
  experiment_name: code_generation_2agents
  logger: ['console', 'wandb']
  train_batch_size: 32
  train_sample_num: 8
  validate_sample_num: 1
  sample_temperature: 0.8
  val_freq: 10
  max_prompt_length: 2048
  max_response_length: 4096
```

### **Single Agent Baseline**

```yaml
specialization: "lora"
lora_rank: 16
lora_alpha: 32

resource:
  nnodes: 1
  n_gpus_per_node: 8

env:
  name: math_env
  dataset: "polaris"
  benchmark: "AIME24"
  max_turns: 1  # Single turn execution
  batched_init: true

base_models:
  policy_0:
    path: "Qwen/Qwen2.5-7B-Instruct"
    name: "single_agent_model"

agent_policy_configs:
  num_agents: 1
  policy_list: ["single_agent"]
  agent_configs:
    agent_0:
      name: "single_agent"
      policy_name: "single_agent_model"

multi_agent_interaction:
  turn_order: ["single_agent"]
  num_interacting_agents: 1

training:
  total_training_steps: 200
  train_batch_size: 64
  max_prompt_length: 4096
  max_response_length: 2048
```

---

## **Summary**

Configuration files are the bridge connecting the framework's core components. Proper configuration requires understanding:

1. **Environment (Env)**: Defines tasks and execution constraints
2. **Agents**: Specifies participants and their models
3. **Interaction**: Determines collaboration patterns
4. **Training**: Sets learning hyperparameters
5. **Models**: Configures inference and optimization

**Configuration Workflow**:
1. Register environments and agents → [Registration](registration.md)
2. Create/modify configuration file (this guide)
3. Validate configuration correctness
4. Run training → [Training Guide](../getting-started/training.md)
5. Monitor and tune

---

## **Next Steps**

After understanding configuration:

- Set up component registrations: [Registration](registration.md)
- Learn about agent implementation: [Agent Functions](agent-functions.md)
- Understand environment state: [Environment State](environment-state.md)

---

## **Related Documentation**

- [Core Architecture](core-architecture.md) - Framework core concepts
- [Data Preparation](data-preparation.md) - Dataset setup
- [Registration](registration.md) - Environment and agent registration
