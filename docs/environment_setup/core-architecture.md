# **Core Architecture**

PettingLLMs implements a multi-agent reinforcement learning system where agents interact with task-specific environments. The framework follows a standardized agent-environment interface pattern.

---

## **System Overview**

In PettingLLMs, the **Environment serves as the central hub** for storing and delivering shared information between multiple agents. Each agent is controlled by an LLM, and all inter-agent communication flows through the environment state.

```
┌─────────────┐         ┌─────────────┐         ┌──────────────────────┐
│             │         │   Agent 1   │         │                      │
│             │────────►│             │◄───────►│                      │
│             │  prompt │ - action    │ state   │   Environment        │
│             │         │ - reward    │ update  │                      │
│             │◄────────│             │────────►│  - Shared State      │
│     LLM     │response └─────────────┘         │  - Task Definition   │
│             │         ┌─────────────┐         │  - History Tracking  │
│  Controls   │────────►│   Agent 2   │◄───────►│  - Coordination      │
│  Multiple   │  prompt │             │ state   │                      │
│   Agents    │         │ - action    │ update  │  Information Storage │
│             │◄────────│ - reward    │────────►│  & Delivery Hub      │
│             │response │             │         │                      │
│             │         └─────────────┘         │                      │
│             │              ...                │                      │
│             │         ┌─────────────┐         │                      │
│             │────────►│   Agent N   │◄───────►│                      │
│             │         │             │         │                      │
└─────────────┘         └─────────────┘         └──────────────────────┘
```

**Key Design**: 
- **LLM (Left)**: Single language model that controls all agents by generating responses to their prompts
- **Agents (Middle)**: Multiple specialized agents that interact sequentially
- **Environment (Right)**: Central information hub that stores shared state and enables coordination

---

## **Core Components**

### **1. Agent**

Represents an AI agent specialized for specific tasks.



**Base Location**: `pettingllms/multi_agent_env/base/agent.py`

**Key Properties**:
```python
@dataclass
class AgentData:
    current_prompt: Optional[Dict[str, Any]]  # Prompt to send to LLM (text/image)
    current_action: Optional[Any]             # Parsed action from LLM response
    agent_reward: Optional[float]             # Reward received by this agent
    success: bool                             # Whether agent succeeded
    answer_history: Optional[List[Any]]       # History of agent answers
    action_history: Optional[List[Any]]       # History of agent actions
    reward_history: Optional[List[float]]     # History of rewards received
```

---

### **2. Environment**

Task-specific environment that coordinates agent interactions.



**Base Location**: `pettingllms/multi_agent_env/base/env.py`

**Key Properties**:
```python
@dataclass
class Env:
    env_idx: int                # Environment index for parallel execution
    rollout_idx: int            # Rollout index for tracking
    max_turns: int              # Maximum number of interaction turns
    current_turn: int           # Current turn counter
    state: Optional[Any]        # Environment state (task-specific, stores shared info)
    done: bool                  # Whether environment terminated
    task: Any                   # Current task/problem definition
    history: List               # Complete interaction history
    config: dict                # Environment configuration
```

---

### **3. Environment State**

Shared data structure for inter-agent communication.

**Purpose**:
- Central information hub for all agents
- Preserves interaction history
- Enables coordination without direct agent-to-agent communication

**Example (CodeEnvState)**:
```python
@dataclass
class CodeEnvState:
    # Problem definition
    problem: str
    golden_code: str
    
    # Generated artifacts
    generated_code: str
    generated_test_input: List[str]
    generated_test_output: List[str]
    
    # Ground truth tests
    ground_truth_test_input: List[str]
    ground_truth_test_output: List[str]
    
    # Evaluation results
    ground_truth_test_vs_generated_code_match_ratio: float
    generated_test_vs_generated_code_match_ratio: float
    
    # History tracking
    generated_code_history: List[str]
    generated_test_vs_generated_code_mismatch_cases_history: List[Dict]
```

See [Environment State](environment-state.md) for detailed documentation.

---

### **4. Model (LLM)**

Language model that generates responses to agent prompts.

**Supported Backends**:
- **vLLM**: High-throughput inference (default)
- **SGLang**: Structured generation
- **HuggingFace**: Direct model loading

**Integration**:
- Models are managed by the execution engine
- Agents send prompts via `current_prompt`
- Responses are parsed by `update_from_model()`

---

## **Interaction Flow**

The agent-environment interaction follows a standardized cycle:

```
1. update_from_env()  → Agent reads environment state, creates prompt
2. Model.generate()   → LLM generates response to prompt
3. update_from_model()→ Agent parses response into action
4. step()            → Agent executes action, updates environment state
5. Repeat for next agent/turn
```

See [Agent Functions](agent-functions.md) for detailed explanations.

---

## **Multi-Agent Coordination**

### **Environment as Information Hub**

**Key Principle**: The environment stores and delivers all shared information between agents. Agents never communicate directly - all coordination happens through the environment state.

```
Agent 1 ─(writes)─► Environment State ─(reads)─► Agent 2
                         │
                    [Persistent Storage]
                    - Actions taken
                    - Results produced
                    - History tracking
                         │
Agent N ◄─(reads)─ Environment State ◄─(writes)─ Agent 3
```

### **Sequential Execution**

Agents take turns in a defined order:

```yaml
multi_agent_interaction:
  turn_order: ["code_generator", "test_generator"]
```

**Information Flow**:
1. Turn 0: `code_generator` reads problem from environment → generates code → **stores in environment**
2. Turn 1: `test_generator` **reads code from environment** → creates tests → **stores results in environment**
3. Turn 2: `code_generator` **reads test results from environment** → refines code → **updates environment**
4. Turn 3: `test_generator` **reads updated code from environment** → validates → **stores validation in environment**
5. Repeat until success or max_turns reached

### **Environment-Mediated Communication**

All information exchange happens through environment state:

```python
# Agent 1: Writes code to environment state
env_data.state.generated_code = "def factorial(n): ..."
# Environment stores this information

# Agent 2: Reads from environment state
code = env_data.state.generated_code  # Retrieved from environment
# Agent 2: Writes test results to environment state
env_data.state.generated_test_vs_generated_code_match_ratio = 0.8
# Environment stores and delivers this to future agents

# Agent 1 (next turn): Reads feedback from environment
feedback = env_data.state.generated_test_vs_generated_code_match_ratio
# Uses feedback to improve code
```

**Benefits**:
- **Centralized Storage**: All agent outputs are preserved in one place
- **Transparent Coordination**: Any agent can access information from any previous agent
- **History Tracking**: Environment maintains complete interaction history
- **Flexible Composition**: Add/remove agents without changing communication logic

---

## **Training System Integration**

The framework integrates with reinforcement learning trainers:

### **PPO Trainer**

```python
# Located in: pettingllms/trainer/multi_agents_ppo_trainer.py
class MultiAgentsPPOTrainer:
    def __init__(self, config):
        self.execution_engine = MultiAgentsExecutionEngine(config)
        self.actor_model = ...      # Policy model
        self.critic_model = ...     # Value model
        self.optimizer = ...        # RL optimizer
```

### **Execution Engine**

```python
# Located in: pettingllms/trainer/multi_agents_execution_engine.py
class MultiAgentsExecutionEngine:
    def __init__(self, config):
        self.env_class = ENV_CLASS_MAPPING[env_name]
        self.agent_class_list = [AGENT_CLASS_MAPPING[name] for name in turn_order]
        self.server_address_dict = ...  # Model servers
```

---

## **Ray-Based Distributed Execution**

For code execution and environment interaction:

### **Docker Workers**

Sandboxed code execution using Docker containers with optimized resource allocation:

```python
# Located in: pettingllms/multi_agent_env/code/code_worker.py
@ray.remote(num_cpus=0.001, max_concurrency=10000)
class RayDockerWorker:
    async def run(
        self,
        script: str,
        input_val: str,
        expected_output: str,
        timeout: float = 40.0,
        image: str = "python:3.11-slim"
    ) -> Dict[str, Any]:
        # Execute code in isolated Docker container
        ...
```

**Resource Configuration**:
- **`num_cpus=0.001`**: Each worker requests minimal CPU resources (0.1% of a CPU)
  - Workers are I/O-bound (waiting for code execution), not CPU-bound
  - Allows spawning thousands of workers on a single machine
  - Efficient resource utilization for high-throughput parallel execution
  
- **`max_concurrency=10000`**: Each worker can handle up to 10,000 concurrent tasks
  - Supports massive parallel code execution across multiple environments
  - Asynchronous execution allows high concurrency without blocking

**Usage**:
- **Sandboxing**: Prevents malicious code from affecting the system
- **Consistency**: Provides identical execution environment across all workers
- **Scalability**: Enables parallel execution across hundreds/thousands of workers
- **Efficiency**: Minimal CPU overhead enables dense worker deployment

---

## **Design Principles**

### **1. Environment-Centric Information Flow**

**Core Idea**: The environment is the **sole storage and delivery mechanism** for multi-agent shared information.

- **Agents**: Generate prompts and parse responses - no internal state storage
- **Environment**: Central repository for all shared data, coordination, and history
- **Models (LLM)**: Handle inference only - stateless
- **Trainer**: Orchestrates RL training

**Why This Matters**:
- Single source of truth for all agent interactions
- Clear separation between computation (agents) and storage (environment)
- Simplified debugging - all state changes are tracked in one place

### **2. Environment-Mediated Communication**

**Strict Rule**: Agents never communicate directly with each other.

```python
# ❌ NOT ALLOWED: Direct agent-to-agent communication
agent2.receive_message(agent1.send_message())

# ✅ REQUIRED: Environment-mediated communication
agent1.step()  # Writes to env_data.state
env_data.state  # Environment stores information
agent2.update_from_env()  # Reads from env_data.state
```

**Benefits**:
- Enables flexible agent composition without coupling
- Makes multi-agent coordination transparent and debuggable
- Allows dynamic agent addition/removal

### **3. Standardized Interface**

All agents implement three core functions that interact with environment:

- `update_from_env()`: **Read** shared state from environment → Create prompt
- `update_from_model()`: Parse LLM response → Extract action
- `step()`: Execute action → **Write** results back to environment

### **4. Modular Architecture**

- Easy to add new environments (just define new state structure)
- Easy to add new agents (implement three standard functions)
- Easy to compose multi-agent systems (environment handles coordination)

---

## **File Organization**

```
pettingllms/
├── multi_agent_env/
│   ├── base/
│   │   ├── agent.py           # Base Agent class
│   │   └── env.py             # Base Env class
│   ├── code/
│   │   ├── code_env.py        # CodeEnv and CodeEnvState
│   │   ├── agents/
│   │   │   ├── code_agent.py  # CodeGenerationAgent
│   │   │   └── unit_test_agent.py  # UnitTestGenerationAgent
│   │   └── code_worker.py     # Ray Docker workers
│   ├── math/
│   │   ├── math_env.py
│   │   └── agents/
│   └── stateful/
│       ├── stateful_env.py
│       └── agents/
├── trainer/
│   ├── train.py                      # Training entry point
│   ├── multi_agents_ppo_trainer.py   # PPO trainer
│   ├── multi_agents_execution_engine.py  # Execution engine
│   └── multiagentssys_register.py    # Environment/Agent registration
└── config/
    ├── code/
    ├── math/
    └── ppo_trainer/
```

---

## **Next Steps**

Continue exploring environment setup:

- Learn about agent functions: [Agent Functions](agent-functions.md)
- Understand environment state: [Environment State](environment-state.md)
- Configure the system: [Configuration](configuration.md)

