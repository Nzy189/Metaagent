# **Environment Setup**

This guide provides a comprehensive overview of setting up the PettingLLMs framework and understanding its core agent-environment interaction mechanisms.

---

## **Overview**

The PettingLLMs framework enables multi-agent reinforcement learning with large language models. This section covers everything you need to know to set up environments, configure agents, and understand the interaction patterns.

---

## **Documentation Structure**

This guide is organized into the following sections:

### **1. [Data Preparation](data-preparation.md)**

Learn how to prepare datasets for training and evaluation.

### **2. [Core Architecture](core-architecture.md)**

Explore the framework's core components.






### **3. [Agent Functions](agent-functions.md)**

Deep dive into the three fundamental agent functions:

- `update_from_env()`: Reading state and creating prompts
- `update_from_model()`: Parsing model responses
- `step()`: Executing actions and updating state
- Complete interaction cycle
- Best practices and debugging tips

### **4. [Environment State](environment-state.md)**

Understand the shared state mechanism:

- State structure for different environments
- Communication patterns between agents
- Multi-agent coordination via shared state
- State design principles
- Custom state definition



### **5. [Configuration](configuration.md)**

Understand the framework's configuration system


### **6. [Workflow Example](workflow-example.md)**

Walk through a complete multi-agent interaction:

- Turn-by-turn execution trace
- State evolution over time
- Feedback loops and iterative refinement
- Reward calculation and termination
- Training implications

### **7. [Registration](registration.md)**

Learn about the environment and agent registration system:

- Current registrations (environments, agents, workers)
- Safe import pattern
- Adding custom environments and agents
- Registration best practices
- Debugging registration issues

---

## **Quick Start**

If you're new to PettingLLMs, we recommend following this path:

1. **Start here**: Read [Core Architecture](core-architecture.md) for a high-level overview
2. **Prepare data**: Follow [Data Preparation](data-preparation.md) to set up datasets
3. **Configure**: Review [Configuration](configuration.md) to understand config files
4. **Register**: Check [Registration](registration.md) to see available environments and agents
5. **Deep dive**: Study [Agent Functions](agent-functions.md) and [Environment State](environment-state.md)
6. **See it in action**: Walk through [Workflow Example](workflow-example.md)

---

## **Key Concepts**

### **Agent-Environment Interaction**

The framework follows a standardized interaction pattern:

```
1. update_from_env()  → Agent reads environment state, creates prompt
2. Model.generate()   → LLM generates response to prompt
3. update_from_model()→ Agent parses response into action
4. step()            → Agent executes action, updates environment state
5. Repeat for next agent/turn
```

### **State-Mediated Communication**

Agents don't communicate directly—they share information through environment state:

```python
# Agent 1 writes to state
env_data.state.generated_code = "def factorial(n): ..."

# Agent 2 reads from state
code = env_data.state.generated_code
```

### **Multi-Agent Coordination**

Agents take turns in a defined order, building on each other's outputs:

```yaml
multi_agent_interaction:
  turn_order: ["code_generator", "test_generator"]
```

---

## **Supported Environments**

The framework currently supports the following environments:

### **Code Generation**
- **code_env**: Multi-agent code generation with test-driven development
- Agents: Code generator, Test generator
- Tasks: APPS, CodeContests, LiveCodeBench

### **Mathematical Reasoning**
- **math_env**: Multi-agent mathematical problem solving
- Agents: Reasoning agent, Tool agent
- Tasks: AIME, OlympiadBench

### **Planning Tasks**
- **stateful_env**: Sequential decision-making in stateful environments
- Agents: Plan agent, Tool call agent
- Tasks: Sokoban, Sudoku

### **Web Search**
- **search_env**: Multi-agent web search and reasoning
- Agents: Web search agent, Reasoning agent
- Tasks: HotpotQA, Bamboogle

### **Interactive Environments**
- **alfworld_env**: Embodied AI in household environments
- **web_env**: Web navigation tasks

---

## **Example: Code Generation Environment**

Here's a minimal example of how agents interact in the code generation environment:

```python
# Turn 0: Code Generator
agent = CodeGenerationAgent()
agent.update_from_env(turn_idx=0, env_data)  # Read problem
# ... model generates code ...
agent.update_from_model(model_response)      # Parse code
await agent.step(env_data, env_worker)       # Execute and evaluate

# Turn 1: Test Generator
agent = UnitTestGenerationAgent()
agent.update_from_env(turn_idx=1, env_data)  # Read problem + code
# ... model generates tests ...
agent.update_from_model(model_response)      # Parse tests
await agent.step(env_data, env_worker)       # Execute tests

# Turn 2: Code Generator (refinement)
agent = CodeGenerationAgent()
agent.update_from_env(turn_idx=2, env_data)  # Read feedback
# ... model refines code ...
agent.update_from_model(model_response)      # Parse refined code
await agent.step(env_data, env_worker)       # Execute and evaluate
```

---

## **Training Workflow**

The complete training workflow:

1. **Prepare data**: Run `python scripts/dataprocess/load_*.py`
2. **Configure**: Edit config files in `pettingllms/config/`
3. **Register**: Ensure environments/agents are in `multiagentssys_register.py`
4. **Train**: Run `bash scripts/train/*/train.sh`
5. **Evaluate**: Run `bash scripts/evaluate/evaluate.sh`

---

## **Additional Resources**

### **Code References**

- Base classes: `pettingllms/multi_agent_env/base/`
- Environments: `pettingllms/multi_agent_env/{task}/`
- Agents: `pettingllms/multi_agent_env/{task}/agents/`
- Trainer: `pettingllms/trainer/`
- Config: `pettingllms/config/`

### **Related Documentation**

- [Installation Guide](../getting-started/installation.md)
- [Training Guide](../getting-started/training.md)
- [Evaluation Guide](../getting-started/evaluation.md)
- [Core Concepts](../core-concepts/overview.md)

---

## **Getting Help**

If you encounter issues:

1. Check the specific section's documentation for details
2. Review the [Workflow Example](workflow-example.md) for a complete walkthrough
3. Examine the actual code in the codebase
4. Refer to the [Registration](registration.md) guide for available components

---

**Ready to get started? Begin with [Core Architecture](core-architecture.md) →**
