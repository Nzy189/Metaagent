# PettingLLMs: Environment Setup and Core Concepts Guide

This guide provides a comprehensive overview of setting up the PettingLLMs framework and understanding its core agent-environment interaction mechanisms, using the **Code Generation Environment** as the primary example.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Core Architecture Overview](#core-architecture-overview)
3. [Key Functions Explained](#key-functions-explained)
   - [update_from_env](#update_from_env)
   - [update_from_model](#update_from_model)
   - [step](#step)
4. [Environment State as Shared Information Hub](#environment-state-as-shared-information-hub)
5. [Complete Workflow Example](#complete-workflow-example)

---

## Environment Setup

### Prerequisites

- Python 3.10 or higher
- CUDA 12.x (for GPU support)
- Docker (for code execution sandbox)
- Git

### Installation Steps

1. **Clone the repository:**

```bash
git clone https://github.com/NorahYujieZhao/PettingLLMs.git
cd PettingLLMs
```

2. **Run the setup script:**

```bash
bash setup.bash
```

This script will:
- Create a Python virtual environment
- Install all dependencies from `requirements_venv.txt`
- Set up the package in development mode

3. **Activate the virtual environment:**

```bash
source pettingllms_venv/bin/activate
```

4. **Verify installation:**

```bash
python -c "import pettingllms; print('Installation successful!')"
```

### Key Dependencies

The framework relies on several major dependencies:

- **transformers** (4.55.2) - LLM model loading and inference
- **vllm** (0.10.0) - High-performance LLM serving
- **ray** (2.48.0) - Distributed computing and environment parallelization
- **wandb** (0.21.1) - Experiment tracking
- **docker** (7.1.0) - Sandboxed code execution
- **datasets** (4.0.0) - Dataset loading and processing
- **accelerate** (1.10.0) - Training acceleration

### Dataset Preparation

Before training, prepare the datasets:

```bash
# For code generation tasks (APPS, CodeContests, LiveCodeBench)
python scripts/dataprocess/load_code.py

# For math reasoning tasks (AIME24/25, OlympiadBench)
python scripts/dataprocess/load_math.py

# For planning/game tasks (Sokoban, Sudoku)
python scripts/dataprocess/load_sokoban.py
```

Datasets will be saved to:
- `datasets/code/` - Code generation problems
- `datasets/math/` - Math problems
- `datasets/sudoku_environments/` - Game environments

---

## Core Architecture Overview

PettingLLMs implements a multi-agent reinforcement learning system where agents interact with task-specific environments. The framework follows a standardized agent-environment interface pattern:

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────┐
│   Agent     │ ◄─────► │  Environment     │ ◄─────► │   Model     │
│             │         │                  │         │   (LLM)     │
│ - state     │         │  - state         │         │             │
│ - action    │         │  - observations  │         │             │
│ - reward    │         │  - rewards       │         │             │
└─────────────┘         └──────────────────┘         └─────────────┘
```

### Core Components

1. **Agent**: Represents an AI agent (e.g., CodeGenerationAgent, UnitTestGenerationAgent)
2. **Environment**: Task-specific environment (e.g., CodeEnv for code generation)
3. **Environment State**: Shared data structure for agent communication
4. **Model**: LLM that generates responses to agent prompts

---

## Key Functions Explained

Using the **CodeGenerationAgent** as our primary example, let's explore the three fundamental functions that enable agent-environment interaction.

### update_from_env

**Purpose**: Updates the agent's internal state and prompt based on the current environment state.

**Location**: `pettingllms/multi_agent_env/code/agents/code_agent.py`

#### Function Signature

```python
def update_from_env(self, turn_idx: int, env_data: Env):
    """
    Updates the agent based on environment feedback.
    
    Args:
        turn_idx: Current turn number (0-indexed)
        env_data: Environment object containing state and observations
    """
```

#### What It Does

1. **Extracts environment state information**:
   - Problem description
   - Previously generated code
   - Test inputs and outputs
   - Execution results
   - Mismatch cases from previous attempts

2. **Constructs context-appropriate prompts**:
   - **Turn 0** (Initial generation): Creates a prompt asking for initial code solution
   - **Turn > 0** (Refinement): Creates a prompt with feedback on previous attempts, including:
     - Failed test cases
     - Execution errors
     - Mismatches between expected and actual outputs

3. **Stores the prompt** in `self.current_prompt` for the model to process

#### Code Example from CodeGenerationAgent

```python
def update_from_env(self, turn_idx: int, env_data: Env):
    # Save environment data
    self.env_data = env_data
    
    # Extract state information
    state = getattr(env_data, "state", None)
    
    # Get problem and current code status
    question = getattr(state, "problem", None)
    current_code = getattr(state, "generated_code", None)
    
    # Build history of failed attempts
    formatted_prompt_for_mismatch_cases = ""
    for idx, code in enumerate(state.generated_code_history):
        if state.generated_test_vs_generated_code_mismatch_cases_history[idx]:
            formatted_prompt_for_mismatch_cases += f"Code {idx+1}:\n{code}\n"
            for mismatch_case in state.generated_test_vs_..._history[idx]:
                formatted_prompt_for_mismatch_cases += f"Input: {mismatch_case['test_input']}\n"
                formatted_prompt_for_mismatch_cases += f"Expected: {mismatch_case['generated_test_output']}\n"
                formatted_prompt_for_mismatch_cases += f"Got: {mismatch_case['code_execution_output']}\n"
    
    # Create appropriate prompt based on turn
    if turn_idx == 0:
        # Initial generation prompt
        formatted_prompt = (
            f"You are a helpful assistant that generates python code...\n"
            f"Problem:\n{question}\n\n"
            f"Please generate the code.\n"
        )
    else:
        # Refinement prompt with feedback
        formatted_prompt = (
            f"Previous attempts:\n{formatted_prompt_for_mismatch_cases}\n"
            f"Refine the code to pass all tests.\n"
        )
    
    # Store prompt for model consumption
    self.current_prompt = {"text": formatted_prompt, "image": None}
```

#### Key Insights

- **State-driven prompting**: The prompt adapts based on environment state
- **History tracking**: Past failures inform future attempts
- **Turn-aware behavior**: Different strategies for initial vs. refinement turns
- **Reads from env.state**: Accesses shared environment state for context

---

### update_from_model

**Purpose**: Parses the model's response and converts it into an actionable format.

**Location**: `pettingllms/multi_agent_env/code/agents/code_agent.py`

#### Function Signature

```python
def update_from_model(self, response: str) -> str:
    """
    Parses model response and extracts the action.
    
    Args:
        response: Raw text response from the LLM
        
    Returns:
        Extracted action (e.g., code string)
    """
```

#### What It Does

1. **Receives raw model output**: Takes the LLM's text response as input
2. **Parses structured content**: Extracts the relevant action using regex or parsing logic
3. **Handles errors**: Provides fallback behavior if parsing fails
4. **Stores the action**: Saves extracted action in `self.current_action`
5. **Returns the action**: Makes it available for the next step

#### Code Example from CodeGenerationAgent

```python
def update_from_model(self, response: str):
    """Parse the model response and extract code."""
    import re
    
    # Try to match Python code block in markdown format
    matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
    
    if matches:
        # Extract the last code block (in case of multiple)
        code = matches[-1].strip()
    else:
        # Fallback if no code block found
        code = "We can not extract the code in the output."
    
    # Store action for next step
    self.current_action = code
    
    return self.current_action
```

#### Different Parsing for Different Agents

**UnitTestGenerationAgent** example:

```python
def update_from_model(self, response: str):
    """Parse model response and extract test cases."""
    import re
    
    # Extract test cases in specific format
    test_action = extract_test_cases(response)
    
    # Structured action: {"input": [...], "output": [...]}
    self.current_action = test_action
    
    return self.current_action
```

#### Key Insights

- **Format conversion**: Transforms unstructured text into structured actions
- **Agent-specific parsing**: Different agents parse different formats
- **Error handling**: Graceful degradation when parsing fails
- **Action storage**: Makes action available for the step function

---

### step

**Purpose**: Executes the agent's action in the environment and updates the environment state.

**Location**: `pettingllms/multi_agent_env/code/agents/code_agent.py`

#### Function Signature

```python
async def step(self, env_data: Env, env_worker: Any = None):
    """
    Executes the agent's action and updates environment state.
    
    Args:
        env_data: Environment object with state to be modified
        env_worker: Ray actor for sandboxed code execution (optional)
    """
```

#### What It Does

1. **Retrieves the current action**: Uses `self.current_action` set by `update_from_model`
2. **Executes the action in the environment**: Runs code, evaluates tests, etc.
3. **Updates environment state**: Modifies `env_data.state` with results
4. **Calculates rewards**: Determines agent performance metrics
5. **Checks termination conditions**: Decides if the task is complete

#### Code Example from CodeGenerationAgent

```python
async def step(self, env_data: Env, env_worker: Any = None):
    """Execute generated code and evaluate against tests."""
    
    # 1) Get the action (generated code)
    gen_code = self.current_action
    
    # 2) Update environment state with generated code
    env_data.state.generated_code = gen_code
    
    # 3) Get test cases from environment state
    ground_truth_test_input = env_data.state.ground_truth_test_input or []
    ground_truth_test_output = env_data.state.ground_truth_test_output or []
    
    # 4) Evaluate code against test cases
    passed_ratio = 0.0
    if ground_truth_test_input and ground_truth_test_output:
        try:
            # Run code in sandboxed environment
            passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                gen_code, 
                ground_truth_test_input, 
                ground_truth_test_output, 
                timeout=30.0, 
                ray_actor=env_worker,
                rollout_idx=self.rollout_idx
            )
            
            # 5) Update environment state with evaluation results
            env_data.state.ground_truth_test_vs_generated_code_match_cases = passed_cases
            env_data.state.ground_truth_test_vs_generated_code_mismatch_cases = failed_cases
            env_data.state.ground_truth_test_vs_generated_code_match_ratio = passed_ratio
            
            # 6) Check for termination (all tests passed)
            if passed_ratio >= 1.0 and len(ground_truth_test_input) > 0:
                self.done = True
                self.is_pass = True
                
        except Exception as e:
            print(f"Warning: Failed to evaluate code: {e}")
            passed_ratio = 0.0
    
    # 7) Calculate reward (improvement over previous attempts)
    if len(self.reward_history) > 0:
        self.agent_reward = passed_ratio - self.reward_history[-1]
    else:
        self.agent_reward = passed_ratio
    
    # 8) Store metrics
    self.reward_history.append(passed_ratio)
    self.value = passed_ratio
```

#### Key Insights

- **Writes to env.state**: Updates shared environment state with action results
- **Action execution**: Performs actual computational work (code execution, test evaluation)
- **State mutation**: Modifies environment state to be shared with other agents
- **Reward calculation**: Provides learning signal for RL training
- **Termination detection**: Controls episode completion

---

## Environment State as Shared Information Hub

The **Environment State** (`env_data.state`) serves as the central communication mechanism between agents in multi-agent scenarios.

### The CodeEnvState Structure

Located in `pettingllms/multi_agent_env/code/code_env.py`:

```python
@dataclass
class CodeEnvState:
    # Problem definition
    problem: str = None
    golden_code: str = None
    
    # Generated artifacts
    generated_code_history: List[str] = field(default_factory=list)
    generated_code: str = None
    generated_test_input: List[str] = None
    generated_test_output: List[str] = None
    
    # Ground truth test cases
    ground_truth_test_input: List[str] = None
    ground_truth_test_output: List[str] = None
    
    # Execution results
    exe_code_generated_test_output: List[str] = None
    exe_code_ground_truth_test_output: List[str] = None
    
    # Evaluation results: ground truth tests vs generated code
    ground_truth_test_vs_generated_code_mismatch_cases: List[Dict] = None
    ground_truth_test_vs_generated_code_match_cases: List[Dict] = None
    ground_truth_test_vs_generated_code_match_ratio: float = None
    
    # Evaluation results: generated tests vs generated code
    generated_test_vs_generated_code_match_cases: List[Dict] = None
    generated_test_vs_generated_code_mismatch_cases: List[Dict] = None
    generated_test_vs_generated_code_mismatch_cases_history: List[Dict] = field(default_factory=list)
    generated_test_vs_generated_code_match_ratio: float = None
    
    # Evaluation results: generated tests vs golden code
    generated_test_vs_golden_code_match_cases: List[Dict] = None
    generated_test_vs_golden_code_mismatch_cases: List[Dict] = None
    generated_test_vs_golden_code_match_ratio: float = None
```

### How Agents Share Information via State

#### Multi-Agent Code Generation Workflow

In a multi-agent code generation scenario with **CodeGenerationAgent** and **UnitTestGenerationAgent**:

1. **Turn 0 - Coder generates initial code:**
   - `CodeGenerationAgent.update_from_env()` reads: `state.problem`
   - `CodeGenerationAgent.step()` writes: `state.generated_code`

2. **Turn 1 - Tester generates test cases:**
   - `UnitTestGenerationAgent.update_from_env()` reads: `state.problem`, `state.generated_code`
   - `UnitTestGenerationAgent.step()` writes: 
     - `state.generated_test_input`
     - `state.generated_test_output`
     - `state.generated_test_vs_generated_code_match_cases`
     - `state.generated_test_vs_generated_code_mismatch_cases`

3. **Turn 2 - Coder refines code based on test results:**
   - `CodeGenerationAgent.update_from_env()` reads:
     - `state.generated_code` (previous attempt)
     - `state.generated_test_vs_generated_code_mismatch_cases` (failed tests)
     - `state.generated_code_history` (all previous attempts)
   - `CodeGenerationAgent.step()` writes: `state.generated_code` (updated)

4. **Turn 3 - Tester validates refined code:**
   - `UnitTestGenerationAgent.update_from_env()` reads: `state.generated_code` (updated)
   - `UnitTestGenerationAgent.step()` writes: updated evaluation results
   - If all tests pass: `agent.done = True` triggers termination

### Key Principles

1. **Shared State Pattern**: 
   - All agents read from and write to the same `env_data.state` object
   - No direct agent-to-agent communication
   - Environment mediates all information exchange

2. **Read-Write Pattern**:
   - `update_from_env()` → **READS** from state
   - `step()` → **WRITES** to state

3. **Immutable History**:
   - History fields (e.g., `generated_code_history`) preserve all attempts
   - Enables learning from past mistakes

4. **Structured Feedback**:
   - Detailed evaluation results (match/mismatch cases) enable precise error correction
   - Agents can understand exactly what went wrong

5. **Turn-Based Coordination**:
   - Agents take turns modifying state
   - Each agent builds upon previous agents' outputs

---

## Complete Workflow Example

Let's trace a complete interaction cycle in the code generation environment:

### Scenario: Generate Python Code with Tests

```
Initial State:
  - state.problem = "Write a function to compute factorial"
  - state.ground_truth_test_input = ["5", "3", "0"]
  - state.ground_truth_test_output = ["120", "6", "1"]
```

### Turn 0: Code Generation Agent

```python
# 1. update_from_env (READ from state)
agent.update_from_env(turn_idx=0, env_data)
  → Reads: state.problem
  → Creates: Initial code generation prompt
  → Stores: self.current_prompt = {"text": "Generate code for: ...", ...}

# 2. Model inference (external)
response = model.generate(agent.current_prompt)
  → Returns: "```python\ndef factorial(n):\n  return n * factorial(n-1)\n```"

# 3. update_from_model (PARSE response)
agent.update_from_model(response)
  → Parses: Extracts code from markdown
  → Stores: self.current_action = "def factorial(n):\n  return n * factorial(n-1)"

# 4. step (WRITE to state)
await agent.step(env_data, env_worker)
  → Executes: Runs code against ground_truth_test_input
  → Result: RecursionError (missing base case)
  → Writes to state:
      - state.generated_code = "def factorial(n):\n  return n * factorial(n-1)"
      - state.ground_truth_test_vs_generated_code_match_ratio = 0.0
      - state.ground_truth_test_vs_generated_code_mismatch_cases = [...]
  → Sets: self.agent_reward = 0.0, self.done = False
```

### Turn 1: Code Generation Agent (Refinement)

```python
# 1. update_from_env (READ from state)
agent.update_from_env(turn_idx=1, env_data)
  → Reads: 
      - state.problem
      - state.generated_code_history
      - state.generated_test_vs_generated_code_mismatch_cases_history
  → Creates: Refinement prompt with error feedback
  → Stores: self.current_prompt = {"text": "Previous attempt failed with RecursionError...", ...}

# 2. Model inference
response = model.generate(agent.current_prompt)
  → Returns: "```python\ndef factorial(n):\n  if n == 0: return 1\n  return n * factorial(n-1)\n```"

# 3. update_from_model
agent.update_from_model(response)
  → Stores: self.current_action = "def factorial(n):\n  if n == 0: return 1\n  return n * factorial(n-1)"

# 4. step (WRITE to state)
await agent.step(env_data, env_worker)
  → Executes: Runs corrected code
  → Result: All tests pass (3/3)
  → Writes to state:
      - state.generated_code = "def factorial(n):\n  if n == 0: return 1\n  return n * factorial(n-1)"
      - state.ground_truth_test_vs_generated_code_match_ratio = 1.0
      - state.ground_truth_test_vs_generated_code_match_cases = [all 3 test cases]
  → Sets: self.agent_reward = 1.0, self.done = True, self.is_pass = True
```

### Key Observations

1. **State Evolution**: 
   - Environment state accumulates information across turns
   - Each agent adds its contribution to the shared state

2. **Feedback Loop**:
   - Failed attempts inform future attempts via state
   - Detailed error information enables targeted corrections

3. **Termination**:
   - Agent sets `self.done = True` when success criteria met
   - Environment can check this flag to end the episode

4. **Reward Signal**:
   - Rewards reflect improvement (delta in pass ratio)
   - Enables RL training with sparse rewards

---

## Summary

### Core Functions Recap

| Function | Purpose | State Interaction | Input/Output |
|----------|---------|-------------------|--------------|
| **update_from_env** | Prepare agent for action | **READS** from `env.state` | Input: `env_data` → Output: `self.current_prompt` |
| **update_from_model** | Parse model response | No state interaction | Input: `response` → Output: `self.current_action` |
| **step** | Execute action | **WRITES** to `env.state` | Input: `env_data` → Output: Updates `env.state` |

### Environment State Role

- **Central communication hub** for multi-agent systems
- **Preserves history** for learning from mistakes
- **Enables coordination** without direct agent-to-agent communication
- **Provides rich feedback** for refinement and RL training

### Design Principles

1. **Separation of concerns**: Each function has a clear, single responsibility
2. **State-mediated communication**: All agent interactions go through environment state
3. **Turn-based execution**: Agents take turns reading and writing state
4. **Flexible architecture**: Same pattern works across different task domains (code, math, planning)

---

## Next Steps

To explore further:

1. **Try other environments**: Math (`pettingllms/multi_agent_env/math/`), Planning (`pettingllms/multi_agent_env/stateful/`)
2. **Implement custom agents**: Extend the `Agent` base class with your own logic
3. **Run training**: Use the provided scripts in `scripts/train/` to train agents
4. **Analyze trajectories**: Examine how state evolves during multi-agent interactions

For more details, see:
- [Training Documentation](docs/training/overview.md)
- [API Reference](docs/api/index.md)
- [Original Paper](https://github.com/NorahYujieZhao/PettingLLMs)

---

**Happy agent training! 🚀**

