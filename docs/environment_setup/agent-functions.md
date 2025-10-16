# **Agent Functions**

This guide explains the three fundamental functions that enable agent-environment interaction, using the **CodeGenerationAgent** and **UnitTestGenerationAgent** as primary examples from the actual source code.

---

## **Agent Class Structure**

All agents inherit from the base `Agent` class and `AgentData` dataclass:

```python
# From: pettingllms/pettingllms/multi_agent_env/base/agent.py

@dataclass
class AgentData:
    current_prompt: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"text": None, "image": None}
    )
    current_action: Optional[Any] = None
    agent_reward: Optional[float] = 0.0
    success: bool = False
    answer_history: Optional[List[Any]] = field(default_factory=list)
    action_history: Optional[List[Any]] = field(default_factory=list)
    reward_history: Optional[List[float]] = field(default_factory=list)


class Agent(AgentData):
    """Base agent class with three core abstract methods."""
    
    @abstractmethod
    def update_from_env(self, turn_idx: int, env_data: Env) -> Env:
        """Read from environment state and create prompt."""
        raise NotImplementedError
    
    @abstractmethod
    def update_from_model(self, response: str, **kwargs) -> Env:
        """Parse model response and extract action."""
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """Reset agent state for new episode."""
        raise NotImplementedError
```

### **Agent Initialization**

```python
# CodeGenerationAgent initialization
class CodeGenerationAgent(Agent):
    def __init__(self, rollout_idx: int | None = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx  # Used for Ray worker identification
        # Additional kwargs for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)


# UnitTestGenerationAgent initialization  
class UnitTestGenerationAgent(Agent):
    def __init__(self, rollout_idx: int | None = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)
```

---

## **Function Overview**

All agents implement three core functions:

| **Function** | **Purpose** | **State Interaction** | **Input/Output** |
| --- | --- | --- | --- |
| **update_from_env** | Prepare agent for action | **READS** from `env.state` | Input: `turn_idx, env_data` → Output: Sets `self.current_prompt` |
| **update_from_model** | Parse model response | No state interaction | Input: `response` → Output: Sets `self.current_action` |
| **step** | Execute action | **WRITES** to `env.state` | Input: `env_data, env_worker` → Output: Updates `env.state` |
| **calculate_reward** | Calculate agent reward | **READS** from `env.state` | Input: `env_data` → Output: Sets `self.agent_reward` |

**Note**: `calculate_reward()` is called separately after `step()` to compute the agent's reward based on environment state.

---

## **1. update_from_env**

**Purpose**: Updates the agent's internal state and prompt based on the current environment state.

**Location**: `pettingllms/pettingllms/multi_agent_env/code/agents/code_agent.py:43`

### **What It Does**

1. **Reads environment state information** from `env_data.state`
2. **Constructs context-appropriate prompts** based on turn index and state
3. **Stores the prompt** in `self.current_prompt` for the model to process
4. **Adapts behavior** based on whether it's initial generation or refinement

### **Function Signature**

```python
def update_from_env(self, turn_idx: int, env_data: Env):
    """
    Update the agent's internal prompt based on environment state.
    
    Args:
        turn_idx: Current turn number (0-indexed)
        env_data: Environment object containing state
    
    Returns:
        None (updates self.current_prompt in place)
    """
```

### **Key Insights**

- **State-driven prompting**: The prompt adapts based on environment state
- **History tracking**: Past failures inform future attempts
- **Turn-aware behavior**: Different strategies for initial vs. refinement turns
- **Reads from env.state**: Accesses shared environment state for context

### **State Fields Read**

For `CodeGenerationAgent`:
- `state.problem`: Programming problem description
- `state.generated_code`: Previous code attempt
- `state.generated_code_history`: All previous code attempts
- `state.generated_test_vs_generated_code_mismatch_cases_history`: Failed test cases

---

## **2. update_from_model**

**Purpose**: Parses the model's response and converts it into an actionable format.

**Location**: `pettingllms/pettingllms/multi_agent_env/code/agents/code_agent.py:112`

### **What It Does**

1. **Receives raw model output**: Takes the LLM's text response as input
2. **Parses structured content**: Extracts the relevant action using regex or custom parsing logic
3. **Handles errors**: Provides fallback behavior if parsing fails
4. **Stores the action**: Saves extracted action in `self.current_action`
5. **Returns the action**: Makes it available for the `step()` function

### **Function Signature**

```python
def update_from_model(self, response: str):
    """
    Parse the model response and extract action.
    
    Args:
        response: Raw text response from the LLM
        
    Returns:
        Extracted action (format depends on agent type)
        - CodeGenerationAgent: returns str (code)
        - UnitTestGenerationAgent: returns dict ({"input": [...], "output": [...]})
    """
```

### **Code Example from CodeGenerationAgent**

```python
def update_from_model(self, response: str):
    """Parse the model response and extract code."""
    import re
    
    # Initialize empty code
    code = ""
    
    # Try to match Python code block in markdown format
    matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
    
    if matches:
        # Extract the last code block (in case of multiple)
        code = matches[-1].strip()
    else:
        # Fallback if no code block found
        code = "We can not extract the code in the output. "
    
    # Store action for next step
    self.current_action = code
    
    return self.current_action
```

### **Code Example from UnitTestGenerationAgent**

```python
def update_from_model(self, response: str):
    """Parse the model response and extract test cases."""
    import re
    
    # Use custom extraction function to parse test cases
    test_action = extract_test_cases(response)
    
    # test_action format: {"input": ["test1_input", "test2_input", ...], 
    #                      "output": ["test1_output", "test2_output", ...]}
    
    # Store action for next step
    self.current_action = test_action
    
    return self.current_action
```

### **Parsing Strategies by Agent Type**

Different agents return different action formats:

#### **CodeGenerationAgent** → Returns `str` (code)
```python
# Extract code from markdown blocks using regex
matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
code = matches[-1].strip() if matches else "We can not extract the code in the output. "
return code
```

#### **UnitTestGenerationAgent** → Returns `dict` (test cases)
```python
# Extract test cases using custom parsing function
from pettingllms.multi_agent_env.code.code_utils import extract_test_cases

test_action = extract_test_cases(response)
# Returns: {"input": ["input1", "input2", ...], "output": ["output1", "output2", ...]}
return test_action
```

### **Error Handling**

Always provide fallback behavior to handle parsing failures:

```python
# Example from CodeGenerationAgent
if matches:
    action = matches[-1].strip()
else:
    # Fallback: return error indicator
    action = "We can not extract the code in the output. "

# This ensures the agent can continue even if parsing fails
```

---

## **3. step**

**Purpose**: Executes the agent's action in the environment and updates the environment state.

**Location**: `pettingllms/pettingllms/multi_agent_env/code/agents/code_agent.py:131`

### **What It Does**

1. **Retrieves the current action**: Uses `self.current_action` set by `update_from_model`
2. **Executes the action**: Runs code, evaluates tests using Ray workers
3. **Updates environment state**: Writes results to `env_data.state` (shared across agents)
4. **Sets success status**: Updates `self.success` based on task completion
5. **Calculates rewards**: Determines agent performance via separate `calculate_reward()` method

### **Function Signature**

```python
async def step(self, env_data: Env, env_worker: Any = None):
    """
    Execute the agent's action and update environment state.
    
    Args:
        env_data: Environment object with state to be modified
        env_worker: Ray actor for sandboxed code execution (RayDockerWorker)
        
    Returns:
        None (updates env_data.state and self.success in place)
    """
```

### **Code Example from CodeGenerationAgent**

```python
async def step(self, env_data: Env, env_worker: Any = None):
    """
    Execute generated code and evaluate against tests.
    The action is the generated code, execute it and update the state.
    """
    
    # 1) Get the action (generated code from update_from_model)
    gen_code = self.current_action
    
    # 2) WRITE to environment state: store generated code
    env_data.state.generated_code = gen_code
    
    # 3) READ from environment state: get ground truth test cases
    ground_truth_test_input = env_data.state.ground_truth_test_input or []
    ground_truth_test_output = env_data.state.ground_truth_test_output or []
    
    passed_ratio = 0.0
    
    # 4) Evaluate code against test cases (if tests exist)
    if isinstance(ground_truth_test_input, list) and isinstance(ground_truth_test_output, list) \
       and ground_truth_test_input and ground_truth_test_output:
        try:
            # Run code in sandboxed Docker environment using Ray worker
            passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                gen_code,
                ground_truth_test_input,
                ground_truth_test_output,
                timeout=30.0,
                ray_actor=env_worker,  # RayDockerWorker for sandboxed execution
                rollout_idx=self.rollout_idx
            )
        except Exception as e:
            print(f"Warning: Failed to evaluate code against tests: {e}")
            passed_ratio, passed_cases, failed_cases = 0.0, [], [f"error: {e}"]
        
        # 5) WRITE to environment state: store evaluation results
        env_data.state.ground_truth_test_vs_generated_code_match_cases = passed_cases
        env_data.state.ground_truth_test_vs_generated_code_mismatch_cases = failed_cases
        env_data.state.ground_truth_test_vs_generated_code_match_ratio = passed_ratio
        
        # 6) Check for termination (all tests passed)
        if passed_ratio >= 1.0 and len(ground_truth_test_input) > 0:
            self.success = True
        else:
            self.success = False


def calculate_reward(self, env_data: Env):
    """Calculate reward based on test pass ratio (called separately)."""
    self.agent_reward = env_data.state.ground_truth_test_vs_generated_code_match_ratio
    self.reward_history.append(self.agent_reward)
```


### **Key Insights**

- **Async execution**: Uses `async/await` for non-blocking code execution
- **Ray workers**: Uses `env_worker` (RayDockerWorker) for sandboxed code execution
- **WRITES to env.state**: Updates shared environment state with action results
- **READS from env.state**: Accesses data written by other agents
- **State mutation**: All agents share state through environment
- **Reward calculation**: Separate `calculate_reward()` method called after `step()`
- **Termination control**: Sets `self.success` (agent-level) and `env_data.done` (environment-level)
- **Error handling**: Graceful fallback for execution failures

### **State Fields Written by CodeGenerationAgent**

The agent WRITES the shared information and the current environment state to `env_data.state`.

## **4. calculate_reward**

**Purpose**: Calculates the agent's reward based on the current environment state after action execution.

**Location**: 
- `pettingllms/pettingllms/multi_agent_env/code/agents/code_agent.py:161`
- `pettingllms/pettingllms/multi_agent_env/code/agents/unit_test_agent.py:198`

### **What It Does**

1. **Reads environment state**: Accesses evaluation metrics from `env_data.state`
2. **Calculates reward**: Assigns a numerical reward value based on performance
3. **Updates agent reward**: Stores reward in `self.agent_reward`
4. **Tracks reward history**: Appends reward to `self.reward_history` for RL training

### **Function Signature**

```python
def calculate_reward(self, env_data: Env):
    """
    Calculate agent reward based on environment state.
    
    Args:
        env_data: Environment object with state containing evaluation metrics
        
    Returns:
        None (updates self.agent_reward and self.reward_history in place)
    """
```

### **Key Insights**

- **Called after step()**: Always executed after `step()` completes
- **READS from env.state**: Accesses metrics written by `step()`
- **No state modification**: Only reads from environment, doesn't modify it
- **RL training signal**: Provides learning signal for policy optimization
- **Agent-specific**: Different agents can have different reward calculations


### **Reward Design Principles**

1. **Immediate feedback**: Reward reflects immediate task performance
2. **Normalized range**: Typically 0.0 to 1.0 for stable training
3. **Task-aligned**: Reward directly measures task success
4. **Differentiable**: Smooth reward signal for gradient-based learning

---

## **Complete Interaction Cycle**

Here's how the three functions work together in a complete agent-environment interaction:

### **Single Turn Example**

```python
# Turn 0: CodeGenerationAgent generates initial code

# 1. update_from_env (READ from environment state)
code_agent.update_from_env(turn_idx=0, env_data)
# → READS: env_data.state.problem
# → Creates: Initial code generation prompt with problem description
# → STORES: code_agent.current_prompt = {"text": "...", "image": None}

# 2. Model inference (handled by execution engine)
response = await model.generate(code_agent.current_prompt["text"])
# → LLM generates: "**Code:**\n```python\ndef solution(n):\n  return n*2\n```"

# 3. update_from_model (PARSE model response)
code_agent.update_from_model(response)
# → Parses: Extracts code from markdown using regex
# → STORES: code_agent.current_action = "def solution(n):\n  return n*2"

# 4. step (EXECUTE and WRITE to environment state)
await code_agent.step(env_data, env_worker)
# → WRITES: env_data.state.generated_code = "def solution(n):\n  return n*2"
# → Executes: Runs code against env_data.state.ground_truth_test_input/output
# → WRITES: env_data.state.ground_truth_test_vs_generated_code_match_ratio = 0.8
# → WRITES: env_data.state.ground_truth_test_vs_generated_code_match_cases = [...]
# → WRITES: env_data.state.ground_truth_test_vs_generated_code_mismatch_cases = [...]
# → Sets: code_agent.success = False (only 80% passed)

# 5. calculate_reward (CALCULATE agent reward)
code_agent.calculate_reward(env_data)
# → READS: env_data.state.ground_truth_test_vs_generated_code_match_ratio
# → Sets: code_agent.agent_reward = 0.8
# → Appends to: code_agent.reward_history.append(0.8)
```

### **Multi-Agent Multi-Turn Example**

```python
# === Turn 0: CodeGenerationAgent ===
code_agent.update_from_env(turn_idx=0, env_data)
# READS: state.problem = "Write a function to double a number"
response = await model.generate(code_agent.current_prompt["text"])
code_agent.update_from_model(response)
await code_agent.step(env_data, env_worker)
# WRITES: state.generated_code = "def solution(n): return n*2"
code_agent.calculate_reward(env_data)

# === Turn 1: UnitTestGenerationAgent ===
test_agent.update_from_env(turn_idx=1, env_data)
# READS: state.problem, state.generated_code (written by code_agent)
response = await model.generate(test_agent.current_prompt["text"])
test_agent.update_from_model(response)
# current_action = {"input": ["5", "10"], "output": ["10", "20"]}
await test_agent.step(env_data, env_worker)
# WRITES: state.generated_test_input = ["5", "10"]
# WRITES: state.generated_test_output = ["10", "20"]
# Evaluates tests against state.generated_code (from Turn 0)
# WRITES: state.generated_test_vs_generated_code_match_ratio = 1.0
# WRITES: state.generated_code_history.append(state.generated_code)
# WRITES: state.generated_test_vs_generated_code_mismatch_cases_history.append([])
test_agent.calculate_reward(env_data)

# === Turn 2: CodeGenerationAgent (refinement) ===
code_agent.update_from_env(turn_idx=2, env_data)
# READS: state.problem
# READS: state.generated_code_history (from Turn 1)
# READS: state.generated_test_vs_generated_code_mismatch_cases_history (from Turn 1)
# Creates refinement prompt with mismatch history
response = await model.generate(code_agent.current_prompt["text"])
code_agent.update_from_model(response)
await code_agent.step(env_data, env_worker)
# WRITES: state.generated_code = "refined code"
# Evaluates against state.ground_truth_test_input/output
code_agent.calculate_reward(env_data)
```

---

## **Implementation Patterns**

### **Pattern 1: Sequential Multi-Agent**

Agents take turns building on each other's outputs:

```python
# Turn 0: Code Generator
code_agent.update_from_env(0, env_data)  # Reads: problem
# ... model generates code ...
code_agent.step(env_data)                # Writes: generated_code

# Turn 1: Test Generator
test_agent.update_from_env(1, env_data)  # Reads: problem, generated_code
# ... model generates tests ...
test_agent.step(env_data)                # Writes: generated_test_input/output

# Turn 2: Code Generator (refinement)
code_agent.update_from_env(2, env_data)  # Reads: mismatch_cases_history
# ... model refines code ...
code_agent.step(env_data)                # Writes: updated generated_code
```

### **Pattern 2: Feedback Loop**

Agents iteratively improve based on evaluation results:

```python
for turn in range(max_turns):
    # Generate action
    agent.update_from_env(turn, env_data)
    response = model.generate(agent.current_prompt)
    agent.update_from_model(response)
    
    # Execute and evaluate
    await agent.step(env_data, env_worker)
    agent.calculate_reward(env_data)
    
    # Check termination
    if agent.success:
        break
```

---

## **Next Steps**

Continue exploring environment setup:

- Learn about environment state structure: [Environment State](environment-state.md)
- Understand registration system: [Registration](registration.md)
- Review configuration options: [Configuration](configuration.md)

