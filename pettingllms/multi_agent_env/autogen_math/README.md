# Autogen Math - Multi-Agent Mathematical Problem Solving

This module implements a multi-agent system for solving mathematical problems using [Autogen](https://github.com/microsoft/autogen), where a reasoning agent and a tool agent discuss and collaborate to reach consensus on the answer.

## Overview

The system features:
- **Reasoning Agent**: Solves problems through step-by-step logical reasoning
- **Tool Agent**: Solves problems by writing and executing Python code
- **Discussion Protocol**: Agents discuss for up to 3 rounds until their answers agree

## Architecture

```
ReasoningAgent (logical reasoning)
         ↕ (discuss and compare)
ToolAgent (code execution)
```

The agents exchange solutions and critique each other until:
1. Their answers agree (consensus reached)
2. Maximum rounds (3) are exhausted

## Components

### 1. Reasoning Agent
- Uses natural language reasoning
- Provides step-by-step explanations
- Outputs answers in `\boxed{}` format

### 2. Tool Agent
- Writes Python code to solve problems
- Executes code safely
- Extracts numerical answers from execution output

### 3. Math Discussion Orchestrator
- Manages the discussion between agents
- Tracks discussion history
- Determines when consensus is reached
- Handles answer verification

## Usage

### Basic Example

```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pettingllms.multi_agent_env.autogen_math import MathDiscussion

async def main():
    # Create OpenAI client
    client = OpenAIChatCompletionClient(
        model="gpt-4o",
        # api_key="your-api-key",  # If not in environment
    )

    # Create discussion orchestrator
    discussion = MathDiscussion(model_client=client, max_rounds=3)

    # Solve a problem
    problem = "What is 15% of 80?"
    result = await discussion.solve_problem(problem)

    print(f"Final Answer: {result['final_answer']}")
    print(f"Rounds Used: {result['rounds_used']}")
    print(f"Answers Agree: {result['answers_agree']}")

asyncio.run(main())
```

### Running the Example

```bash
cd /lp-dev/user/yujie/PettingLLMs
python pettingllms/multi_agent_env/autogen_math/math_discussion.py
```

## Discussion Protocol

### Round 1
1. Reasoning agent receives the problem and provides a solution
2. Tool agent receives the problem and writes code to solve it
3. Compare answers:
   - If agree: Done ✓
   - If disagree: Continue to Round 2

### Round 2+
1. Both agents receive:
   - Original problem
   - History of all previous solutions
   - Request to review and provide corrected solution
2. Agents provide new solutions
3. Compare answers:
   - If agree: Done ✓
   - If disagree and rounds < 3: Continue
   - If rounds = 3: Use last answers

## Result Structure

The `solve_problem` method returns a dictionary with:

```python
{
    "problem": str,                    # Original problem
    "final_answer": str,               # Final consensus answer
    "reasoning_answer": str,           # Last reasoning agent answer
    "tool_answer": str,                # Last tool agent answer
    "answers_agree": bool,             # Whether agents reached consensus
    "rounds_used": int,                # Number of discussion rounds
    "discussion_history": list[dict],  # Full discussion history
    "is_correct": bool,                # Correctness vs ground truth (if provided)
    "ground_truth": str                # Ground truth answer (if provided)
}
```

## Dependencies

- `autogen-agentchat`
- `autogen-ext`
- `math_verify` (for answer parsing and verification)

## Comparison with Original Math Environment

| Feature | Original Math Env | Autogen Math |
|---------|------------------|--------------|
| Framework | Custom Agent base class | Autogen AssistantAgent |
| Communication | Sequential turns | Interactive discussion |
| Consensus | Checked after each turn | Multi-round negotiation |
| Max Rounds | Configurable | 3 (default) |
| Code Execution | Ray worker | Direct execution |

## Future Enhancements

- [ ] Add support for more complex tool calling
- [ ] Implement memory/context management for longer discussions
- [ ] Add visualization of discussion flow
- [ ] Support batch problem solving
- [ ] Integration with existing math dataset loaders
