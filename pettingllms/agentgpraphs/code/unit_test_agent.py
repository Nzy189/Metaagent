import copy
import logging
from typing import Any

from pettingllms.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class UnitTestGenerationAgent(BaseAgent):
    """
    Agent specialized for generating unit test cases.
    """

    def __init__(self, remove_thinking=False, max_tests=2):
        """
        Initialize the Unit Test Generation Agent.
        """
        self.revise_instruction = "Based on feedback from the previous attempt, revise test cases to improve test coverage and accuracy."
        self._trajectory = Trajectory()
        self.messages = []
        self.remove_thinking = remove_thinking
        self.max_tests = max_tests
        self.current_observation = None

    def format_test_feedback(self, validation_results: dict) -> str:
        """Format test validation result feedback"""
        if "error" in validation_results:
            return f"Error occurred during test validation: {validation_results['error']}\nPlease check and modify your test cases."
        
        if not validation_results.get("all_passed", False):
            test_results = validation_results.get("test_results", [])
            if not test_results:
                return "Test cases did not pass validation, but no detailed results available. Please recheck your test cases."
            
            formatted_feedback = "Test case validation results:\n"
            invalid_count = 0
            
            for test in test_results:
                if not test.get("is_valid", False):
                    formatted_feedback += f"### Test case {test.get('test_index', 'unknown') + 1} is invalid\n"
                    formatted_feedback += f"  Input: {truncatefn(test.get('input', 'N/A'))}\n"
                    formatted_feedback += f"  Your expected output: {truncatefn(test.get('generated_output', 'N/A'))}\n"
                    if 'reference_output' in test:
                        formatted_feedback += f"  Reference code actual output: {truncatefn(test['reference_output'])}\n"
                    elif 'code_output' in test:
                        formatted_feedback += f"  Current code actual output: {truncatefn(test['code_output'])}\n"
                    formatted_feedback += "\n"
                    
                    invalid_count += 1
                    if invalid_count >= self.max_tests:
                        break
            
            passed_tests = validation_results.get("passed_tests", 0)
            total_tests = validation_results.get("total_tests", 0)
            formatted_feedback += f"Valid test cases: {passed_tests}/{total_tests}\n"
            formatted_feedback += "Please analyze the mismatch reasons and modify test cases to ensure consistency between inputs and expected outputs."
            
            return formatted_feedback
        else:
            return "Congratulations! All your generated test cases are valid. Consider adding more test cases to cover edge cases if needed."

    def format_initial_prompt(self, question: str, generated_codes: list = None) -> str:
        """Format initial test generation prompt"""
        prompt = f"Please generate test cases for the following programming problem:\n\n{question}\n\n"
        
        if generated_codes:
            prompt += "Current available code solutions:\n"
            for i, code in enumerate(generated_codes):
                prompt += f"### Solution {i + 1}:\n```python\n{code}\n```\n\n"
        
        prompt += """Please generate multiple test cases to verify code correctness. Each test case should include:

**Test Input:**
```
[input data]
```

**Test Output:**
```
[expected output]
```

Please ensure:
1. Test cases cover various scenarios (normal cases, edge cases, special cases)
2. Input format matches the problem description
3. Expected outputs are correct
4. Generate at least 3-5 test cases"""
        
        return prompt

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update the agent's internal state after an environment step.
        """
        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self._trajectory.steps:
            # Initial problem statement
            assert isinstance(observation, dict) and "question" in observation, "Initial observation must be a dict with a 'question' key."
            question = observation["question"]
            state = observation.get("state", {})
            generated_codes = state.get("generated_codes", [])
            formatted_observation = self.format_initial_prompt(question, generated_codes)
        else:
            if "validation_results" in observation:
                validation_results = observation["validation_results"]
                formatted_observation = self.format_test_feedback(validation_results)
            elif "error" in observation:
                formatted_observation = f"Execution error: {observation['error']}\nPlease modify your test cases."
            else:
                formatted_observation = str(observation)

        if done:
            return

        self.messages.append({"role": "user", "content": formatted_observation})
        self.current_observation = formatted_observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update the agent's internal state based on the model's response.
        """
        content = response
        action = response

        # Handle thinking removal if needed
        if self.remove_thinking and content.count("</think>") == 1:
            thought, action = response.split("</think>")
            thought += "</think>"
            action = action.strip()
            self.messages.append({"role": "assistant", "content": action})
        else:
            self.messages.append({"role": "assistant", "content": response})

        # Create new step
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions), 
            action=action, 
            model_response=response, 
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)

        return Action(action=action)

    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.current_observation = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return the history of messages for chat completion."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Return the trajectory object."""
        return self._trajectory

    def get_current_state(self) -> Step | None:
        """Return the current step/state of the agent."""
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1] 