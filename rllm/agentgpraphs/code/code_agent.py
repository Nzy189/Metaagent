import copy
import logging
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CodeGenerationAgent(BaseAgent):
    """
    Agent specialized for generating code to solve programming problems.
    """

    def __init__(self, remove_thinking=False, max_tests=2, public_test_only=True):
        """
        Initialize the Code Generation Agent.
        """
        self.revise_instruction = "Based on feedback from the previous attempt, revise the code to fix errors and improve the solution."
        self._trajectory = Trajectory()
        self.messages = []
        self.remove_thinking = remove_thinking
        self.max_tests = max_tests
        self.public_test_only = public_test_only
        self.current_observation = None

    def format_code_feedback(self, validation_results: dict) -> str:
        """Format code validation result feedback"""
        if "error" in validation_results:
            return f"Error occurred during code validation: {validation_results['error']}\nPlease check and modify your code."
        
        if not validation_results.get("all_passed", False):
            test_results = validation_results.get("test_results", [])
            if not test_results:
                return "Code did not pass validation, but no detailed test results available. Please recheck your solution."
            
            formatted_feedback = "Code validation results:\n"
            failed_count = 0
            
            for test in test_results:
                if not test.get("passed", False):
                    formatted_feedback += f"### Test {test.get('test_index', 'unknown') + 1} failed\n"
                    formatted_feedback += f"  Input: {truncatefn(test.get('input', 'N/A'))}\n"
                    formatted_feedback += f"  Expected output: {truncatefn(test.get('expected', 'N/A'))}\n"
                    if 'actual' in test:
                        formatted_feedback += f"  Actual output: {truncatefn(test['actual'])}\n"
                    formatted_feedback += "\n"
                    
                    failed_count += 1
                    if failed_count >= self.max_tests:
                        break
            
            passed_tests = validation_results.get("passed_tests", 0)
            total_tests = validation_results.get("total_tests", 0)
            formatted_feedback += f"Passed {passed_tests}/{total_tests} test cases.\n"
            formatted_feedback += "Please analyze the error patterns, modify the code to address these issues, and ensure the solution handles all test cases correctly."
            
            return formatted_feedback
        else:
            return "Congratulations! Your code has successfully passed all test cases. Please carefully review your solution once more to ensure it handles all edge cases properly."

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update the agent's internal state after an environment step.
        """
        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self._trajectory.steps:
            # Initial problem statement
            assert isinstance(observation, dict) and "question" in observation, "Initial observation must be a dict with a 'question' key."
            question = observation["question"]
            formatted_observation = f"Please write Python code for the following programming problem:\n\n{question}\n\nProvide your solution between ```python and ```."
        else:
            if "validation_results" in observation:
                validation_results = observation["validation_results"]
                formatted_observation = self.format_code_feedback(validation_results)
            elif "error" in observation:
                formatted_observation = f"Execution error: {observation['error']}\nPlease modify your code."
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
