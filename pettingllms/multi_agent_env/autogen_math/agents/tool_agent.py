"""
Autogen-based Tool Agent for mathematical problem solving using Python code.
"""
import re
from autogen_agentchat.agents import AssistantAgent
from math_verify import parse


class ToolAgent(AssistantAgent):
    """
    Tool agent that solves math problems by writing and executing Python code.
    Inherits from autogen's AssistantAgent.
    """

    def __init__(self, model_client, name="tool_agent", executor=None):
        system_message = (
            "You are a programming expert specializing in solving mathematical problems. "
            "Write clear, executable Python code to solve problems step by step. "
            "Always use print() to output the final answer. "
            "Print the variable, not just the digit. "
            "Example: If the answer is stored in variable x, write print(x)\n\n"
            "Format your response as:\n"
            "**Code:**\n"
            "```python\n"
            "# your code here\n"
            "```\n\n"
            "When reviewing others' solutions:\n"
            "- If a solution is correct, confirm it\n"
            "- If solutions differ, write code to verify which is correct\n"
            "- If all are incorrect, provide corrected code\n"
            "- Explain your reasoning"
        )

        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
        )

        self.executor = executor
        self.extracted_answer = None
        self.last_code = None
        self.last_output = None

    def extract_code(self, response: str) -> str:
        """Extract Python code from the response"""
        # Look for code blocks
        code_pattern = r"```python\n(.*?)```"
        matches = re.findall(code_pattern, response, re.DOTALL)
        if matches:
            self.last_code = matches[-1].strip()
            return self.last_code
        return None

    async def execute_code(self, code: str, timeout: float = 20.0) -> str:
        """Execute the code and return the output"""
        if not code:
            return "No code to execute"

        try:
            # Create a safe execution environment
            local_vars = {}
            exec_globals = {
                "__builtins__": __builtins__,
                "print": print,
            }

            # Capture stdout
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                exec(code, exec_globals, local_vars)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            self.last_output = output.strip()
            return self.last_output

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            self.last_output = error_msg
            return error_msg

    def extract_answer(self, output: str) -> str:
        """Extract the answer from execution output using math_verify.parse"""
        try:
            answer = parse(output)
            self.extracted_answer = answer
            return answer
        except Exception:
            self.extracted_answer = None
            return None

    def get_answer(self) -> str:
        """Get the current extracted answer"""
        return self.extracted_answer
