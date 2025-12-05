"""
Autogen-based Reasoning Agent for mathematical problem solving.
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from math_verify import parse


class ReasoningAgent(AssistantAgent):
    """
    Reasoning agent that solves math problems through step-by-step reasoning.
    Inherits from autogen's AssistantAgent.
    """

    def __init__(self, model_client, name="reasoning_agent"):
        system_message = (
            "You are a mathematical reasoning expert. "
            "Solve problems step by step using clear logical reasoning. "
            "Always provide your final answer in \\boxed{} format. "
            "Example: \\boxed{123}\n\n"
            "When reviewing others' solutions:\n"
            "- If a solution is correct, confirm it\n"
            "- If solutions differ, carefully analyze which is correct\n"
            "- If all are incorrect, provide the correct solution\n"
            "- Explain your reasoning clearly"
        )

        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
        )

        self.extracted_answer = None
        self.solution_history = []

    def extract_answer(self, response: str) -> str:
        """Extract the answer from the response using math_verify.parse"""
        try:
            answer = parse(response)
            self.extracted_answer = answer
            return answer
        except Exception as e:
            self.extracted_answer = None
            return None

    def get_answer(self) -> str:
        """Get the current extracted answer"""
        return self.extracted_answer
