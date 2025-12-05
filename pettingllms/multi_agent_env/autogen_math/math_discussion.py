"""
Main runner for autogen-based math problem solving with reasoning and tool agents.
The two agents discuss for up to 3 rounds until they agree on the answer.
"""
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from math_verify import verify

from pettingllms.multi_agent_env.autogen_math.agents.reasoning_agent import ReasoningAgent
from pettingllms.multi_agent_env.autogen_math.agents.tool_agent import ToolAgent


class MathDiscussion:
    """
    Orchestrates discussion between reasoning and tool agents to solve math problems.
    The agents discuss for up to 3 rounds until their answers agree.
    """

    def __init__(self, model_client, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.reasoning_agent = ReasoningAgent(model_client=model_client)
        self.tool_agent = ToolAgent(model_client=model_client)
        self.model_client = model_client

    async def solve_problem(self, problem: str, ground_truth: str = None) -> dict:
        """
        Solve a math problem through agent discussion.

        Args:
            problem: The math problem to solve
            ground_truth: Optional ground truth answer for verification

        Returns:
            dict with solution details including final answer and discussion history
        """
        print(f"\n{'='*80}")
        print(f"Problem: {problem}")
        print(f"{'='*80}\n")

        discussion_history = []
        reasoning_answer = None
        tool_answer = None

        for round_idx in range(self.max_rounds):
            print(f"\n--- Round {round_idx + 1}/{self.max_rounds} ---\n")

            # Build context from previous rounds
            if round_idx == 0:
                # First round: fresh problem
                reasoning_prompt = (
                    f"Problem:\n{problem}\n\n"
                    f"Please think step by step and output the final answer in \\boxed{{}} format.\n"
                    f"Example: \\boxed{{123}}"
                )
                tool_prompt = (
                    f"Problem:\n{problem}\n\n"
                    f"Please write Python code to solve this problem.\n"
                    f"Use print() to output the final answer."
                )
            else:
                # Subsequent rounds: include previous solutions
                history_text = self._build_history_text(discussion_history)
                reasoning_prompt = (
                    f"Problem:\n{problem}\n\n"
                    f"{history_text}\n\n"
                    f"The previous solutions disagree. Please review them carefully and provide your solution.\n"
                    f"If one is correct, confirm it. If both are wrong, provide the correct solution.\n"
                    f"Output your final answer in \\boxed{{}} format."
                )
                tool_prompt = (
                    f"Problem:\n{problem}\n\n"
                    f"{history_text}\n\n"
                    f"The previous solutions disagree. Please write Python code to solve this problem.\n"
                    f"Use print() to output the final answer."
                )

            # Reasoning agent's turn
            print(f"[Reasoning Agent] Thinking...")
            reasoning_response = await self._get_agent_response(
                self.reasoning_agent, reasoning_prompt
            )
            reasoning_answer = self.reasoning_agent.extract_answer(reasoning_response)
            print(f"[Reasoning Agent] Answer: {reasoning_answer}")
            print(f"[Reasoning Agent] Response: {reasoning_response[:300]}...\n")

            # Tool agent's turn
            print(f"[Tool Agent] Writing code...")
            tool_response = await self._get_agent_response(self.tool_agent, tool_prompt)
            code = self.tool_agent.extract_code(tool_response)

            if code:
                print(f"[Tool Agent] Executing code...")
                output = await self.tool_agent.execute_code(code)
                tool_answer = self.tool_agent.extract_answer(output)
                print(f"[Tool Agent] Code output: {output}")
                print(f"[Tool Agent] Answer: {tool_answer}\n")
            else:
                print(f"[Tool Agent] No code found in response")
                tool_answer = None

            # Record this round
            discussion_history.append(
                {
                    "round": round_idx + 1,
                    "reasoning_response": reasoning_response,
                    "reasoning_answer": reasoning_answer,
                    "tool_response": tool_response,
                    "tool_code": code,
                    "tool_output": self.tool_agent.last_output,
                    "tool_answer": tool_answer,
                }
            )

            # Check if answers agree
            if reasoning_answer is not None and tool_answer is not None:
                answers_agree = verify(reasoning_answer, tool_answer)
                print(f"[Discussion] Answers agree: {answers_agree}")

                if answers_agree:
                    print(f"[Discussion] ✓ Consensus reached in round {round_idx + 1}!")
                    break
            else:
                print(
                    f"[Discussion] Could not extract answers from one or both agents"
                )

            if round_idx < self.max_rounds - 1:
                print(f"[Discussion] Answers differ, continuing to next round...\n")
            else:
                print(
                    f"[Discussion] Max rounds reached without consensus. Using last answers."
                )

        # Final result
        final_answer = reasoning_answer if reasoning_answer is not None else tool_answer
        is_correct = None
        if ground_truth and final_answer:
            is_correct = verify(final_answer, ground_truth)

        result = {
            "problem": problem,
            "final_answer": final_answer,
            "reasoning_answer": reasoning_answer,
            "tool_answer": tool_answer,
            "answers_agree": (
                verify(reasoning_answer, tool_answer)
                if reasoning_answer and tool_answer
                else False
            ),
            "rounds_used": len(discussion_history),
            "discussion_history": discussion_history,
            "is_correct": is_correct,
            "ground_truth": ground_truth,
        }

        print(f"\n{'='*80}")
        print(f"Final Answer: {final_answer}")
        if ground_truth:
            print(f"Ground Truth: {ground_truth}")
            print(f"Correct: {is_correct}")
        print(f"Rounds Used: {len(discussion_history)}/{self.max_rounds}")
        print(f"Answers Agree: {result['answers_agree']}")
        print(f"{'='*80}\n")

        return result

    async def _get_agent_response(self, agent, prompt: str) -> str:
        """Get response from an agent given a prompt"""
        # Use the agent's on_messages method to generate response
        messages = [TextMessage(content=prompt, source="user")]

        response = await agent.on_messages(messages, cancellation_token=None)

        # Extract the text content from the response
        if hasattr(response, "chat_message") and hasattr(
            response.chat_message, "content"
        ):
            return response.chat_message.content
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)

    def _build_history_text(self, history: list) -> str:
        """Build a text summary of discussion history"""
        text = "History of previous solutions:\n"
        for entry in history:
            round_num = entry["round"]
            text += f"\n--- Round {round_num} ---\n"
            text += f"Reasoning solution: {entry['reasoning_response'][:200]}...\n"
            text += f"Reasoning answer: {entry['reasoning_answer']}\n"
            if entry["tool_code"]:
                text += f"Code solution:\n{entry['tool_code'][:200]}...\n"
            text += f"Code output: {entry['tool_output']}\n"
            text += f"Code answer: {entry['tool_answer']}\n"
        return text


async def run_example():
    """Example usage of the MathDiscussion system"""
    # Create OpenAI client
    client = OpenAIChatCompletionClient(
        model="gpt-4o",
        # api_key="your-api-key",  # If not in environment
    )

    # Create discussion orchestrator
    discussion = MathDiscussion(model_client=client, max_rounds=3)

    # Example problem
    problem = (
        "A rectangular box has a volume of 120 cubic units. "
        "Its length is twice its width, and its height is 5 units. "
        "What is the width of the box?"
    )
    ground_truth = "2\\sqrt{3}"

    # Solve the problem
    result = await discussion.solve_problem(problem, ground_truth)

    return result


if __name__ == "__main__":
    asyncio.run(run_example())
