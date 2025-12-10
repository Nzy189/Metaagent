import asyncio
import re
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

from autogen_ext.models.openai import OpenAIChatCompletionClient

from pettingllms.multi_agent_env.autogen_graph.math_graph.math_env import MathEnv, MathEnvBatch
from pettingllms.multi_agent_env.math.math_utils import extract_code


def extract_answer(text: str) -> str:
    """
    Extract the final answer from solution text.
    Looks for patterns like:
    - "The answer is X"
    - "Final answer: X"
    - "Answer: X"
    - Last boxed expression \\boxed{X}
    
    Args:
        text: Solution text
        
    Returns:
        Extracted answer or empty string
    """
    if not text:
        return ""
    
    # Try to find boxed answer (LaTeX style)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # Try to find explicit answer statements
    answer_patterns = [
        r'[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$)',
        r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
        r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    
    # If no pattern matched, try to extract last line with numbers
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Check if last line contains numbers
        if re.search(r'\d', last_line):
            return last_line
    
    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    - Remove extra whitespace
    - Convert to lowercase
    - Remove common punctuation
    - Extract numbers if present
    
    Args:
        answer: Answer string
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
    
    # Convert to lowercase and strip
    normalized = answer.lower().strip()
    
    # Remove common punctuation and symbols
    normalized = re.sub(r'[,\$\s]+', '', normalized)
    
    # Try to extract numeric value if present
    numeric_match = re.search(r'-?\d+\.?\d*', normalized)
    if numeric_match:
        return numeric_match.group(0)
    
    return normalized


def check_answer_correctness(generated_answer: str, ground_truth_answer: str) -> bool:
    """
    Check if generated answer matches ground truth.
    
    Args:
        generated_answer: Generated answer string
        ground_truth_answer: Ground truth answer string
        
    Returns:
        True if answers match, False otherwise
    """
    if not generated_answer or not ground_truth_answer:
        return False
    
    # Normalize both answers
    gen_norm = normalize_answer(generated_answer)
    gt_norm = normalize_answer(ground_truth_answer)
    
    # Direct comparison
    if gen_norm == gt_norm:
        return True
    
    # Try comparing as floats if both are numeric
    try:
        gen_float = float(gen_norm)
        gt_float = float(gt_norm)
        # Allow small floating point differences
        return abs(gen_float - gt_float) < 1e-6
    except (ValueError, TypeError):
        pass
    
    return False


async def math_graph(env: Optional[MathEnv] = None, model_client_dict: Optional[dict] = None):
    """
    Main function for math problem solving workflow using autogen.
    
    This workflow:
    1. Math solver generates a step-by-step solution
    2. Verifier checks the solution and provides feedback
    3. Loop continues until solution is approved or max iterations reached
    4. Extract final answer and compare with ground truth
    5. Assign final_reward (1.0 if correct, 0.0 otherwise)
    
    Args:
        env: Optional MathEnv instance with problem and ground truth
        model_client_dict: Dictionary of model clients for each agent
        
    Returns:
        env: Updated environment with final_reward
    """
    # Get problem from env
    task = env.state.problem
    if not task:
        raise ValueError("Environment provided but no problem found in env.state.problem")
    
    # Define solver agent
    solver = AssistantAgent(
        "math_solver",
        model_client=model_client_dict["math_solver"],
        system_message=(
            "You are an expert mathematician. "
            "Given a mathematical problem, provide a detailed step-by-step solution. "
            "Show your reasoning clearly and conclude with the final answer in the format:\n"
            "Final Answer: <your answer>\n"
            "Or use LaTeX boxed notation: \\boxed{<your answer>}"
        ),
    )
    
    # Define verifier agent
    verifier = AssistantAgent(
        "math_verifier",
        model_client=model_client_dict["math_verifier"],
        system_message=(
            "You are a strict mathematics verifier. "
            "Review the solution provided and check for logical errors, calculation mistakes, or unclear reasoning. "
            "If the solution is correct and complete, reply with exactly:\n"
            "APPROVE\n"
            "Otherwise, reply with:\n"
            "NEEDS_REVISION: <brief explanation of the issue>\n"
            "Suggest how to fix the problem."
        ),
    )
    
    # Build graph: solver -> verifier -> (solver or end)
    builder = DiGraphBuilder()
    builder.add_node(solver).add_node(verifier)
    
    builder.add_edge(solver, verifier)
    
    def approved(msg):
        return "APPROVE" in msg.to_model_text()
    
    def needs_revision(msg):
        return "NEEDS_REVISION" in msg.to_model_text()
    
    builder.add_edge(verifier, solver, condition=needs_revision)
    
    graph = builder.build()
    
    team = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=MaxMessageTermination(15),
    )
    
    # Run the workflow
    await Console(team.run_stream(task=task))
    
    # Extract the final solution from solver's messages
    final_solution: Optional[str] = None
    try:
        # Try to get solver's message history
        possible_histories = []
        for attr in ("messages", "_messages", "chat_history", "history"):
            if hasattr(solver, attr):
                possible_histories.append(getattr(solver, attr))
        
        # Scan messages for solution text
        for hist in possible_histories:
            if not hist:
                continue
            for m in hist:
                try:
                    text = (
                        m.to_model_text() if hasattr(m, "to_model_text") else (
                            m.get("content", "") if isinstance(m, dict) else str(m)
                        )
                    )
                except Exception:
                    text = str(m)
                
                if text:
                    final_solution = text
    except Exception:
        pass
    
    # If env is provided, evaluate the solution
    if env is not None:
        try:
            ground_truth = env.state.ground_truth_answer or ""
            extracted_answer = ""
            is_correct = False
            
            if final_solution:
                # Extract answer from solution
                extracted_answer = extract_answer(final_solution)
                
                # Check correctness
                is_correct = check_answer_correctness(extracted_answer, ground_truth)
                
                # Update env state
                env.state.reasoning_generated_solution = final_solution
                env.state.reasoning_generated_solution_history.append(final_solution)
                env.state.reasoning_extracted_answer = extracted_answer
                env.state.reasoning_extracted_answer_history.append(extracted_answer)
                env.state.reasoning_is_correct = is_correct
            
            # Assign final reward: 1.0 if correct, 0.0 otherwise
            final_reward = 1.0 if is_correct else 0.0
            env.state.final_reward = final_reward
            env.final_reward = final_reward
            
        except Exception as e:
            # In case of any evaluation failure, assign zero reward
            print(f"Warning: Failed to evaluate math solution: {e}")
            env.state.final_reward = 0.0
            env.final_reward = 0.0
    
    # Return env with final_reward
    if env is not None:
        return env
