import copy
import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.logger_config import get_multi_logger
from typing import List
from pettingllms.multi_agent_env.math_single_agent.math_utils import evaluate_math_solution

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class MathSolvingAgent(Agent):
    """
    Agent specialized for solving mathematical problems.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Math Solving Agent's data.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)
        
        # 初始化多日志系统
        self.multi_logger = get_multi_logger()

    def update_from_env(self, env_data: Env):
        # Save environment data
        self.env_data = env_data

        # Support passing either the raw environment (with state) or a wrapped Env
        state = getattr(env_data, "state", None)
        agent_obs = getattr(env_data, "agent_observations", None)

        def as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join([str(v) for v in value])
            return str(value)
        

        problem = getattr(state, "problem", None)
        current_solution = getattr(state, "generated_solution", None)
        ground_truth_answer = getattr(state, "ground_truth_answer", None)
        current_extracted_answer = getattr(state, "extracted_answer", None)
        current_correctness = getattr(state, "is_correct", None)
        
        need_generate = current_solution in (None, "") or current_extracted_answer in (None, "")

        if need_generate:
            # Generation mode
            formatted_prompt = (
                f"You are a helpful assistant that solves mathematical problems step by step.\n\n"
                f"You need to think step by step and provide a complete solution with clear reasoning.\n"
                f"Please provide your final answer in #### format.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Put your final answer in the format of `#### <answer>`.\n\n For the final answer, only output the answer after ####, no other text."
                f"example: `#### 123`"
            )
        else:
            # Refinement mode - if the current answer is incorrect
            if current_correctness is False:
                formatted_prompt = (
                    f"You are a helpful assistant that refines mathematical solutions. You need to think step by step and provide a complete solution with clear reasoning.\n\n"
                    f"Problem:\n{problem}\n\n"
                    f"Your previous solution:\n{as_text(current_solution)}\n\n "
                    f"Your extracted answer: {current_extracted_answer}\n"
                    f"Your previous answer was incorrect. Please solve the problem again with more careful reasoning.\n"
                    "Put your final answer in the format of `#### <answer>`."
                    f"For the final answer, only output the answer after ####, no other text."
                    f"example: `#### 123`"
                )
        self.current_prompt = {"text": formatted_prompt, "image": None}
        
    
    def update_from_model(self, response: str):
        # Parse the response and update agent_data
        self.current_action = response.strip()
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Process the generated solution and evaluate it against the ground truth.
        """
        # 1) Update generated solution
        generated_solution = self.current_action
        env_data.state.generated_solution = generated_solution

        # 2) Extract answer from the solution
        extracted_answer = extract_answer(generated_solution)
        env_data.state.extracted_answer = extracted_answer

        # 3) Evaluate correctness
        ground_truth_answer = env_data.state.ground_truth_answer
        is_correct = False
        
        if extracted_answer is not None and ground_truth_answer is not None:
            try:
                is_correct = evaluate_math_solution(generated_solution, ground_truth_answer)
                env_data.state.is_correct = is_correct
                if is_correct:
                    self.done = True
                    self.is_pass = True
                    
            except Exception as e:
                print(f"Warning: Failed to evaluate math solution: {e}")
                is_correct = False
                env_data.state.is_correct = False
        else:
            env_data.state.is_correct = False

        # 4) Update reward based on correctness
        if len(self.reward_history) > 0:
            self.agent_reward = float(is_correct) - self.reward_history[-1]
        else:
            self.agent_reward = float(is_correct)
        self.reward_history.append(float(is_correct))

    def calculate_reward(self, env_data: List[Env]) -> float:
        """
        Compute reward based on environment state.
        Uses correctness for reward calculation.
        """
        state = getattr(env_data[0], "state", None)
        correctness = 0.0

        if state is not None:
            is_correct = getattr(state, "is_correct", None)
            if isinstance(is_correct, bool):
                correctness = float(is_correct)

        # Record and return
        self.agent_reward = correctness
        self.reward_history.append(self.agent_reward)
        
        return self.agent_reward
    
    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
