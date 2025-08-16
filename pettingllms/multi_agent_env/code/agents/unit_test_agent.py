import copy
import logging
from typing import Any

from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.multi_agent_env.code.code_utils import extract_test_cases

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class UnitTestGenerationAgent(Agent):
    """
    Agent specialized for generating unit test cases.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Unit Test Generation Agent.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)
        
        # 初始化多日志系统
        self.multi_logger = get_multi_logger()

    def update_from_env(self, env_data: Env):
        """
        Update the agent's internal prompt after an environment step.
        Rules:
        - If either state.current_code or state.current_test_input is None/empty, prompt to generate test cases.
        - Otherwise, refine or correct tests based on existing code and test cases.
        """
        # Save environment data
        self.env_data = env_data

        state = getattr(env_data, "state", None)
        agent_obs = getattr(env_data, "agent_observations", None)

        def as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join([str(v) for v in value])
            return str(value)

        if state is not None:
            question = as_text(getattr(state, "problem", ""))
            current_code = getattr(state, "current_code", None)
            current_test_input = getattr(state, "current_test_input", None)
            current_code_output = getattr(state, "current_code_output", None)
            current_test_output = getattr(state, "current_test_output", None)
            mismatch_testcases = getattr(state, "mismatch_testcases", None)
        elif agent_obs is not None:
            question = as_text(agent_obs.get("question", ""))
            current_code = agent_obs.get("current_code", None)
            current_test_input = agent_obs.get("current_test_input", None)
            current_code_output = agent_obs.get("current_code_output", None)
            current_test_output = agent_obs.get("current_test_output", None)
            mismatch_testcases = agent_obs.get("mismatch_testcases", None)
        else:
            question = ""
            current_code = None
            current_test_input = None
            current_code_output = None
            current_test_output = None
            mismatch_testcases = None

        need_generate = current_code in (None, "") or current_test_input in (None, "")

        if need_generate:
            # Test-case generation mode
            formatted_prompt = (
                f"<|im_start|>You are a helpful assistant that generates test examples for coding tasks.<|im_end|>\n"
                f"<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).\n"
                f"This is the problem:\n{question}\n\n"
                f"You need to provide a new test example. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.\n"
                f"Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.\n"
                f"Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n\n"
                f"**Test Input:**\n```\ninput here\n```\n\n"
                f"**Test Output:**\n```\noutput here\n```\n\n"
                f"**Explanation:**\nexplanation here.<|im_end|>\n"
                f"<|im_start|>Assistant:"
            )
        else:
            # Test-case refinement mode
            formatted_prompt = (
                f"<|im_start|>You are a helpful assistant that refines or corrects test examples for coding tasks.<|im_end|>\n"
                f"<|im_start|>User: Given a coding task, instead of providing the final script, your task is to refine or correct test examples.\n"
                f"This is the problem:\n{question}\n\n"
                f"Current code output:\n{as_text(current_code_output)}\n\n"
                f"Current tests (inputs):\n{as_text(current_test_input)}\n\n"
                f"Current tests (expected outputs):\n{as_text(current_test_output)}\n\n"
                f"Mismatch summary (if any):\n{as_text(mismatch_testcases)}\n\n"
                f"You need to provide corrected or more discriminative tests while keeping format consistent. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.\n"
                f"Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.\n"
                f"Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n\n"
                f"**Test Input:**\n```\ninput here\n```\n\n"
                f"**Test Output:**\n```\noutput here\n```\n\n"
                f"**Explanation:**\nexplanation here.<|im_end|>\n"
                f"<|im_start|>Assistant:"
            )

        self.current_prompt = {"text": formatted_prompt, "image": None}
          
    def update_from_model(self, response: str):
        # Parse the response and update agent_data
        import re
        test_action = extract_test_cases(response)
        
        # Parse test cases
        self.current_action = test_action
   
        
        return self.current_action
    
    def calculate_reward(self, env_data: Env, mode: str = "sum") -> float:
        """
        Compute reward based on environment state, supporting three modes:
        - generated: use generated_pass_ratio (prefer generated_test_vs_generated_code_match_ratio, fallback to generated_test_vs_golden_code_match_ratio)
        - golden: use golden_pass_ratio (golden_test_vs_generated_code_match_ratio)
        - sum/both/others: sum of both
        """
        state = getattr(env_data, "state", None)
        generated_pass_ratio = 0.0
        golden_pass_ratio = 0.0

        if state is not None:
            # Generated tests vs generated code
            gen_vs_gen = getattr(state, "generated_test_vs_generated_code_match_ratio", None)
            # Generated tests vs golden code (as fallback)
            gold_vs_gen = getattr(state, "golden_test_vs_generated_code_match_ratio", None)

            if isinstance(gen_vs_gen, (int, float)):
                generated_pass_ratio = float(gen_vs_gen)
            if isinstance(gold_vs_gen, (int, float)):
                golden_pass_ratio = float(gold_vs_gen)

        m = (mode or "sum").lower()
        if m in ("generated", "gen"):
            reward = generated_pass_ratio
        elif m in ("golden", "gold"):
            reward = golden_pass_ratio
        else:
            reward = generated_pass_ratio + golden_pass_ratio

        # Record and return
        self.agent_reward = reward
        if self.info is None:
            self.info = {}
        self.info.update({
            "generated_pass_ratio": generated_pass_ratio,
            "golden_pass_ratio": golden_pass_ratio,
            "reward_mode": m,
        })
        
        return reward

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