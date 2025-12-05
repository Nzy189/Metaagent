from pettingllms.dyevolve.workflow.nodes.agent_node import AgentNode
from typing import List, Dict, Any, Optional
import json
import logging
import re
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.dyevolve.prompt_code import (
    CODE_WORKFLOW_EXAMPLES,
    CODE_GENERATION_PROMPT_TEMPLATE,
    get_code_generation_prompt
)
from pettingllms.multi_agent_env.dyevolve.prompt_math import (
    get_math_generation_prompt
)

logger = logging.getLogger(__name__)


class MASCodeGenerationAgent(Agent):
    

    def __init__(self, task_type: str = "code", rollout_idx: Optional[int] = None, **kwargs):
        """
        Initialize the MAS Code Generation Agent.
        
        Args:
            task_type: 任务类型，"code" 或 "math"
            rollout_idx: Rollout index for tracking
            **kwargs: Additional configuration
        """
        super().__init__()
        self.task_type = task_type.lower()
        self.rollout_idx = rollout_idx
        
        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def reset(self):
        """重置 agent 状态"""
        super().reset()
        self.generated_code = None
        self.generated_code_history = []

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        根据环境数据更新 agent，生成相应的 prompt。
        
        Args:
            turn_idx: 当前 turn 索引
            env_data: 环境数据
        """
        # Save environment data
        self.env_data = env_data

        # Get task description from environment state
        state = getattr(env_data, "state", None)
        task_description = getattr(state, "task_description", "")
        previous_code = getattr(state, "generated_mas_code", None)
        feedback = getattr(state, "code_feedback", None)

        # 根据任务类型选择 prompt 生成方法
        if self.task_type == "code":
            prompt_text = self._generate_code_prompt(
                task_description, previous_code, feedback, turn_idx
            )
        elif self.task_type == "math":
            prompt_text = self._generate_math_prompt(
                task_description, previous_code, feedback, turn_idx
            )
        else:
            logger.warning(f"Unknown task type: {self.task_type}, defaulting to code")
            prompt_text = self._generate_code_prompt(
                task_description, previous_code, feedback, turn_idx
            )

        self.current_prompt = {"text": prompt_text, "image": None}

    def _generate_code_prompt(
        self,
        task_description: str,
        previous_code: Optional[str],
        feedback: Optional[str],
        turn_idx: int
    ) -> str:
        """
        为 code 任务生成 prompt。
        
        Args:
            task_description: 任务描述
            previous_code: 之前生成的代码
            feedback: 反馈信息
            turn_idx: 当前 turn
            
        Returns:
            生成的 prompt 字符串
        """
        if turn_idx == 0:
            # First turn - 使用 prompt_code.py 中的函数
            prompt, _ = get_code_generation_prompt(task_description)
            formatted_prompt = (
                "<|im_start|>system\nYou are an expert in designing Multi-Agent System workflows.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            # Refinement turn - 提供反馈并要求改进
            formatted_prompt = (
                "<|im_start|>system\nYou are an expert in designing Multi-Agent System workflows.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Task: {task_description}\n\n"
                f"Previous code:\n```python\n{previous_code}\n```\n\n"
                f"Feedback: {feedback}\n\n"
                "Please improve the code based on the feedback. "
                "Make sure to:\n"
                "1. Fix any errors or issues mentioned in the feedback\n"
                "2. Improve the workflow design if needed\n"
                "3. Ensure the code is complete and runnable\n\n"
                "Respond with your reasoning in <think> block, then provide the improved code in <code> block.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        return formatted_prompt

    def _generate_math_prompt(
        self,
        task_description: str,
        previous_code: Optional[str],
        feedback: Optional[str],
        turn_idx: int
    ) -> str:
        """
        为 math 任务生成 prompt。
        
        Args:
            task_description: 任务描述
            previous_code: 之前生成的代码
            feedback: 反馈信息
            turn_idx: 当前 turn
            
        Returns:
            生成的 prompt 字符串
        """
        if turn_idx == 0:
            # First turn - 使用 prompt_math.py 中的函数（需要实现）
            # 如果 prompt_math.py 为空，使用默认的 math workflow prompt
            formatted_prompt = (
                "<|im_start|>system\nYou are an expert in designing Multi-Agent System workflows for mathematical problem solving.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Design a multi-agent workflow to solve the following mathematical task:\n\n"
                f"{task_description}\n\n"
                "Your workflow should:\n"
                "1. Break down the problem into steps\n"
                "2. Use appropriate agents for different reasoning stages\n"
                "3. Include verification and checking mechanisms\n"
                "4. Generate complete, runnable Python code\n\n"
                "Respond with your reasoning in <think> block, then provide the code in <code> block.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            # Refinement turn
            formatted_prompt = (
                "<|im_start|>system\nYou are an expert in designing Multi-Agent System workflows for mathematical problem solving.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Task: {task_description}\n\n"
                f"Previous code:\n```python\n{previous_code}\n```\n\n"
                f"Feedback: {feedback}\n\n"
                "Please improve the code based on the feedback. "
                "Ensure the mathematical reasoning is sound and the code is correct.\n\n"
                "Respond with your reasoning in <think> block, then provide the improved code in <code> block.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        return formatted_prompt

    def update_from_model(self, response: str):
        """
        解析模型响应，提取生成的 MAS 代码。
        
        Args:
            response: 模型生成的响应
            
        Returns:
            提取的代码字符串
        """
        # 解析响应，提取 <code> 块中的代码
        code = ""

        # 首先尝试匹配 <code> 标签
        code_match = re.search(r"<code>\s*```python(.*?)```\s*</code>", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # 尝试匹配普通的 ```python 代码块
            matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
            if matches:
                code = matches[-1].strip()
            else:
                # 如果没有代码块，返回错误消息
                code = "# Error: Could not extract code from the model response."
                logger.warning("Failed to extract code from model response")

        # 保存生成的代码
        self.generated_code = code
        self.generated_code_history.append(code)
        
        # Update current action
        self.current_action = code

        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        执行 agent 的 step，将生成的代码保存到环境状态。
        
        Args:
            env_data: 环境数据
            env_worker: 环境 worker（可选）
        """
        # Save generated code to environment state
        generated_code = self.current_action
        env_data.state.generated_mas_code = generated_code
        
        # 将代码添加到历史记录
        if not hasattr(env_data.state, "generated_mas_code_history"):
            env_data.state.generated_mas_code_history = []
        env_data.state.generated_mas_code_history.append(generated_code)

        # 验证代码是否有效（基本语法检查）
        try:
            compile(generated_code, '<string>', 'exec')
            self.success = True
            env_data.state.code_syntax_valid = True
            logger.info("Generated code passed syntax check")
        except SyntaxError as e:
            self.success = False
            env_data.state.code_syntax_valid = False
            env_data.state.code_feedback = f"Syntax Error: {str(e)}"
            logger.error(f"Syntax error in generated code: {e}")
        except Exception as e:
            self.success = False
            env_data.state.code_syntax_valid = False
            env_data.state.code_feedback = f"Error: {str(e)}"
            logger.error(f"Error compiling generated code: {e}")

    def calculate_reward(self, env_data: Env):
        """
        计算奖励。基于代码质量和有效性。
        
        Args:
            env_data: 环境数据
        """
        # 基础奖励：语法是否正确
        if getattr(env_data.state, "code_syntax_valid", False):
            self.agent_reward = 1.0
        else:
            self.agent_reward = 0.0

        # 可以添加更多的奖励维度，例如：
        # - 代码完整性
        # - 是否包含必要的 import
        # - 是否包含 workflow 定义
        # - 代码长度是否合理
        
        code = getattr(env_data.state, "generated_mas_code", "")
        
        # 检查是否包含关键元素
        if "AgentNode" in code:
            self.agent_reward += 0.2
        if "Workflow" in code or "AgentGraph" in code:
            self.agent_reward += 0.2
        if "tool_registry" in code.lower():
            self.agent_reward += 0.1
        if len(code) > 100:  # 代码长度合理
            self.agent_reward += 0.1

        self.agent_reward = min(1.0, self.agent_reward)  # Cap at 1.0
        self.reward_history.append(self.agent_reward)