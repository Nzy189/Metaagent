import asyncio
import sys
from typing import Optional

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pettingllms.multi_agent_env.autogen_graph.code_graph.code_env import CodeEnv, CodeEnvBatch


async def code_graph(env: Optional[CodeEnv] = None, model_client_dict: Optional[OpenAIChatCompletionClient] = None):
    """
    Main function for code generation and testing workflow.
    
    Args:
        env: Optional CodeEnv instance with problem and test cases
        problem: Optional problem description (used if env is None)
    """
    # Get problem from env or use provided problem

    task = env.state.problem
    if not task:
        raise ValueError("Environment provided but no problem found in env.state.problem")
   

    # 1. Docker code executor
    code_executor = DockerCommandLineCodeExecutor(work_dir="code_workdir")
    await code_executor.start()

    # 2. 定义 Coder / Executor / Reviewer
    coder = AssistantAgent(
        "code_coder",
        model_client=model_client_dict["code_coder"],
        system_message=(
            "You are a senior Python engineer. "
            "Given a feature request and review feedback, "
            "write or update code in ```python``` blocks only. "
            "Include minimal comments; make code runnable as-is."
        ),
    )

    executor = CodeExecutorAgent(
        "code_executor",
        code_executor=code_executor,
        sources=["code_coder"],  # 只执行 coder 的 code
    )

    reviewer = AssistantAgent(
        "code_reviewer",
        model_client=model_client_dict["code_reviewer"],
        system_message=(
            "You are a strict code reviewer and tester. "
            "Read the latest code and execution output. "
            "If everything passes, reply with exactly:\n"
            "APPROVE\n"
            "Otherwise, reply with:\n"
            "NEEDS_CHANGE: <one-sentence reason>\n"
            "Optionally propose a short fix suggestion."
        ),
    )

    # 3. GraphFlow：coder -> executor -> reviewer -> (coder or end)
    builder = DiGraphBuilder()
    builder.add_node(coder).add_node(executor).add_node(reviewer)

    builder.add_edge(coder, executor)
    builder.add_edge(executor, reviewer)

    def approved(msg):
        return "APPROVE" in msg.to_model_text()

    def needs_change(msg):
        return "NEEDS_CHANGE" in msg.to_model_text()

    builder.add_edge(reviewer, coder, condition=needs_change)


    graph = builder.build()

    team = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=MaxMessageTermination(20),
    )

    await Console(team.run_stream(task=task))
    await code_executor.stop()
    
    # Return result if env was provided
    if env is not None:
        return env

