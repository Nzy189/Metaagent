import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main(env=None, model_client_dict=None):
    """
    Web research workflow using autogen graph.

    Args:
        env: Environment instance (optional, for graph-based execution)
        model_client_dict: Dictionary of model clients for each agent {agent_name: client}
    """
    # Use provided model_client or create default
    # In graph-based execution, model_client_dict will be provided by the engine
    if model_client_dict and 'web_planner' in model_client_dict:
        model_client = model_client_dict['web_planner']
    else:
        # Fallback for standalone execution
        model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

    # 2. 定义三个 agent
    planner = AssistantAgent(
        "web_planner",
        model_client=model_client,
        system_message=(
            "You are a research planner. Given a user query, "
            "decide which sites to search, which pages to open, "
            "and what information to extract. "
            "Output concise instructions for a web browsing agent."
        ),
    )

    web_surfer = MultimodalWebSurfer(
        name="web_surfer",
        model_client=model_client,
        headless=True,  # 调试时可以设为 False 看浏览器
    )

    summarizer = AssistantAgent(
        "web_summarizer",
        model_client=model_client,
        system_message=(
            "You read the browsing transcript and page texts from the web_surfer agent. "
            "Synthesize a concise, well-structured answer to the original user query. "
            "Use bullet points and cite sources by domain name (e.g., arxiv.org, microsoft.com)."
        ),
    )

    # 3. GraphFlow：planner -> web_surfer -> summarizer
    builder = DiGraphBuilder()
    builder.add_node(planner).add_node(web_surfer).add_node(summarizer)
    builder.add_edge(planner, web_surfer)
    builder.add_edge(web_surfer, summarizer)
    graph = builder.build()

    team = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=MaxMessageTermination(12),
    )

    task = (
        "Find 2–3 recent papers or blog posts (2024–2025) that survey or benchmark "
        "LLM-based multi-agent systems (AutoGen, LangGraph, CrewAI, etc.). "
        "Summarize key comparisons in <=8 bullet points."
    )

    stream = team.run_stream(task=task)
    await Console(stream)

    # 结束时记得关浏览器
    await web_surfer.close()


if __name__ == "__main__":
    asyncio.run(main())
