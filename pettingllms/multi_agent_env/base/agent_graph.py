import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def run_simple_agent_graph():
    """
    最简单的 agent_graph 示例：writer -> reviewer
    """
    # 1. 创建 OpenAI 客户端
    client = OpenAIChatCompletionClient(
        model="gpt-4",
        # api_key="your-api-key",  # 如果环境变量中没有设置
    )

    # 2. 创建两个 agent
    writer = AssistantAgent(
        name="writer",
        model_client=client,
        system_message="You are a research paper writer. Write clear, concise paragraphs.",
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=client,
        system_message=(
            "You are a senior reviewer. "
            "Point out weaknesses and then give a revised version."
        ),
    )

    # 3. 用 DiGraphBuilder 搭一个有向图：writer -> reviewer
    builder = DiGraphBuilder()
    builder.add_node(writer).add_node(reviewer)
    builder.add_edge(writer, reviewer)  # 顺序执行

    graph = builder.build()

    # 4. 创建 GraphFlow 团队
    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
    )

    # 5. 运行：用 Console 包一层输出更好看
    task = "Write a related-work paragraph about LLM-based multi-agent RL systems."
    await Console(flow.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(run_simple_agent_graph())
