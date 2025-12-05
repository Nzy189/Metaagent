"""
Modern, robust workflow system for multi-agent orchestration.

Quick Start:
-----------
```python
from workflow import AgentNode, Workflow, ToolRegistry

# Setup tools
tools = ToolRegistry()
tools.register("my_tool", my_function, description="...", parameters={...})

# Create agent
agent = AgentNode(name="Agent", system_prompt="...", tool_registry=tools)

# Create and run workflow
workflow = Workflow()
workflow.add_node(agent)
result = workflow.run("Your question here")
```

Main Components:
---------------
- Core: Message, Context, WorkflowNode, ToolRegistry
- Nodes: AgentNode, EnsembleNode, DebateNode, ReflectionNode, RouterNode
- Workflows: Workflow, ConditionalWorkflow, LoopWorkflow
- Compat: CompatBaseAgent, CompatBaseWorkflow (for migration)

See README.md for detailed documentation.
"""

from .core import (
    Message,
    MessageType,
    Context,
    WorkflowNode,
    ToolRegistry
)

# Import workflow first as it doesn't depend on nodes
from .workflow import (
    Workflow,
    ConditionalWorkflow,
    LoopWorkflow
)

# Import graph for flexible agent interactions
from .graph import (
    AgentGraph,
    create_simple_chain
)

# Lazy import nodes to avoid import errors if dependencies are missing
def _import_nodes():
    """Lazy import nodes to handle missing dependencies gracefully."""
    try:
        from .nodes import (
            AgentNode,
            EnsembleNode,
            DebateNode,
            ReflectionNode,
            RouterNode
        )
        return AgentNode, EnsembleNode, DebateNode, ReflectionNode, RouterNode
    except ImportError as e:
        # Return None if dependencies are missing
        return None, None, None, None, None

# Try to import nodes
AgentNode, EnsembleNode, DebateNode, ReflectionNode, RouterNode = _import_nodes()

__version__ = "1.0.0"

__all__ = [
    # Core
    'Message',
    'MessageType',
    'Context',
    'WorkflowNode',
    'ToolRegistry',
    
    # Nodes
    'AgentNode',
    'EnsembleNode',
    'DebateNode',
    'ReflectionNode',
    'RouterNode',
    
    # Workflows (Linear)
    'Workflow',
    'ConditionalWorkflow',
    'LoopWorkflow',
    
    # Graph (Flexible)
    'AgentGraph',
    'create_simple_chain',
]

