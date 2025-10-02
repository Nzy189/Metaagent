from pettingllms.tools.code_tools import (
    PythonInterpreter,
)
from pettingllms.tools.registry import ToolRegistry
from pettingllms.tools.web_tools import (
    FirecrawlTool,
    GoogleSearchTool,
    TavilyExtractTool,
    TavilySearchTool,
)

# Define default tools dict
DEFAULT_TOOLS = {
    "python": PythonInterpreter,
    "google_search": GoogleSearchTool,
    "firecrawl": FirecrawlTool,
    "tavily_extract": TavilyExtractTool,
    "tavily_search": TavilySearchTool,
}

# Create the singleton registry instance and register all default tools
tool_registry = ToolRegistry()
tool_registry.register_all(DEFAULT_TOOLS)

__all__ = [
    "PythonInterpreter",
    "LocalRetrievalTool",
    "GoogleSearchTool",
    "FirecrawlTool",
    "TavilyExtractTool",
    "TavilySearchTool",
    "ToolRegistry",
    "tool_registry",
]
