from pettingllms.agents.math_agent import MathAgent
from pettingllms.agents.tool_agent import ToolAgent

__all__ = ["MathAgent", "ToolAgent"]


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


# Define all agent imports
AGENT_IMPORTS = [
    ("pettingllms.agents.miniwob_agent", "MiniWobAgent"),
    ("pettingllms.agents.frozenlake_agent", "FrozenLakeAgent"),
    # ("pettingllms.agents.swe_agent", "SWEAgent"),
    ("pettingllms.agents.code_agent", "CompetitionCodingAgent"),
    ("pettingllms.agents.webarena_agent", "WebArenaAgent"),
]

for module_path, class_name in AGENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
