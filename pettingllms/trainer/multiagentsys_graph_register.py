"""
Registration module for graph-based multi-agent systems.
Imports classes from pettingllms/mas_graph/
"""

def safe_import(module_path, class_name):
    """Safely import a class from a module, returning None if import fails."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


# ============================================================================
# Environment Classes (Graph-based)
# ============================================================================
ENV_CLASSES = {
    "code_env": safe_import("pettingllms.mas_graph.code_graph.code_env", "CodeEnv"),
    "math_env": safe_import("pettingllms.mas_graph.math_graph.math_env", "MathEnv"),
}

ENV_BATCH_CLASSES = {
    "code_env": safe_import("pettingllms.mas_graph.code_graph.code_env", "CodeEnvBatch"),
    "math_env": safe_import("pettingllms.mas_graph.math_graph.math_env", "MathEnvBatch"),
}



# ============================================================================
# Workflow Functions (Graph-based)
# These are the main entry points for graph-based workflows
# ============================================================================
AGENT_WORKER_FLOW_FUNCTIONS = {
    # Import from the multi_agent_env.autogen_graph for backward compatibility
    "code_graph": safe_import("pettingllms.mas_graph.code_graph.code_graph", "code_graph"),
    "math_graph": safe_import("pettingllms.mas_graph.math_graph.math_graph", "math_graph"),
    # Also support old import paths if needed
    "code_graph_legacy": safe_import("pettingllms.multi_agent_env.autogen_graph.code_graph.code_graph", "code_graph"),
    "math_graph_legacy": safe_import("pettingllms.multi_agent_env.autogen_graph.math_graph.math_graph", "math_graph"),
}

# ============================================================================
# Filter out None values and create final mappings
# ============================================================================
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
ENV_BATCH_CLASS_MAPPING = {k: v for k, v in ENV_BATCH_CLASSES.items() if v is not None}
AGENT_WORKER_FLOW_FUNCTIONS_MAPPING = {k: v for k, v in AGENT_WORKER_FLOW_FUNCTIONS.items() if v is not None}
