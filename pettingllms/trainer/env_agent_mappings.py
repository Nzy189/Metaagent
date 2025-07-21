def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


# Import environment classes
ENV_CLASSES = {
    "browsergym": safe_import("pettingllms.environments.browsergym.browsergym", "BrowserGymEnv"),
    "frozenlake": safe_import("pettingllms.environments.frozenlake.frozenlake", "FrozenLakeEnv"),
    "tool": safe_import("pettingllms.environments.tools.tool_env", "ToolEnvironment"),
    "math": safe_import("pettingllms.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "code": safe_import("pettingllms.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "swe": safe_import("pettingllms.environments.swe.swe", "SWEEnv"),
    "competition_coding": safe_import("pettingllms.environments.code.competition_coding", "CompetitionCodingEnv"),
}

# Import agent classes
AGENT_CLASSES = {
    "miniwobagent": safe_import("pettingllms.agents.miniwob_agent", "MiniWobAgent"),
    "frozenlakeagent": safe_import("pettingllms.agents.frozenlake_agent", "FrozenLakeAgent"),
    "tool_agent": safe_import("pettingllms.agents.tool_agent", "ToolAgent"),
    "sweagent": safe_import("pettingllms.agents.swe_agent", "SWEAgent"),
    "math_agent": safe_import("pettingllms.agents.math_agent", "MathAgent"),
    "code_agent": safe_import("pettingllms.agents.code_agent", "CompetitionCodingAgent"),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
