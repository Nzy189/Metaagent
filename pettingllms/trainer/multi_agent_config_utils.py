"""
Utilities for handling multi-agent configuration parsing and merging.
"""

from copy import deepcopy
from omegaconf import OmegaConf, DictConfig
from typing import List, Dict, Any


def parse_multi_agent_config(config: DictConfig) -> List[DictConfig]:
    """
    Parse the multi-agent configuration and create individual configs for each agent.
    
    Args:
        config: The main configuration containing agents section
        
    Returns:
        List of configurations, one for each agent
    """
    # Check if multi-agent configuration exists
    if not hasattr(config, 'agents') or config.agents is None:
        # Single agent mode - return the original config
        return [config]
    
    agents_config = config.agents
    num_agents = agents_config.num_agents
    default_config = agents_config.default_config
    agent_configs = agents_config.get('agent_configs', {})
    
    agent_config_list = []
    
    for i in range(num_agents):
        # Start with the base config (everything except the agents section)
        agent_config = deepcopy(config)
        
        # Remove the agents section from the individual agent config to avoid confusion
        if hasattr(agent_config, 'agents'):
            delattr(agent_config, 'agents')
        
        # Apply default agent configuration
        agent_config = merge_agent_config(agent_config, default_config, f"agent_{i}")
        
        # Apply specific agent configuration if exists
        agent_key = f"agent_{i}"
        if agent_key in agent_configs:
            specific_config = agent_configs[agent_key]
            agent_config = merge_agent_specific_config(agent_config, specific_config)
        
        agent_config_list.append(agent_config)
    
    return agent_config_list


def merge_agent_config(base_config: DictConfig, agent_default_config: DictConfig, agent_id: str) -> DictConfig:
    """
    Merge the default agent configuration into the base configuration.
    
    Args:
        base_config: The base configuration
        agent_default_config: Default configuration for agents
        agent_id: Identifier for this specific agent
        
    Returns:
        Merged configuration
    """
    # Update actor_rollout_ref with agent default config
    if hasattr(agent_default_config, 'model'):
        base_config.actor_rollout_ref.model.update(agent_default_config.model)
    
    if hasattr(agent_default_config, 'actor'):
        base_config.actor_rollout_ref.actor.update(agent_default_config.actor)
    
    if hasattr(agent_default_config, 'rollout'):
        base_config.actor_rollout_ref.rollout.update(agent_default_config.rollout)
    
    if hasattr(agent_default_config, 'ref'):
        base_config.actor_rollout_ref.ref.update(agent_default_config.ref)
    
    # Update agent configuration
    if hasattr(agent_default_config, 'agent'):
        base_config.agent.update(agent_default_config.agent)
        
        # Set default agent name if not specified
        if not hasattr(base_config.agent, 'name') or base_config.agent.name is None:
            base_config.agent.name = f"agent_{agent_id}"
    
    return base_config


def merge_agent_specific_config(base_config: DictConfig, specific_config: DictConfig) -> DictConfig:
    """
    Merge agent-specific configuration overrides.
    
    Args:
        base_config: The base configuration with defaults applied
        specific_config: Agent-specific configuration overrides
        
    Returns:
        Configuration with specific overrides applied
    """
    # Update agent name if specified
    if hasattr(specific_config, 'name'):
        base_config.agent.name = specific_config.name
    
    # Update model configuration if specified
    if hasattr(specific_config, 'model'):
        base_config.actor_rollout_ref.model.update(specific_config.model)
    
    # Update actor configuration if specified
    if hasattr(specific_config, 'actor'):
        merge_nested_config(base_config.actor_rollout_ref.actor, specific_config.actor)
    
    # Update rollout configuration if specified
    if hasattr(specific_config, 'rollout'):
        merge_nested_config(base_config.actor_rollout_ref.rollout, specific_config.rollout)
    
    # Update ref configuration if specified
    if hasattr(specific_config, 'ref'):
        merge_nested_config(base_config.actor_rollout_ref.ref, specific_config.ref)
    
    # Update agent configuration if specified
    if hasattr(specific_config, 'agent'):
        merge_nested_config(base_config.agent, specific_config.agent)
    
    return base_config


def merge_nested_config(base_dict: DictConfig, override_dict: DictConfig) -> None:
    """
    Recursively merge nested configuration dictionaries.
    
    Args:
        base_dict: Base configuration dictionary to be updated
        override_dict: Override configuration dictionary
    """
    for key, value in override_dict.items():
        if key in base_dict and isinstance(base_dict[key], DictConfig) and isinstance(value, DictConfig):
            # Recursively merge nested dictionaries
            merge_nested_config(base_dict[key], value)
        else:
            # Direct assignment for non-dict values or new keys
            base_dict[key] = value


def get_agent_configs_summary(agent_configs: List[DictConfig]) -> Dict[str, Any]:
    """
    Generate a summary of agent configurations for logging.
    
    Args:
        agent_configs: List of agent configurations
        
    Returns:
        Summary dictionary
    """
    summary = {
        "num_agents": len(agent_configs),
        "agents": []
    }
    
    for i, config in enumerate(agent_configs):
        agent_summary = {
            "agent_id": i,
            "agent_name": config.agent.get('name', f'agent_{i}'),
            "model_path": config.actor_rollout_ref.model.path,
            "learning_rate": config.actor_rollout_ref.actor.optim.lr
        }
        summary["agents"].append(agent_summary)
    
    return summary 