"""
Core OpenAI API Patch Module for PettingLLMs

Patch autogen and langchain LLM engines to use llm_async_generate from async_generate.py
"""

import asyncio
import functools
from typing import Dict, List, Optional, Union
import random
import numpy as np


# Global state
_server_address_dict: Dict[str, Union[str, List[str]]] = {}
_tokenizer_dict: Dict[str, any] = {}
_ppo_trainer_config_dict: Dict[str, any] = {}
_agent_policy_mapping: Dict[str, str] = {}
_current_turn_idx: int = 0
_current_rollout_idx: int = 0
_current_env_idx: int = 0
_patched: bool = False

# Trajectory storage: {(rollout_idx, turn_idx, policy_name): (output_dpr, response)}
_trajectory_store: Dict[tuple, tuple] = {}


def init_patch_context(
    server_address_dict: Dict[str, Union[str, List[str]]],
    tokenizer_dict: Dict[str, any],
    ppo_trainer_config_dict: Dict[str, any],
    agent_policy_mapping: Dict[str, str]
):
    """
    Initialize patch context with engine attributes.
    
    Args:
        server_address_dict: {policy_name: address or [addresses]}
        tokenizer_dict: {policy_name: tokenizer}
        ppo_trainer_config_dict: {policy_name: ppo_config}
        agent_policy_mapping: {agent_name: policy_name}
    """
    global _server_address_dict, _tokenizer_dict, _ppo_trainer_config_dict, _agent_policy_mapping
    _server_address_dict = server_address_dict
    _tokenizer_dict = tokenizer_dict
    _ppo_trainer_config_dict = ppo_trainer_config_dict
    _agent_policy_mapping = agent_policy_mapping
    print(f"[Patch] Initialized context with policies: {list(server_address_dict.keys())}")


def set_rollout_context(rollout_idx: int, env_idx: int):
    """
    Set current rollout and env indices for trajectory collection.
    
    Args:
        rollout_idx: Current rollout index
        env_idx: Current environment index
    """
    global _current_rollout_idx, _current_env_idx
    _current_rollout_idx = rollout_idx
    _current_env_idx = env_idx
    print(f"[Patch] Set rollout context: rollout_idx={rollout_idx}, env_idx={env_idx}")


def get_rollout_idx() -> int:
    """Get current rollout index."""
    return _current_rollout_idx


def get_env_idx() -> int:
    """Get current environment index."""
    return _current_env_idx


def get_turn_idx() -> int:
    """Get current turn index (node number in agent graph flow)."""
    return _current_turn_idx


def increment_turn_idx():
    """Increment turn index when transitioning to next node."""
    global _current_turn_idx
    _current_turn_idx += 1
    print(f"[Patch] Turn index incremented to {_current_turn_idx}")


def reset_turn_idx():
    """Reset turn index to 0 for new graph execution."""
    global _current_turn_idx, _trajectory_store
    _current_turn_idx = 0
    _trajectory_store = {}  # Clear trajectory store for new rollout
    print(f"[Patch] Turn index reset to 0, trajectory store cleared")


def clear_trajectory_store():
    """Clear the trajectory store."""
    global _trajectory_store
    _trajectory_store = {}
    print(f"[Patch] Trajectory store cleared")


def get_trajectory_store() -> Dict[tuple, tuple]:
    """
    Get collected trajectories for current rollout.
    
    Returns:
        Dict mapping (rollout_idx, turn_idx, policy_name) to (output_dpr, response)
    """
    return _trajectory_store.copy()


def get_server_address(policy_name: str) -> str:
    """Get vLLM server address for policy."""
    addresses = _server_address_dict[policy_name]
    if isinstance(addresses, list):
        return random.choice(addresses)
    return addresses


async def _patched_generate(
    messages: List[Dict[str, str]],
    policy_name: str,
    model_name: str,
    agent_name: Optional[str] = None,
    **kwargs
) -> str:
    """
    Core patched generate function that calls llm_async_generate.
    
    Args:
        messages: Chat messages
        policy_name: Policy name for server/tokenizer lookup
        model_name: Model name
        agent_name: Agent name for reward attribution
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text response
    """
    from pettingllms.trainer.async_generate import llm_async_generate, convert_prompt_to_dpr
    
    # Get context
    address = get_server_address(policy_name)
    tokenizer = _tokenizer_dict[policy_name]
    ppo_config = _ppo_trainer_config_dict[policy_name]
    turn_idx = get_turn_idx()
    rollout_idx = get_rollout_idx()
    env_idx = get_env_idx()
    
    # Convert messages to prompt
    if isinstance(messages, list) and len(messages) > 0:
        if isinstance(messages[0], dict):
            # Chat format
            prompt_text = messages[-1].get('content', '')
        else:
            prompt_text = str(messages[-1])
    else:
        prompt_text = str(messages)
    
    # Create prompt DataProto
    prompt_dpr = convert_prompt_to_dpr(
        tokenizer=tokenizer,
        processor=None,
        prompts={"text": prompt_text, "image": None},
        max_prompt_length=ppo_config.data.max_prompt_length,
        multi_modal=False,
        enable_thinking=False
    )
    
    # Call llm_async_generate
    output_dpr, response = await llm_async_generate(
        rollout_idx=rollout_idx,
        turn_idx=turn_idx,
        agent_idx=0,
        prompt_dpr=prompt_dpr,
        ppo_trainer_config=ppo_config,
        address=address,
        model_name=model_name,
        tokenizer=tokenizer,
        enable_thinking=False,
        image_data=None,
        application_id=f"autogen_graph_r{rollout_idx}_t{turn_idx}",
        env_idx=env_idx,
        policy_name=policy_name,
        timeout=kwargs.get('timeout', 60.0),
        mode=kwargs.get('mode', 'inference'),
        lora_id=None,
        agent_config=None,
    )
    
    # Store trajectory with agent_name for later reward attribution
    # Note: reward will be added later by the environment step
    if output_dpr is not None:
        # Add agent_name to output_dpr for trajectory collection
        output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name or "unknown"], dtype=object)
        output_dpr.non_tensor_batch["turn_idx"] = np.array([turn_idx], dtype=np.int32)
        
        # Store in trajectory store
        global _trajectory_store
        key = (rollout_idx, turn_idx, policy_name)
        _trajectory_store[key] = (output_dpr, response)
        print(f"[Patch] Stored trajectory for key={key}")
    
    # Increment turn index after generation
    increment_turn_idx()
    
    return response


def patch_autogen():
    """
    Patch autogen's OpenAIChatCompletionClient to use llm_async_generate.
    """
    global _patched
    if _patched:
        return
    
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    
    original_create = OpenAIChatCompletionClient.create
    
    @functools.wraps(original_create)
    async def patched_create(self, messages, **kwargs):
        # Get policy name from model or use first available
        model_name = kwargs.get('model') or self._model_id
        policy_name = model_name  # model_name is set to policy_name in generate_single_rollout
        
        # Try to infer agent_name from model_name or policy mapping
        agent_name = None
        for a_name, p_name in _agent_policy_mapping.items():
            if p_name == policy_name:
                agent_name = a_name
                break
        
        # Call patched generate
        response_text = await _patched_generate(
            messages=messages,
            policy_name=policy_name,
            model_name=model_name,
            agent_name=agent_name,
            **kwargs
        )
        
        # Return in autogen format
        from autogen_core.models import CreateResult, LLMMessage
        return CreateResult(
            content=response_text,
            usage={},
            finish_reason="stop",
            cached=False,
        )
    
    OpenAIChatCompletionClient.create = patched_create
    _patched = True
    print("[Patch] Patched autogen OpenAIChatCompletionClient.create")


def patch_langchain():
    """
    Patch langchain's ChatOpenAI to use llm_async_generate.
    """
    from langchain_openai import ChatOpenAI
    
    original_generate = ChatOpenAI._generate
    
    @functools.wraps(original_generate)
    async def patched_generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Convert langchain messages
        msg_dicts = [{"role": m.type, "content": m.content} for m in messages]
        
        # Get policy
        policy_name = list(_server_address_dict.keys())[0]
        model_name = self.model_name
        
        # Call patched generate
        response_text = await _patched_generate(
            messages=msg_dicts,
            policy_name=policy_name,
            model_name=model_name,
            **kwargs
        )
        
        # Return in langchain format
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    ChatOpenAI._generate = patched_generate
    print("[Patch] Patched langchain ChatOpenAI._generate")


def patch_all():
    """Apply all patches (autogen and langchain)."""
    patch_autogen()
    print("[Patch] All patches applied")


def wrap_autogen_graph(graph_callable):
    """
    Wrap an autogen graph callable to track node transitions.
    
    Args:
        graph_callable: The graph's main() function or callable
        
    Returns:
        Wrapped callable with turn_idx tracking
    """
    @functools.wraps(graph_callable)
    async def wrapped_graph(*args, **kwargs):
        reset_turn_idx()
        print(f"[Patch] Starting autogen graph execution")
        result = await graph_callable(*args, **kwargs)
        print(f"[Patch] Graph completed after {get_turn_idx()} turns")
        return result
    
    return wrapped_graph
