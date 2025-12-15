"""
Core algorithms for multi-agent RL training.

This module contains core algorithms for reward calculation and other 
training-related computations that need to be easily debuggable and modifiable.
"""

import numpy as np
try:
    from verl.protocol import DataProto
except Exception:
    from verl import DataProto


def calculate_reward(data_proto: DataProto, algorithm: str = "default", **kwargs) -> DataProto:
    """
    Calculate rewards for trajectories based on the specified algorithm.

    This function is the core reward calculation logic that can be easily
    debugged and modified. It takes trajectories with env_final_reward marked
    and assigns final rewards based on the selected algorithm.

    Args:
        data_proto: DataProto containing trajectory data with env_final_reward in non_tensor_batch.
                   Expected fields in non_tensor_batch:
                   - env_final_reward: Final reward from environment (required)
                   - hop_idx: Hop index for hop-based weighting (optional, for graph-based systems)
                   - turn_idx: Turn index for turn-based weighting (optional, for turn-based systems)
                   - agent_name: Agent name for debugging (optional)
        algorithm: Reward calculation algorithm to use. Options:
            - "default": Directly assign env_final_reward to reward
            - "hop_weighted": Weight rewards by hop (later hops get higher rewards)
            - "hop_discount": Apply discount factor to rewards based on hop
            - "turn_weighted": Weight rewards by turn (later turns get higher rewards)
            - "discount": Apply discount factor to rewards based on turn
            - "binary": Binary rewards based on threshold
            - "normalized": Normalize rewards to [-1, 1] range
        **kwargs: Additional algorithm-specific parameters:
            - max_hops: Maximum number of hops (for hop_weighted, hop_discount)
            - max_turns: Maximum number of turns (for turn_weighted, discount)
            - discount_factor: Discount factor (for discount/hop_discount, default: 0.99)
            - reward_threshold: Threshold for binary rewards (for binary, default: 0.5)
    
    Returns:
        DataProto: Same DataProto with "reward" field added to non_tensor_batch
        
    Raises:
        ValueError: If env_final_reward not found or algorithm is unknown
        
    Examples:
        >>> # Default algorithm
        >>> output_dpr = calculate_reward(output_dpr, algorithm="default")
        
        >>> # Turn-weighted algorithm
        >>> output_dpr = calculate_reward(
        ...     output_dpr, 
        ...     algorithm="turn_weighted",
        ...     max_turns=8
        ... )
        
        >>> # Discount algorithm
        >>> output_dpr = calculate_reward(
        ...     output_dpr,
        ...     algorithm="discount",
        ...     max_turns=8,
        ...     discount_factor=0.95
        ... )
    """
    if "env_final_reward" not in data_proto.non_tensor_batch:
        raise ValueError(
            "env_final_reward not found in non_tensor_batch. "
            "Make sure to mark env_final_reward before calling calculate_reward."
        )
    
    env_final_reward = data_proto.non_tensor_batch["env_final_reward"]
    
    # ==================== Algorithm: default ====================
    if algorithm == "default":
        """
        Default algorithm: directly assign env_final_reward to reward.
        This is the simplest approach where each trajectory gets the final
        environment reward without any modifications.
        """
        data_proto.non_tensor_batch["reward"] = env_final_reward.copy()

    # ==================== Algorithm: hop_weighted ====================
    elif algorithm == "hop_weighted":
        """
        Hop-weighted algorithm: weight rewards by hop index.
        Later hops receive higher weights, encouraging the model to focus
        on actions that are closer to the final outcome.
        This is designed for graph-based multi-agent systems where each
        LLM request is counted as one "hop".

        Formula: reward = env_final_reward * (hop_idx + 1) / max_hops
        """
        hop_idx = data_proto.non_tensor_batch.get("hop_idx", np.zeros_like(env_final_reward))
        max_hops = kwargs.get("max_hops", 10)

        # Linear weighting: weight increases linearly with hop index
        weights = (hop_idx + 1) / max_hops
        # Clip weights to [0, 1] in case hop_idx exceeds max_hops
        weights = np.clip(weights, 0.0, 1.0)
        data_proto.non_tensor_batch["reward"] = env_final_reward * weights

        # Debug info
        print(f"[RewardCalc] hop_weighted: hop_idx={hop_idx[0] if len(hop_idx) > 0 else 'N/A'}, "
              f"weight={weights[0] if len(weights) > 0 else 'N/A'}, "
              f"reward={data_proto.non_tensor_batch['reward'][0] if len(data_proto.non_tensor_batch['reward']) > 0 else 'N/A'}")

    # ==================== Algorithm: hop_discount ====================
    elif algorithm == "hop_discount":
        """
        Hop-discount algorithm: apply discount factor based on hop.
        Earlier hops are discounted more heavily, similar to temporal
        difference learning in RL.
        This is designed for graph-based multi-agent systems.

        Formula: reward = env_final_reward * discount_factor^(max_hops - hop_idx)
        """
        hop_idx = data_proto.non_tensor_batch.get("hop_idx", np.zeros_like(env_final_reward))
        max_hops = kwargs.get("max_hops", 10)
        discount_factor = kwargs.get("discount_factor", 0.99)

        # Discount from final hop backwards
        discount = np.power(discount_factor, np.maximum(0, max_hops - hop_idx))
        data_proto.non_tensor_batch["reward"] = env_final_reward * discount

        # Debug info
        print(f"[RewardCalc] hop_discount: hop_idx={hop_idx[0] if len(hop_idx) > 0 else 'N/A'}, "
              f"discount={discount[0] if len(discount) > 0 else 'N/A'}, "
              f"reward={data_proto.non_tensor_batch['reward'][0] if len(data_proto.non_tensor_batch['reward']) > 0 else 'N/A'}")

    # ==================== Algorithm: turn_weighted ====================
    elif algorithm == "turn_weighted":
        """
        Turn-weighted algorithm: weight rewards by turn index.
        Later turns receive higher weights, encouraging the model to focus
        on actions that are closer to the final outcome.
        
        Formula: reward = env_final_reward * (turn_idx + 1) / max_turns
        """
        turn_idx = data_proto.non_tensor_batch.get("turn_idx", np.zeros_like(env_final_reward))
        max_turns = kwargs.get("max_turns", 8)
        
        # Linear weighting: weight increases linearly with turn index
        weights = (turn_idx + 1) / max_turns
        data_proto.non_tensor_batch["reward"] = env_final_reward * weights
        
        # Debug info
        print(f"[RewardCalc] turn_weighted: turn_idx={turn_idx[0] if len(turn_idx) > 0 else 'N/A'}, "
              f"weight={weights[0] if len(weights) > 0 else 'N/A'}, "
              f"reward={data_proto.non_tensor_batch['reward'][0] if len(data_proto.non_tensor_batch['reward']) > 0 else 'N/A'}")
    
    # ==================== Algorithm: discount ====================
    elif algorithm == "discount":
        """
        Discount algorithm: apply discount factor based on turn.
        Earlier turns are discounted more heavily, similar to temporal
        difference learning in RL.
        
        Formula: reward = env_final_reward * discount_factor^(max_turns - turn_idx)
        """
        turn_idx = data_proto.non_tensor_batch.get("turn_idx", np.zeros_like(env_final_reward))
        max_turns = kwargs.get("max_turns", 8)
        discount_factor = kwargs.get("discount_factor", 0.99)
        
        # Discount from final turn backwards
        discount = np.power(discount_factor, max_turns - turn_idx)
        data_proto.non_tensor_batch["reward"] = env_final_reward * discount
        
        # Debug info
        print(f"[RewardCalc] discount: turn_idx={turn_idx[0] if len(turn_idx) > 0 else 'N/A'}, "
              f"discount={discount[0] if len(discount) > 0 else 'N/A'}, "
              f"reward={data_proto.non_tensor_batch['reward'][0] if len(data_proto.non_tensor_batch['reward']) > 0 else 'N/A'}")
    
    # ==================== Algorithm: binary ====================
    elif algorithm == "binary":
        """
        Binary algorithm: convert rewards to binary values.
        Rewards above threshold become 1.0, others become 0.0.
        Useful for sparse reward environments.
        
        Formula: reward = 1.0 if env_final_reward > threshold else 0.0
        """
        threshold = kwargs.get("reward_threshold", 0.5)
        
        data_proto.non_tensor_batch["reward"] = np.where(
            env_final_reward > threshold,
            1.0,
            0.0
        )
        
        # Debug info
        success_rate = np.mean(data_proto.non_tensor_batch["reward"])
        print(f"[RewardCalc] binary: threshold={threshold}, "
              f"success_rate={success_rate:.2%}")
    
    # ==================== Algorithm: normalized ====================
    elif algorithm == "normalized":
        """
        Normalized algorithm: clip rewards to [-1, 1] range.
        Prevents extreme reward values from dominating training.
        
        Formula: reward = clip(env_final_reward, -1.0, 1.0)
        """
        data_proto.non_tensor_batch["reward"] = np.clip(env_final_reward, -1.0, 1.0)
        
        # Debug info
        print(f"[RewardCalc] normalized: mean_reward={np.mean(data_proto.non_tensor_batch['reward']):.4f}, "
              f"std_reward={np.std(data_proto.non_tensor_batch['reward']):.4f}")
    
    # ==================== Algorithm: shaped ====================
    elif algorithm == "shaped":
        """
        Shaped algorithm: combine multiple reward shaping techniques.
        This can include turn weighting + normalization, or other combinations.
        
        Example: Apply turn weighting first, then normalize.
        """
        turn_idx = data_proto.non_tensor_batch.get("turn_idx", np.zeros_like(env_final_reward))
        max_turns = kwargs.get("max_turns", 8)
        
        # First apply turn weighting
        weights = (turn_idx + 1) / max_turns
        shaped_reward = env_final_reward * weights
        
        # Then normalize to [-1, 1]
        shaped_reward = np.clip(shaped_reward, -1.0, 1.0)
        
        data_proto.non_tensor_batch["reward"] = shaped_reward
        
        # Debug info
        print(f"[RewardCalc] shaped: mean_reward={np.mean(shaped_reward):.4f}")
    
    # ==================== Unknown algorithm ====================
    else:
        raise ValueError(
            f"Unknown reward algorithm: {algorithm}. "
            f"Supported algorithms: default, hop_weighted, hop_discount, turn_weighted, discount, binary, normalized, shaped"
        )
    
    return data_proto

