# AT-GRPO Algorithm

**AT-GRPO** (Agent- and Turn-wise Group Relative Policy Optimization) extends the GRPO algorithm with multi-agent support.

## Background: GRPO

Group Relative Policy Optimization (GRPO) is an on-policy RL algorithm that:

- Samples multiple rollouts per prompt
- Computes advantages relative to the group
- Updates policy using PPO-style objectives

GRPO is effective for single-agent LLM training but doesn't handle multi-agent scenarios.

## AT-GRPO: Multi-Agent Extension

AT-GRPO extends GRPO with two key innovations:

### 1. Agent-wise Grouping

**Problem**: Different agents have different roles and should learn differently.

**Solution**: Group rollouts by agent role before computing advantages.

```python
# Standard GRPO
advantages = rewards - rewards.mean()

# AT-GRPO with agent-wise grouping
for agent_role in agent_roles:
    agent_rewards = rewards[agent_role]
    agent_advantages = agent_rewards - agent_rewards.mean()
```

**Benefits**:
- Each agent learns relative to its own role
- Prevents interference between agent types
- Enables role specialization

### 2. Turn-wise Grouping

**Problem**: Multi-turn conversations have temporal dependencies.

**Solution**: Group rollouts by conversation turn for temporal credit assignment.

```python
# AT-GRPO with turn-wise grouping
for turn_idx in range(num_turns):
    turn_rewards = rewards[:, turn_idx]
    turn_advantages = turn_rewards - turn_rewards.mean()
```

**Benefits**:
- Proper credit assignment across turns
- Learns turn-specific strategies
- Handles long-horizon tasks

### 3. Combined Grouping

AT-GRPO combines both groupings:

```python
# Agent- and Turn-wise Grouping
for agent_role in agent_roles:
    for turn_idx in range(num_turns):
        group_indices = get_group(agent_role, turn_idx)
        group_rewards = rewards[group_indices]
        group_advantages = group_rewards - group_rewards.mean()
        advantages[group_indices] = group_advantages
```

## Tree-Structured Sampling

AT-GRPO uses tree-structured sampling for efficient exploration:

```
                    Root Prompt
                        |
        +---------------+---------------+
        |               |               |
    Rollout 1      Rollout 2      Rollout 3
        |               |               |
    Turn 1 (shared prefix)
        |
    +---+---+---+
    |   |   |   |
   A1  A2  A3  A4  (Agent actions)
    |
    Turn 2
    ...
```

**Benefits**:
- Explores multiple paths from shared states
- More efficient than independent sampling
- Natural variance for advantage estimation

## Mixed Reward Structure

AT-GRPO combines global and local rewards:

### Global Rewards

Based on overall task success:

```python
# Example: Code task
global_reward = test_pass_rate
```

All agents receive the same global reward to encourage coordination.

### Local Rewards

Based on individual agent contributions:

```python
# Example: Code task
tester_local_reward = test_quality_score
coder_local_reward = code_correctness_score
```

Each agent receives role-specific local rewards for specialization.

### Combined Reward

```python
final_reward = alpha * global_reward + (1 - alpha) * local_reward
```

The mixing coefficient `alpha` balances coordination vs. specialization.

## Algorithm Pseudocode

```python
def AT_GRPO(env, policies, num_iterations):
    for iteration in range(num_iterations):
        # 1. Collect rollouts
        rollouts = []
        for prompt in batch:
            # Tree-structured sampling
            tree_rollouts = sample_tree(env, policies, prompt)
            rollouts.extend(tree_rollouts)
        
        # 2. Compute rewards
        for rollout in rollouts:
            rollout.global_reward = compute_global_reward(rollout)
            rollout.local_rewards = compute_local_rewards(rollout)
            rollout.reward = combine_rewards(
                rollout.global_reward, 
                rollout.local_rewards
            )
        
        # 3. Group rollouts and compute advantages
        for agent_role in agent_roles:
            for turn_idx in range(max_turns):
                # Get rollouts for this group
                group = get_group(rollouts, agent_role, turn_idx)
                
                # Compute advantages
                group_rewards = [r.reward for r in group]
                group_mean = np.mean(group_rewards)
                
                for rollout in group:
                    rollout.advantage = rollout.reward - group_mean
        
        # 4. Update policies
        for policy in policies:
            # Get data for this policy
            policy_data = filter_by_policy(rollouts, policy)
            
            # PPO update
            for epoch in range(ppo_epochs):
                for batch in create_batches(policy_data):
                    # Compute policy loss
                    ratio = policy(batch) / old_policy(batch)
                    clipped_ratio = clip(ratio, 1-eps, 1+eps)
                    loss = -min(
                        ratio * batch.advantage,
                        clipped_ratio * batch.advantage
                    )
                    
                    # Update
                    loss.backward()
                    optimizer.step()
```

## Implementation Details

### Advantage Normalization

```python
# Normalize advantages within each group
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### Value Function

AT-GRPO can optionally use a value function:

```python
# With value function
advantages = rewards + gamma * values_next - values

# Without value function (default)
advantages = rewards - rewards.mean()
```

### Clipping

PPO-style clipping prevents large policy updates:

```python
epsilon = 0.2  # Clipping parameter
clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
```

## Hyperparameters

Key hyperparameters for AT-GRPO:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.7 | Global/local reward mixing |
| `epsilon` | 0.2 | PPO clipping parameter |
| `ppo_epochs` | 3 | Training epochs per batch |
| `batch_size` | 128 | Batch size for training |
| `gamma` | 0.99 | Discount factor |
| `lr` | 1e-5 | Learning rate |
| `tree_width` | 4 | Rollouts per tree node |

## Comparison with Other Algorithms

| Algorithm | Multi-Agent | Turn-wise | Tree Sampling |
|-----------|-------------|-----------|---------------|
| PPO | ❌ | ❌ | ❌ |
| GRPO | ❌ | ❌ | ✅ |
| AT-GRPO | ✅ | ✅ | ✅ |

## Results

AT-GRPO significantly outperforms single-agent GRPO:

### Planning Tasks (Plan-Path)
- Single GRPO: 11% → **47%**
- AT-GRPO: 11% → **96-97%**

### Code Tasks (LiveCodeBench, Qwen3-8B)
- Single GRPO: 22.8% → **25.7%**
- AT-GRPO: 22.8% → **30.3-33.1%**

### Math Tasks (AIME24, Qwen3-8B)
- Single GRPO: 18.3% → **18.3%** (no improvement)
- AT-GRPO: 18.3% → **50-57%**

See [Benchmark Results](../results/benchmarks.md) for full results.

## Next Steps

- Learn about [Multi-Agent Workflows](workflows.md)
- Understand the [Training System](training-system.md)
- Explore [Training Guides](../training/overview.md)

