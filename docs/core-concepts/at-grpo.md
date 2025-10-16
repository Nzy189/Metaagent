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

AT-GRPO uses tree-structured sampling with best-of-N selection at each agent step:

### Training Algorithm Flow

For each environment prompt, AT-GRPO maintains N parallel rollouts and builds a tree structure:

```
Initial State (env_idx)
    |
    |
    |
Turn 1:
    Agent 1 acts:
        - Generate responses in parallel
        - Execute environment steps
        - Calculate rewards for all N rollouts
        - Select best rollout (highest reward)
        - Copy best state to all N rollouts  ← Tree branch selection
    
    Agent 2 acts:
        - Generate responses in parallel (from shared state)
        - Execute environment steps
        - Calculate rewards
        - Select best rollout
        - Copy best state to all N rollouts  ← Tree branch selection
    
    ... (continue for all agents in turn)

Turn 2:
    (Repeat same process with shared state from Turn 1)
    ...
```

### Key Implementation Details

**Step-by-Step Execution** (from `generate_env_idx_rollout`):

1. . **Sequential Environment Execution**: Execute agent actions in environment
   ```python
   for idx in range(N):
       current_agent.update_from_model(response)
       await current_agent.step(env)
   ```

3. **Reward Calculation**: Calculate rewards for all N rollouts
   ```python
   for idx in range(N):
       current_agent.calculate_reward(env)
   ```

4. **Best-of-N Selection**: Select rollout with highest reward
   ```python
   if if_greedy:
       best_i = argmax([agent.agent_reward for agent in agents])
   else:
       best_i = 0
   ```

5. **State Broadcasting**: Copy best state to all rollouts
   ```python
   selected_env = envs_list[best_i]
   selected_agent_group = agent_groups[best_i]
   
   # Broadcast to all rollouts
   envs_list = [deepcopy(selected_env) for _ in envs_list]
   agent_groups = [deepcopy(selected_agent_group) for _ in agent_groups]
   ```

### Tree Structure Visualization

```
                    Initial State
                         |
        +----------------+----------------+
        |                |                |
    Rollout 0       Rollout 1       Rollout N-1
        |                |                |
    Agent 1 generates N different responses
        ↓                ↓                ↓
    [Reward: 0.5]   [Reward: 0.8]   [Reward: 0.3]
        |                |                |
        +-------→ Select Best (1) ←-------+
                         |
            All rollouts copy state from Rollout 1
                         |
        +----------------+----------------+
        |                |                |
    Rollout 0       Rollout 1       Rollout N-1
    (all same)      (all same)      (all same)
        |                |                |
    Agent 2 generates N different responses
        ↓                ↓                ↓
    [Reward: 0.6]   [Reward: 0.4]   [Reward: 0.9]
        |                |                |
        +-------→ Select Best (N-1) ←-----+
                         |
                   (Continue...)
```

**Benefits**:
- Efficient exploration: N parallel attempts per agent
- Progressive refinement: Best decisions cascade forward
- Memory efficient: Only one state tree instead of N independent trajectories
- Credit assignment: Each agent's contribution is evaluated separately
- Natural variance for advantage estimation through parallel sampling

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








## Next Steps

Continue exploring core concepts:

- Learn about distributed architecture: [Training System](training-system.md)
- Understand agent specialization: [Three-Level Specialization](three-level-specialization.md)
- Return to concepts overview: [Core Concepts](overview.md)

