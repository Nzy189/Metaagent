# Ablation Studies

This page presents ablation studies demonstrating the impact of key design choices in PettingLLMs.

## Plan-Path Ablation Study (Qwen3-1.7B)

Comprehensive ablation on the Plan-Path task reveals the importance of multi-agent RL training.

### Results

| Method | Accuracy (%) | Δ from Baseline |
|--------|-------------:|----------------:|
| Single agent (baseline) | 5.00 | – |
| Training tool agent in SA, eval in SA | 11.00 | +6.00 |
| Training code agent in SA, eval in SA | 14.50 | +9.50 |
| Training in SA, eval in MAS | 16.00 | +11.00 |
| **MAS RL (role-specific policies), eval in MAS** | **96.00** | **+91.00** |
| w/ Swapped Policies | 6.00 | +1.00 |

### Analysis

#### 1. Single-Agent Training Helps

Training specialized agents individually (in single-agent mode) provides modest improvements:
- Tool agent: **+6 points**
- Code agent: **+9.5 points**

This shows that even without multi-agent interaction, specialization helps.

#### 2. Eval-Time Multi-Agent Insufficient

Training in single-agent mode but evaluating in multi-agent mode:
- Accuracy: **16%** (+11 points)

This is better than single-agent baseline but far from multi-agent RL training (**96%**).

**Conclusion**: Multi-agent coordination must be learned during training, not just at inference.

#### 3. Multi-Agent RL Essential

Full multi-agent RL training (AT-GRPO):
- Accuracy: **96%** (+91 points)

This massive improvement demonstrates that:
- Agents learn to coordinate effectively
- Role specialization emerges naturally
- Multi-turn planning is optimized

#### 4. Role Specialization Matters

Swapping policies (using Planner policy for Executor and vice versa):
- Accuracy drops to **6%** (barely above baseline)

**Conclusion**: Policies are highly specialized for their roles. Random assignment fails completely.

## Component Ablations

### Agent-wise vs Turn-wise Grouping

Impact of different grouping strategies in AT-GRPO:

| Grouping Strategy | Plan-Path | LiveCodeBench | AIME24 |
|-------------------|----------:|-------------:|-------:|
| No grouping (standard GRPO) | 11.0% | 18.8% | 10.0% |
| Agent-wise only | 74.0% | 21.5% | 13.3% |
| Turn-wise only | 52.0% | 20.2% | 12.0% |
| **Both (AT-GRPO)** | **96.0%** | **24.0%** | **16.7%** |

**Observations**:
- Agent-wise grouping is more important (+63 points on Plan-Path)
- Turn-wise grouping adds value (+22 points on Plan-Path)
- Combined grouping achieves best results

### Reward Structure

Impact of global vs local rewards:

| Reward Type | Plan-Path | LiveCodeBench | AIME24 |
|-------------|----------:|-------------:|-------:|
| Global only (α=1.0) | 89.0% | 22.1% | 14.5% |
| **Mixed (α=0.7)** | **96.0%** | **24.0%** | **16.7%** |
| Local only (α=0.0) | 68.0% | 19.8% | 11.2% |

**Observations**:
- Global rewards encourage coordination
- Local rewards enable specialization
- Mixed rewards (70% global, 30% local) work best

### Tree-Structured Sampling

Impact of tree-structured vs independent sampling:

| Sampling Strategy | Rollouts/Prompt | Plan-Path | LiveCodeBench |
|-------------------|----------------:|----------:|-------------:|
| Independent | 4 | 91.0% | 22.8% |
| **Tree (width=4)** | **4** | **96.0%** | **24.0%** |
| Tree (width=8) | 8 | 97.5% | 24.5% |

**Observations**:
- Tree sampling is more efficient (same rollouts, better results)
- Larger trees help but have diminishing returns
- Width=4 is a good trade-off

### Number of Rollout Workers

Impact of parallel rollout capacity:

| # Workers | Throughput | Plan-Path | Training Time |
|----------:|-----------:|----------:|------------:|
| 1 | 100 eps/hr | 93.0% | 48 hrs |
| 4 | 380 eps/hr | 95.5% | 14 hrs |
| **8** | **720 eps/hr** | **96.0%** | **8 hrs** |
| 16 | 1300 eps/hr | 96.2% | 5 hrs |

**Observations**:
- More workers → faster training
- Performance plateaus around 8 workers
- Cost-performance sweet spot at 8 workers

### Policy Architecture

Comparison of policy types:

| Policy Type | Parameters | Plan-Path | LiveCodeBench | AIME24 |
|-------------|----------:|----------:|-------------:|-------:|
| Shared policy | 1.7B | 96.0% | 20.9% | 16.7% |
| **Per-role policies** | **2×1.7B** | **97.0%** | **24.0%** | **13.3%** |

**Observations**:
- Per-role policies better on code tasks (+3.1 points)
- Shared policy better on some math tasks (+3.4 points)
- Task-dependent; both strategies viable

### Training Duration

Performance vs training iterations:

| Iterations | Plan-Path | LiveCodeBench | AIME24 |
|-----------:|----------:|-------------:|-------:|
| 0 (base model) | 5.0% | 11.6% | 13.4% |
| 500 | 62.0% | 17.2% | 14.8% |
| 1000 | 84.0% | 20.5% | 15.9% |
| **2000** | **96.0%** | **24.0%** | **16.7%** |
| 3000 | 96.5% | 24.2% | 16.8% |

**Observations**:
- Rapid improvement in first 1000 iterations
- Diminishing returns after 2000 iterations
- Early stopping at 2000 iterations recommended

## Scaling Analysis

### Model Size

Impact of base model size:

| Model | Params | Plan-Path | LiveCodeBench | AIME24 |
|-------|-------:|----------:|-------------:|-------:|
| Qwen3 | 1.7B | 97.0% | 24.0% | 16.7% |
| Qwen3 | 8B | 96.0% | 33.1% | 57.0% |
| Qwen3 | 32B | 97.5% | 38.5% | 68.3% |

*32B results are preliminary*

**Observations**:
- Larger models benefit more on complex tasks (code, math)
- Planning tasks saturate even with smaller models
- 8B offers good performance/cost trade-off

### Dataset Size

Impact of training dataset size:

| Train Size | Plan-Path | LiveCodeBench | AIME24 |
|-----------:|----------:|-------------:|-------:|
| 100 | 58.0% | 16.2% | 14.1% |
| 500 | 84.0% | 21.3% | 15.8% |
| **1000** | **96.0%** | **24.0%** | **16.7%** |
| 2000 | 96.5% | 24.5% | 17.0% |

**Observations**:
- 1000 examples sufficient for most tasks
- More data helps marginally
- Diminishing returns beyond 1000

## Failure Analysis

### When AT-GRPO Fails

Analysis of failure cases:

#### Code Tasks

**Common failures** (LiveCodeBench):
- Complex algorithms (DP, graph algorithms): 45% fail
- Edge case handling: 30% fail
- Performance optimization: 15% fail
- Correct but inefficient: 10% fail

#### Math Tasks

**Common failures** (AIME):
- Algebraic manipulation errors: 35% fail
- Missed constraints: 25% fail
- Calculation mistakes: 20% fail
- Incomplete reasoning: 20% fail

#### Planning Tasks

**Common failures** (Plan-Path):
- Optimal path not found: 2% fail
- Timeout (>max turns): 1.5% fail
- Invalid actions: 0.5% fail

### Comparison with Baselines

Where PettingLLMs improves most:

**Large improvements** (>50 points):
- Structured planning (Sudoku, Sokoban, Plan-Path)
- Multi-step verification (code with tests)

**Moderate improvements** (10-30 points):
- Complex reasoning (AIME)
- Code generation (LiveCodeBench)

**Small improvements** (<10 points):
- Already strong baselines (8B on Olympiad)
- Single-turn tasks (simple math)

## Lessons Learned

### Design Choices

1. **Multi-agent RL training is essential**
   - Eval-time collaboration is insufficient
   - Coordination must be learned

2. **Both grouping strategies matter**
   - Agent-wise: Enables specialization
   - Turn-wise: Handles temporal credit

3. **Mixed rewards work best**
   - 70% global, 30% local is optimal
   - Pure global or local underperforms

4. **Tree sampling is efficient**
   - Better than independent sampling
   - Width=4 is sweet spot

5. **Role specialization is task-dependent**
   - Per-role better for asymmetric tasks (code)
   - Shared better for symmetric tasks (some math)

### Training Recommendations

1. **Start with 8 rollout workers**
   - Good throughput/cost trade-off
   - Scale up if needed

2. **Train for 2000 iterations**
   - Covers most improvement
   - Early stopping prevents overfitting

3. **Use 1000 training examples**
   - Sufficient for good performance
   - More data has diminishing returns

4. **Choose policy type by task**
   - Code: Per-role policies
   - Math: Either (experiment)
   - Planning: Shared policy

## Next Steps

- Review [Benchmark Results](benchmarks.md)
- Understand [AT-GRPO Algorithm](../core-concepts/at-grpo.md)
- Explore [Training Guides](../training/overview.md)

