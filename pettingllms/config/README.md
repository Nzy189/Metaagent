# 配置文件架构说明

## 概述

本项目采用了模块化的配置文件架构，避免了重复配置，使配置更加科学和易于维护。

## 配置文件结构

### 1. 基础配置文件 (`code_base.yaml`)
- **作用**: 包含所有共享的配置项
- **内容**: 
  - 数据集配置
  - 模型配置
  - Actor/Rollout/Ref配置
  - 算法配置
  - 训练器配置
  - 环境配置
  - 多智能体交互配置

### 2. 单策略配置文件 (`code_single_policy.yaml`)
- **作用**: 配置两个智能体使用同一个策略进行训练
- **特点**: 
  - 继承自 `code_base.yaml`
  - 两个智能体都使用 `policy_0`
  - 使用相同的模型 (Qwen/Qwen2.5-Coder-3B)

### 3. 双策略配置文件 (`code_two_policies.yaml`)
- **作用**: 配置两个智能体使用不同的策略进行训练
- **特点**:
  - 继承自 `code_base.yaml`
  - `code_generator` 使用 `policy_0` (Qwen/Qwen2.5-Coder-3B)
  - `test_generator` 使用 `policy_1` (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

## 配置继承关系

```
code_base.yaml (基础配置)
├── code_single_policy.yaml (单策略)
└── code_two_policies.yaml (双策略)
```

## 使用方法

### 单策略训练
```bash
python train.py --config config/code_single_policy.yaml
```

### 双策略训练
```bash
python train.py --config config/code_two_policies.yaml
```

## 配置项说明

### 共享配置项 (code_base.yaml)
- `data`: 数据集配置
- `models`: 模型路径配置
- `actor_config`: Actor网络配置
- `rollout_config`: Rollout配置
- `ref_config`: 参考模型配置
- `agent_config`: 智能体配置
- `algorithm`: 算法配置
- `trainer`: 训练器配置
- `env`: 环境配置
- `multi_agent_interaction`: 多智能体交互配置

### 特定配置项
- `agents`: 智能体定义和策略分配
- `actor_rollout_ref_configs`: 策略特定的Actor/Rollout/Ref配置

## 优势

1. **避免重复**: 共享配置项只需在基础配置中定义一次
2. **易于维护**: 修改共享配置只需修改基础配置文件
3. **清晰结构**: 每个配置文件职责明确
4. **易于扩展**: 新增配置只需继承基础配置并添加特定项 