# SFT Training Scripts

这个目录包含用于收集SFT数据和训练模型的脚本。

## 脚本说明

### 1. 完整流程（收集+训练）

#### `code_sft_collect_and_train.sh`
完整的SFT流程：从code环境收集数据并训练模型

```bash
cd /path/to/PettingLLMs
bash scripts/sft_train/code_sft_collect_and_train.sh
```

**配置项：**
- 收集100个episodes的数据
- 只保留`env.success == True`的数据
- 使用LoRA训练3个epochs
- 输出目录：`./sft_data_code` (数据), `./sft_model_code` (模型)

#### `math_sft_collect_and_train.sh`
完整的SFT流程：从math环境收集数据并训练模型

```bash
cd /path/to/PettingLLMs
bash scripts/sft_train/math_sft_collect_and_train.sh
```

### 2. 仅收集数据

#### `code_sft_collect_only.sh`
只收集数据，不进行训练

```bash
cd /path/to/PettingLLMs
bash scripts/sft_train/code_sft_collect_only.sh
```

**用途：**
- 适合需要先收集大量数据，稍后再训练的场景
- 可以在数据收集完成后检查数据质量
- 设置 `training.run_sft_training=False`

### 3. 仅训练模型

#### `code_sft_train_only.sh`
使用已收集的数据进行训练

```bash
cd /path/to/PettingLLMs
# 先修改脚本中的 TRAIN_DATA_PATH 指向你的数据文件
bash scripts/sft_train/code_sft_train_only.sh
```

**注意：**
- 需要先运行数据收集脚本
- 修改脚本中的 `TRAIN_DATA_PATH` 变量指向实际的数据文件

### 4. 快速测试

#### `code_sft_quick_test.sh`
快速测试脚本，用于调试和验证

```bash
cd /path/to/PettingLLMs
bash scripts/sft_train/code_sft_quick_test.sh
```

**特点：**
- 只收集10个episodes（快速）
- 使用单GPU
- 训练1个epoch
- 适合快速验证流程是否正常

## 使用步骤

### 方式1: 一键运行（推荐用于生产）

```bash
# 收集数据并训练
bash scripts/sft_train/code_sft_collect_and_train.sh
```

### 方式2: 分步运行（推荐用于调试）

```bash
# 步骤1: 收集数据
bash scripts/sft_train/code_sft_collect_only.sh

# 步骤2: 检查收集的数据
ls -lh ./sft_data_code/
cat ./sft_data_code/sft_data_*_stats.json

# 步骤3: 训练模型
# 编辑 code_sft_train_only.sh，设置正确的数据文件路径
bash scripts/sft_train/code_sft_train_only.sh
```

### 方式3: 快速测试（推荐用于首次使用）

```bash
# 快速测试整个流程
bash scripts/sft_train/code_sft_quick_test.sh
```

## 配置参数说明

### 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 基础模型路径 | `/home/nvidia/data/models/Qwen3-1.7B` |
| `GPU_num` | 使用的GPU数量 | 2 |
| `training.sft_num_episodes` | 收集的episode数量 | 100 |
| `training.only_success` | 只保留成功的数据 | True |
| `env.max_turns` | 每个episode的最大轮数 | 3 |

### SFT训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `training.sft_config.num_train_epochs` | 训练epochs | 3 |
| `training.sft_config.per_device_train_batch_size` | 每设备batch size | 2 |
| `training.sft_config.gradient_accumulation_steps` | 梯度累积步数 | 8 |
| `training.sft_config.learning_rate` | 学习率 | 5e-5 |
| `training.sft_config.use_lora` | 是否使用LoRA | True |
| `training.sft_config.lora_r` | LoRA rank | 64 |
| `training.sft_config.lora_alpha` | LoRA alpha | 16 |

## 自定义配置

### 修改模型路径

编辑脚本中的 `MODEL_PATH` 变量：

```bash
MODEL_PATH="/your/model/path"
```

### 修改GPU设置

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU
GPU_num=4
```

### 修改收集的数据量

```bash
training.sft_num_episodes=500  # 收集500个episodes
```

### 使用不同的环境

```bash
# 使用不同的config文件
--config-path ../config/your_env \
--config-name your_config_name \
```

## 输出文件

### 数据收集输出

```
./sft_data_code/
├── sft_data_20241203_123456.jsonl       # 收集的训练数据
└── sft_data_20241203_123456_stats.json  # 数据统计信息
```

### 训练输出

```
./sft_model_code/
├── config.json                    # 模型配置
├── adapter_config.json            # LoRA配置（如果使用LoRA）
├── adapter_model.safetensors      # LoRA权重（如果使用LoRA）
├── tokenizer_config.json          # Tokenizer配置
├── special_tokens_map.json
├── tokenizer.json
└── training_args.bin              # 训练参数
```

## 数据格式

收集的数据格式为JSONL，每行一个样本：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "问题内容..."
    },
    {
      "role": "assistant",
      "content": "回答内容..."
    }
  ],
  "metadata": {
    "agent_name": "code_single_agent",
    "policy_name": "shared_model",
    "reward": 1.0,
    "env_success": true,
    "env_name": "code_env",
    "env_idx": 0,
    "rollout_idx": 0,
    "turn_idx": 0
  }
}
```

## 环境兼容性

这些脚本支持所有已注册的多智能体环境：

- ✅ `code_env` - 代码生成环境
- ✅ `math_env` - 数学问题环境
- ✅ `search_env` - 搜索环境
- ✅ `stateful_env` - 有状态环境
- ✅ 其他自定义环境

只需要：
1. 环境继承自 `pettingllms.multi_agent_env.base.Env`
2. 环境有 `success` 属性
3. 环境已在 `ENV_CLASS_MAPPING` 中注册

## 故障排查

### 问题1: CUDA内存不足

**解决方案：**
```bash
# 减少batch size
training.sft_config.per_device_train_batch_size=1

# 或增加梯度累积
training.sft_config.gradient_accumulation_steps=16

# 或使用更小的LoRA rank
training.sft_config.lora_r=32
```

### 问题2: 没有收集到数据

**检查：**
1. 确认环境能够成功完成任务（`env.success == True`）
2. 查看统计文件 `*_stats.json` 了解成功率
3. 如果成功率过低，考虑设置 `training.only_success=False`

### 问题3: 训练数据文件找不到

**解决方案：**
```bash
# 列出所有收集的数据文件
ls -lh ./sft_data_code/sft_data_*.jsonl

# 更新训练脚本中的路径
TRAIN_DATA_PATH="./sft_data_code/sft_data_20241203_123456.jsonl"
```

## 最佳实践

1. **首次使用**：先运行 `code_sft_quick_test.sh` 验证流程
2. **生产环境**：使用 `code_sft_collect_only.sh` + `code_sft_train_only.sh` 分步运行
3. **大规模收集**：增加 `training.sft_num_episodes` 到 1000+
4. **质量保证**：保持 `training.only_success=True` 确保数据质量
5. **资源优化**：根据GPU内存调整batch size和LoRA rank

## 扩展到其他环境

复制并修改脚本：

```bash
# 复制模板
cp code_sft_collect_and_train.sh your_env_sft_collect_and_train.sh

# 修改关键参数
# 1. --config-path 和 --config-name
# 2. env相关参数
# 3. 输出目录名称
```

## 支持和反馈

如有问题，请查看：
- 主README: `/path/to/PettingLLMs/pettingllms/sft_train/README.md`
- 配置文件: `/path/to/PettingLLMs/pettingllms/config/code/code_L0_single_agent.yaml`
