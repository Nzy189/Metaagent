# SFT 数据收集 API 使用说明

本文档说明如何使用外部 API（OpenAI、DeepSeek、Claude）进行 SFT 数据收集。

## 功能概述

现在 `code_sft_collect_only.sh` 支持两种模式：
1. **本地模型模式**（默认）：使用本地 vLLM 服务器
2. **API 模式**：使用外部 API（OpenAI、DeepSeek、Claude）

## 支持的 API

### 1. OpenAI API
- **API 类型**: `openai`
- **默认模型**: `gpt-4o`
- **环境变量**: `OPENAI_API_KEY`
- **支持的模型**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo` 等

### 2. DeepSeek API
- **API 类型**: `deepseek`
- **默认模型**: `deepseek-chat`
- **默认 Base URL**: `https://api.deepseek.com`
- **环境变量**: `DEEPSEEK_API_KEY`
- **支持的模型**: `deepseek-chat`, `deepseek-coder` 等

### 3. Claude API (Anthropic)
- **API 类型**: `claude`
- **默认模型**: `claude-3-5-sonnet-20241022`
- **环境变量**: `ANTHROPIC_API_KEY`
- **支持的模型**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307` 等

## 使用方法

### 方法 1: 使用环境变量（推荐）

```bash
# 1. 设置 API Key
export OPENAI_API_KEY="your-openai-api-key-here"
# 或
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
# 或
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# 2. 运行脚本并启用 API 模式
cd scripts/sft_train
USE_API=true API_TYPE=openai API_MODEL=gpt-4o bash code_sft_collect_only.sh
```

### 方法 2: 在脚本中配置

编辑 `code_sft_collect_only.sh`，修改以下配置：

```bash
# 启用 API 模式
USE_API=true
API_TYPE="openai"  # 或 "deepseek" 或 "claude"
API_MODEL="gpt-4o"
API_TEMPERATURE=0.7
API_MAX_TOKENS=2048
API_TIMEOUT=60.0

# 设置 API Key
export OPENAI_API_KEY="your-key-here"
```

然后运行：
```bash
bash code_sft_collect_only.sh
```

## 配置参数详解

### 必需参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `USE_API` | 是否启用 API 模式 | `true` / `false` |
| `API_TYPE` | API 类型 | `openai` / `deepseek` / `claude` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `API_MODEL` | 模型名称 | 根据 API_TYPE 自动设置 |
| `API_BASE_URL` | 自定义 Base URL | 空（使用默认） |
| `API_TEMPERATURE` | 采样温度 | `0.7` |
| `API_MAX_TOKENS` | 最大生成 tokens | `2048` |
| `API_TIMEOUT` | 请求超时时间（秒） | `60.0` |

## 使用示例

### 示例 1: 使用 OpenAI GPT-4o

```bash
export OPENAI_API_KEY="sk-..."
USE_API=true \
API_TYPE=openai \
API_MODEL=gpt-4o \
API_TEMPERATURE=0.8 \
bash code_sft_collect_only.sh
```

### 示例 2: 使用 DeepSeek

```bash
export DEEPSEEK_API_KEY="sk-..."
USE_API=true \
API_TYPE=deepseek \
API_MODEL=deepseek-chat \
API_MAX_TOKENS=4096 \
bash code_sft_collect_only.sh
```

### 示例 3: 使用 Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
USE_API=true \
API_TYPE=claude \
API_MODEL=claude-3-5-sonnet-20241022 \
API_TEMPERATURE=0.7 \
bash code_sft_collect_only.sh
```

### 示例 4: 使用自定义 Base URL（例如代理）

```bash
export OPENAI_API_KEY="sk-..."
USE_API=true \
API_TYPE=openai \
API_MODEL=gpt-4o \
API_BASE_URL="https://your-proxy.com/v1" \
bash code_sft_collect_only.sh
```

## Python 直接调用

如果需要在 Python 代码中直接使用 API 客户端：

```python
import asyncio
from pettingllms.utils.api_client import create_api_client

async def main():
    # 创建 API 客户端
    client = create_api_client(
        api_type="openai",  # 或 "deepseek" 或 "claude"
        api_key="your-api-key",  # 可选，会从环境变量读取
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2048,
        timeout=60.0
    )

    # 生成响应
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
    ]

    response = await client.generate(messages)
    print(response)

# 运行
asyncio.run(main())
```

## 批量生成

API 客户端也支持批量生成（并行处理）：

```python
from pettingllms.utils.api_client import create_api_client, batch_generate

async def batch_example():
    client = create_api_client(api_type="openai")

    prompts = [
        "Write a hello world program in Python",
        "Write a hello world program in JavaScript",
        "Write a hello world program in Rust"
    ]

    responses = await batch_generate(
        client=client,
        prompts=prompts,
        system_prompt="You are a coding assistant."
    )

    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

asyncio.run(batch_example())
```

## 安装依赖

确保已安装相应的 Python 包：

```bash
# OpenAI 和 DeepSeek
pip install openai

# Claude (Anthropic)
pip install anthropic
```

## 注意事项

1. **API Key 安全**: 不要将 API Key 硬编码在脚本中，建议使用环境变量
2. **成本控制**: 使用 API 会产生费用，建议先用小规模数据测试
3. **速率限制**: 注意各 API 提供商的速率限制
4. **超时设置**: 根据网络情况调整 `API_TIMEOUT` 参数
5. **数据隐私**: 注意不要将敏感数据发送到外部 API

## 故障排查

### 1. API Key 错误
```
ValueError: API key not provided for openai
```
**解决方案**: 确保设置了正确的环境变量（`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`）

### 2. 导入错误
```
ImportError: openai package is required
```
**解决方案**: 安装相应的包 `pip install openai` 或 `pip install anthropic`

### 3. 超时错误
```
Error getting API response: timeout
```
**解决方案**: 增加 `API_TIMEOUT` 参数值

### 4. 速率限制
```
RateLimitError: Rate limit exceeded
```
**解决方案**:
- 减少并发请求数量
- 降低 `training.sft_num_episodes` 参数
- 等待一段时间后重试

## 输出结果

API 模式和本地模式的输出格式完全相同：
- SFT 数据保存在 `OUTPUT_DIR` 目录（默认: `./sft_data_code`）
- 包含 JSONL 格式的训练数据和统计信息

示例输出：
```
SFT data collection completed!
Collected data saved to: ./sft_data_code
To train on this data, use code_sft_train_only.sh
```

## 扩展开发

如果需要支持其他 API 提供商，可以在 `pettingllms/utils/api_client.py` 中添加新的客户端类：

```python
class NewAPIClient(BaseAPIClient):
    """New API client"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        # 初始化客户端

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 实现生成逻辑
        pass
```

然后在 `create_api_client` 函数中注册新的 API 类型。
