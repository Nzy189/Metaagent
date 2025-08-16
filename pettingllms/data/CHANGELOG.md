# 变更日志

## 新增功能

### 1. CodeContests 数据集下载功能

- **文件**: `download_codecontests.py`
- **功能**: 从 Hugging Face 下载 Gen-Verse/CodeContests_train 数据集
- **特性**:
  - 支持命令行参数配置
  - 自动创建目录结构
  - 保存为 Parquet 和 JSON 格式
  - 支持不同数据集分割 (train/validation/test)

### 2. 本地数据集优先加载机制

- **文件**: `pettingllms/multi_agent_env/code/code_utils.py`
- **功能**: 修改 `load_problem_batch` 函数，支持本地优先加载
- **特性**:
  - 自动检查本地 `datasets` 目录
  - 智能路径解析和转换
  - 本地加载失败时自动回退到 Hugging Face
  - 保持原有的 streaming 和传统加载方式

### 3. 测试和示例脚本

- **文件**: `test_local_loading.py`
- **功能**: 测试本地数据集加载功能
- **文件**: `usage_example.py`
- **功能**: 展示如何使用本地数据集加载功能

### 4. 文档和说明

- **文件**: `README.md`
- **内容**: 详细的使用说明和示例
- **文件**: `CHANGELOG.md`
- **内容**: 本变更日志

## 目录结构

```
pettingllms/data/
├── datasets/                          # 数据集存储目录
│   └── codecontests_train/           # CodeContests 数据集
│       ├── train.parquet             # 训练集
│       ├── train.json                # JSON 格式
│       └── ...                       # 其他分割
├── download_codecontests.py          # 数据集下载脚本
├── test_local_loading.py             # 本地加载测试脚本
├── usage_example.py                  # 使用示例脚本
├── README.md                         # 详细说明文档
└── CHANGELOG.md                      # 变更日志
```

## 使用方法

### 步骤 1: 下载数据集
```bash
cd pettingllms/data
python download_codecontests.py
```

### 步骤 2: 测试本地加载
```bash
python test_local_loading.py
```

### 步骤 3: 在代码中使用
```python
from pettingllms.multi_agent_env.code.code_utils import load_problem_batch

# 自动从本地加载（如果存在），否则从 Hugging Face 下载
problems = load_problem_batch(
    dataset_name="CodeContests_train",
    batch_size=5,
    split="train"
)
```

## 技术细节

### 本地路径解析
- 自动将数据集名称转换为本地路径格式
- 例如: `CodeContests_train` → `codecontests_train`
- 支持包含 `/` 的数据集名称

### 回退机制
1. 优先检查本地 `pettingllms/data/datasets/` 目录
2. 本地存在且可加载时，使用本地数据
3. 本地不存在或加载失败时，自动从 Hugging Face 下载
4. 支持 streaming 和传统加载方式

### 错误处理
- 本地数据集检查失败时继续执行
- 本地加载失败时自动回退
- 详细的错误日志和状态提示

## 兼容性

- 保持与原有代码的完全兼容
- 不影响现有的 Hugging Face 数据集加载功能
- 支持所有原有的参数和返回值格式

## 依赖要求

- `datasets` 库 (用于数据集加载)
- `pathlib` (Python 3.4+ 内置)
- 其他原有依赖保持不变


