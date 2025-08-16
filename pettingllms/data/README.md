# 数据集下载说明

## CodeContests 数据集

### 下载方式

#### 方式1: 使用命令行脚本
```bash
cd pettingllms/data
python download_codecontests.py
```

可选参数：
- `--local_dir`: 指定本地保存目录
- `--split`: 指定数据集分割 (train/validation/test)

示例：
```bash
# 下载到指定目录
python download_codecontests.py --local_dir /path/to/your/data

# 下载验证集
python download_codecontests.py --split validation
```

#### 方式2: 使用简单示例脚本
```bash
cd pettingllms/data
python download_example.py
```

#### 方式3: 在代码中调用
```python
from download_codecontests import download_codecontests_dataset

# 下载到默认目录 (pettingllms/data/codecontests)
success = download_codecontests_dataset()

# 下载到指定目录
success = download_codecontests_dataset(local_dir="/path/to/your/data")

# 下载指定分割
success = download_codecontests_dataset(split="validation")
```

### 数据集信息

- **来源**: Hugging Face - Gen-Verse/CodeContests_train
- **格式**: 自动保存为 Parquet 和 JSON 格式
- **默认保存位置**: `pettingllms/data/codecontests/`
- **文件结构**:
  ```
  codecontests/
  ├── train.parquet
  ├── train.json
  ├── validation.parquet (如果下载)
  ├── validation.json (如果下载)
  └── ...
  ```

### 依赖要求

确保已安装以下包：
```bash
pip install datasets
```

### 注意事项

1. 首次下载可能需要较长时间，取决于网络速度
2. 数据集会自动缓存到 Hugging Face 的默认缓存目录
3. 如果下载失败，请检查网络连接和 Hugging Face 访问权限

## 本地数据集加载

### 自动本地优先加载

修改后的 `load_problem_batch` 函数现在支持自动从本地加载数据集：

1. **优先检查本地**: 函数会首先检查 `pettingllms/data/datasets/` 目录下是否存在对应的数据集
2. **自动回退**: 如果本地数据集不存在或加载失败，会自动回退到从 Hugging Face 下载
3. **智能路径解析**: 自动将数据集名称转换为本地路径格式

### 使用示例

```python
from pettingllms.multi_agent_env.code.code_utils import load_problem_batch

# 自动从本地加载（如果存在），否则从 Hugging Face 下载
problems = load_problem_batch(
    dataset_name="CodeContests_train",
    batch_size=5,
    split="train"
)
```

### 测试本地加载功能

运行测试脚本验证功能：

```bash
cd pettingllms/data
python test_local_loading.py
```

### 目录结构

下载后的数据集将保存在以下结构：

```
pettingllms/data/
├── datasets/
│   └── codecontests_train/          # 自动生成的目录名
│       ├── train.parquet            # 训练集数据
│       ├── train.json               # JSON 格式（便于查看）
│       ├── validation.parquet       # 验证集（如果下载）
│       └── test.parquet             # 测试集（如果下载）
├── download_codecontests.py         # 下载脚本
├── test_local_loading.py            # 测试脚本
└── README.md                        # 说明文档
```
