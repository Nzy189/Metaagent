.PHONY: install install-basic install-torch install-ml install-compiled install-other install-dev clean

# 按顺序安装所有依赖
install: install-basic install-torch install-ml install-compiled install-other install-dev install-project
	@echo "🎉 所有依赖安装完成！"

# 1. 基础构建工具
install-basic:
	@echo "🔧 安装基础构建工具..."
	pip install wheel setuptools>=80.0.0 packaging ninja>=1.11.0

# 2. PyTorch生态系统
install-torch:
	@echo "🔧 安装PyTorch生态系统..."
	pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 triton==3.3.0

# 3. 基础ML库
install-ml:
	@echo "🔧 安装基础机器学习库..."
	pip install "numpy>=2.2.0,<2.3.0" scipy scikit-learn pandas datasets
	pip install "transformers>=4.53.0,<4.54.0" "tokenizers>=0.21.0,<0.22.0" "tiktoken>=0.9.0" accelerate

# 4. 需要编译的包
install-compiled:
	@echo "🔧 安装需要编译的包..."
	pip install "flash-attn>=2.8.0"
	pip install deepspeed vllm==0.9.2 torchao==0.9.0 xgrammar==0.1.19

# 5. 其他依赖
install-other:
	@echo "🔧 安装其他依赖..."
	pip install "sgl-kernel>=0.2.0" sglang==0.4.9.post2 sglang-router peft
	pip install sentence-transformers torchmetrics "pillow>=11.3.0" "safetensors>=0.5.3"
	pip install polars dm-tree "pyarrow>=15.0.0" "fsspec>=2023.1.0,<=2025.3.0"
	pip install google-cloud-aiplatform vertexai kubernetes ray
	pip install "requests>=2.32.0" "aiohttp>=3.12.0" gradio selenium browsergym firecrawl
	pip install fastapi uvicorn latex2sympy2 pylatexenc nltk scikit-image
	pip install swebench e2b_code_interpreter jupyter ipython notebook
	pip install fire gdown tabulate sortedcontainers PyMuPDF together wandb pybind11 gym
    pip install "tqdm>=4.67.0" rich "antlr4-python3-runtime==4.7.2" "pydantic>=2.11.0,<3.0.0"

# 6. 开发工具
install-dev:
	@echo "🔧 安装开发工具..."
	pip install pytest pre-commit ruff mypy
	pip install "mkdocs>=1.5.0" "mkdocs-material>=9.0.0" "mkdocstrings[python]>=0.24.0"
	pip install "mkdocs-autorefs>=0.5.0" "pymdown-extensions>=10.0.0"

# 7. 安装项目本身
install-project:
	@echo "📦 以可编辑模式安装项目..."
	pip install -e . --no-deps

# 清理
clean:
	pip uninstall -y pettingllms
	pip freeze | grep -v "^-e" | xargs pip uninstall -y

# 快速重装（跳过大包）
reinstall-quick: clean install-basic install-project
	@echo "🚀 快速重装完成！" 