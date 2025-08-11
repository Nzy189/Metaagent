#!/usr/bin/env python3
"""
Install PettingLLMs dependencies in order.
Resolve build dependencies for packages like flash-attn.
"""

import subprocess
import sys
import time

def run_pip_install(packages, description=""):
    """Install a list of packages with pip."""
    if description:
        print(f"\n🔧 {description}")
    
    for package in packages:
        print(f"📦 正在安装 {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True, text=True)
            print(f"✅ 安装成功 {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败 {package}")
            print(f"错误信息: {e.stderr}")
            return False
        time.sleep(1)  # Short delay to avoid potential concurrency issues
    return True

def main():
    print("🚀 开始按顺序安装 PettingLLMs 依赖...")
    
    # Group 1: Basic build tools and core deps
    basic_deps = [
        "wheel",
        "setuptools>=80.0.0",
        "packaging",
        "ninja>=1.11.0",
    ]
    
    # Group 2: PyTorch ecosystem
    torch_deps = [
        "torch==2.7.0",
        "torchaudio==2.7.0", 
        "torchvision==0.22.0",
        "triton==3.3.0",
    ]
    
    # Group 3: Basic ML libraries
    ml_deps = [
        "numpy>=2.2.0,<2.3.0",
        "scipy",
        "scikit-learn",
        "pandas",
        "datasets",
        "transformers>=4.53.0,<4.54.0",
        "tokenizers>=0.21.0,<0.22.0",
        "tiktoken>=0.9.0",
        "accelerate",
    ]
    
    # Group 4: Packages requiring compilation
    compiled_deps = [
        "flash-attn>=2.8.0",
        "deepspeed", 
        "vllm==0.9.2",
        "torchao==0.9.0",
        "xgrammar==0.1.19",
    ]
    
    # Group 5: Other dependencies
    other_deps = [
        "sgl-kernel>=0.2.0",
        "sglang==0.4.9.post2", 
        "sglang-router",
        "peft",
        "sentence-transformers",
        "torchmetrics",
        "pillow>=11.3.0",
        "safetensors>=0.5.3",
        "polars",
        "dm-tree",
        "pyarrow>=15.0.0",
        "fsspec>=2023.1.0,<=2025.3.0",
        "google-cloud-aiplatform",
        "vertexai",
        "kubernetes",
        "ray",
        "requests>=2.32.0",
        "aiohttp>=3.12.0",
        "gradio",
        "selenium",
        "browsergym",
        "firecrawl",
        "fastapi",
        "uvicorn",
        "latex2sympy2",
        "pylatexenc",
        "nltk",
        "scikit-image", 
        "swebench",
        "e2b_code_interpreter",
        "jupyter",
        "ipython",
        "notebook",
        "fire",
        "gdown",
        "tabulate",
        "sortedcontainers",
        "PyMuPDF",
        "together",
        "wandb",
        "pybind11",
        "gym",
        "tqdm>=4.67.0",
        "rich",
        "antlr4-python3-runtime==4.7.2",
        "pydantic>=2.11.0,<3.0.0",
    ]
    
    # Dev tools
    dev_deps = [
        "pytest",
        "pre-commit", 
        "ruff",
        "mypy",
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.0.0",
        "mkdocstrings[python]>=0.24.0",
        "mkdocs-autorefs>=0.5.0",
        "pymdown-extensions>=10.0.0",
    ]
    
    # 按顺序安装各组
    install_groups = [
        (basic_deps, "安装基础构建工具"),
        (torch_deps, "安装 PyTorch 生态"),
        (ml_deps, "安装基础机器学习库"),
        (compiled_deps, "安装需要编译的包"),
        (other_deps, "安装其他依赖"),
        (dev_deps, "安装开发工具"),
    ]
    
    for deps, description in install_groups:
        if not run_pip_install(deps, description):
            print(f"❌ 安装失败，停止于: {description}")
            return False
    
    print("\n🎉 所有依赖安装完成！")
    
    # Finally install the project itself in editable mode
    print("\n📦 以可编辑模式安装项目...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"
        ], check=True)
        print("✅ 项目安装成功！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 项目安装失败: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 