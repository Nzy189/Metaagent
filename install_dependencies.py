#!/usr/bin/env python3
"""
按顺序安装PettingLLMs的依赖包
解决flash-attn等包的构建依赖问题
"""

import subprocess
import sys
import time

def run_pip_install(packages, description=""):
    """安装指定的包列表"""
    if description:
        print(f"\n🔧 {description}")
    
    for package in packages:
        print(f"📦 Installing {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True, text=True)
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}")
            print(f"Error: {e.stderr}")
            return False
        time.sleep(1)  # 短暂延迟，避免并发问题
    return True

def main():
    print("🚀 开始按顺序安装PettingLLMs依赖...")
    
    # 第1组：基础构建工具和核心依赖
    basic_deps = [
        "wheel",
        "setuptools>=80.0.0",
        "packaging",
        "ninja>=1.11.0",
    ]
    
    # 第2组：PyTorch生态系统
    torch_deps = [
        "torch==2.7.0",
        "torchaudio==2.7.0", 
        "torchvision==0.22.0",
        "triton==3.3.0",
    ]
    
    # 第3组：基础ML库
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
    
    # 第4组：需要编译的包
    compiled_deps = [
        "flash-attn>=2.8.0",
        "deepspeed", 
        "vllm==0.9.2",
        "torchao==0.9.0",
        "xgrammar==0.1.19",
    ]
    
    # 第5组：其他依赖
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
        "antlr4-python3-runtime>=4.9.0,<5.0.0",
        "pydantic>=2.11.0,<3.0.0",
    ]
    
    # 开发工具
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
        (torch_deps, "安装PyTorch生态系统"),
        (ml_deps, "安装基础机器学习库"),
        (compiled_deps, "安装需要编译的包"),
        (other_deps, "安装其他依赖"),
        (dev_deps, "安装开发工具"),
    ]
    
    for deps, description in install_groups:
        if not run_pip_install(deps, description):
            print(f"❌ 安装失败，停止在: {description}")
            return False
            
    print("\n🎉 所有依赖安装完成！")
    
    # 最后以可编辑模式安装项目本身
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