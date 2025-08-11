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
        time.sleep(1)  # Short delay to avoid potential concurrency issues
    return True

def main():
    print("🚀 Start installing PettingLLMs dependencies in order...")
    
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
    
    # Install each group in sequence
    install_groups = [
        (basic_deps, "Install basic build tools"),
        (torch_deps, "Install PyTorch ecosystem"),
        (ml_deps, "Install core ML libraries"),
        (compiled_deps, "Install packages requiring compilation"),
        (other_deps, "Install other dependencies"),
        (dev_deps, "Install development tools"),
    ]
    
    for deps, description in install_groups:
        if not run_pip_install(deps, description):
            print(f"❌ Installation failed, stopping at: {description}")
            return False
    
    print("\n🎉 All dependencies installed!")
    
    # 最后以可编辑模式安装项目本身
    print("\n📦 Installing project in editable mode...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"
        ], check=True)
        print("✅ Project installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Project installation failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 