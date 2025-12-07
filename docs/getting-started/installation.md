# Installation

This guide will help you install PettingLLMs and its dependencies.

## Prerequisites

Before installing PettingLLMs, ensure you have:

- **Python 3.8+**
- **CUDA 11.8+** (for GPU support)
- **Git**

## Quick Installation

The easiest way to install PettingLLMs is using the provided setup script:

```bash
git clone https://github.com/pettingllms-ai/PettingLLMs.git
cd PettingLLMs
bash setup.bash
```

This script will:

1. Create a virtual environment
2. Install all required dependencies
3. Set up the PettingLLMs package in development mode

## Manual Installation

If you prefer to install manually, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/pettingllms-ai/PettingLLMs.git
cd PettingLLMs
```

### 2. Create Virtual Environment

```bash
python -m venv pettingllms_venv
source pettingllms_venv/bin/activate  # On Linux/Mac
# or
pettingllms_venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

For CUDA 12.8:
```bash
pip install -r requirements_venv_cu128.txt
```

For other CUDA versions:
```bash
pip install -r requirements_venv.txt
```

### 4. Install PettingLLMs

```bash
pip install -e .
```

## Verify Installation

To verify that PettingLLMs is installed correctly:

```bash
python -c "import pettingllms; print('PettingLLMs installed successfully!')"
```

## Docker Installation (Optional)

If you prefer using Docker, you can build a container with all dependencies:

```bash
# Coming soon
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Check your CUDA version: `nvcc --version`
2. Install the appropriate PyTorch version for your CUDA
3. Verify GPU is accessible: `python -c "import torch; print(torch.cuda.is_available())"`

### Import Errors

If you get import errors:

1. Ensure the virtual environment is activated
2. Reinstall the package: `pip install -e .`
3. Check Python version compatibility

### Dependency Conflicts

If you encounter dependency conflicts:

1. Try creating a fresh virtual environment
2. Update pip: `pip install --upgrade pip`
3. Install dependencies one at a time to identify conflicts

## Next Steps

Once installation is complete, proceed to the [Quick Start Guide](quick-start.md) to run your first training session.

