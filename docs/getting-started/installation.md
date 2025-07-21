# Installation Guide

This guide will help you set up pettingllms on your system.

## Prerequisites

Before installing pettingllms, ensure you have the following:

- Python 3.10 or higher
- CUDA version >= 12.1
- [uv](https://docs.astral.sh/uv/) package manager

## Installing uv

If you don't have uv installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Basic Installation

pettingllms uses [verl](https://github.com/volcengine/verl) as its training backend. Follow these steps to install pettingllms and our custom fork of verl:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/agentica-project/pettingllms.git
cd pettingllms

# create a conda environment
conda create -n pettingllms python=3.10
conda activate pettingllms

# Install all dependencies
pip install -e ./verl
pip install -e .
```

This will install pettingllms and all its dependencies in development mode.

For more help, refer to the [GitHub issues page](https://github.com/agentica-project/pettingllms/issues). 