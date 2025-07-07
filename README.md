# Multi-Agent LLM Training Framework

# PettingLLMs Quick Start Guide

This guide will help you quickly set up and test the PettingLLMs environment.

## 1. Environment Setup

First, set up the environment for PettingLLMs:

```bash
bash scripts/setup_pettingllms_test.sh
```

## 2. Model Choice

The system supports various language models through SGLang. You can choose from:

### Recommended Models:
- **Qwen/Qwen2.5-1.5B-Instruct** (default, lightweight)
- **Qwen/Qwen2.5-7B-Instruct** (better performance)
- **Qwen/Qwen2.5-14B-Instruct** (best performance, requires more GPU memory)
- **meta-llama/Llama-3.1-8B-Instruct**
- **microsoft/DialoGPT-medium**

### Model Configuration:
To use a different model, modify the `--model-path` parameter in `fix_sglang.sh`:

```bash
# Example: Using Qwen2.5-7B-Instruct
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000 \
    --tp 1 \
    --dtype float16 \
    --trust-remote-code \
    --mem-fraction-static 0.7 \
    --attention-backend triton \
    --disable-cuda-graph
```

## 3. Testing Environment

### Step 1: Launch Local Server
Start the SGLang server with your chosen model:

```bash
bash fix_sglang.sh
```

**Server Status:** The server will be available at `http://localhost:30000`

### Step 2: Test Tic-Tac-Toe Environment
Navigate to the environment directory and run the test:

```bash
cd pettingllms/env/tic_tac_toe
python env.py
```

## 4. Log Information

The system automatically generates comprehensive logs during testing:

### Log Files Location:
All logs are stored in the `logs/` directory with timestamp suffixes:

```
logs/
├── game_results_YYYYMMDD_HHMMSS.log     # Game outcomes and statistics
├── llm_conversation_YYYYMMDD_HHMMSS.log # LLM interactions and responses
└── game_summary_<model_name>.txt        # Summary of all game rounds
```

### Log Contents:

#### Game Results Log:
- Game initialization status
- Player actions and moves
- Game state after each step
- Final game outcomes (win/draw/incomplete)
- Performance statistics

#### LLM Conversation Log:
- Full prompts sent to the model
- Model responses and generated actions
- Error messages and debugging information
- Request/response timing information

#### Game Summary:
- Aggregated results from multiple game rounds
- Win/loss statistics per model
- Overall performance metrics





## 📖 About The Project

This project is dedicated to exploring the training of Large Language Models (LLMs) in multi-agent environments. Our core approach involves leveraging agent symmetry and self-play techniques to train a single, robust policy for agents in symmetric games. We utilize state-of-the-art models and reinforcement learning algorithms to achieve intelligent and coordinated agent behavior.

## ✨ Core Components

Here is an overview of the current supported features and components of our framework.

* **Symmetry Strategy**
    * [] **Agent Symmetry:** Using self-play to train a single policy for symmetric games.

* **Training Environments**
    * [] Tic-Tac
    * [] Hanabi

* **Supported Models**
    * [] LLM-QWen-7b
    * [] LLM-QWen-0.5b

* **Data Modalities**
    * [] VLM (Vision Language Model)
    * [] LLM (Large Language Model)

* **Reinforcement Learning Algorithms**
    * [] PPO (Proximal Policy Optimization)

## 🗺️ Roadmap

We have an active development roadmap to enhance the capabilities of this framework. Our future goals include:

* [ ] **Model Expansion:**
    * Integrate a wider variety of open-source LLMs.
    * Develop and test novel model architectures specifically for multi-agent tasks.

* [ ] **Modality Enhancement:**
    * Expand to include other sensory modalities like audio or structured data.
    * Improve the fusion techniques between different modalities (e.g., VLM and LLM).

* [ ] **Algorithm Diversification:**
    * Implement and test other multi-agent RL algorithms (e.g., MADDPG, QMIX).
    * Explore curriculum learning and automated environment generation.

* [ ] **Environment Suite Growth:**
    * Add more complex and diverse game environments.
    * Develop environments for co-operative and competitive non-symmetric scenarios.

* [ ] **Evaluation and Benchmarking:**
    * Establish a comprehensive benchmark for evaluating multi-agent LLM performance.
    * Conduct extensive experiments and publish findings.
