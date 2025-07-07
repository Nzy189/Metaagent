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


```
