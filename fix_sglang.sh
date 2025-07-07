#!/bin/bash

# SGLang Server Launch Script for PettingLLMs
# Usage: bash fix_sglang.sh [MODEL_NAME]
# Example: bash fix_sglang.sh Qwen/Qwen2.5-7B-Instruct

# Default model if no parameter provided
DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Use provided model or default
MODEL_PATH=${1:-$DEFAULT_MODEL}

echo "=================== SGLang Server Setup ==================="
echo "Model: $MODEL_PATH"
echo "Port: 30000"
echo "=========================================================="

# Set CUDA environment variables
export CUDA_HOME="/home/yujie/miniconda3/envs/pettingllms"
export LD_LIBRARY_PATH="/home/yujie/miniconda3/envs/pettingllms/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/yujie/miniconda3/envs/pettingllms/lib:$LD_LIBRARY_PATH"

# Create cudart symbolic link (if not exists)
CUDA_LIB_DIR="/home/yujie/miniconda3/envs/pettingllms/lib"
if [ ! -f "$CUDA_LIB_DIR/libcudart.so" ]; then
    echo "Creating CUDA runtime symbolic link..."
    ln -sf /home/yujie/miniconda3/envs/pettingllms/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 $CUDA_LIB_DIR/libcudart.so
fi

echo "Starting SGLang server..."
echo "Note: Server will be available at http://localhost:30000"
echo "Use Ctrl+C to stop the server"

# Launch SGLang server with recommended parameters
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port 30000 \
    --tp 1 \
    --dtype float16 \
    --trust-remote-code \
    --mem-fraction-static 0.7 \
    --attention-backend triton \
    --disable-cuda-graph 