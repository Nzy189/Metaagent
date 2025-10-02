#!/bin/bash
# launch_multi_vllm.sh - Multi-Agent LLM Launch Script
# 
# Usage:
#   ./launch_multi_vllm.sh
#
# Features:
#   - Launches code_generator model on port 8201
#   - Launches test_generator model on port 8202
#   - Each model uses independent proxy service (ports 8200 and 8210)

set -x

export CUDA_VISIBLE_DEVICES=2,3
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Model configuration
CODE_GENERATOR_MODEL="/home/lah003/workspace/verl_efficient/checkpoints/verl_examples/gsm8k/plan_path_two_policy_1.7B_3_turns_policy_tool_call_agent_model/global_step_101/actor/checkpoint"
TEST_GENERATOR_MODEL="/home/lah003/workspace/verl_efficient/checkpoints/verl_examples/gsm8k/plan_path_two_policy_1.7B_3_turns_policy_plan_agent_model/global_step_101/actor/checkpoint"  # can be changed to a different model

# Port configuration
CODE_VLLM_PORT=8201
CODE_PROXY_PORT=8220
TEST_VLLM_PORT=8202
TEST_PROXY_PORT=8210

# Cleanup function: stop all related processes
cleanup() {
    echo "Cleaning up processes..."
    
    # Stop proxy processes
    if [ ! -z "$CODE_PROXY_PID" ]; then
        echo "Stopping code_generator proxy process $CODE_PROXY_PID"
        kill $CODE_PROXY_PID 2>/dev/null
    fi
    
    if [ ! -z "$TEST_PROXY_PID" ]; then
        echo "Stopping test_generator proxy process $TEST_PROXY_PID"
        kill $TEST_PROXY_PID 2>/dev/null
    fi
    
    # Stop vLLM processes
    if [ ! -z "$CODE_VLLM_PID" ]; then
        echo "Stopping code_generator vLLM process $CODE_VLLM_PID"
        kill $CODE_VLLM_PID 2>/dev/null
    fi
    
    if [ ! -z "$TEST_VLLM_PID" ]; then
        echo "Stopping test_generator vLLM process $TEST_VLLM_PID"
        kill $TEST_VLLM_PID 2>/dev/null
    fi
    
    # Force kill all processes using the ports
    echo "Force killing processes using the ports"
    for port in $CODE_VLLM_PORT $CODE_PROXY_PORT $TEST_VLLM_PORT $TEST_PROXY_PORT; do
        lsof -ti:$port | xargs -r kill -9 2>/dev/null
    done
    
    echo "Cleanup complete"
    exit 0
}

# Register signal handlers
trap cleanup EXIT INT TERM

# First clean up any existing processes
echo "Cleaning up existing processes..."
for port in $CODE_VLLM_PORT $CODE_PROXY_PORT $TEST_VLLM_PORT $TEST_PROXY_PORT; do
    lsof -ti:$port | xargs -r kill -9 2>/dev/null
done
sleep 3

echo "=== Starting Code Generator vLLM Engine (port $CODE_VLLM_PORT) ==="
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
    --model "$CODE_GENERATOR_MODEL" \
    --served-model-name "$CODE_GENERATOR_MODEL" \
    --host 127.0.0.1 --port $CODE_VLLM_PORT \
    --gpu-memory-utilization 0.8 --tensor-parallel-size 1 \
    --max-model-len 32768 &

CODE_VLLM_PID=$!
echo "Code Generator vLLM Engine process ID: $CODE_VLLM_PID"

echo "=== Starting Test Generator vLLM Engine (port $TEST_VLLM_PORT) ==="
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model "$TEST_GENERATOR_MODEL" \
    --served-model-name "$TEST_GENERATOR_MODEL" \
    --host 127.0.0.1 --port $TEST_VLLM_PORT \
    --gpu-memory-utilization 0.8 --tensor-parallel-size 1 \
    --max-model-len 32768 &

TEST_VLLM_PID=$!
echo "Test Generator vLLM Engine process ID: $TEST_VLLM_PID"

# Wait for vLLM engines to start
echo "Waiting for vLLM engines to start..."
sleep 15

# Check if vLLM engines started successfully
if ! kill -0 $CODE_VLLM_PID 2>/dev/null; then
    echo "Error: Code Generator vLLM Engine failed to start"
    exit 1
fi

if ! kill -0 $TEST_VLLM_PID 2>/dev/null; then
    echo "Error: Test Generator vLLM Engine failed to start"
    exit 1
fi

echo "=== Starting Code Generator Proxy Service (port $CODE_PROXY_PORT) ==="
export VLLM_BACKEND_ADDRESS=127.0.0.1:$CODE_VLLM_PORT
export PROXY_PORT=$CODE_PROXY_PORT
python scripts/vllm_token_id_proxy.py &
CODE_PROXY_PID=$!
echo "Code Generator Proxy Service process ID: $CODE_PROXY_PID"

echo "=== Starting Test Generator Proxy Service (port $TEST_PROXY_PORT) ==="
export VLLM_BACKEND_ADDRESS=127.0.0.1:$TEST_VLLM_PORT
export PROXY_PORT=$TEST_PROXY_PORT
python scripts/vllm_token_id_proxy.py &
TEST_PROXY_PID=$!
echo "Test Generator Proxy Service process ID: $TEST_PROXY_PID"

echo "=== All Services Started ==="
echo "Code Generator: vLLM port $CODE_VLLM_PORT, proxy port $CODE_PROXY_PORT"
echo "Test Generator: vLLM port $TEST_VLLM_PORT, proxy port $TEST_PROXY_PORT"
echo "Press Ctrl+C to stop all services"

# Export service address environment variable for other scripts
export VLLM_SERVICE_ADDRESS="127.0.0.1:$CODE_PROXY_PORT"
echo "Environment variable set: VLLM_SERVICE_ADDRESS=$VLLM_SERVICE_ADDRESS"

# Wait for all processes
wait
