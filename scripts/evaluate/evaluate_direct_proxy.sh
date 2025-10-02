#!/bin/bash
# evaluate_direct_proxy.sh - Integrated vLLM Launch and Evaluation Script
# 
# Usage:
#   ./evaluate_direct_proxy.sh [agent_name]
#
# Features:
#   - Automatically launches vLLM services and proxies
#   - Validates service startup before evaluation
#   - Runs multi-agent evaluation
#   - Auto-cleanup on completion or interruption
#
set -x

export CUDA_VISIBLE_DEVICES=0,1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Model configuration
CODE_GENERATOR_MODEL="/home/lah003/workspace/verl_efficient/checkpoints/verl_examples/gsm8k/plan_path_two_policy_1.7B_3_turns_policy_tool_call_agent_model/global_step_101/actor/checkpoint"
TEST_GENERATOR_MODEL="/home/lah003/workspace/verl_efficient/checkpoints/verl_examples/gsm8k/plan_path_two_policy_1.7B_3_turns_policy_plan_agent_model/global_step_101/actor/checkpoint"

# Port configuration
CODE_VLLM_PORT=8201
CODE_PROXY_PORT=8220
TEST_VLLM_PORT=8202
TEST_PROXY_PORT=8210

CODE_ADDRESS="127.0.0.1:${CODE_PROXY_PORT}"
TEST_ADDRESS="127.0.0.1:${TEST_PROXY_PORT}"

# Global PIDs for cleanup
CODE_VLLM_PID=""
TEST_VLLM_PID=""
CODE_PROXY_PID=""
TEST_PROXY_PID=""

# Get command-line arguments
SPECIFIC_AGENT=${1:-""}

# Cleanup function: stop all related processes
cleanup() {
    echo "======================================"
    echo "开始清理所有服务进程..."
    echo "======================================"
    
    # Stop proxy processes
    if [ ! -z "$CODE_PROXY_PID" ]; then
        echo "停止 code_generator proxy 进程 $CODE_PROXY_PID"
        kill $CODE_PROXY_PID 2>/dev/null
    fi
    
    if [ ! -z "$TEST_PROXY_PID" ]; then
        echo "停止 test_generator proxy 进程 $TEST_PROXY_PID"
        kill $TEST_PROXY_PID 2>/dev/null
    fi
    
    # Stop vLLM processes
    if [ ! -z "$CODE_VLLM_PID" ]; then
        echo "停止 code_generator vLLM 进程 $CODE_VLLM_PID"
        kill $CODE_VLLM_PID 2>/dev/null
    fi
    
    if [ ! -z "$TEST_VLLM_PID" ]; then
        echo "停止 test_generator vLLM 进程 $TEST_VLLM_PID"
        kill $TEST_VLLM_PID 2>/dev/null
    fi
    
    # Force kill all processes using the ports
    echo "强制终止占用端口的所有进程"
    for port in $CODE_VLLM_PORT $CODE_PROXY_PORT $TEST_VLLM_PORT $TEST_PROXY_PORT; do
        lsof -ti:$port | xargs -r kill -9 2>/dev/null
    done
    
    echo "清理完成"
}

# Register signal handlers
trap cleanup EXIT INT TERM

echo "======================================"
echo "=== 多代理评估脚本（集成启动） ==="
echo "======================================"
echo "Code Generator Model: $CODE_GENERATOR_MODEL"
echo "Test Generator Model: $TEST_GENERATOR_MODEL"
echo "Code Generator Address: $CODE_ADDRESS"
echo "Test Generator Address: $TEST_ADDRESS"
if [ ! -z "$SPECIFIC_AGENT" ]; then
    echo "指定评估代理: $SPECIFIC_AGENT"
else
    echo "评估所有代理"
fi
echo

# First clean up any existing processes
echo "======================================"
echo "清理已存在的进程..."
echo "======================================"
for port in $CODE_VLLM_PORT $CODE_PROXY_PORT $TEST_VLLM_PORT $TEST_PROXY_PORT; do
    lsof -ti:$port | xargs -r kill -9 2>/dev/null
done
sleep 3

# Launch vLLM services
echo "======================================"
echo "=== 步骤 1: 启动 vLLM 服务 ==="
echo "======================================"

echo "启动 Code Generator vLLM 引擎 (端口 $CODE_VLLM_PORT)..."
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
    --model "$CODE_GENERATOR_MODEL" \
    --served-model-name "$CODE_GENERATOR_MODEL" \
    --host 127.0.0.1 --port $CODE_VLLM_PORT \
    --gpu-memory-utilization 0.8 --tensor-parallel-size 1 \
    --max-model-len 32768 &

CODE_VLLM_PID=$!
echo "Code Generator vLLM 进程 ID: $CODE_VLLM_PID"

echo "启动 Test Generator vLLM 引擎 (端口 $TEST_VLLM_PORT)..."
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model "$TEST_GENERATOR_MODEL" \
    --served-model-name "$TEST_GENERATOR_MODEL" \
    --host 127.0.0.1 --port $TEST_VLLM_PORT \
    --gpu-memory-utilization 0.8 --tensor-parallel-size 1 \
    --max-model-len 32768 &

TEST_VLLM_PID=$!
echo "Test Generator vLLM 进程 ID: $TEST_VLLM_PID"

# Wait for vLLM engines to start
echo "等待 vLLM 引擎启动..."
sleep 15

# Check if vLLM engines started successfully
if ! kill -0 $CODE_VLLM_PID 2>/dev/null; then
    echo "错误: Code Generator vLLM 引擎启动失败"
    exit 1
fi

if ! kill -0 $TEST_VLLM_PID 2>/dev/null; then
    echo "错误: Test Generator vLLM 引擎启动失败"
    exit 1
fi

echo "✓ vLLM 引擎启动成功"
echo

# Launch proxy services
echo "======================================"
echo "=== 步骤 2: 启动代理服务 ==="
echo "======================================"

echo "启动 Code Generator 代理服务 (端口 $CODE_PROXY_PORT)..."
export VLLM_BACKEND_ADDRESS=127.0.0.1:$CODE_VLLM_PORT
export PROXY_PORT=$CODE_PROXY_PORT
python scripts/vllm_token_id_proxy.py &
CODE_PROXY_PID=$!
echo "Code Generator 代理服务进程 ID: $CODE_PROXY_PID"

echo "启动 Test Generator 代理服务 (端口 $TEST_PROXY_PORT)..."
export VLLM_BACKEND_ADDRESS=127.0.0.1:$TEST_VLLM_PORT
export PROXY_PORT=$TEST_PROXY_PORT
python scripts/vllm_token_id_proxy.py &
TEST_PROXY_PID=$!
echo "Test Generator 代理服务进程 ID: $TEST_PROXY_PID"

echo "等待代理服务启动..."
sleep 5
echo

# Validate services
echo "======================================"
echo "=== 步骤 3: 验证服务状态 ==="
echo "======================================"

check_server() {
    local address=$1
    local name=$2
    local max_retries=30
    local retry=0
    
    echo "检查 $name 服务器 ($address)..."
    while [ $retry -lt $max_retries ]; do
        if curl -s "http://$address/v1/models" > /dev/null; then
            echo "✓ $name 服务器运行正常"
            return 0
        fi
        retry=$((retry + 1))
        echo "等待 $name 服务器启动... (尝试 $retry/$max_retries)"
        sleep 2
    done
    
    echo "✗ $name 服务器启动失败或无法访问"
    return 1
}

CODE_SERVER_OK=false
TEST_SERVER_OK=false

if check_server "$CODE_ADDRESS" "Code Generator"; then
    CODE_SERVER_OK=true
fi

if check_server "$TEST_ADDRESS" "Test Generator"; then
    TEST_SERVER_OK=true
fi

# Check required servers based on the specified agent
if [ ! -z "$SPECIFIC_AGENT" ]; then
    if [ "$SPECIFIC_AGENT" = "code_generator" ] && [ "$CODE_SERVER_OK" = false ]; then
        echo "错误: Code Generator 服务器未能成功启动"
        exit 1
    elif [ "$SPECIFIC_AGENT" = "test_generator" ] && [ "$TEST_SERVER_OK" = false ]; then
        echo "错误: Test Generator 服务器未能成功启动"
        exit 1
    fi
else
    # Evaluating all agents requires all servers to be running
    if [ "$CODE_SERVER_OK" = false ] || [ "$TEST_SERVER_OK" = false ]; then
        echo "错误: 部分服务器未能成功启动"
        exit 1
    fi
fi

echo "✓ 所有服务器验证通过"
echo

# Evaluation function
evaluate_multi_agents() {
    echo "======================================"
    echo "=== 步骤 4: 开始多代理协作评估 ==="
    echo "======================================"
    echo "Code Generator: $CODE_ADDRESS -> $CODE_GENERATOR_MODEL"
    echo "Test Generator: $TEST_ADDRESS -> $TEST_GENERATOR_MODEL"
    echo
    
    # Configuration
    CONFIG_PATH="../config/code"
    CONFIG_NAME="code_two_policies"
    config_path="../config/math"
    config_name="math_two_policies"
    config_name_aggretion="math_aggretion" 
    config_path_aggretion="../config/math"
    config_path_code="../config/code"
    config_path_plan_path="../config/stateful"
    config_name_plan_path="plan_path_two_policies"
    config_name_code="code_eval"
    train_data_size=32
    val_data_size=32
    data_dir=~/data/math/model_0
    USE_GRPO="models.model_0.ppo_trainer_config.algorithm.adv_estimator=grpo models.model_0.ppo_trainer_config.actor_rollout_ref.actor.use_kl_loss=False"
    RESOURCE="resource.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.nnodes=1 models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=2"
    DATA="+models.model_0.ppo_trainer_config.data.train_files=$data_dir/text/train.parquet +models.model_0.ppo_trainer_config.data.val_files=$data_dir/text/test.parquet"
        
    python3 -m pettingllms.utils.async_vllm_code_eval \
        --config-path "$config_path_plan_path" --config-name "$config_name_plan_path" \
        +parallel=false \
        enable_thinking=false \
        benchmark="plan_path" \
        data.epoch_size=120 \
        +map_size=10 \
        data.max_prompt_length=20000 \
        data.max_response_length=8192\
        data.resample_freq=4 \
        data.filter_method=std \
        data.filter_ratio=0.5 \
        sample_mode=tree \
        env.max_turns=4 \
        models.model_0.path="$CODE_GENERATOR_MODEL" \
        models.model_1.path="$TEST_GENERATOR_MODEL"
    
    local eval_exit_code=$?
    
    if [ $eval_exit_code -eq 0 ]; then
        echo "======================================"
        echo "✓ 评估成功完成"
        echo "======================================"
    else
        echo "======================================"
        echo "✗ 评估失败，退出代码: $eval_exit_code"
        echo "======================================"
    fi
    
    return $eval_exit_code
}

# Run evaluation - perform full multi-agent evaluation regardless of whether a specific agent is specified
if [ ! -z "$SPECIFIC_AGENT" ]; then
    echo "注意: 已指定 '$SPECIFIC_AGENT'，但多代理评估需要两个代理协作；将运行完整评估"
fi

evaluate_multi_agents

EVAL_RESULT=$?

echo "======================================"
echo "=== 所有评估已完成 ==="
echo "======================================"

# Exit with the evaluation result code
exit $EVAL_RESULT
