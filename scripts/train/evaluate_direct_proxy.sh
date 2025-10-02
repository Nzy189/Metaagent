#!/bin/bash
# evaluate_direct_proxy.sh - 直接调用FastAPI代理的多Agent评估脚本
# 
# 用法:
#   ./evaluate_direct_proxy.sh [AGENT_NAME]
#
# 参数:
#   AGENT_NAME  可选，指定要评估的agent名称
#               可选值: "code_generator", "test_generator"  
#               如果不指定，将评估所有agent
#
# 前提条件:
#   需要先运行 ./scripts/launch_multi_vllm.sh 启动两个vLLM服务器

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

# 关联的启动脚本（从中解析模型路径与端口）
LAUNCH_FILE="scripts/launch_multi_vllm.sh"

# 从启动脚本解析变量值
read_var() {
    local var_name="$1"
    local file_path="$2"
    local line
    line=$(grep -E "^${var_name}=" "$file_path" | head -n1)
    if [ -z "$line" ]; then
        echo ""; return 1
    fi
    echo "$line" | sed -E 's/^[^=]+=\s*"?([^"#]+)"?.*/\1/' | tr -d '\n'
}

# 解析模型与端口
CODE_GENERATOR_MODEL=$(read_var "CODE_GENERATOR_MODEL" "$LAUNCH_FILE")
TEST_GENERATOR_MODEL=$(read_var "TEST_GENERATOR_MODEL" "$LAUNCH_FILE")
CODE_PROXY_PORT=$(read_var "CODE_PROXY_PORT" "$LAUNCH_FILE")
TEST_PROXY_PORT=$(read_var "TEST_PROXY_PORT" "$LAUNCH_FILE")

if [ -z "$CODE_GENERATOR_MODEL" ] || [ -z "$TEST_GENERATOR_MODEL" ]; then
    echo "错误: 无法从 $LAUNCH_FILE 解析模型路径"
    exit 1
fi

if [ -z "$CODE_PROXY_PORT" ] || [ -z "$TEST_PROXY_PORT" ]; then
    echo "错误: 无法从 $LAUNCH_FILE 解析代理端口"
    exit 1
fi

CODE_ADDRESS="127.0.0.1:${CODE_PROXY_PORT}"
TEST_ADDRESS="127.0.0.1:${TEST_PROXY_PORT}"

# 获取命令行参数
SPECIFIC_AGENT=${1:-""}

echo "=== 直接FastAPI代理多Agent评估脚本 ==="
echo "Code Generator 地址: $CODE_ADDRESS, 模型: $CODE_GENERATOR_MODEL"
echo "Test Generator 地址: $TEST_ADDRESS, 模型: $TEST_GENERATOR_MODEL"
if [ ! -z "$SPECIFIC_AGENT" ]; then
    echo "指定评估Agent: $SPECIFIC_AGENT"
else
    echo "评估所有Agent"
fi
echo

# 检查服务器状态
check_server() {
    local address=$1
    local name=$2
    echo "检查 $name 服务器 ($address)..."
    if curl -s "http://$address/v1/models" > /dev/null; then
        echo "✓ $name 服务器运行正常"
        return 0
    else
        echo "✗ $name 服务器未运行或无法访问"
        return 1
    fi
}

echo "=== 检查服务器状态 ==="
CODE_SERVER_OK=false
TEST_SERVER_OK=false

if check_server "$CODE_ADDRESS" "Code Generator"; then
    CODE_SERVER_OK=true
fi

if check_server "$TEST_ADDRESS" "Test Generator"; then
    TEST_SERVER_OK=true
fi

# 根据指定的agent检查必要的服务器
if [ ! -z "$SPECIFIC_AGENT" ]; then
    if [ "$SPECIFIC_AGENT" = "code_generator" ] && [ "$CODE_SERVER_OK" = false ]; then
        echo "错误: Code Generator服务器未运行，请先启动 ./scripts/launch_multi_vllm.sh"
        exit 1
    elif [ "$SPECIFIC_AGENT" = "test_generator" ] && [ "$TEST_SERVER_OK" = false ]; then
        echo "错误: Test Generator服务器未运行，请先启动 ./scripts/launch_multi_vllm.sh"
        exit 1
    fi
else
    # 评估所有agent时需要所有服务器都运行
    if [ "$CODE_SERVER_OK" = false ] || [ "$TEST_SERVER_OK" = false ]; then
        echo "错误: 部分服务器未运行，请先启动 ./scripts/launch_multi_vllm.sh"
        exit 1
    fi
fi

echo "服务器检查通过"
echo

# 使用 async_vllm_code_eval.py 进行多Agent协作评估
evaluate_multi_agents() {
    echo "=== 开始多Agent协作评估 ==="
    echo "Code Generator: $CODE_ADDRESS -> $CODE_GENERATOR_MODEL"
    echo "Test Generator: $TEST_ADDRESS -> $TEST_GENERATOR_MODEL"
    echo
    
    # 配置参数
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
        
    python3 -m pettingllms.scripts.async_vllm_code_eval \
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
}

# 执行评估 - 不管是否指定特定agent，都进行完整的多agent评估
if [ ! -z "$SPECIFIC_AGENT" ]; then
    echo "注意: 指定了 '$SPECIFIC_AGENT'，但多Agent评估需要两个agent协作，将进行完整评估"
fi

evaluate_multi_agents

echo "=== 所有评估完成 ==="
