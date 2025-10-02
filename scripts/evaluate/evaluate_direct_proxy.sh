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

 
LAUNCH_FILE="scripts/launch_multi_vllm.sh"


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


CODE_GENERATOR_MODEL=$(read_var "CODE_GENERATOR_MODEL" "$LAUNCH_FILE")
TEST_GENERATOR_MODEL=$(read_var "TEST_GENERATOR_MODEL" "$LAUNCH_FILE")
CODE_PROXY_PORT=$(read_var "CODE_PROXY_PORT" "$LAUNCH_FILE")
TEST_PROXY_PORT=$(read_var "TEST_PROXY_PORT" "$LAUNCH_FILE")

if [ -z "$CODE_GENERATOR_MODEL" ] || [ -z "$TEST_GENERATOR_MODEL" ]; then
    echo "Error: Failed to parse model paths from $LAUNCH_FILE"
    exit 1
fi

if [ -z "$CODE_PROXY_PORT" ] || [ -z "$TEST_PROXY_PORT" ]; then
    echo "Error: Failed to parse proxy ports from $LAUNCH_FILE"
    exit 1
fi

CODE_ADDRESS="127.0.0.1:${CODE_PROXY_PORT}"
TEST_ADDRESS="127.0.0.1:${TEST_PROXY_PORT}"

# Get command-line arguments
SPECIFIC_AGENT=${1:-""}

echo "=== Direct FastAPI Proxy Multi-Agent Evaluation Script ==="
echo "Code Generator Address: $CODE_ADDRESS, Model: $CODE_GENERATOR_MODEL"
echo "Test Generator Address: $TEST_ADDRESS, Model: $TEST_GENERATOR_MODEL"
if [ ! -z "$SPECIFIC_AGENT" ]; then
    echo "Evaluate specified agent: $SPECIFIC_AGENT"
else
    echo "Evaluate all agents"
fi
echo

# Check server status
check_server() {
    local address=$1
    local name=$2
    echo "Checking $name server ($address)..."
    if curl -s "http://$address/v1/models" > /dev/null; then
        echo "✓ $name server is running"
        return 0
    else
        echo "✗ $name server is not running or unreachable"
        return 1
    fi
}

echo "=== Checking multi-agent server status ==="
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
        echo "Error: Code Generator server is not running. Please start ./scripts/launch_multi_vllm.sh first"
        exit 1
    elif [ "$SPECIFIC_AGENT" = "test_generator" ] && [ "$TEST_SERVER_OK" = false ]; then
        echo "Error: Test Generator server is not running. Please start ./scripts/launch_multi_vllm.sh first"
        exit 1
    fi
else
    # Evaluating all agents requires all servers to be running
    if [ "$CODE_SERVER_OK" = false ] || [ "$TEST_SERVER_OK" = false ]; then
        echo "Error: Some servers are not running. Please start ./scripts/launch_multi_vllm.sh first"
        exit 1
    fi
fi

echo "Server checks passed"
echo

# Use async_vllm_code_eval.py for multi-agent cooperative evaluation
evaluate_multi_agents() {
    echo "=== Starting multi-agent cooperative evaluation ==="
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
}

# Run evaluation - perform full multi-agent evaluation regardless of whether a specific agent is specified
if [ ! -z "$SPECIFIC_AGENT" ]; then
    echo "Note: '$SPECIFIC_AGENT' was specified, but multi-agent evaluation requires cooperation of two agents; running full evaluation"
fi

evaluate_multi_agents

echo "=== All evaluations completed ==="
