set -x

export CUDA_VISIBLE_DEVICES=4
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

# 通用配置
train_data_size=32
val_data_size=32
data_dir=~/data/math/model_0
USE_GRPO="models.model_0.ppo_trainer_config.algorithm.adv_estimator=grpo models.model_0.ppo_trainer_config.actor_rollout_ref.actor.use_kl_loss=False"
RESOURCE="resource.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.nnodes=1 models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=2"
DATA="+models.model_0.ppo_trainer_config.data.train_files=$data_dir/text/train.parquet +models.model_0.ppo_trainer_config.data.val_files=$data_dir/text/test.parquet"

# 模型列表
models=(
  "/home/lah003/models/Qwen3-4B"
)

# 任务配置映射
declare -A task_config_map
task_config_map["math"]="../config/math math_single_policy"
task_config_map["code"]="../config/code code_eval"
task_config_map["plan_path"]="../config/stateful plan_path_single_policy"

# 基准测试映射
declare -A benchmark_map
benchmark_map["math"]="AIME24 AIME25 OlympiadBench_test gsm8k_test"
benchmark_map["code"]="apps livecodebench code_contests"
benchmark_map["plan_path"]="plan_path_benchmark"

# 推理模式
reasoning_modes=("false")

# 任务列表
tasks=("math" "code")
max_turns=("1" "3" "5")
# 记录测试结果的函数
log_test_info() {
    local task=$1
    local benchmark=$2
    local reasoning=$3
    local model=$4
    local max_turns=$5
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 简单直接地追加记录到JSON文件
    echo "{\"timestamp\":\"$timestamp\",\"task\":\"$task\",\"benchmark\":\"$benchmark\",\"reasoning\":$reasoning,\"model\":\"$model\",\"max_turns\":$max_turns}" >> success_rollout_rate_dict.json
    
    # 同时记录到TXT文件，格式更易读
    echo "[$timestamp] Task: $task | Benchmark: $benchmark | Reasoning: $reasoning | Model: $model | MaxTurns: $max_turns" >> success_rollout_rate_dict.txt
    
    echo "已记录测试信息: Task=$task, Benchmark=$benchmark, Reasoning=$reasoning, Model=$model, MaxTurns=$max_turns"
}

# 主循环：遍历所有组合
for task in "${tasks[@]}"; do
    echo "=== 开始测试任务: $task ==="
    
    # 获取任务对应的配置
    config_info=${task_config_map[$task]}
    config_path=$(echo $config_info | cut -d' ' -f1)
    config_name=$(echo $config_info | cut -d' ' -f2)
    
    # 获取该任务对应的基准测试列表
    benchmarks=${benchmark_map[$task]}
    
    for benchmark in $benchmarks; do
        echo "=== 测试基准: $benchmark ==="
        
        for reasoning in "${reasoning_modes[@]}"; do
            echo "=== 推理模式: $reasoning ==="
            
            for model in "${models[@]}"; do
                echo "=== 评估模型: $model ==="
                for max_turn in "${max_turns[@]}"; do
                  # 在测试开始前记录信息
                  log_test_info "$task" "$benchmark" "$reasoning" "$model" "$max_turn"
                  
                  # 根据任务类型设置特定参数
                  if [ "$task" == "math" ]; then
                      python3 -m pettingllms.scripts.async_vllm_code_eval \
                          --config-path "$config_path" --config-name "$config_name" \
                          $USE_GRPO $RESOURCE $DATA \
                          +difficulty=test \
                          +parallel=false \
                          enable_thinking=$reasoning \
                          models.model_0.path="$model" \
                          benchmark="$benchmark" \
                          data.epoch_size=120 \
                          data.max_prompt_length=8192 \
                          data.max_response_length=8192\
                          data.resample_freq=4 \
                          data.filter_method=std \
                          data.filter_ratio=0 \
                          sample_mode=tree \
                          env.max_turns=$max_turn
                  elif [ "$task" == "code" ]; then
                      python3 -m pettingllms.scripts.async_vllm_code_eval \
                          --config-path "$config_path" --config-name "$config_name" \
                          $USE_GRPO $RESOURCE $DATA \
                          +difficulty=test \
                          +parallel=false \
                          enable_thinking=$reasoning \
                          models.model_0.path="$model" \
                          benchmark="$benchmark" \
                          data.epoch_size=120 \
                          data.max_prompt_length=8192 \
                          data.max_response_length=8192 \
                          data.resample_freq=4 \
                          data.filter_method=std \
                          data.filter_ratio=0 \
                          sample_mode=tree \
                          env.max_turns=$max_turn
                 
            
                 
                  fi
                  
                  echo "完成评估: Task=$task, Benchmark=$benchmark, Reasoning=$reasoning, Model=$model, MaxTurns=$max_turn"
                  echo "----------------------------------------"
                done
                  
            done
        done
    done
done

echo "=== 所有测试完成 ==="