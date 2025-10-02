#!/bin/bash

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 多GPU并行验证脚本启动 ===${NC}"

# 工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# GPU配置
declare -A GPU_CONFIGS
GPU_CONFIGS[0]="0,8100,8101,/home/lah003/models/Qwen3-1.7B"
GPU_CONFIGS[1]="1,8102,8103,/home/lah003/models/Qwen3-1.7B" 
GPU_CONFIGS[2]="2,8104,8105,/home/lah003/models/Qwen3-8B"

# 存储进程ID
declare -A LAUNCH_PIDS
declare -A VALIDATE_PIDS

# 清理函数
cleanup() {
    echo -e "${YELLOW}正在清理所有进程...${NC}"
    
    # 停止所有validate进程
    for gpu in "${!VALIDATE_PIDS[@]}"; do
        if [ ! -z "${VALIDATE_PIDS[$gpu]}" ]; then
            echo -e "${YELLOW}停止GPU $gpu 的validate进程 ${VALIDATE_PIDS[$gpu]}${NC}"
            kill -TERM "${VALIDATE_PIDS[$gpu]}" 2>/dev/null || true
        fi
    done
    
    # 停止所有launch进程
    for gpu in "${!LAUNCH_PIDS[@]}"; do
        if [ ! -z "${LAUNCH_PIDS[$gpu]}" ]; then
            echo -e "${YELLOW}停止GPU $gpu 的launch进程 ${LAUNCH_PIDS[$gpu]}${NC}"
            kill -TERM "${LAUNCH_PIDS[$gpu]}" 2>/dev/null || true
        fi
    done
    
    # 强制清理端口
    for port in 8100 8101 8102 8103 8104 8105; do
        lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
    done
    
    echo -e "${GREEN}清理完成${NC}"
    exit 0
}

# 注册信号处理器
trap cleanup EXIT INT TERM

# 创建GPU特定的launch脚本
create_launch_script() {
    local gpu=$1
    local cuda_device=$2
    local vllm_port=$3
    local proxy_port=$4
    local model_path=$5
    
    local script_name="validate_launch_gpu${gpu}.sh"
    
    cat > "$SCRIPT_DIR/$script_name" << EOF
#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$cuda_device
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

export CUDA_HOME=\${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=\$CUDA_HOME/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# 清理函数
cleanup() {
    echo "GPU $gpu: 正在清理进程..."
    
    if [ ! -z "\$PROXY_PID" ]; then
        echo "GPU $gpu: 关闭代理进程 \$PROXY_PID"
        kill \$PROXY_PID 2>/dev/null || true
    fi
    
    if [ ! -z "\$VLLM_PID" ]; then
        echo "GPU $gpu: 关闭 vLLM 引擎进程 \$VLLM_PID"
        kill \$VLLM_PID 2>/dev/null || true
    fi
    
    lsof -ti:$vllm_port | xargs -r kill -9 2>/dev/null || true
    lsof -ti:$proxy_port | xargs -r kill -9 2>/dev/null || true
    
    echo "GPU $gpu: 清理完成"
    exit 0
}

trap cleanup EXIT INT TERM

# 清理旧进程
echo "GPU $gpu: 清理可能存在的旧进程..."
lsof -ti:$vllm_port | xargs -r kill -9 2>/dev/null || true
lsof -ti:$proxy_port | xargs -r kill -9 2>/dev/null || true
sleep 2

# 启动 vLLM 后端
echo "GPU $gpu: 启动 vLLM 引擎在端口 $vllm_port..."
python -m vllm.entrypoints.openai.api_server \\
    --model $model_path \\
    --host 127.0.0.1 --port $vllm_port \\
    --gpu-memory-utilization 0.9 --tensor-parallel-size 1 \\
    --max-model-len 32768 &

VLLM_PID=\$!
echo "GPU $gpu: vLLM 引擎进程 ID: \$VLLM_PID"

# 等待 vLLM 引擎启动
echo "GPU $gpu: 等待 vLLM 引擎启动..."
sleep 15

# 检查 vLLM 引擎是否成功启动
if ! kill -0 \$VLLM_PID 2>/dev/null; then
    echo "GPU $gpu: 错误：vLLM 引擎启动失败"
    exit 1
fi

# 启动代理
export VLLM_BACKEND_ADDRESS=127.0.0.1:$vllm_port
export PROXY_PORT=$proxy_port
echo "GPU $gpu: 启动代理服务在端口 $proxy_port..."
python scripts/vllm_token_id_proxy.py &

PROXY_PID=\$!
echo "GPU $gpu: 代理服务进程 ID: \$PROXY_PID"

# 等待代理进程结束
wait \$PROXY_PID
EOF

    chmod +x "$SCRIPT_DIR/$script_name"
    echo "$script_name"
}

# 创建GPU特定的validate脚本
create_validate_script() {
    local gpu=$1
    local cuda_device=$2
    local model_path=$3
    local proxy_port=$4
    
    local script_name="validate_gpu${gpu}.sh"
    
    cat > "$SCRIPT_DIR/$script_name" << EOF
#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$cuda_device
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

export CUDA_HOME=\${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=\$CUDA_HOME/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# 通用配置
train_data_size=32
val_data_size=32
data_dir=~/data/math/model_0
USE_GRPO="models.model_0.ppo_trainer_config.algorithm.adv_estimator=grpo models.model_0.ppo_trainer_config.actor_rollout_ref.actor.use_kl_loss=False"
RESOURCE="resource.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.nnodes=1 models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=2"
DATA="+models.model_0.ppo_trainer_config.data.train_files=\$data_dir/text/train.parquet +models.model_0.ppo_trainer_config.data.val_files=\$data_dir/text/test.parquet"

# 模型列表
models=(
  "$model_path"
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
reasoning_modes=("true" "false")

# 任务列表
tasks=("math" "code")
max_turns=("1" "3" "5")

# 记录测试结果的函数
log_test_info() {
    local task=\$1
    local benchmark=\$2
    local reasoning=\$3
    local model=\$4
    local max_turns=\$5
    
    local timestamp=\$(date '+%Y-%m-%d %H:%M:%S')
    local gpu_info="GPU$gpu"
    
    # 简单直接地追加记录到JSON文件
    echo "{\\"timestamp\\":\\"\\$timestamp\\",\\"gpu\\":\\"\\$gpu_info\\",\\"task\\":\\"\\$task\\",\\"benchmark\\":\\"\\$benchmark\\",\\"reasoning\\":\\$reasoning,\\"model\\":\\"\\$model\\",\\"max_turns\\":\\$max_turns}" >> success_rollout_rate_dict_gpu${gpu}.json
    
    # 同时记录到TXT文件，格式更易读
    echo "[\\$timestamp] GPU$gpu | Task: \\$task | Benchmark: \\$benchmark | Reasoning: \\$reasoning | Model: \\$model | MaxTurns: \\$max_turns" >> success_rollout_rate_dict_gpu${gpu}.txt
    
    echo "GPU$gpu: 已记录测试信息: Task=\\$task, Benchmark=\\$benchmark, Reasoning=\\$reasoning, Model=\\$model, MaxTurns=\\$max_turns"
}

# 主循环：遍历所有组合
for task in "\${tasks[@]}"; do
    echo "GPU$gpu: === 开始测试任务: \\$task ==="
    
    # 获取任务对应的配置
    config_info=\${task_config_map[\$task]}
    config_path=\$(echo \$config_info | cut -d' ' -f1)
    config_name=\$(echo \$config_info | cut -d' ' -f2)
    
    # 获取该任务对应的基准测试列表
    benchmarks=\${benchmark_map[\$task]}
    
    for benchmark in \$benchmarks; do
        echo "GPU$gpu: === 测试基准: \\$benchmark ==="
        
        for reasoning in "\${reasoning_modes[@]}"; do
            echo "GPU$gpu: === 推理模式: \\$reasoning ==="
            
            for model in "\${models[@]}"; do
                echo "GPU$gpu: === 评估模型: \\$model ==="
                for max_turn in "\${max_turns[@]}"; do
                  # 在测试开始前记录信息
                  log_test_info "\\$task" "\\$benchmark" "\\$reasoning" "\\$model" "\\$max_turn"
                  
                  # 根据任务类型设置特定参数
                  if [ "\\$task" == "math" ]; then
                      python3 -m pettingllms.scripts.async_vllm_code_eval \\
                          --config-path "\\$config_path" --config-name "\\$config_name" \\
                          \\$USE_GRPO \\$RESOURCE \\$DATA \\
                          +difficulty=test \\
                          +parallel=false \\
                          enable_thinking=\\$reasoning \\
                          models.model_0.path="\\$model" \\
                          benchmark="\\$benchmark" \\
                          data.epoch_size=120 \\
                          data.max_prompt_length=8192 \\
                          data.max_response_length=2048 \\
                          data.resample_freq=4 \\
                          data.filter_method=std \\
                          data.filter_ratio=0 \\
                          sample_mode=tree \\
                          env.max_turns=\\$max_turn
                  elif [ "\\$task" == "code" ]; then
                      python3 -m pettingllms.scripts.async_vllm_code_eval \\
                          --config-path "\\$config_path" --config-name "\\$config_name" \\
                          \\$USE_GRPO \\$RESOURCE \\$DATA \\
                          +difficulty=test \\
                          +parallel=false \\
                          enable_thinking=\\$reasoning \\
                          models.model_0.path="\\$model" \\
                          benchmark="\\$benchmark" \\
                          data.epoch_size=120 \\
                          data.max_prompt_length=4096 \\
                          data.max_response_length=4096 \\
                          data.resample_freq=4 \\
                          data.filter_method=std \\
                          data.filter_ratio=0 \\
                          sample_mode=tree \\
                          env.max_turns=\\$max_turn
                  elif [ "\\$task" == "plan_path" ]; then
                      python3 -m pettingllms.scripts.async_vllm_code_eval \\
                          --config-path "\\$config_path" --config-name "\\$config_name" \\
                          \\$USE_GRPO \\$RESOURCE \\$DATA \\
                          +difficulty=test \\
                          +parallel=false \\
                          enable_thinking=\\$reasoning \\
                          models.model_0.path="\\$model" \\
                          benchmark="\\$benchmark" \\
                          data.epoch_size=120 \\
                          data.max_prompt_length=8192 \\
                          data.max_response_length=2048 \\
                          data.resample_freq=4 \\
                          data.filter_method=std \\
                          data.filter_ratio=0 \\
                          sample_mode=tree \\
                          env.max_turns=\\$max_turn
                  fi
                  
                  echo "GPU$gpu: 完成评估: Task=\\$task, Benchmark=\\$benchmark, Reasoning=\\$reasoning, Model=\\$model, MaxTurns=\\$max_turn"
                  echo "GPU$gpu: ----------------------------------------"
                done
                  
            done
        done
    done
done

echo "GPU$gpu: === 所有测试完成 ==="
EOF

    chmod +x "$SCRIPT_DIR/$script_name"
    echo "$script_name"
}

# 主执行流程
echo -e "${BLUE}开始创建GPU特定的脚本...${NC}"

# 为每个GPU创建脚本
for gpu in "${!GPU_CONFIGS[@]}"; do
    IFS=',' read -r cuda_device vllm_port proxy_port model_path <<< "${GPU_CONFIGS[$gpu]}"
    
    echo -e "${BLUE}为GPU $gpu 创建脚本 (CUDA_DEVICE=$cuda_device, VLLM_PORT=$vllm_port, PROXY_PORT=$proxy_port, MODEL=$model_path)${NC}"
    
    launch_script=$(create_launch_script "$gpu" "$cuda_device" "$vllm_port" "$proxy_port" "$model_path")
    validate_script=$(create_validate_script "$gpu" "$cuda_device" "$model_path" "$proxy_port")
    
    echo -e "${GREEN}GPU $gpu 脚本创建完成: $launch_script, $validate_script${NC}"
done

echo -e "${BLUE}开始启动所有GPU的launch脚本...${NC}"

# 启动所有launch脚本
for gpu in "${!GPU_CONFIGS[@]}"; do
    launch_script="validate_launch_gpu${gpu}.sh"
    echo -e "${BLUE}启动GPU $gpu 的launch脚本: $launch_script${NC}"
    
    cd "$WORKSPACE_DIR"
    bash "$SCRIPT_DIR/$launch_script" > "launch_gpu${gpu}.log" 2>&1 &
    LAUNCH_PIDS[$gpu]=$!
    
    echo -e "${GREEN}GPU $gpu launch脚本已启动，PID: ${LAUNCH_PIDS[$gpu]}${NC}"
done

# 等待所有launch脚本启动完成
echo -e "${YELLOW}等待所有launch脚本启动完成 (60秒)...${NC}"
sleep 60

# 检查所有launch进程是否还在运行
for gpu in "${!LAUNCH_PIDS[@]}"; do
    if ! kill -0 "${LAUNCH_PIDS[$gpu]}" 2>/dev/null; then
        echo -e "${RED}警告: GPU $gpu 的launch进程已退出${NC}"
    else
        echo -e "${GREEN}GPU $gpu 的launch进程运行正常${NC}"
    fi
done

echo -e "${BLUE}开始启动所有GPU的validate脚本...${NC}"

# 启动所有validate脚本
for gpu in "${!GPU_CONFIGS[@]}"; do
    validate_script="validate_gpu${gpu}.sh"
    echo -e "${BLUE}启动GPU $gpu 的validate脚本: $validate_script${NC}"
    
    cd "$WORKSPACE_DIR"
    bash "$SCRIPT_DIR/$validate_script" > "validate_gpu${gpu}.log" 2>&1 &
    VALIDATE_PIDS[$gpu]=$!
    
    echo -e "${GREEN}GPU $gpu validate脚本已启动，PID: ${VALIDATE_PIDS[$gpu]}${NC}"
done

echo -e "${GREEN}所有GPU的validate脚本已启动，等待完成...${NC}"
echo -e "${YELLOW}你可以通过以下命令查看各GPU的日志:${NC}"
for gpu in "${!GPU_CONFIGS[@]}"; do
    echo -e "${YELLOW}  GPU $gpu launch日志: tail -f launch_gpu${gpu}.log${NC}"
    echo -e "${YELLOW}  GPU $gpu validate日志: tail -f validate_gpu${gpu}.log${NC}"
done

# 等待所有validate进程完成
for gpu in "${!VALIDATE_PIDS[@]}"; do
    echo -e "${BLUE}等待GPU $gpu 的validate进程完成...${NC}"
    wait "${VALIDATE_PIDS[$gpu]}"
    echo -e "${GREEN}GPU $gpu 的validate进程已完成${NC}"
done

echo -e "${GREEN}=== 所有GPU测试完成 ===${NC}"

# 显示结果文件
echo -e "${BLUE}结果文件:${NC}"
for gpu in "${!GPU_CONFIGS[@]}"; do
    if [ -f "success_rollout_rate_dict_gpu${gpu}.json" ]; then
        echo -e "${GREEN}  GPU $gpu JSON结果: success_rollout_rate_dict_gpu${gpu}.json${NC}"
    fi
    if [ -f "success_rollout_rate_dict_gpu${gpu}.txt" ]; then
        echo -e "${GREEN}  GPU $gpu TXT结果: success_rollout_rate_dict_gpu${gpu}.txt${NC}"
    fi
done
