set -x

export CUDA_VISIBLE_DEVICES=6
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

# 清理函数：关闭所有相关进程
cleanup() {
    echo "正在清理进程..."
    
    # 关闭代理进程
    if [ ! -z "$PROXY_PID" ]; then
        echo "关闭代理进程 $PROXY_PID"
        kill $PROXY_PID 2>/dev/null
    fi
    
    
    
    # 强制关闭所有占用 8101 和 8100 端口的进程
    echo "强制关闭占用端口 8201 和 8200 的进程"
    lsof -ti:8201 | xargs -r kill -9 2>/dev/null
    lsof -ti:8200 | xargs -r kill -9 2>/dev/null
    
    echo "清理完成"
    exit 0
}

# 注册信号处理器
trap cleanup EXIT INT TERM

# 首先清理可能存在的旧进程
echo "清理可能存在的旧进程..."
lsof -ti:8101 | xargs -r kill -9 2>/dev/null
lsof -ti:8100 | xargs -r kill -9 2>/dev/null
sleep 2

# 启动 vLLM 后端在 8201 端口，使用 livecodebench_baseline checkpoint
echo "启动 vLLM 引擎，使用 livecodebench_baseline checkpoint..."
python -m vllm.entrypoints.openai.api_server \
    --model /home/lah003/workspace/verl_efficient/checkpoints/verl_examples/gsm8k/plan_path_single_policy_1.7B_policy_plan_agent_model/global_step_101/actor/checkpoint\
    --host 127.0.0.1 --port 8201 \
    --gpu-memory-utilization 0.9 --tensor-parallel-size 1 \
    --max-model-len 32000 &

VLLM_PID=$!
echo "vLLM 引擎进程 ID: $VLLM_PID"

# 等待 vLLM 引擎启动
echo "等待 vLLM 引擎启动..."
sleep 10

# 检查 vLLM 引擎是否成功启动
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "错误：vLLM 引擎启动失败"
    exit 1
fi

# 启动代理，将 /v1/completions 的 tokens 转换为 token_id:<id>
export VLLM_BACKEND_ADDRESS=127.0.0.1:8201
export PROXY_PORT=8200
echo "启动代理服务..."
python scripts/vllm_token_id_proxy.py &

PROXY_PID=$!
echo "代理服务进程 ID: $PROXY_PID"

# 等待代理进程结束（或接收到信号）
wait $PROXY_PID