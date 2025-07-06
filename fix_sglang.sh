#!/bin/bash

# 设置 CUDA 环境变量
export CUDA_HOME="/home/yujie/miniconda3/envs/pettingllms"
export LD_LIBRARY_PATH="/home/yujie/miniconda3/envs/pettingllms/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/yujie/miniconda3/envs/pettingllms/lib:$LD_LIBRARY_PATH"

# 创建 cudart 符号链接（如果不存在）
CUDA_LIB_DIR="/home/yujie/miniconda3/envs/pettingllms/lib"
if [ ! -f "$CUDA_LIB_DIR/libcudart.so" ]; then
    ln -sf /home/yujie/miniconda3/envs/pettingllms/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 $CUDA_LIB_DIR/libcudart.so
fi

# 启动 SGLang 服务器，使用推荐的参数
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --port 30000 \
    --tp 1 \
    --dtype float16 \
    --trust-remote-code \
    --mem-fraction-static 0.7 \
    --attention-backend triton \
    --disable-cuda-graph 