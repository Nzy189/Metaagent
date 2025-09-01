#!/bin/bash

# 设置环境变量
source set_env.sh

# 清理triton缓存
rm -rf /tmp/triton_cache
mkdir -p /tmp/triton_cache

# 确保CUDA初始化
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# 运行训练
echo "开始训练..."
python -m pettingllms.trainer.train "$@"
