#!/bin/bash

# Quick Test Script for Frontend Design Agent Graph
# 快速测试脚本

set -e

# 配置
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CODE_PORT=8000
VISUAL_PORT=8001

echo "🚀 启动快速测试..."

# 清理现有进程
echo "🧹 清理现有进程..."
pkill -f "sglang.*--port.*(8000|8001)" || true
sleep 2

# 启动两个模型服务器
echo "📡 启动代码生成服务器 (端口: $CODE_PORT)..."
nohup python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --port $CODE_PORT \
    --host localhost \
    --trust-remote-code \
    --dtype auto > code_server.log 2>&1 &
CODE_PID=$!

echo "📡 启动视觉分析服务器 (端口: $VISUAL_PORT)..."
nohup python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --port $VISUAL_PORT \
    --host localhost \
    --trust-remote-code \
    --dtype auto > visual_server.log 2>&1 &
VISUAL_PID=$!

# 清理函数
cleanup() {
    echo "🧹 清理进程..."
    kill $CODE_PID $VISUAL_PID 2>/dev/null || true
    pkill -f "sglang.*--port.*(8000|8001)" || true
}
trap cleanup EXIT

# 等待服务器启动
echo "⏳ 等待服务器启动 (60秒)..."
sleep 60

# 测试连接
echo "🔗 测试服务器连接..."
if curl -s "http://localhost:$CODE_PORT/health" >/dev/null && \
   curl -s "http://localhost:$VISUAL_PORT/health" >/dev/null; then
    echo "✅ 服务器连接正常"
else
    echo "❌ 服务器连接失败"
    echo "代码服务器日志:"
    tail -20 code_server.log
    echo "视觉服务器日志:"
    tail -20 visual_server.log
    exit 1
fi

# 运行测试
echo "🧪 运行graph测试..."
python agent_collaboration_graph.py \
    --hostname localhost \
    --code_port $CODE_PORT \
    --visual_port $VISUAL_PORT \
    --num_samples 2 \
    --max_iterations 2 \
    --output_path "quick_test_results.json"

echo "🎉 快速测试完成！"
echo "查看结果: quick_test_results.json" 