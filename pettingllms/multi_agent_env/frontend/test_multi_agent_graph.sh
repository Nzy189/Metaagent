#!/bin/bash

# 简化版前端设计智能体图测试脚本
# 只包含启动SGLang和执行Python脚本的核心功能

set -e  # 如果任何命令失败则退出


VENV_PATH="pettingllms"



# 配置参数
HOSTNAME="localhost"
CODE_PORT=8000
VISUAL_PORT=8000
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_SAMPLES=5
MAX_ITERATIONS=3
OUTPUT_DIR="./test_results"
TEMP_DIR="./temp"


# 设置Python路径
PYTHON_CMD="python"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "[INFO] $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查端口占用并清理
cleanup_port() {
    local port=$1
    echo "清理端口 $port..."
    pkill -f ".*python.*sglang.*--port.*$port" 2>/dev/null || true
    sleep 2
}

# 启动SGLang服务器
start_sglang_server() {
    local port=$1
    local server_name=$2
    
    echo "启动 $server_name SGLang服务器 (端口: $port)..."
    
    # 清理端口
    cleanup_port $port
    
    # 启动服务器
    local cmd="$PYTHON_CMD -m sglang.launch_server \
        --model-path $MODEL_NAME \
        --port $port \
        --host $HOSTNAME \
        --trust-remote-code \
        --dtype auto\
        --mem-fraction-static 0.15"
    
    echo "执行命令: $cmd"
    $cmd &
    
    local pid=$!
    echo "服务器进程已启动 (PID: $pid)"
    
    # 等待服务器启动
    echo "等待服务器启动..."
    local max_wait=60
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        if curl -s --max-time 3 "http://$HOSTNAME:$port/health" >/dev/null 2>&1; then
            log_success "$server_name 服务器启动成功!"
            return 0
        fi
        sleep 3
        wait_time=$((wait_time + 3))
        echo "等待中... (${wait_time}s/${max_wait}s)"
    done
    
    log_error "$server_name 服务器启动超时"
    return 1
}

# 停止SGLang服务器
stop_sglang_servers() {
    echo "停止SGLang服务器..."
    cleanup_port $CODE_PORT
    cleanup_port $VISUAL_PORT
}

# 运行Python脚本
run_python_script() {
    echo "运行Agent Graph测试..."
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$TEMP_DIR"
    
    local output_file="$OUTPUT_DIR/graph_test_results_$(date +%Y%m%d_%H%M%S).json"
    
    local python_cmd="$PYTHON_CMD agent_collaboration_graph.py \
        --hostname $HOSTNAME \
        --code_port $CODE_PORT \
        --visual_port $VISUAL_PORT \
        --num_samples $NUM_SAMPLES \
        --max_iterations $MAX_ITERATIONS \
        --output_path $output_file"
    
    echo "执行命令: $python_cmd"
    
    if $python_cmd; then
        log_success "Python脚本执行成功"
        echo "结果保存至: $output_file"
    else
        log_error "Python脚本执行失败"
        return 1
    fi
}

# 清理函数
cleanup() {
    echo ""
    echo "执行清理操作..."
    stop_sglang_servers
    echo "清理完成"
}

# 信号处理
#trap cleanup EXIT INT TERM

# 主函数
main() {
    echo "========================================"
    echo "  Frontend Design Agent Graph 测试"
    echo "========================================"
    echo "模型: $MODEL_NAME"
    echo "代码生成端口: $CODE_PORT"
    echo "视觉分析端口: $VISUAL_PORT"
    echo "测试样本: $NUM_SAMPLES"
    echo ""
    
    # 启动SGLang服务器
    start_sglang_server $CODE_PORT "code_generation" || exit 1
    echo ""
    if [ $VISUAL_PORT -ne $CODE_PORT ]; then
        start_sglang_server $VISUAL_PORT "visual_analysis" || exit 1
    fi
    echo ""
    
    # 等待服务器稳定
    echo "等待服务器稳定..."
    sleep 5
    
    # 运行Python脚本
    run_python_script || exit 1
    
    log_success "所有任务完成！"
    return 0
}

# 参数处理
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --code-port)
            CODE_PORT="$2"
            shift 2
            ;;
        --visual-port)
            VISUAL_PORT="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "简化版 Frontend Design Agent Graph 测试脚本"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL_NAME        模型名称"
            echo "  --code-port PORT          代码生成服务器端口 (默认: 8000)"
            echo "  --visual-port PORT        视觉分析服务器端口 (默认: 8001)"
            echo "  --samples NUM             测试样本数 (默认: 5)"
            echo "  --iterations NUM          最大迭代次数 (默认: 3)"
            echo "  --output-dir DIR          输出目录 (默认: ./test_results)"
            echo "  --help, -h                显示此帮助信息"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 运行主函数
main 