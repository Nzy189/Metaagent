#!/bin/bash

# 代码生成智能体协作图测试脚本 - 简化版
# 功能：启动SGLang、清理退出、测试5个样本

set -e

# 基本配置
HOSTNAME="localhost"
CODE_PORT=8000
TEST_PORT=8000
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_SAMPLES=5
OUTPUT_DIR="./test_results"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 清理端口
cleanup_port() {
    local port=$1
    log_info "清理端口 $port..."
    pkill -f ".*python.*sglang.*--port.*$port" 2>/dev/null || true
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    sleep 2
}

# 启动SGLang服务器
start_sglang() {
    local port=$1
    local server_name=$2
    
    log_info "启动 $server_name SGLang服务器 (端口: $port)..."
    
    # 清理端口
    cleanup_port $port
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 启动服务器
    local cmd="python -m sglang.launch_server \
        --model-path $MODEL_NAME \
        --port $port \
        --host $HOSTNAME \
        --trust-remote-code \
        --dtype auto"
    
    log_info "执行命令: $cmd"
    nohup $cmd > "${OUTPUT_DIR}/${server_name}_${port}.log" 2>&1 &
    
    local pid=$!
    echo $pid > "${OUTPUT_DIR}/${server_name}_${port}.pid"
    log_success "服务器已启动 (PID: $pid)"
    
    # 等待服务器启动
    log_info "等待服务器启动..."
    local max_wait=60
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        if curl -s --max-time 3 "http://$HOSTNAME:$port/health" >/dev/null 2>&1; then
            log_success "$server_name 服务器就绪!"
            return 0
        fi
        sleep 3
        wait_time=$((wait_time + 3))
        printf "等待中... (${wait_time}s/${max_wait}s)\r"
    done
    
    echo ""
    log_error "$server_name 服务器启动超时"
    return 1
}

# 停止所有SGLang服务器
stop_sglang() {
    log_info "停止SGLang服务器..."
    
    for port in $CODE_PORT $TEST_PORT; do
        cleanup_port $port
        
        # 删除PID文件
        rm -f "${OUTPUT_DIR}"/code_${CODE_PORT}.pid
        rm -f "${OUTPUT_DIR}"/test_${TEST_PORT}.pid
    done
    
    log_success "SGLang服务器已停止"
}

# 测试连接
test_connection() {
    log_info "测试连接..."
    
    python agent_collaboration_graph.py \
        --code-port $CODE_PORT \
        --test-port $TEST_PORT \
        --test-mode connectivity
}

# 运行测试
run_test() {
    log_info "运行 $NUM_SAMPLES 个样本测试..."
    
    local output_file="$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S).json"
    
    python agent_collaboration_graph.py \
        --code-port $CODE_PORT \
        --test-port $TEST_PORT \
        --output-path "$output_file"
    
    log_success "测试完成，结果保存到: $output_file"
}

# 清理函数
cleanup() {
    echo ""
    log_info "执行清理..."
    stop_sglang
}

# 显示帮助
show_help() {
    echo "代码生成智能体协作测试脚本 - 简化版"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start     启动SGLang服务器"
    echo "  stop      停止SGLang服务器"
    echo "  test      运行5个样本测试"
    echo "  clean     清理所有进程和文件"
    echo "  conn      测试连接"
    echo "  help      显示帮助"
    echo ""
    echo "无参数运行: 启动服务器 → 测试连接 → 运行测试 → 停止服务器"
}

# 主函数
main() {
    local command=${1:-"full"}
    
    case $command in
        "start")
            echo "启动SGLang服务器..."
            start_sglang $CODE_PORT "code"
            if [ $TEST_PORT -ne $CODE_PORT ]; then
                start_sglang $TEST_PORT "test"
            fi
            ;;
        "stop")
            stop_sglang
            ;;
        "test")
            run_test
            ;;
        "clean")
            cleanup
            rm -rf "$OUTPUT_DIR"
            log_success "清理完成"
            ;;
        "conn")
            test_connection
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        "full")
            echo "========================================"
            echo "  代码生成智能体协作测试 - 简化版"
            echo "========================================"
            echo "模型: $MODEL_NAME"
            echo "端口: $CODE_PORT, $TEST_PORT"
            echo "样本数: $NUM_SAMPLES"
            echo ""
            
            
            #trap cleanup EXIT INT TERM
            
            # 启动服务器
            #start_sglang $CODE_PORT "code" || exit 1
            if [ $TEST_PORT -ne $CODE_PORT ]; then
                start_sglang $TEST_PORT "test" || exit 1
            fi
            
            # 测试连接
            if test_connection; then
                log_success "连接测试通过"
            else
                log_warning "连接测试失败，使用模拟模式"
            fi
            
            # 运行测试
            run_test || exit 1
            
            log_success "所有任务完成!"
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 