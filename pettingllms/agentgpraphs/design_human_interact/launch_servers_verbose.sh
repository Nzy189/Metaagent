#!/bin/bash

# SGLang Servers Verbose Launch Script
# 详细显示SGLang服务器启动过程的专用脚本

set -e

# 配置参数
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CODE_PORT=8000
VISUAL_PORT=8001
HOSTNAME="localhost"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# 日志函数
log() {
    local level=$1
    shift
    local timestamp=$(date '+%H:%M:%S')
    case $level in
        "INFO") echo -e "${BLUE}[INFO $timestamp]${NC} $*" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS $timestamp]${NC} $*" ;;
        "WARNING") echo -e "${YELLOW}[WARNING $timestamp]${NC} $*" ;;
        "ERROR") echo -e "${RED}[ERROR $timestamp]${NC} $*" ;;
        "DEBUG") echo -e "${CYAN}[DEBUG $timestamp]${NC} $*" ;;
    esac
}

# 显示横幅
show_banner() {
    clear
    echo -e "${PURPLE}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    SGLang 服务器详细启动监控器                              ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║  📋 功能: 启动两个SGLang服务器并实时显示详细启动过程                        ║"
    echo "║  🤖 模型: $MODEL_NAME                      ║"
    echo "║  🔌 端口: $CODE_PORT (代码生成), $VISUAL_PORT (视觉分析)                                     ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

# 清理现有进程
cleanup_existing() {
    log INFO "清理现有SGLang进程..."
    
    # 查找并显示现有进程
    local existing_procs=$(ps aux | grep "sglang.launch_server" | grep -v grep || true)
    if [ -n "$existing_procs" ]; then
        log WARNING "发现现有SGLang进程:"
        echo "$existing_procs"
        log INFO "正在终止现有进程..."
        pkill -f "sglang.launch_server" || true
        sleep 3
        pkill -9 -f "sglang.launch_server" || true
    else
        log SUCCESS "没有发现现有SGLang进程"
    fi
    
    # 检查端口占用
    for port in $CODE_PORT $VISUAL_PORT; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            local pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
            log WARNING "端口 $port 被进程 $pid 占用，正在清理..."
            kill $pid 2>/dev/null || true
            sleep 2
        fi
    done
    
    log SUCCESS "清理完成"
    echo ""
}

# 启动单个服务器 (前台模式，完整输出)
launch_server_foreground() {
    local port=$1
    local server_name=$2
    local log_suffix=$3
    
    echo -e "${PURPLE}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                启动 $server_name SGLang服务器 (端口: $port)               ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    local cmd="python -m sglang.launch_server \
        --model-path $MODEL_NAME \
        --port $port \
        --host $HOSTNAME \
        --trust-remote-code \
        --dtype auto \
        --log-level info"
    
    log INFO "启动命令: $cmd"
    log INFO "服务器类型: $server_name"
    log INFO "监听端口: $port"
    log INFO "启动时间: $(date)"
    echo ""
    
    echo -e "${CYAN}${BOLD}==================== SGLang 实时启动输出 ====================${NC}"
    echo -e "${YELLOW}💡 提示: 这里将显示SGLang服务器的完整启动过程，包括模型下载、加载等详细信息${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    
    # 直接执行命令，显示所有输出
    if $cmd; then
        log SUCCESS "$server_name 服务器启动成功！"
    else
        log ERROR "$server_name 服务器启动失败"
        return 1
    fi
}

# 启动单个服务器 (后台模式，但显示详细进度)
launch_server_background() {
    local port=$1
    local server_name=$2
    local server_id=$3
    
    echo -e "${PURPLE}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║            启动 $server_name SGLang服务器 (端口: $port) - 后台模式           ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    local log_file="${server_name}_${port}_detailed.log"
    local cmd="python -m sglang.launch_server \
        --model-path $MODEL_NAME \
        --port $port \
        --host $HOSTNAME \
        --trust-remote-code \
        --dtype auto \
        --log-level debug"
    
    log INFO "启动命令: $cmd"
    log INFO "日志文件: $log_file"
    log INFO "后台启动，实时跟踪日志..."
    echo ""
    
    # 后台启动服务器
    $cmd > "$log_file" 2>&1 &
    local pid=$!
    
    log INFO "服务器进程已启动 (PID: $pid)"
    echo ""
    
    # 实时跟踪日志文件
    echo -e "${CYAN}==================== 实时启动日志跟踪 ====================${NC}"
    echo -e "${YELLOW}📋 正在实时显示 $server_name 的启动日志...${NC}"
    echo -e "${CYAN}============================================================${NC}"
    
    # 启动日志跟踪
    tail -f "$log_file" &
    local tail_pid=$!
    
    # 等待服务器启动
    local max_wait=300
    local wait_time=0
    local check_interval=5
    
    while [ $wait_time -lt $max_wait ]; do
        # 检查进程状态
        if ! kill -0 $pid 2>/dev/null; then
            log ERROR "服务器进程 $pid 已退出"
            kill $tail_pid 2>/dev/null || true
            return 1
        fi
        
        # 检查服务器是否就绪
        if curl -s --max-time 3 "http://$HOSTNAME:$port/health" >/dev/null 2>&1; then
            kill $tail_pid 2>/dev/null || true
            echo ""
            echo -e "${GREEN}${BOLD}🎉 $server_name 服务器启动成功！ (用时: ${wait_time}s)${NC}"
            
            # 快速功能测试
            log INFO "进行快速功能测试..."
            local test_response=$(curl -s --max-time 10 -X POST "http://$HOSTNAME:$port/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"model": "default", "messages": [{"role": "user", "content": "测试"}], "max_tokens": 5}' 2>/dev/null || echo "")
            
            if [ -n "$test_response" ]; then
                log SUCCESS "✅ API功能测试通过"
            else
                log WARNING "⚠️  API功能测试失败，但服务器已启动"
            fi
            
            echo -e "${GREEN}🚀 $server_name 服务器 (端口:$port) 完全就绪！${NC}"
            echo ""
            return 0
        fi
        
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    kill $tail_pid 2>/dev/null || true
    log ERROR "$server_name 服务器启动超时"
    return 1
}

# 主函数
main() {
    show_banner
    
    # 解析参数
    local mode="foreground"  # 默认前台模式
    while [[ $# -gt 0 ]]; do
        case $1 in
            --background|-b)
                mode="background"
                shift
                ;;
            --foreground|-f)
                mode="foreground"
                shift
                ;;
            --help|-h)
                echo "SGLang 详细启动脚本"
                echo ""
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --foreground, -f    前台模式 (默认) - 依次启动服务器，显示完整输出"
                echo "  --background, -b    后台模式 - 并行启动服务器，跟踪日志文件"
                echo "  --help, -h          显示此帮助"
                echo ""
                echo "前台模式: 逐个启动服务器，完整显示启动过程，需要手动停止"
                echo "后台模式: 并行启动两个服务器，实时跟踪日志文件"
                exit 0
                ;;
            *)
                log ERROR "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    log INFO "启动模式: $mode"
    log INFO "目标模型: $MODEL_NAME"
    echo ""
    
    # 清理现有进程
    cleanup_existing
    
    if [ "$mode" = "foreground" ]; then
        # 前台模式 - 逐个启动
        log INFO "🚀 前台模式启动 - 依次显示每个服务器的完整启动过程"
        echo ""
        
        echo -e "${YELLOW}🔔 注意: 前台模式下，第一个服务器启动完成后需要按 Ctrl+C 来启动第二个服务器${NC}"
        echo -e "${YELLOW}   或者使用后台模式: $0 --background${NC}"
        echo ""
        
        read -p "按回车键开始启动第一个服务器 (代码生成)..." -r
        launch_server_foreground $CODE_PORT "代码生成" "code"
        
        echo ""
        echo -e "${CYAN}第一个服务器已启动，现在启动第二个服务器...${NC}"
        read -p "按回车键开始启动第二个服务器 (视觉分析)..." -r
        launch_server_foreground $VISUAL_PORT "视觉分析" "visual"
        
    else
        # 后台模式 - 并行启动
        log INFO "🚀 后台模式启动 - 并行启动两个服务器"
        echo ""
        
        # 启动第一个服务器
        launch_server_background $CODE_PORT "代码生成" "code" &
        local pid1=$!
        
        sleep 5  # 稍微错开启动时间
        
        # 启动第二个服务器
        launch_server_background $VISUAL_PORT "视觉分析" "visual" &
        local pid2=$!
        
        # 等待两个服务器都启动完成
        wait $pid1 && wait $pid2
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}${BOLD}"
            echo "🎉 两个SGLang服务器都已成功启动！"
            echo ""
            echo "📍 服务器信息:"
            echo "  - 代码生成服务器: http://$HOSTNAME:$CODE_PORT"
            echo "  - 视觉分析服务器: http://$HOSTNAME:$VISUAL_PORT"
            echo ""
            echo "📋 现在可以运行测试脚本:"
            echo "  ./test_multi_agent_graph.sh"
            echo -e "${NC}"
        else
            log ERROR "一个或多个服务器启动失败"
            exit 1
        fi
    fi
}

# 清理函数
cleanup() {
    echo ""
    log INFO "正在清理..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "tail -f.*_detailed.log" 2>/dev/null || true
    log SUCCESS "清理完成"
}

# 设置清理信号处理
trap cleanup EXIT INT TERM

# 运行主函数
main "$@" 