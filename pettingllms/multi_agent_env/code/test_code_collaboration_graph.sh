#!/bin/bash

# Code Generation Agent Collaboration Graph Test Script - Simplified Version
# Function: Start SGLang, cleanup exit, test 5 samples

set -e

# Basic configuration
HOSTNAME="localhost"
CODE_PORT=8000
TEST_PORT=8000
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_SAMPLES=5
OUTPUT_DIR="./test_results"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Log functions
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

# Cleanup port
cleanup_port() {
    local port=$1
    log_info "Cleaning up port $port..."
    pkill -f ".*python.*sglang.*--port.*$port" 2>/dev/null || true
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    sleep 2
}

# Start server
start_sglang() {
    local port=$1
    local server_name=$2
    
    log_info "Starting $server_name SGLang server (port: $port)..."
    
    # Clean up port
    cleanup_port $port
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Start server
    local cmd="python -m sglang.launch_server \
        --model-path $MODEL_NAME \
        --port $port \
        --host $HOSTNAME \
        --trust-remote-code \
        --dtype auto"
    
    log_info "Executing command: $cmd"
    nohup $cmd > "${OUTPUT_DIR}/${server_name}_${port}.log" 2>&1 &
    
    local pid=$!
    echo $pid > "${OUTPUT_DIR}/${server_name}_${port}.pid"
    log_success "Server started (PID: $pid)"
    
    # Wait for server to start
    log_info "Waiting for server to start..."
    local max_wait=60
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        if curl -s --max-time 3 "http://$HOSTNAME:$port/health" >/dev/null 2>&1; then
            log_success "$server_name server ready!"
            return 0
        fi
        sleep 3
        wait_time=$((wait_time + 3))
        printf "Waiting... (${wait_time}s/${max_wait}s)\r"
    done
    
    echo ""
    log_error "$server_name server startup timeout"
    return 1
}

# Stop all SGLang servers
stop_sglang() {
    log_info "Stopping SGLang servers..."
    
    for port in $CODE_PORT $TEST_PORT; do
        cleanup_port $port
        
        # Delete PID files
        rm -f "${OUTPUT_DIR}"/code_${CODE_PORT}.pid
        rm -f "${OUTPUT_DIR}"/test_${TEST_PORT}.pid
    done
    
    log_success "SGLang servers stopped"
}

# Test connection
test_connection() {
    log_info "Testing connection..."
    
    python agent_collaboration_graph.py \
        --code-port $CODE_PORT \
        --test-port $TEST_PORT \
        --test-mode connectivity
}

# Run test
run_test() {
    log_info "Running $NUM_SAMPLES sample tests..."
    
    local output_file="$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S).json"
    
    python agent_collaboration_graph.py \
        --code-port $CODE_PORT \
        --test-port $TEST_PORT \
        --output-path "$output_file"
    
    log_success "Test completed, results saved to: $output_file"
}

# Cleanup function
cleanup() {
    echo ""
    log_info "Executing cleanup..."
    stop_sglang
}

# Show help
show_help() {
    echo "Code Generation Agent Collaboration Test Script - Simplified Version"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start SGLang servers"
    echo "  stop      Stop SGLang servers"
    echo "  test      Run 5 sample tests"
    echo "  clean     Clean up all processes and files"
    echo "  conn      Test connection"
    echo "  help      Show help"
    echo ""
    echo "No parameter run: Start servers → Test connection → Run tests → Stop servers"
}

# Main function
main() {
    local command=${1:-"full"}
    
    case $command in
        "start")
            echo "Starting SGLang servers..."
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
            echo "  Code Generation Agent Collaboration Test - Simplified Version"
            echo "========================================"
            echo "Model: $MODEL_NAME"
            echo "Ports: $CODE_PORT, $TEST_PORT"
            echo "Samples: $NUM_SAMPLES"
            echo ""
            
            
            #trap cleanup EXIT INT TERM
            
            # Start servers
            #start_sglang $CODE_PORT "code" || exit 1
            if [ $TEST_PORT -ne $CODE_PORT ]; then
                start_sglang $TEST_PORT "test" || exit 1
            fi
            
            # Test connection
            if test_connection; then
                log_success "Connection test passed"
            else
                log_warning "Connection test failed, using simulation mode"
            fi
            
            # Run test
            run_test || exit 1
            
            log_success "All tasks completed!"
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 