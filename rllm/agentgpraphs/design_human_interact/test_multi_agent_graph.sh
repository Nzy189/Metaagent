#!/bin/bash

# Multi-Agent Graph Testing Script
# This script launches sglang servers and tests the FrontendDesignAgentGraph

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CODE_PORT=8000
VISUAL_PORT=8001
MODEL_PATH=${MODEL_PATH:-"microsoft/DialoGPT-medium"}  # Default model, can be overridden
HOSTNAME="localhost"
NUM_SAMPLES=1
TEST_TIMEOUT=300  # 5 minutes timeout
LOG_DIR="./logs"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for server on port $port to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://${HOSTNAME}:${port}/v1/models" >/dev/null 2>&1; then
            print_success "Server on port $port is ready!"
            return 0
        fi
        
        print_info "Attempt $attempt/$max_attempts - Server not ready yet, waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    print_error "Server on port $port failed to start within timeout"
    return 1
}

# Function to start sglang server
start_sglang_server() {
    local port=$1
    local server_name=$2
    local log_file="${LOG_DIR}/sglang_${server_name}_${port}.log"
    
    print_info "Starting sglang server for $server_name on port $port..."
    
    # Check if port is available
    if ! check_port $port; then
        print_warning "Port $port is already in use. Attempting to kill existing process..."
        kill_process_on_port $port
        sleep 5
    fi
    
    # Create log directory
    mkdir -p $LOG_DIR
    
    # Start sglang server in background
    nohup python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port $port \
        --host 0.0.0.0 \
        --disable-radix-cache \
        --max-running-requests 10 \
        > "$log_file" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "${LOG_DIR}/sglang_${server_name}_${port}.pid"
    
    print_info "Started sglang server (PID: $server_pid) for $server_name on port $port"
    print_info "Logs: $log_file"
    
    # Wait for server to be ready
    if wait_for_server $port; then
        print_success "sglang server for $server_name is ready!"
        return 0
    else
        print_error "Failed to start sglang server for $server_name"
        return 1
    fi
}

# Function to kill process on port
kill_process_on_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        print_info "Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null || true
        sleep 2
    fi
}

# Function to cleanup servers
cleanup_servers() {
    print_info "Cleaning up sglang servers..."
    
    # Kill servers using PID files
    for pid_file in ${LOG_DIR}/sglang_*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ps -p $pid > /dev/null 2>&1; then
                print_info "Killing sglang server (PID: $pid)"
                kill -TERM $pid 2>/dev/null || true
                sleep 2
                # Force kill if still running
                if ps -p $pid > /dev/null 2>&1; then
                    kill -9 $pid 2>/dev/null || true
                fi
            fi
            rm -f "$pid_file"
        fi
    done
    
    # Additional cleanup by port
    kill_process_on_port $CODE_PORT
    kill_process_on_port $VISUAL_PORT
    
    print_success "Cleanup completed"
}

# Function to test server connectivity
test_server_connectivity() {
    print_info "Testing server connectivity..."
    
    # Test code generation server
    print_info "Testing code generation server (port $CODE_PORT)..."
    local code_response=$(curl -s -X POST "http://${HOSTNAME}:${CODE_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "default",
            "messages": [{"role": "user", "content": "Hello, please respond with OK"}],
            "max_tokens": 10,
            "temperature": 0.1
        }' 2>/dev/null || echo "ERROR")
    
    if [[ "$code_response" == *"OK"* ]] || [[ "$code_response" == *"choices"* ]]; then
        print_success "Code generation server is responding"
    else
        print_error "Code generation server is not responding properly"
        print_error "Response: $code_response"
        return 1
    fi
    
    # Test visual analysis server
    print_info "Testing visual analysis server (port $VISUAL_PORT)..."
    local visual_response=$(curl -s -X POST "http://${HOSTNAME}:${VISUAL_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "default",
            "messages": [{"role": "user", "content": "Hello, please respond with OK"}],
            "max_tokens": 10,
            "temperature": 0.1
        }' 2>/dev/null || echo "ERROR")
    
    if [[ "$visual_response" == *"OK"* ]] || [[ "$visual_response" == *"choices"* ]]; then
        print_success "Visual analysis server is responding"
    else
        print_error "Visual analysis server is not responding properly"
        print_error "Response: $visual_response"
        return 1
    fi
    
    print_success "Both servers are responding correctly!"
    return 0
}

# Function to run loop test
test_loop_function() {
    print_info "Testing loop function..."
    
    cat > test_loop.py << 'EOF'
import asyncio
import sys
import tempfile
import traceback
from rllm.agentgpraphs.design_human_interact.agent_collaboration_graph import FrontendDesignAgentGraph
from rllm.agentgpraphs.design_human_interact.websight_env import WebEnv

async def test_loop():
    try:
        print("🔄 Testing loop function...")
        
        # Create agent graph
        agent_graph = FrontendDesignAgentGraph(
            hostname="localhost",
            code_port=8000,
            visual_port=8001,
            max_iterations=1,
            temp_path=tempfile.gettempdir()
        )
        
        # Test server connections
        if not agent_graph.client.test_connections():
            print("❌ Failed to connect to servers")
            return False
        
        # Create mock environment
        mock_sample = {
            "task_id": "test_loop",
            "problem_description": "Create a simple HTML page",
            "ground_truth": "<html><body><h1>Test</h1></body></html>"
        }
        
        env = WebEnv(task=mock_sample, max_turns=2, temp_path=tempfile.gettempdir())
        
        # Reset agents
        agents_info = agent_graph._get_agents_list()
        print(f"🔍 Detected {len(agents_info)} agents: {[name for name, _, _ in agents_info]}")
        
        for _, agent_instance, _ in agents_info:
            agent_instance.reset()
        
        # Get initial observation
        obs, _ = env.reset()
        
        # Test loop function
        print("📞 Calling loop function...")
        step_data = await agent_graph.loop(obs, step_idx=0)
        
        print("✅ Loop function completed successfully!")
        print(f"📊 Returned data for {len(step_data)} agents:")
        
        for agent_name, agent_data in step_data.items():
            print(f"  {agent_name} ({agent_data['original_name']}):")
            print(f"    Action Type: {agent_data['action_type']}")
            print(f"    Action Length: {len(str(agent_data['action']))} chars")
            print(f"    Response Length: {len(str(agent_data['response']))} chars")
        
        # Test reward update
        env_results = {"default_reward": 0.8}
        updated_data = await agent_graph.update_rewards(step_data, env_results)
        
        print("✅ Reward update completed!")
        for agent_name, agent_data in updated_data.items():
            print(f"  {agent_name}: reward = {agent_data['reward']}")
        
        env.cleanup()
        agent_graph.cleanup()
        
        print("🎉 Loop function test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Loop function test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_loop())
    sys.exit(0 if success else 1)
EOF
    
    python test_loop.py
    local exit_code=$?
    rm -f test_loop.py
    
    if [ $exit_code -eq 0 ]; then
        print_success "Loop function test passed!"
        return 0
    else
        print_error "Loop function test failed!"
        return 1
    fi
}

# Function to run evaluate test
test_evaluate_function() {
    print_info "Testing evaluate function with $NUM_SAMPLES sample..."
    
    cat > test_evaluate.py << 'EOF'
import sys
import tempfile
import traceback
import json
from rllm.agentgpraphs.design_human_interact.agent_collaboration_graph import FrontendDesignAgentGraph

def test_evaluate():
    try:
        print("🧪 Testing evaluate function...")
        
        # Create agent graph
        agent_graph = FrontendDesignAgentGraph(
            hostname="localhost",
            code_port=8000,
            visual_port=8001,
            max_iterations=1,  # Keep it small for testing
            temp_path=tempfile.gettempdir()
        )
        
        # Test server connections
        if not agent_graph.client.test_connections():
            print("❌ Failed to connect to servers")
            return False
        
        # Run evaluation with 1 sample
        print("🚀 Running evaluation...")
        results = agent_graph.run_evaluation(
            num_samples=1,
            output_path="test_results.json"
        )
        
        if "error" in results:
            print(f"❌ Evaluation failed: {results['error']}")
            return False
        
        print("✅ Evaluation completed successfully!")
        print(f"📊 Results:")
        print(f"  Total tasks: {results['total_tasks']}")
        print(f"  Successful tasks: {results['successful_tasks']}")
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Average iterations: {results['average_iterations']:.1f}")
        
        # Show detailed results for the single task
        if results['detailed_results']:
            task_result = results['detailed_results'][0]
            print(f"\n📝 Task Details:")
            print(f"  Task ID: {task_result.get('task_id', 'N/A')}")
            print(f"  Success: {task_result.get('success', False)}")
            print(f"  Total iterations: {task_result.get('total_iterations', 0)}")
            print(f"  Termination reason: {task_result.get('termination_reason', 'N/A')}")
            
            if 'agent_data' in task_result:
                print(f"  Agent data: {len(task_result['agent_data'])} agents")
                for agent_name, agent_data in task_result['agent_data'].items():
                    print(f"    {agent_name}: {len(agent_data['steps'])} steps, reward = {agent_data['total_reward']}")
        
        agent_graph.cleanup()
        
        print("🎉 Evaluate function test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluate function test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluate()
    sys.exit(0 if success else 1)
EOF
    
    python test_evaluate.py
    local exit_code=$?
    rm -f test_evaluate.py
    
    if [ $exit_code -eq 0 ]; then
        print_success "Evaluate function test passed!"
        return 0
    else
        print_error "Evaluate function test failed!"
        return 1
    fi
}

# Main execution function
main() {
    print_info "Starting Multi-Agent Graph Testing Script"
    print_info "=========================================="
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-path)
                MODEL_PATH="$2"
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
            --num-samples)
                NUM_SAMPLES="$2"
                shift 2
                ;;
            --cleanup-only)
                cleanup_servers
                exit 0
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --model-path PATH     Model path for sglang servers (default: microsoft/DialoGPT-medium)"
                echo "  --code-port PORT      Port for code generation server (default: 8000)"
                echo "  --visual-port PORT    Port for visual analysis server (default: 8001)"
                echo "  --num-samples N       Number of samples to test (default: 1)"
                echo "  --cleanup-only        Only cleanup existing servers and exit"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_info "Configuration:"
    print_info "  Model Path: $MODEL_PATH"
    print_info "  Code Port: $CODE_PORT"
    print_info "  Visual Port: $VISUAL_PORT"
    print_info "  Hostname: $HOSTNAME"
    print_info "  Number of samples: $NUM_SAMPLES"
    print_info "  Log Directory: $LOG_DIR"
    
    # Set up cleanup trap
    trap cleanup_servers EXIT INT TERM
    
    # Clean up any existing servers
    cleanup_servers
    
    # Start sglang servers
    print_info "Starting sglang servers..."
    
    if ! start_sglang_server $CODE_PORT "code"; then
        print_error "Failed to start code generation server"
        exit 1
    fi
    
    if ! start_sglang_server $VISUAL_PORT "visual"; then
        print_error "Failed to start visual analysis server"
        exit 1
    fi
    
    # Test server connectivity
    if ! test_server_connectivity; then
        print_error "Server connectivity test failed"
        exit 1
    fi
    
    # Run tests
    print_info "Running tests..."
    
    # Test loop function
    if ! test_loop_function; then
        print_error "Loop function test failed"
        exit 1
    fi
    
    # Test evaluate function
    if ! test_evaluate_function; then
        print_error "Evaluate function test failed"
        exit 1
    fi
    
    print_success "All tests completed successfully! 🎉"
    print_info "Check logs in $LOG_DIR for detailed server output"
    print_info "Results saved to test_results.json"
    
    # Keep servers running for manual testing
    print_info "Servers are still running for manual testing..."
    print_info "Code generation server: http://${HOSTNAME}:${CODE_PORT}"
    print_info "Visual analysis server: http://${HOSTNAME}:${VISUAL_PORT}"
    print_info "Press Ctrl+C to stop servers and exit"
    
    # Wait for user interrupt
    while true; do
        sleep 60
        print_info "Servers still running... (Press Ctrl+C to stop)"
    done
}

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        missing_deps+=("python")
    fi
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    # Check if lsof is available
    if ! command -v lsof &> /dev/null; then
        missing_deps+=("lsof")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again"
        exit 1
    fi
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    main "$@"
fi 