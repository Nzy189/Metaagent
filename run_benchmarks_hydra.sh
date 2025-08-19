#!/bin/bash

# 使用 Hydra 的简化版多 benchmark 运行脚本
# 逻辑：一个 benchmark 在四张 GPU 卡上运行，运行完成后再运行下一个
# 不保存任何 log 文件

set -e  # 遇到错误时退出

# benchmark 列表
BENCHMARKS=(   "MBPP" "CodeForces"  "CodeContests" "LiveCodeBench")

export CUDA_VISIBLE_DEVICES=4,5,6,7

# 计算总数用于进度显示
total_benchmarks=${#BENCHMARKS[@]}
current_benchmark=0

run_benchmark_on_gpu() {
    local benchmark_name="$1"
    current_benchmark=$((current_benchmark + 1))
    
    echo "[$current_benchmark/$total_benchmarks] 开始运行 benchmark: $benchmark_name"
    
    python pettingllms/trainer/multi_agents_execution_engine.py \
        env.benchmark="$benchmark_name" \
        experiment_name="code_test_${benchmark_name}" \
        trainer.experiment_name="code_eval_${benchmark_name}" 
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$current_benchmark/$total_benchmarks] benchmark $benchmark_name 运行完成"
    else
        echo "[$current_benchmark/$total_benchmarks] benchmark $benchmark_name 运行失败，退出码: $exit_code"
        exit $exit_code
    fi
    
    return $exit_code
}



for benchmark in "${BENCHMARKS[@]}"; do
    run_benchmark_on_gpu "$benchmark"
done



