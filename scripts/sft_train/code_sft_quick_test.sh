#!/bin/bash
# Quick Test Script for SFT Data Collection (Small Scale)
# This script runs a quick test with minimal episodes for debugging

set -x

# GPU Configuration (single GPU for quick test)
export CUDA_VISIBLE_DEVICES=0
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

# CUDA Environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# Configuration
GPU_num=1
MODEL_PATH="/home/nvidia/data/models/Qwen3-1.7B"
EXPERIMENT_NAME="code_sft_quick_test"
OUTPUT_DIR="./sft_data_test"
SFT_MODEL_OUTPUT="./sft_model_test"

# Model resource configuration
model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"

# Run quick test with only 10 episodes and 1 training epoch
python3 -m pettingllms.sft_train.train \
    --config-path ../config/code \
    --config-name code_L0_single_agent \
    $model_0_resource \
    base_models.policy_0.path="$MODEL_PATH" \
    training.experiment_name=$EXPERIMENT_NAME \
    training.max_prompt_length=2048 \
    training.max_response_length=1024 \
    env.dataset=code_contests \
    env.benchmark=code_contests \
    env.max_turns=2 \
    training.sft_num_episodes=10 \
    training.sft_data_dir="$OUTPUT_DIR" \
    training.collect_mode=env \
    training.run_sft_training=True \
    training.sft_config.model_name_or_path="$MODEL_PATH" \
    training.sft_config.output_dir="$SFT_MODEL_OUTPUT" \
    training.sft_config.num_train_epochs=1 \
    training.sft_config.per_device_train_batch_size=1 \
    training.sft_config.gradient_accumulation_steps=2 \
    training.sft_config.learning_rate=5e-5 \
    training.sft_config.use_lora=True \
    training.sft_config.lora_r=32 \
    training.sft_config.lora_alpha=16

echo "Quick test completed!"
echo "Collected data saved to: $OUTPUT_DIR"
echo "Trained model saved to: $SFT_MODEL_OUTPUT"
