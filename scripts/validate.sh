set -x

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

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 通用配置
config_path="../config/math"
config_name="math_single_policy"
config_path_code="../config/code"
config_name_code="code_eval"
train_data_size=32
val_data_size=32
data_dir=~/data/math/model_0
USE_GRPO="models.model_0.ppo_trainer_config.algorithm.adv_estimator=grpo models.model_0.ppo_trainer_config.actor_rollout_ref.actor.use_kl_loss=False"
RESOURCE="resource.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.nnodes=1 models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=2"
DATA="+models.model_0.ppo_trainer_config.data.train_files=$data_dir/text/train.parquet +models.model_0.ppo_trainer_config.data.val_files=$data_dir/text/test.parquet"

# 模型列表（Hugging Face 仓库名）

#models=(
#  "Qwen/Qwen3-1.7B"
#  "Qwen/Qwen3-4B"
#  "Qwen/Qwen3-4B-Instruct"
#  "Qwen/Qwen3-8B"
#  "Qwen/Qwen2.5-7B-Instruct"
#  "Qwen/Qwen2.5-3B-Instruct"
#)
models=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
)




for model in "${models[@]}"; do
  echo "=== Evaluating model: $model ==="

  python3 -m pettingllms.scripts.async_vllm_code_eval \
    --config-path "$config_path_code" --config-name "$config_name_code" \
    $USE_GRPO $RESOURCE $DATA \
    models.model_0.path="$model" \
    benchmark="apps" \
    data.epoch_size=120 \
    data.max_prompt_length=8192 \
    data.max_response_length=6144 \
    data.resample_freq=4 \
    data.filter_method=std \
    data.filter_ratio=0.5 \
    sample_mode=tree \
    env.max_turns=3

done
