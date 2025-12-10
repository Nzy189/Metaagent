set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}


GPU_num=8

model_0_config_path="models.model_0.ppo_trainer_config"

model_0_resource="resource.n_gpus_per_node=$GPU_num \
  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num \
  $model_0_config_path.trainer.nnodes=1 \
  $model_0_config_path.trainer.n_training_gpus_per_node=$GPU_num \
  $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=4"

rollout_memory_fix="$model_0_config_path.actor_rollout_ref.rollout.max_num_seqs=64 \
  $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  $model_0_config_path.actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  $model_0_config_path.actor_rollout_ref.rollout.max_model_len=12288 \
  $model_0_config_path.actor_rollout_ref.rollout.enable_chunked_prefill=True"

memory_optimization="$model_0_config_path.actor_rollout_ref.model.enable_gradient_checkpointing=True \
  $model_0_config_path.actor_rollout_ref.model.enable_activation_offload=True \
  $model_0_config_path.actor_rollout_ref.model.use_remove_padding=True \
  $model_0_config_path.actor_rollout_ref.actor.use_dynamic_bsz=True"


python3 -m pettingllms.trainer.train --config-path ../config/math --config-name math_L2_lora \
    $model_0_resource \
    $rollout_memory_fix \
    $memory_optimization \
    base_models.policy_0.path="your base model path"\
    training.experiment_name=math_1.7B_lora_8gpu\
    training.total_training_steps=200\
    training.train_batch_size=32\
    training.train_sample_num=8\
    training.validate_sample_num=5\
    training.max_prompt_length=4096\
    training.max_response_length=8192\
    training.val_freq=10\
    training.resample_freq=3\
    env.dataset=polaris\
    env.benchmark=AIME24\


