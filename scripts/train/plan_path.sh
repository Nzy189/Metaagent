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

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_data_dir=~/data/math/model_0



model_0_resource="resource.n_gpus_per_node=8  $model_0_config_path.trainer.n_gpus_per_node=8 $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=8"


python3 -m pettingllms.trainer.train --config-path ../config/stateful --config-name stateful_single_policy \
    $model_0_resource $model_0_data\
    models.model_0.path=/home/nvidia/data/models/Qwen3-8B\
    experiment_name=sokoban_single_policy_8B\
    if_dapo=True\
    benchmark=sokoban\
    env.map_size=10\
    trainer.total_training_steps=400\
    trainer.save_freq=50\
    data.epoch_size=20\
    data.gen_batch_size=32\
    data.gen_n_samples=4\
    data.max_prompt_length=8192\
    data.max_response_length=2048\
    data.resample_freq=1\
    data.filter_method=std\
    data.filter_ratio=0\
    $model_0_config_path.actor_rollout_ref.actor.ppo_mini_batch_size=32\
    $model_0_config_path.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.5\
    $model_0_config_path.actor_rollout_ref.rollout.max_num_batched_tokens=65536\
    sample_mode=tree\
    env.max_turns=3\