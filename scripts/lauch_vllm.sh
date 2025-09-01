set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m pettingllms.scripts.launch_vllm_servers \
  --trainer-config pettingllms/config/code/ppo_trainer/eval.yaml \
  --registry-path logs/ray_vllm_registry.json \
  --num-gpus 4 \
  --num-cpus 100 \
  --actor-name async_llm_server

