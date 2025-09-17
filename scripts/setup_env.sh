python3 -m venv verl_efficient_venv

pip uninstall torch -y && pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# INSTALL VLLM
# (1) from source
cd /home/lah003/workspace/vllm
python3 use_existing_torch.py
pip install -r requirements/build.txt
pip uninstall vllm -y && pip install --no-build-isolation -e .
# (2) pre-built
# pip install vllm==0.10.0 --no-deps

cd /home/lah003/workspace/verl_efficient
pip install -r requirements_venv_cu128.txt

git submodule update --init --recursive
cd /home/lah003/workspace/verl_efficient/verl
pip install -e .
cd /home/lah003/workspace/verl_efficient
pip install -e .

# INSTALL FLASH_ATTN
#pip install flash_attn==2.8.2
cd /home/lah003/workspace/flash-attention
MAX_JOBS=8 python3 setup.py install

pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu128
pip uninstall tensordict && pip install tensordict==0.9.1

# install models
git lfs clone https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ~/Qwen3-4B-Instruct-2507
git lfs clone https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct ~/Qwen2.5-Coder-3B-Instruct