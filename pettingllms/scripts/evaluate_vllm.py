import argparse
import asyncio
import json
import os
import sys
import time
import uuid
import subprocess

from omegaconf import OmegaConf
from omegaconf import DictConfig
from transformers import AutoTokenizer

from pettingllms.trainer.utils import (
    convert_prompt_to_dpr,
    convert_dpr_to_response,
    llm_async_generate,
)
from pettingllms.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


def _read_registry_address(registry_path: str) -> str | None:
    if not os.path.exists(registry_path):
        return None
    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
        addresses = data.get("addresses") or []
        return addresses[0] if addresses else None
    except Exception:
        return None


def _maybe_override_model_path_from_ckpt(config: DictConfig) -> DictConfig:
    """
    如果最新 checkpoint 中包含 HuggingFace 格式的模型（huggingface/），
    则用它覆盖 actor_rollout_ref.model.path，以便用 vLLM 直接加载合并后的权重。
    """
    ckpt_root = config.trainer.get("default_local_dir")
    if not ckpt_root:
        return config
    latest = find_latest_ckpt_path(ckpt_root)
    if not latest:
        return config
    hf_dir = os.path.join(latest, "huggingface")
    if os.path.isdir(hf_dir) and os.path.exists(os.path.join(hf_dir, "config.json")):
        # 覆盖 actor_rollout_ref.model.path
        try:
            config = OmegaConf.to_container(config, resolve=True)
            cfg = OmegaConf.create(config)
            if "actor_rollout_ref" in cfg and "model" in cfg.actor_rollout_ref:
                cfg.actor_rollout_ref.model.path = hf_dir
            return cfg
        except Exception:
            return config
    return config


def _dump_temp_config(cfg: DictConfig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(cfg, path)
    return path


def _launch_vllm_in_background(trainer_config_path: str, registry_path: str, num_gpus: int, num_cpus: int, actor_name: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "pettingllms.scripts.launch_vllm_servers",
        "--trainer-config",
        trainer_config_path,
        "--registry-path",
        registry_path,
        "--num-gpus",
        str(num_gpus),
        "--num-cpus",
        str(num_cpus),
        "--actor-name",
        actor_name,
    ]
    # 后台启动，stdout/stderr 重定向到文件
    log_dir = os.path.dirname(registry_path) or "."
    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(log_dir, "vllm_launch_stdout.log")
    stderr_path = os.path.join(log_dir, "vllm_launch_stderr.log")
    stdout_f = open(stdout_path, "a")
    stderr_f = open(stderr_path, "a")
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f)
    return proc


async def _evaluate_once(address: str, trainer_cfg: DictConfig, prompt: str, mode: str = "validate") -> str:
    # 1) 计算 model_name（vLLM OpenAI server 侧用它作为 models 名称）
    model_path = trainer_cfg.actor_rollout_ref.model.path
    # 如果包含 checkpoint 字样则保留完整路径，否则截取最后两段
    if "checkpoint" in str(model_path):
        model_name = str(model_path)
    else:
        model_name = "/".join(str(model_path).split("/")[-2:])

    # 2) 加载 tokenizer 路径（优先使用 tokenizer_path）
    tokenizer_path = trainer_cfg.actor_rollout_ref.get("tokenizer_path", model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trainer_cfg.actor_rollout_ref.model.get("trust_remote_code", False))

    # 3) 构造 DataProto
    max_prompt_len = trainer_cfg.data.max_prompt_length
    dpr = convert_prompt_to_dpr(
        tokenizer=tokenizer,
        processor=None,
        prompts={"text": prompt, "image": None},
        max_prompt_length=max_prompt_len,
        multi_modal=False,
    )

    # 4) 组装 DataProto 的张量部分（convert_prompt_to_dpr 返回 formatted_prompts + tensors）
    #    convert_prompt_to_dpr 已经完成 input_ids/attention_mask/position_ids 的构造

    # 5) 调用 llm_async_generate（Completion API 路径）
    output_dpr, _ = await llm_async_generate(
        rollout_idx=0,
        turn_idx=0,
        agent_idx=0,
        prompt_dpr=dpr,
        ppo_trainer_config=trainer_cfg,
        enable_thinking=False,
        address=address,
        model_name=model_name,
        tokenizer=tokenizer,
        application_id=str(uuid.uuid4()),
        mode=mode,
    )

    # 6) 解码输出
    resp = convert_dpr_to_response(
        tokenizer=tokenizer,
        chat_parser=None,
        dpr=output_dpr,
        max_prompt_length=max_prompt_len,
    )
    return resp


def main():
    parser = argparse.ArgumentParser(description="Evaluate with vLLM via llm_async_generate")
    parser.add_argument("--trainer-config", type=str, default="pettingllms/config/code/ppo_trainer/eval.yaml")
    parser.add_argument("--registry-path", type=str, default="logs/ray_vllm_registry.json")
    parser.add_argument("--address", type=str, default=None, help="Existing vLLM address like 127.0.0.1:8000")
    parser.add_argument("--prompt", type=str, default="Hello, who are you?", help="Single prompt text")
    parser.add_argument("--mode", type=str, default="validate", choices=["train", "validate"]) 
    parser.add_argument("--launch", action="store_true", help="Launch vLLM server if no address/registry")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--actor-name", type=str, default="async_llm_server")
    args = parser.parse_args()

    # 1) 加载并解析配置
    trainer_cfg = OmegaConf.load(args.trainer_config)
    OmegaConf.resolve(trainer_cfg)

    # 2) address 获取逻辑
    address = args.address or _read_registry_address(args.registry_path)

    proc = None
    tmp_cfg_path = None
    try:
        if address is None and args.launch:
            # 用 ckpt 覆盖 model.path（如存在 HuggingFace 目录）
            patched_cfg = _maybe_override_model_path_from_ckpt(trainer_cfg)
            tmp_cfg_path = os.path.join(os.path.dirname(args.registry_path) or ".", "tmp_eval_vllm_config.yaml")
            _dump_temp_config(patched_cfg, tmp_cfg_path)

            # 后台启动 vLLM server
            proc = _launch_vllm_in_background(tmp_cfg_path, args.registry_path, args.num_gpus, args.num_cpus, args.actor_name)

            # 等待地址写入 registry
            for _ in range(120):
                time.sleep(1.0)
                address = _read_registry_address(args.registry_path)
                if address:
                    break
            if not address:
                raise RuntimeError("Failed to obtain vLLM address from registry. Check logs and GPU availability.")

        if address is None:
            raise RuntimeError("No address provided and no registry found. Use --launch or specify --address.")

        # 3) 执行一次评估
        resp = asyncio.run(_evaluate_once(address, trainer_cfg, args.prompt, args.mode))
        print("==== Evaluation Result ====")
        print(resp)

    finally:
        # 不主动杀掉 vLLM，若用户使用 --launch，可复用该服务。
        pass


if __name__ == "__main__":
    main()


