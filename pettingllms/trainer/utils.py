def convert_prompt_to_dpr(tokenizer, chat_parser, processor, prompts, max_prompt_length, multi_modal=False, **kwargs):
    """
    将 prompt dict 转换为 veRL 的 DataProto。
    
    Args:
        tokenizer: HF tokenizer，需支持 apply_chat_template 与 __call__ 分词
        chat_parser: 预留（当前未使用）
        prompts: dict，{"text": str, "image": None 或 图片路径}
        max_prompt_length: 最长 prompt 长度（左侧 padding）
        multi_modal: 是否多模态（若 True，应同时传入 processor 等必要信息）
        kwargs: 可选参数，如 processor、meta_info 等
    Returns:
        DataProto: 包含张量与非张量信息
    """
    from verl.protocol import DataProto, union_two_dict
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import pad_sequence_to_length
    import numpy as np
    import torch

    if not isinstance(prompts, dict) or "text" not in prompts:
        raise ValueError("prompts 必须是包含 'text' 键的字典: {'text': str, 'image': Optional[path]} ")

    text = prompts.get("text", "") or ""
    image_path = prompts.get("image", None)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        chat = np.array([
            {"content": text, "role": "user"}
        ])
        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = tokenizer(
            prompt_with_chat_template,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # Multimodal (optional): depends on externally provided processor
        multi_modal_inputs = None
        if multi_modal and image_path is not None and "processor" in kwargs:
            
            image_inputs = processor.image_processor([image_path], return_tensors="pt")
            multi_modal_inputs = {k: v for k, v in image_inputs.items()}
           

        # Pad to a unified length
        input_ids = pad_sequence_to_length(
            input_ids,
            max_seq_len=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
        )
        attention_mask = pad_sequence_to_length(
            attention_mask,
            max_seq_len=max_prompt_length,
            pad_token_id=0,
            left_pad=True,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        data.non_tensor_batch["formatted_prompts"] = np.array([prompt_with_chat_template])
        if multi_modal_inputs is not None:
            data.non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

        # Merge meta_info if provided
        meta_info = kwargs.get("meta_info")
        if meta_info:
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data
    finally:
        tokenizer.padding_side = old_padding_side


def convert_dpr_to_response(tokenizer, chat_parser, dpr, max_prompt_length, multi_modal=False, **kwargs):
    attn = dpr.batch["attention_mask"][0, max_prompt_length :]
    tokens = dpr.batch["responses"][0]

    # Find last index where attention == 1
    non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
    if len(non_pad_indices) == 0:
        trimmed = tokens[:0]  # empty
    else:
        last_valid_idx = non_pad_indices[-1].item()
        trimmed = tokens[: last_valid_idx + 1]  # include the last valid token

    response = tokenizer.decode(trimmed, skip_special_tokens=False)

    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token
    response = response.replace(pad_token, "").replace(eos_token, "")

