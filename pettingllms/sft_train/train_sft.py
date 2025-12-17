"""
Qwen3-8B SFT Training Script

This script fine-tunes Qwen3-8B on the collected SFT data from PyChecker agents.
Supports both full fine-tuning and LoRA for efficient training.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QwenSFTConfig:
    """Configuration for Qwen3-8B SFT training"""

    # Model settings
    model_name_or_path: str = "Qwen/Qwen2.5-8B"  # or "Qwen/Qwen3-8B" when available
    tokenizer_name_or_path: Optional[str] = None

    # Data settings
    train_file: str = "./sft_data/sft_data.jsonl"
    val_file: Optional[str] = None
    max_seq_length: int = 4096

    # Training hyperparameters
    output_dir: str = "./qwen_sft_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimization
    optim: str = "adamw_torch"
    fp16: bool = False
    bf16: bool = True  # Use bf16 for better stability with Qwen
    gradient_checkpointing: bool = True

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"

    # Other
    seed: int = 42
    report_to: str = "tensorboard"


def load_qwen_model_and_tokenizer(config: QwenSFTConfig):
    """
    Load Qwen model and tokenizer

    Args:
        config: SFT configuration

    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model: {config.model_name_or_path}")

    # Load tokenizer
    tokenizer_path = config.tokenizer_name_or_path or config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="right",  # Important for training
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
    )

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Required for gradient checkpointing

    # Apply LoRA if enabled
    if config.use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Preprocess examples for Qwen chat format

    Expected format in JSONL:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    # Apply Qwen chat template
    texts = []
    for messages in examples["messages"]:
        # Use Qwen's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding=False,  # Dynamic padding in data collator
    )

    # Labels are the same as input_ids for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def load_and_prepare_dataset(config: QwenSFTConfig, tokenizer):
    """
    Load and prepare dataset for training

    Args:
        config: SFT configuration
        tokenizer: Tokenizer

    Returns:
        train_dataset, eval_dataset (optional)
    """
    logger.info(f"Loading dataset from: {config.train_file}")

    # Load dataset
    data_files = {"train": config.train_file}
    if config.val_file:
        data_files["validation"] = config.val_file

    dataset = load_dataset("json", data_files=data_files)

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation dataset size: {len(dataset['validation'])}")

    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset",
    )

    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset.get("validation", None)

    return train_dataset, eval_dataset


def train_qwen_sft(config: QwenSFTConfig):
    """
    Main training function for Qwen SFT

    Args:
        config: SFT configuration
    """
    logger.info("Starting Qwen SFT training...")
    logger.info(f"Configuration: {config}")

    # Set seed for reproducibility
    torch.manual_seed(config.seed)

    # Load model and tokenizer
    model, tokenizer = load_qwen_model_and_tokenizer(config)

    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)

    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        evaluation_strategy=config.evaluation_strategy if eval_dataset else "no",
        load_best_model_at_end=config.load_best_model_at_end if eval_dataset else False,
        metric_for_best_model=config.metric_for_best_model if eval_dataset else None,
        seed=config.seed,
        report_to=config.report_to,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save state
    trainer.save_state()

    logger.info("Training completed!")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-8B on SFT data")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-8B",
                       help="Model name or path")
    parser.add_argument("--tokenizer-name", type=str, default=None,
                       help="Tokenizer name or path (default: same as model)")

    # Data arguments
    parser.add_argument("--train-file", type=str, required=True,
                       help="Path to training data JSONL file")
    parser.add_argument("--val-file", type=str, default=None,
                       help="Path to validation data JSONL file")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                       help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--output-dir", type=str, default="./qwen_sft_output",
                       help="Output directory for model and logs")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")

    # LoRA arguments
    parser.add_argument("--use-lora", action="store_true", default=True,
                       help="Use LoRA for efficient training")
    parser.add_argument("--lora-r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha")

    # Other arguments
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bfloat16 precision")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Create config
    config = QwenSFTConfig(
        model_name_or_path=args.model_name,
        tokenizer_name_or_path=args.tokenizer_name,
        train_file=args.train_file,
        val_file=args.val_file,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bf16=args.bf16,
        seed=args.seed,
    )

    # Train
    train_qwen_sft(config)


if __name__ == "__main__":
    main()
