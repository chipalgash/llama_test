#!/usr/bin/env python3
"""Train a small QLoRA adapter for the Llama 3 thesis experiment.

This script is intended for Colab/GPU. It reads normalized JSONL rows with
`prompt` and `text` fields and saves only the LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.training.lora_config import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    TARGET_MODULES,
)
from src.utils.logging import StepTimer, get_logger, log_kv, log_stage


LOGGER = get_logger("train_qlora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-train-samples", type=int, default=500)
    parser.add_argument("--max-eval-samples", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def format_chat_example(tokenizer, row: dict[str, str]) -> str:
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["text"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def tokenize_rows(tokenizer, rows: list[dict[str, str]], max_seq_length: int) -> Dataset:
    texts = [format_chat_example(tokenizer, row) for row in rows]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    total_timer = StepTimer()
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    log_stage(LOGGER, "QLoRA training started")
    log_kv(
        LOGGER,
        {
            "train_jsonl": args.train_jsonl,
            "eval_jsonl": args.eval_jsonl,
            "adapter_dir": args.adapter_dir,
            "model_name": args.model_name,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "hf_token_present": bool(hf_token),
        },
    )
    log_kv(
        LOGGER,
        {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": ",".join(TARGET_MODULES),
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        },
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        log_kv(
            LOGGER,
            {
                "cuda_available": True,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            },
        )
    else:
        LOGGER.warning("CUDA is not available. This script is expected to run on a GPU runtime.")

    log_stage(LOGGER, "Loading tokenizer")
    tokenizer_timer = StepTimer()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info("Tokenizer ready in %s. pad_token_id=%s eos_token_id=%s", tokenizer_timer.elapsed(), tokenizer.pad_token_id, tokenizer.eos_token_id)

    log_stage(LOGGER, "Loading base model in 4-bit")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_timer = StepTimer()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        token=hf_token,
    )
    LOGGER.info("Base model loaded in %s.", model_timer.elapsed())
    model.config.use_cache = False
    log_stage(LOGGER, "Preparing LoRA adapter")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    try:
        model.print_trainable_parameters()
    except Exception:
        LOGGER.info("Trainable parameter summary is unavailable for this model object.")

    log_stage(LOGGER, "Reading and tokenizing datasets")
    train_rows = read_jsonl(args.train_jsonl, args.max_train_samples)
    eval_rows = read_jsonl(args.eval_jsonl, args.max_eval_samples)
    LOGGER.info("Loaded train rows: %s", len(train_rows))
    LOGGER.info("Loaded eval rows : %s", len(eval_rows))
    tokenize_timer = StepTimer()
    train_dataset = tokenize_rows(tokenizer, train_rows, args.max_seq_length)
    eval_dataset = tokenize_rows(tokenizer, eval_rows, args.max_seq_length)
    LOGGER.info("Tokenized datasets in %s.", tokenize_timer.elapsed())
    LOGGER.info("Train dataset size: %s", len(train_dataset))
    LOGGER.info("Eval dataset size : %s", len(eval_dataset))

    training_args = TrainingArguments(
        output_dir=str(args.adapter_dir.parent / f"{args.adapter_dir.name}_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        fp16=True,
        seed=args.seed,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    log_stage(LOGGER, "Training")
    train_timer = StepTimer()
    trainer.train()
    LOGGER.info("Training finished in %s.", train_timer.elapsed())

    log_stage(LOGGER, "Saving adapter")
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    LOGGER.info("Saved adapter to %s", args.adapter_dir)
    LOGGER.info("Total elapsed: %s", total_timer.elapsed())


if __name__ == "__main__":
    main()
