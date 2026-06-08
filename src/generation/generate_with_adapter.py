#!/usr/bin/env python3
"""Generate outputs from Llama 3 with a trained LoRA adapter.

This script is intended for Colab/GPU after `src.training.train_qlora` has
produced an adapter directory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from peft import PeftModel

from src.generation.generate_baseline import (
    build_chat_prompt,
    generate_text,
    load_model,
    read_jsonl,
)
from src.generation.prompts import prompt_for_language
from src.training.lora_config import MODEL_NAME
from src.utils.logging import StepTimer, format_seconds, get_logger, log_kv, log_stage

import csv
import torch


LOGGER = get_logger("generate_with_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "ru"], required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--prompt-source", choices=["fixed", "payload"], default="fixed")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    return parser.parse_args()


def main() -> None:
    total_timer = StepTimer()
    args = parse_args()
    log_stage(LOGGER, "Fine-tuned generation started")
    log_kv(
        LOGGER,
        {
            "language": args.language,
            "eval_jsonl": args.eval_jsonl,
            "adapter_dir": args.adapter_dir,
            "output_csv": args.output_csv,
            "model_name": args.model_name,
            "max_samples": args.max_samples,
            "prompt_source": args.prompt_source,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "load_in_4bit": args.load_in_4bit,
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

    log_stage(LOGGER, "Reading evaluation payload")
    payload_rows = read_jsonl(args.eval_jsonl, args.max_samples)
    LOGGER.info("Loaded %s eval rows from %s.", len(payload_rows), args.eval_jsonl)
    tokenizer, base_model = load_model(args.model_name, args.load_in_4bit, args.hf_token)
    log_stage(LOGGER, "Loading LoRA adapter")
    adapter_timer = StepTimer()
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    LOGGER.info("Adapter loaded from %s in %s.", args.adapter_dir, adapter_timer.elapsed())

    log_stage(LOGGER, "Generating fine-tuned outputs")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "language",
        "condition",
        "prompt",
        "generated_text",
        "model_name",
        "adapter_name",
        "generation_temperature",
        "top_p",
        "max_new_tokens",
        "seed",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(payload_rows, start=1):
            row_started_at = time.monotonic()
            prompt = row.get("prompt", "") if args.prompt_source == "payload" else prompt_for_language(args.language)
            writer.writerow(
                {
                    "id": row["id"],
                    "language": args.language,
                    "condition": "finetuned",
                    "prompt": prompt,
                    "generated_text": generate_text(tokenizer, model, prompt, args),
                    "model_name": args.model_name,
                    "adapter_name": str(args.adapter_dir),
                    "generation_temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                }
            )
            if index == 1 or index % 5 == 0 or index == len(payload_rows):
                elapsed = time.monotonic() - row_started_at
                LOGGER.info(
                    "Generated %s/%s rows | last_row=%s | total_elapsed=%s",
                    index,
                    len(payload_rows),
                    format_seconds(elapsed),
                    total_timer.elapsed(),
                )
    log_stage(LOGGER, "Fine-tuned generation finished")
    LOGGER.info("Saved CSV: %s", args.output_csv)
    LOGGER.info("Total elapsed: %s", total_timer.elapsed())


if __name__ == "__main__":
    main()
