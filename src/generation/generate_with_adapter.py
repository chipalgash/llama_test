#!/usr/bin/env python3
"""Generate outputs from Llama 3 with a trained LoRA adapter.

This script is intended for Colab/GPU after `src.training.train_qlora` has
produced an adapter directory.
"""

from __future__ import annotations

import argparse
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

import csv
import torch


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payload_rows = read_jsonl(args.eval_jsonl, args.max_samples)
    tokenizer, base_model = load_model(args.model_name, args.load_in_4bit)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

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
        for row in payload_rows:
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


if __name__ == "__main__":
    main()
