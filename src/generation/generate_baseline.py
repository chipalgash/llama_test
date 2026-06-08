#!/usr/bin/env python3
"""Generate baseline outputs from meta-llama/Meta-Llama-3-8B-Instruct.

This script is intended for Colab/GPU. It does not fine-tune the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.generation.prompts import prompt_for_language
from src.training.lora_config import MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "ru"], required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
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


def read_jsonl(path: Path, limit: int) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def load_model(model_name: str, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    model.eval()
    return tokenizer, model


def build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_text(tokenizer, model, prompt: str, args: argparse.Namespace) -> str:
    chat_prompt = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payload_rows = read_jsonl(args.eval_jsonl, args.max_samples)
    tokenizer, model = load_model(args.model_name, args.load_in_4bit)

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
                    "condition": "baseline",
                    "prompt": prompt,
                    "generated_text": generate_text(tokenizer, model, prompt, args),
                    "model_name": args.model_name,
                    "adapter_name": "",
                    "generation_temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                }
            )


if __name__ == "__main__":
    main()
