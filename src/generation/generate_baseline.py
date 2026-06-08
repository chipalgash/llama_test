#!/usr/bin/env python3
"""Generate baseline outputs from meta-llama/Meta-Llama-3-8B-Instruct.

This script is intended for Colab/GPU. It does not fine-tune the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.generation.prompts import prompt_for_language
from src.training.lora_config import MODEL_NAME
from src.utils.logging import StepTimer, format_seconds, get_logger, log_kv, log_stage


LOGGER = get_logger("generate_baseline")


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


def resolve_hf_token(token: str | None = None) -> str | None:
    return token or os.environ.get("HF_TOKEN")


def load_model(model_name: str, load_in_4bit: bool, hf_token: str | None = None):
    token = resolve_hf_token(hf_token)
    log_stage(LOGGER, "Loading tokenizer")
    log_kv(LOGGER, {"model_name": model_name, "hf_token_present": bool(token)})
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info("Tokenizer ready. pad_token_id=%s eos_token_id=%s", tokenizer.pad_token_id, tokenizer.eos_token_id)

    quantization_config = None
    if load_in_4bit:
        LOGGER.info("Using 4-bit NF4 quantization with float16 compute.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    log_stage(LOGGER, "Loading base model")
    timer = StepTimer()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        token=token,
    )
    model.eval()
    LOGGER.info("Model ready in %s.", timer.elapsed())
    return tokenizer, model


def build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_text(tokenizer, model, prompt: str, args: argparse.Namespace) -> str:
    chat_prompt = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    max_length = inputs["input_ids"].shape[-1] + args.max_new_tokens
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def main() -> None:
    total_timer = StepTimer()
    args = parse_args()
    log_stage(LOGGER, "Baseline generation started")
    log_kv(
        LOGGER,
        {
            "language": args.language,
            "eval_jsonl": args.eval_jsonl,
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
    tokenizer, model = load_model(args.model_name, args.load_in_4bit, args.hf_token)

    log_stage(LOGGER, "Generating baseline outputs")
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
            if index == 1 or index % 5 == 0 or index == len(payload_rows):
                elapsed = time.monotonic() - row_started_at
                LOGGER.info(
                    "Generated %s/%s rows | last_row=%s | total_elapsed=%s",
                    index,
                    len(payload_rows),
                    format_seconds(elapsed),
                    total_timer.elapsed(),
                )
    log_stage(LOGGER, "Baseline generation finished")
    LOGGER.info("Saved CSV: %s", args.output_csv)
    LOGGER.info("Total elapsed: %s", total_timer.elapsed())


if __name__ == "__main__":
    main()
