#!/usr/bin/env python3
"""Convert existing EN/RU payload CSV files to the normalized thesis JSONL format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULTS = {
    "en": {
        "train_csv": Path("archive/old_experiments/payloads/en_csv/train_with_descriptions.csv"),
        "eval_csv": Path("archive/old_experiments/payloads/en_csv/eval_with_descriptions.csv"),
        "train_jsonl": Path("data/processed/en/train_payload.jsonl"),
        "eval_jsonl": Path("data/processed/en/eval_payload.jsonl"),
        "metadata_json": Path("data/processed/en/metadata.json"),
        "source": "english_creepypasta",
    },
    "ru": {
        "train_csv": Path("archive/old_experiments/payloads/ru_csv/train_with_descriptions.csv"),
        "eval_csv": Path("archive/old_experiments/payloads/ru_csv/eval_with_descriptions.csv"),
        "train_jsonl": Path("data/processed/ru/train_payload.jsonl"),
        "eval_jsonl": Path("data/processed/ru/eval_payload.jsonl"),
        "metadata_json": Path("data/processed/ru/metadata.json"),
        "source": "russian_creepypasta",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "ru", "all"], default="all")
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-eval", type=int, default=100)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_id(language: str, split: str, index: int, row: dict[str, str]) -> str:
    raw = "|".join(
        [
            language,
            split,
            str(index),
            row.get("story_id", ""),
            row.get("chunk_id", ""),
            row.get("text_sha256", ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{language}_{split}_{index:06d}_{digest}"


def normalize_row(
    row: dict[str, str],
    *,
    language: str,
    split: str,
    index: int,
    source: str,
) -> dict[str, Any]:
    return {
        "id": stable_id(language, split, index, row),
        "story_id": row.get("story_id", ""),
        "language": language,
        "genre": "horror",
        "prompt": row.get("description", ""),
        "text": row.get("text", ""),
        "source": source,
        "split": split,
        "source_file": row.get("source_file", ""),
        "title": row.get("title", ""),
        "chunk_id": int(row.get("chunk_id") or 0),
        "text_sha256": row.get("text_sha256", ""),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_language(language: str, max_train: int, max_eval: int) -> dict[str, Any]:
    settings = DEFAULTS[language]
    train_source = read_csv(settings["train_csv"])
    eval_source = read_csv(settings["eval_csv"])

    train_rows = [
        normalize_row(row, language=language, split="train", index=index, source=settings["source"])
        for index, row in enumerate(train_source[:max_train], start=1)
    ]
    eval_rows = [
        normalize_row(row, language=language, split="eval", index=index, source=settings["source"])
        for index, row in enumerate(eval_source[:max_eval], start=1)
    ]

    write_jsonl(settings["train_jsonl"], train_rows)
    write_jsonl(settings["eval_jsonl"], eval_rows)

    metadata = {
        "language": language,
        "source_train_csv": str(settings["train_csv"]),
        "source_eval_csv": str(settings["eval_csv"]),
        "train_jsonl": str(settings["train_jsonl"]),
        "eval_jsonl": str(settings["eval_jsonl"]),
        "source_train_rows": len(train_source),
        "source_eval_rows": len(eval_source),
        "written_train_rows": len(train_rows),
        "written_eval_rows": len(eval_rows),
        "story_level_split": True,
        "notes": "Generated from existing story-level split CSV files.",
    }
    settings["metadata_json"].parent.mkdir(parents=True, exist_ok=True)
    settings["metadata_json"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    languages = ["en", "ru"] if args.language == "all" else [args.language]
    summaries = [convert_language(language, args.max_train, args.max_eval) for language in languages]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
