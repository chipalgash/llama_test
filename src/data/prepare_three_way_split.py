#!/usr/bin/env python3
"""Create deterministic train/validation/test JSONL payloads for EN and RU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_SIZE = 500
DEFAULT_VALIDATION_SIZE = 100
DEFAULT_TEST_SIZE = 100
DEFAULT_SEED = 42

LANGUAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "en": {
        "train_csv": Path(
            "archive/old_experiments/payloads/en_csv/train_with_descriptions.csv"
        ),
        "eval_csv": Path(
            "archive/old_experiments/payloads/en_csv/eval_with_descriptions.csv"
        ),
        "output_dir": Path("data/processed/en"),
        "source": "english_creepypasta",
        "default_genre": "horror",
        "system": (
            "You are a literary fiction writer specializing in horror and thriller. "
            "Write only the requested fictional scene without explanations or "
            "commentary."
        ),
    },
    "ru": {
        "train_csv": Path(
            "archive/old_experiments/payloads/ru_csv/train_with_descriptions.csv"
        ),
        "eval_csv": Path(
            "archive/old_experiments/payloads/ru_csv/eval_with_descriptions.csv"
        ),
        "output_dir": Path("data/processed/ru"),
        "source": "russian_creepypasta",
        "default_genre": "хоррор",
        "system": (
            "Ты автор художественной прозы, специализирующийся на ужасах и "
            "триллерах. Пиши только запрошенную сцену без пояснений и "
            "метакомментариев."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["all", "en", "ru"], default="all")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    missing_columns = {"story_id", "text", "description"} - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV file {path} is missing required columns: {missing}")

    return rows


def group_by_story(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for index, row in enumerate(rows, start=1):
        story_id = row.get("story_id", "").strip()
        if not story_id:
            raise ValueError(f"Row {index} has an empty story_id")
        groups[story_id].append(row)

    return dict(groups)


def select_exact_story_groups(
    groups: dict[str, list[dict[str, str]]],
    target_size: int,
    seed: int,
    label: str,
) -> list[str]:
    if target_size < 0:
        raise ValueError(f"{label} target size must be non-negative")

    story_ids = sorted(groups)
    random.Random(seed).shuffle(story_ids)

    choices_by_size: dict[int, list[str]] = {0: []}
    for story_id in story_ids:
        story_size = len(groups[story_id])
        for current_size, current_story_ids in list(choices_by_size.items()):
            new_size = current_size + story_size
            if new_size > target_size or new_size in choices_by_size:
                continue

            choices_by_size[new_size] = current_story_ids + [story_id]
            if new_size == target_size:
                return choices_by_size[new_size]

    available_rows = sum(len(rows) for rows in groups.values())
    raise ValueError(
        f"Could not create {label} split with exactly {target_size} rows "
        f"without splitting story_id groups. Available rows: {available_rows}."
    )


def rows_for_story_ids(
    groups: dict[str, list[dict[str, str]]],
    story_ids: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for story_id in story_ids:
        rows.extend(groups[story_id])
    return rows


def make_three_way_split(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    train_groups = group_by_story(train_rows)
    eval_groups = group_by_story(eval_rows)

    train_story_ids = select_exact_story_groups(
        train_groups,
        train_size,
        seed,
        "train",
    )

    validation_story_ids = select_exact_story_groups(
        eval_groups,
        validation_size,
        seed + 1,
        "validation",
    )
    remaining_eval_groups = {
        story_id: rows
        for story_id, rows in eval_groups.items()
        if story_id not in set(validation_story_ids)
    }
    test_story_ids = select_exact_story_groups(
        remaining_eval_groups,
        test_size,
        seed + 2,
        "test",
    )

    splits = {
        "train": rows_for_story_ids(train_groups, train_story_ids),
        "validation": rows_for_story_ids(eval_groups, validation_story_ids),
        "test": rows_for_story_ids(eval_groups, test_story_ids),
    }
    validate_no_story_overlap(splits)
    return splits


def story_ids_for_rows(rows: list[dict[str, str]]) -> set[str]:
    return {row["story_id"] for row in rows}


def story_overlaps(
    splits: dict[str, list[dict[str, str]]],
) -> dict[str, int]:
    train_story_ids = story_ids_for_rows(splits["train"])
    validation_story_ids = story_ids_for_rows(splits["validation"])
    test_story_ids = story_ids_for_rows(splits["test"])

    return {
        "train_validation": len(train_story_ids & validation_story_ids),
        "train_test": len(train_story_ids & test_story_ids),
        "validation_test": len(validation_story_ids & test_story_ids),
    }


def validate_no_story_overlap(splits: dict[str, list[dict[str, str]]]) -> None:
    overlaps = story_overlaps(splits)
    non_zero_overlaps = {name: count for name, count in overlaps.items() if count != 0}
    if non_zero_overlaps:
        raise ValueError(f"story_id overlap detected: {non_zero_overlaps}")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_payload_record(
    row: dict[str, str],
    language: str,
    split: str,
    index: int,
    config: dict[str, Any],
) -> dict[str, str]:
    text = row.get("text", "")
    story_id = row["story_id"].strip()
    chunk_id = row.get("chunk_id", "").strip()
    genre = (
        row.get("genre", "").strip()
        or row.get("category", "").strip()
        or config["default_genre"]
    )

    return {
        "id": f"{language}_{split}_{index:06d}",
        "story_id": story_id,
        "language": language,
        "genre": genre,
        "system": config["system"],
        "prompt": row.get("description", ""),
        "text": text,
        "source": config["source"],
        "split": split,
        "source_file": row.get("source_file", ""),
        "title": row.get("title", ""),
        "chunk_id": chunk_id,
        "text_sha256": row.get("text_sha256", "").strip() or text_sha256(text),
    }


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(
    output_dir: Path,
    language: str,
    splits: dict[str, list[dict[str, str]]],
    overlaps: dict[str, int],
    seed: int,
) -> None:
    metadata = {
        "language": language,
        "seed": seed,
        "sizes": {
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
        },
        "story_id_overlap": overlaps,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def process_language(
    language: str,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> None:
    config = LANGUAGE_CONFIGS[language]
    train_rows = read_csv_rows(config["train_csv"])
    eval_rows = read_csv_rows(config["eval_csv"])

    splits = make_three_way_split(
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )
    overlaps = story_overlaps(splits)

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_names = {
        "train": "train_payload.jsonl",
        "validation": "validation_payload.jsonl",
        "test": "test_payload.jsonl",
    }

    for split, rows in splits.items():
        records = [
            build_payload_record(row, language, split, index, config)
            for index, row in enumerate(rows, start=1)
        ]
        write_jsonl(output_dir / output_names[split], records)

    write_metadata(output_dir, language, splits, overlaps, seed)


def main() -> None:
    args = parse_args()
    languages = ["en", "ru"] if args.language == "all" else [args.language]

    for language in languages:
        process_language(
            language=language,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
