#!/usr/bin/env python3
"""Create manual evaluation CSV templates from generation outputs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.utils.logging import StepTimer, get_logger, log_kv, log_stage


LOGGER = get_logger("manual_eval_template")


EVAL_COLUMNS = [
    "id",
    "language",
    "condition",
    "prompt",
    "generated_text",
    "genre_score_1_5",
    "suspense_score_1_5",
    "language_naturalness_1_5",
    "coherence_1_5",
    "originality_1_5",
    "comment",
]

BLINDED_COLUMNS = [
    "blind_id",
    "language",
    "prompt",
    "generated_text",
    "genre_score_1_5",
    "suspense_score_1_5",
    "language_naturalness_1_5",
    "coherence_1_5",
    "originality_1_5",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-baseline", type=Path, default=Path("outputs/en/baseline_generations.csv"))
    parser.add_argument("--en-finetuned", type=Path, default=Path("outputs/en/finetuned_generations.csv"))
    parser.add_argument("--ru-baseline", type=Path, default=Path("outputs/ru/baseline_generations.csv"))
    parser.add_argument("--ru-finetuned", type=Path, default=Path("outputs/ru/finetuned_generations.csv"))
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--en-output", type=Path, default=Path("outputs/en/manual_eval_sample.csv"))
    parser.add_argument("--ru-output", type=Path, default=Path("outputs/ru/manual_eval_sample.csv"))
    parser.add_argument("--blinded-output", type=Path, default=Path("outputs/manual_eval_blinded.csv"))
    parser.add_argument("--mapping-output", type=Path, default=Path("outputs/manual_eval_blinded_mapping.csv"))
    return parser.parse_args()


def read_rows(path: Path, language: str, condition: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "id": row.get("id", ""),
                    "language": row.get("language", language),
                    "condition": row.get("condition", condition),
                    "prompt": row.get("prompt", ""),
                    "generated_text": row.get("generated_text") or row.get("text", ""),
                    "genre_score_1_5": "",
                    "suspense_score_1_5": "",
                    "language_naturalness_1_5": "",
                    "coherence_1_5": "",
                    "originality_1_5": "",
                    "comment": "",
                }
            )
        return rows


def sample_rows(rows: list[dict[str, str]], sample_size: int, rng: random.Random) -> list[dict[str, str]]:
    if len(rows) <= sample_size:
        return rows[:]
    return rng.sample(rows, sample_size)


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    timer = StepTimer()
    args = parse_args()
    log_stage(LOGGER, "Manual evaluation template started")
    log_kv(
        LOGGER,
        {
            "sample_size_per_condition": args.sample_size,
            "seed": args.seed,
            "en_output": args.en_output,
            "ru_output": args.ru_output,
            "blinded_output": args.blinded_output,
            "mapping_output": args.mapping_output,
        },
    )
    rng = random.Random(args.seed)

    en_rows = [
        *sample_rows(read_rows(args.en_baseline, "en", "baseline"), args.sample_size, rng),
        *sample_rows(read_rows(args.en_finetuned, "en", "finetuned"), args.sample_size, rng),
    ]
    ru_rows = [
        *sample_rows(read_rows(args.ru_baseline, "ru", "baseline"), args.sample_size, rng),
        *sample_rows(read_rows(args.ru_finetuned, "ru", "finetuned"), args.sample_size, rng),
    ]

    write_csv(args.en_output, en_rows, EVAL_COLUMNS)
    write_csv(args.ru_output, ru_rows, EVAL_COLUMNS)
    LOGGER.info("EN manual eval rows: %s", len(en_rows))
    LOGGER.info("RU manual eval rows: %s", len(ru_rows))

    all_rows = en_rows + ru_rows
    rng.shuffle(all_rows)
    blinded_rows = []
    mapping_rows = []
    for index, row in enumerate(all_rows, start=1):
        blind_id = f"blind_{index:04d}"
        blinded_rows.append(
            {
                "blind_id": blind_id,
                "language": row["language"],
                "prompt": row["prompt"],
                "generated_text": row["generated_text"],
                "genre_score_1_5": "",
                "suspense_score_1_5": "",
                "language_naturalness_1_5": "",
                "coherence_1_5": "",
                "originality_1_5": "",
                "comment": "",
            }
        )
        mapping_rows.append(
            {
                "blind_id": blind_id,
                "id": row["id"],
                "language": row["language"],
                "condition": row["condition"],
            }
        )

    write_csv(args.blinded_output, blinded_rows, BLINDED_COLUMNS)
    write_csv(args.mapping_output, mapping_rows, ["blind_id", "id", "language", "condition"])
    log_stage(LOGGER, "Manual evaluation template finished")
    LOGGER.info("Wrote %s blinded rows to %s", len(blinded_rows), args.blinded_output)
    LOGGER.info("Wrote condition mapping to %s", args.mapping_output)
    LOGGER.info("Total elapsed: %s", timer.elapsed())


if __name__ == "__main__":
    main()
