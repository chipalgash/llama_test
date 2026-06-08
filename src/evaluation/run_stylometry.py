#!/usr/bin/env python3
"""Build stylometric feature and metric summary CSVs for one language."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluation.metrics import FEATURE_COLUMNS, stylometric_features
from src.evaluation.statistical_tests import (
    clean_float,
    mann_whitney_p_value,
    mean,
    median,
    rank_biserial,
    stdev,
)
from src.utils.logging import StepTimer, get_logger, log_kv, log_stage


LOGGER = get_logger("run_stylometry")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "ru"], required=True)
    parser.add_argument("--human-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--finetuned-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    return parser.parse_args()


def read_human_jsonl(path: Path, language: str) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "id": item["id"],
                    "language": language,
                    "condition": "human",
                    "prompt": item.get("prompt", ""),
                    "text": item.get("text", ""),
                }
            )
    return rows


def read_generation_csv(path: Path, condition: str, language: str) -> list[dict[str, str]]:
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
                    "text": row.get("generated_text") or row.get("text", ""),
                }
            )
        return rows


def write_features(path: Path, source_rows: list[dict[str, str]], language: str) -> list[dict[str, str | float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in source_rows:
        features = stylometric_features(row["text"], language)
        rows.append({**row, **features})
    fieldnames = ["id", "language", "condition", "prompt", "text", *FEATURE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(path: Path, feature_rows: list[dict[str, str | float]]) -> None:
    comparisons = [
        ("human", "baseline"),
        ("human", "finetuned"),
        ("baseline", "finetuned"),
    ]
    summary_rows = []
    for left, right in comparisons:
        for feature in FEATURE_COLUMNS:
            left_values = [
                clean_float(row[feature])
                for row in feature_rows
                if row["condition"] == left
            ]
            right_values = [
                clean_float(row[feature])
                for row in feature_rows
                if row["condition"] == right
            ]
            if not left_values or not right_values:
                continue
            summary_rows.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "feature": feature,
                    "left_condition": left,
                    "right_condition": right,
                    "left_n": len(left_values),
                    "right_n": len(right_values),
                    "left_mean": mean(left_values),
                    "right_mean": mean(right_values),
                    "left_median": median(left_values),
                    "right_median": median(right_values),
                    "left_std": stdev(left_values),
                    "right_std": stdev(right_values),
                    "mann_whitney_p_value": mann_whitney_p_value(left_values, right_values),
                    "rank_biserial": rank_biserial(left_values, right_values),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison",
        "feature",
        "left_condition",
        "right_condition",
        "left_n",
        "right_n",
        "left_mean",
        "right_mean",
        "left_median",
        "right_median",
        "left_std",
        "right_std",
        "mann_whitney_p_value",
        "rank_biserial",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    timer = StepTimer()
    args = parse_args()
    log_stage(LOGGER, "Stylometry started")
    log_kv(
        LOGGER,
        {
            "language": args.language,
            "human_jsonl": args.human_jsonl,
            "baseline_csv": args.baseline_csv,
            "finetuned_csv": args.finetuned_csv,
            "features_csv": args.features_csv,
            "summary_csv": args.summary_csv,
        },
    )
    source_rows = [
        *read_human_jsonl(args.human_jsonl, args.language),
        *read_generation_csv(args.baseline_csv, "baseline", args.language),
        *read_generation_csv(args.finetuned_csv, "finetuned", args.language),
    ]
    LOGGER.info("Rows loaded for feature extraction: %s", len(source_rows))
    feature_rows = write_features(args.features_csv, source_rows, args.language)
    write_summary(args.summary_csv, feature_rows)
    log_stage(LOGGER, "Stylometry finished")
    LOGGER.info("Wrote %s feature rows to %s", len(feature_rows), args.features_csv)
    LOGGER.info("Wrote metric summary to %s", args.summary_csv)
    LOGGER.info("Total elapsed: %s", timer.elapsed())


if __name__ == "__main__":
    main()
