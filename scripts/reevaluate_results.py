#!/usr/bin/env python3
"""Re-evaluate completed Colab horror experiment outputs locally.

The script intentionally uses only the Python standard library. It reads the
CSV files produced by the Colab notebook and reports diagnostics that are easy
to miss in the original smoke-run metrics: sample sizes, length bias, artifact
rates, robust stylometric effects, and an inferred detector confusion matrix.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REAL_ARTIFACT_MARKERS = [
    "GraphicsUnit",
    "#echo",
    "AnchorStyles",
    "Compatible",
    "DOWNLOADING",
    "다운받기",
    "وینت",
    "namespace",
    "href=",
    "http://",
    "https://",
    "/**/",
    "Object#",
    "gcاسطة",
    "Hlav",
]

LOOSE_NOTEBOOK_MARKERS = REAL_ARTIFACT_MARKERS + ["FIRST", "LAST"]

FEAR_WORDS = {
    "fear",
    "afraid",
    "scared",
    "terror",
    "horror",
    "dark",
    "darkness",
    "blood",
    "dead",
    "death",
    "scream",
    "screaming",
    "shadow",
    "shadows",
    "night",
    "nightmare",
    "ghost",
    "monster",
    "creature",
    "door",
    "window",
    "whisper",
    "whispers",
    "cold",
    "alone",
    "body",
    "grave",
    "madness",
    "evil",
}

CLICHE_PATTERNS = [
    "shiver down my spine",
    "chill down my spine",
    "darkness seemed to",
    "the darkness seemed",
    "couldn't shake the feeling",
    "could not shake the feeling",
    "heart pounded",
    "heart hammered",
    "blood ran cold",
    "air grew thick",
    "heavy with the scent",
    "like living things",
    "unseen presence",
    "whispered my name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print local diagnostics for a completed horror experiment output directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_success"),
        help="Directory containing detector_summary.csv and stylometric_features.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown report path. If omitted, only stdout is used.",
    )
    parser.add_argument(
        "--length-normalized-words",
        type=int,
        default=0,
        help=(
            "Word count for length-normalized feature recomputation. "
            "Default 0 chooses the smallest label median word count."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']*", str(text).lower())


def trim_to_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    matches = list(re.finditer(r"[A-Za-z][A-Za-z']*", str(text)))
    if len(matches) <= max_words:
        return text
    return str(text)[: matches[max_words - 1].end()]


def extract_features(text: str) -> dict[str, float]:
    word_list = words(text)
    return {
        "n_chars": float(len(str(text))),
        "n_words": float(len(word_list)),
        "type_token_ratio": len(set(word_list)) / max(1, len(word_list)),
        "avg_word_len": statistics.fmean([len(word) for word in word_list]) if word_list else 0.0,
        "exclamation_count": float(str(text).count("!")),
        "question_count": float(str(text).count("?")),
        "ellipsis_count": float(str(text).count("...")),
        "quote_count": float(str(text).count('"') + str(text).count("“") + str(text).count("”")),
        "fear_word_rate": sum(1 for word in word_list if word in FEAR_WORDS) / max(1, len(word_list)),
    }


def marker_hits(text: str, markers: Iterable[str]) -> list[str]:
    lowered = str(text).lower()
    return [marker for marker in markers if marker.lower() in lowered]


def cliche_hits(text: str) -> list[str]:
    lowered = str(text).lower()
    return [pattern for pattern in CLICHE_PATTERNS if pattern in lowered]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def pct(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def rank_biserial(x_values: list[float], y_values: list[float]) -> float:
    """Compute rank-biserial correlation from pairwise order comparisons."""
    if not x_values or not y_values:
        return float("nan")
    greater = 0.0
    for x_value in x_values:
        for y_value in y_values:
            if x_value > y_value:
                greater += 1.0
            elif x_value == y_value:
                greater += 0.5
    u_stat = greater
    return (2 * u_stat) / (len(x_values) * len(y_values)) - 1


def wilson_ci(successes: int, total: int, z_value: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p_hat = successes / total
    denom = 1 + z_value**2 / total
    centre = p_hat + z_value**2 / (2 * total)
    margin = z_value * math.sqrt((p_hat * (1 - p_hat) + z_value**2 / (4 * total)) / total)
    return (centre - margin) / denom, (centre + margin) / denom


def infer_confusion(row: dict[str, str], max_total: int = 500) -> dict[str, int] | None:
    """Infer integer TP/FP/FN/TN from rounded detector metrics when possible."""
    try:
        target_accuracy = float(row["accuracy"])
        target_precision = float(row["precision"])
        target_recall = float(row["recall"])
    except (KeyError, ValueError):
        return None

    best: tuple[float, dict[str, int]] | None = None
    for total in range(2, max_total + 1):
        correct = round(target_accuracy * total)
        if abs(correct / total - target_accuracy) > 1e-9:
            continue
        for positives in range(1, total):
            negatives = total - positives
            for tp in range(0, positives + 1):
                fn = positives - tp
                recall = tp / positives
                predicted_positive = round(tp / target_precision) if target_precision else 0
                if predicted_positive < tp:
                    continue
                fp = predicted_positive - tp
                if fp > negatives:
                    continue
                tn = negatives - fp
                if tp + tn != correct:
                    continue
                precision = tp / predicted_positive if predicted_positive else 0.0
                error = (
                    abs(precision - target_precision)
                    + abs(recall - target_recall)
                    + abs((tp + tn) / total - target_accuracy)
                )
                candidate = {"total": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
                if best is None or error < best[0] or (error == best[0] and total < best[1]["total"]):
                    best = (error, candidate)
    if best is None or best[0] > 1e-8:
        return None
    return best[1]


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def report(results_dir: Path, normalized_words: int) -> str:
    features_path = results_dir / "stylometric_features.csv"
    detector_path = results_dir / "detector_summary.csv"
    train_removed_path = results_dir / "train_removed_by_quality_filter.csv"
    eval_removed_path = results_dir / "eval_removed_by_quality_filter.csv"
    effects_path = results_dir / "rank_biserial_effects.csv"

    feature_rows = read_csv(features_path)
    detector_rows = read_csv(detector_path)
    train_removed_rows = read_csv(train_removed_path)
    eval_removed_rows = read_csv(eval_removed_path)
    existing_effect_rows = read_csv(effects_path)

    if not feature_rows:
        raise SystemExit(f"Missing or empty required file: {features_path}")

    labels = sorted({row["label"] for row in feature_rows})
    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        rows_by_label[row["label"]].append(row)

    word_medians = {
        label: median([float(row["n_words"]) for row in rows])
        for label, rows in rows_by_label.items()
    }
    if normalized_words <= 0:
        normalized_words = max(1, int(min(word_medians.values())))

    lines: list[str] = []
    lines.append("# Local Re-evaluation Report")
    lines.append("")
    lines.append(f"Results directory: `{results_dir}`")
    lines.append(f"Length-normalized recomputation: first `{normalized_words}` words per text")
    lines.append("")

    lines.append("## Files and Sample Sizes")
    size_rows = [
        ["stylometric_features.csv", str(len(feature_rows))],
        ["rank_biserial_effects.csv", str(len(existing_effect_rows))],
        ["detector_summary.csv", str(len(detector_rows))],
        ["train_removed_by_quality_filter.csv", str(len(train_removed_rows))],
        ["eval_removed_by_quality_filter.csv", str(len(eval_removed_rows))],
    ]
    lines.extend(table(["file", "rows"], size_rows))
    lines.append("")

    lines.append("## Label Distribution")
    label_rows = []
    for label in labels:
        label_rows.append([label, str(len(rows_by_label[label]))])
    lines.extend(table(["label", "n"], label_rows))
    lines.append("")

    lines.append("## Length and Surface Metrics")
    metric_rows = []
    for label in labels:
        rows = rows_by_label[label]
        chars = [float(row["n_chars"]) for row in rows]
        word_counts = [float(row["n_words"]) for row in rows]
        ttr = [float(row["type_token_ratio"]) for row in rows]
        fear = [float(row["fear_word_rate"]) for row in rows]
        quotes = [float(row["quote_count"]) for row in rows]
        metric_rows.append(
            [
                label,
                str(len(rows)),
                fmt(mean(chars), 1),
                fmt(median(chars), 1),
                fmt(mean(word_counts), 1),
                fmt(median(word_counts), 1),
                fmt(mean(ttr), 3),
                fmt(mean(fear), 4),
                fmt(mean(quotes), 2),
            ]
        )
    lines.extend(
        table(
            [
                "label",
                "n",
                "mean chars",
                "median chars",
                "mean words",
                "median words",
                "mean TTR",
                "mean fear rate",
                "mean quotes",
            ],
            metric_rows,
        )
    )
    lines.append("")

    lines.append("## Artifact and Cliche Rates")
    artifact_rows = []
    for label in labels:
        rows = rows_by_label[label]
        loose_count = 0
        real_count = 0
        cliche_count = 0
        loose_hits: Counter[str] = Counter()
        real_hits: Counter[str] = Counter()
        cliche_counter: Counter[str] = Counter()
        for row in rows:
            loose = marker_hits(row["text"], LOOSE_NOTEBOOK_MARKERS)
            real = marker_hits(row["text"], REAL_ARTIFACT_MARKERS)
            cliches = cliche_hits(row["text"])
            loose_count += bool(loose)
            real_count += bool(real)
            cliche_count += bool(cliches)
            loose_hits.update(loose)
            real_hits.update(real)
            cliche_counter.update(cliches)
        artifact_rows.append(
            [
                label,
                pct(loose_count / len(rows)),
                pct(real_count / len(rows)),
                pct(cliche_count / len(rows)),
                ", ".join(f"{key}:{value}" for key, value in loose_hits.most_common(4)) or "-",
                ", ".join(f"{key}:{value}" for key, value in real_hits.most_common(4)) or "-",
                ", ".join(f"{key}:{value}" for key, value in cliche_counter.most_common(4)) or "-",
            ]
        )
    lines.extend(
        table(
            [
                "label",
                "loose artifact rows",
                "real artifact rows",
                "cliche rows",
                "loose hits",
                "real hits",
                "top cliches",
            ],
            artifact_rows,
        )
    )
    lines.append("")

    if train_removed_rows or eval_removed_rows:
        lines.append("## Quality Filter Removals")
        removal_rows = []
        for split, rows in [("train", train_removed_rows), ("eval", eval_removed_rows)]:
            marker_counts = Counter(row.get("markers_found", "") or "<empty>" for row in rows)
            stories = {row.get("story_id", "") for row in rows}
            artifact_flag_count = sum(row.get("has_artifact_marker") == "True" for row in rows)
            repeated_count = sum(row.get("repeated_token_pattern") == "True" for row in rows)
            removal_rows.append(
                [
                    split,
                    str(len(rows)),
                    str(len(stories)),
                    pct(artifact_flag_count / len(rows)) if rows else "n/a",
                    str(repeated_count),
                    ", ".join(f"{key}:{value}" for key, value in marker_counts.most_common(5)) or "-",
                ]
            )
        lines.extend(
            table(
                ["split", "removed rows", "unique stories", "artifact flag rate", "repeated rows", "top markers"],
                removal_rows,
            )
        )
        lines.append("")

    feature_names = [
        "n_chars",
        "n_words",
        "type_token_ratio",
        "avg_word_len",
        "exclamation_count",
        "question_count",
        "ellipsis_count",
        "quote_count",
        "fear_word_rate",
    ]

    lines.append("## Human vs AI Effects")
    effect_rows = []
    for ai_label in [label for label in labels if label != "human"]:
        for feature in feature_names:
            human_values = [float(row[feature]) for row in rows_by_label.get("human", [])]
            ai_values = [float(row[feature]) for row in rows_by_label[ai_label]]
            rb = rank_biserial(human_values, ai_values)
            effect_rows.append([f"human_vs_{ai_label}", feature, fmt(rb, 3), fmt(abs(rb), 3)])
    effect_rows.sort(key=lambda row: (row[0], -float(row[3]) if row[3] != "n/a" else 0.0))
    lines.extend(table(["comparison", "feature", "rank biserial", "abs effect"], effect_rows))
    lines.append("")

    lines.append("## Length-normalized Effects")
    normalized_by_label: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in feature_rows:
        normalized_text = trim_to_words(row["text"], normalized_words)
        normalized_by_label[row["label"]].append(extract_features(normalized_text))

    normalized_rows = []
    for ai_label in [label for label in labels if label != "human"]:
        for feature in [
            "type_token_ratio",
            "avg_word_len",
            "exclamation_count",
            "question_count",
            "ellipsis_count",
            "quote_count",
            "fear_word_rate",
        ]:
            human_values = [row[feature] for row in normalized_by_label.get("human", [])]
            ai_values = [row[feature] for row in normalized_by_label[ai_label]]
            rb = rank_biserial(human_values, ai_values)
            normalized_rows.append([f"human_vs_{ai_label}", feature, fmt(rb, 3), fmt(abs(rb), 3)])
    normalized_rows.sort(key=lambda row: (row[0], -float(row[3]) if row[3] != "n/a" else 0.0))
    lines.extend(table(["comparison", "feature", "rank biserial", "abs effect"], normalized_rows))
    lines.append("")

    if detector_rows:
        lines.append("## Detector Summary")
        detector_table_rows = []
        for row in detector_rows:
            confusion = infer_confusion(row)
            if confusion:
                successes = confusion["tp"] + confusion["tn"]
                low, high = wilson_ci(successes, confusion["total"])
                confusion_text = (
                    f"TP={confusion['tp']}, FP={confusion['fp']}, "
                    f"FN={confusion['fn']}, TN={confusion['tn']}, n={confusion['total']}"
                )
                accuracy_ci = f"{pct(low)}-{pct(high)}"
            else:
                confusion_text = "not inferred"
                accuracy_ci = "n/a"
            detector_table_rows.append(
                [
                    row.get("scenario", ""),
                    fmt(float(row.get("accuracy", "nan")), 3),
                    fmt(float(row.get("precision", "nan")), 3),
                    fmt(float(row.get("recall", "nan")), 3),
                    fmt(float(row.get("f1", "nan")), 3),
                    fmt(float(row.get("roc_auc", "nan")), 3),
                    accuracy_ci,
                    confusion_text,
                ]
            )
        lines.extend(
            table(
                ["scenario", "accuracy", "precision", "recall", "f1", "roc_auc", "accuracy 95% CI", "inferred confusion"],
                detector_table_rows,
            )
        )
        lines.append("")

    lines.append("## Practical Reading")
    human_words = word_medians.get("human")
    ai_word_medians = [value for label, value in word_medians.items() if label != "human"]
    min_ai_words = min(ai_word_medians) if ai_word_medians else float("nan")
    if human_words and min_ai_words and human_words > min_ai_words * 1.3:
        lines.append(
            f"- Human texts are much longer than AI texts: median `{human_words:.1f}` vs AI minimum median `{min_ai_words:.1f}` words. "
            "Length is a dominant confounder."
        )
    if train_removed_rows or eval_removed_rows:
        combined_markers = Counter(
            row.get("markers_found", "") or "<empty>"
            for row in train_removed_rows + eval_removed_rows
        )
        if any(marker in combined_markers for marker in ["FIRST", "LAST", "FIRST|LAST"]):
            lines.append(
                "- The quality filter is likely over-removing normal prose because `FIRST` and `LAST` are treated as artifacts."
            )
    if detector_rows:
        lines.append(
            "- Detector metrics are smoke-run diagnostics, not stable evidence, unless the inferred test size is comfortably large."
        )
    lines.append(
        "- For the next run, prioritize a stricter artifact marker list, length-matched generation, and a larger eval set."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    text = report(args.results_dir, args.length_normalized_words)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
