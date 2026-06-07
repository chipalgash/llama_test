#!/usr/bin/env python3
"""Prepare Russian creepypasta Markdown stories for SFT/QLoRA experiments.

Inputs are one-story-per-file Markdown files with optional YAML-like front
matter. Outputs mirror the existing English pipeline: story-level split
metadata, full chunk CSVs, capped train/eval payload CSVs with descriptions,
a short corpus report, and a zip payload for Colab.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


RU_STOPWORDS = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть",
    "в", "вам", "вас", "вдруг", "ведь", "во", "вот", "все", "всего",
    "всем", "всех", "вы", "где", "да", "даже", "для", "до", "его",
    "ее", "если", "есть", "еще", "же", "за", "здесь", "и", "из",
    "или", "им", "их", "к", "как", "какая", "какой", "когда", "конечно",
    "кто", "ли", "лишь", "мне", "мной", "много", "может", "мой", "моя",
    "мы", "на", "над", "надо", "нас", "не", "него", "нее", "ней",
    "нет", "ни", "них", "но", "ну", "о", "об", "один", "она", "они",
    "оно", "он", "от", "очень", "по", "под", "после", "потом", "потому",
    "при", "про", "раз", "с", "сам", "себя", "сейчас", "со", "так",
    "такая", "такой", "там", "тебя", "тем", "то", "тогда", "того",
    "тоже", "только", "том", "тот", "тут", "ты", "у", "уже", "хотя",
    "чего", "чем", "через", "что", "чтобы", "это", "этого", "этой",
    "этом", "этот", "я",
}


@dataclass(frozen=True)
class Story:
    story_id: str
    title: str
    author: str
    category: str
    source_url: str
    fetched_at: str
    path: Path
    text: str
    n_chars: int
    n_words: int
    sha256: str


class ChunkRow(TypedDict):
    split: str
    story_id: str
    source_file: str
    title: str
    author: str
    category: str
    source_url: str
    chunk_id: int
    text: str
    description: str
    n_chars: int
    n_words: int
    text_sha256: str


class SplitRow(TypedDict):
    split: str
    story_id: str
    source_file: str
    title: str
    author: str
    category: str
    source_url: str
    n_chars: int
    n_words: int
    story_sha256: str


class SummaryRow(TypedDict):
    metric: str
    value: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Russian creepypasta SFT payload artifacts.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/russian_creepypasta_stories_md"),
        help="Directory with one Markdown file per Russian story.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_ru"),
        help="Directory for Russian CSV outputs.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for human-readable reports.",
    )
    parser.add_argument(
        "--payload-dir",
        type=Path,
        default=Path("data/russian_colab_payload"),
        help="Directory for files that will be zipped for Colab.",
    )
    parser.add_argument(
        "--payload-zip",
        type=Path,
        default=Path("data/russian_horror_experiment_payload.zip"),
        help="Output zip path for Colab upload.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-story-chars", type=int, default=500)
    parser.add_argument("--min-story-words", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=700)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument(
        "--max-train-chunks",
        type=int,
        default=2000,
        help="Cap train chunks included in the Colab payload. Use 0 for all.",
    )
    parser.add_argument(
        "--max-eval-chunks",
        type=int,
        default=500,
        help="Cap eval chunks included in the Colab payload. Use 0 for all.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def parse_front_matter(raw_text: str) -> tuple[dict[str, str], str]:
    if not raw_text.startswith("---\n"):
        return {}, raw_text.strip()

    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text.strip()

    meta_block = parts[1]
    body = parts[2]
    metadata: dict[str, str] = {}
    for line in meta_block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        metadata[key.strip()] = value
    return metadata, body.strip()


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", text))


def title_from_filename(path: Path) -> str:
    stem = re.sub(r"-\d+$", "", path.stem)
    return stem.replace("-", " ").strip().capitalize()


def load_stories(input_dir: Path, min_story_chars: int, min_story_words: int) -> tuple[list[Story], dict[str, int]]:
    files = sorted(input_dir.rglob("*.md"))
    stories: list[Story] = []
    seen_hashes: set[str] = set()
    skipped_short = 0
    skipped_low_words = 0
    skipped_duplicate = 0

    for path in files:
        metadata, body = parse_front_matter(read_text(path))
        text = clean_text(body)
        if len(text) < min_story_chars:
            skipped_short += 1
            continue
        n_words = word_count(text)
        if n_words < min_story_words:
            skipped_low_words += 1
            continue

        text_hash = sha256_text(text)
        if text_hash in seen_hashes:
            skipped_duplicate += 1
            continue
        seen_hashes.add(text_hash)

        stories.append(
            Story(
                story_id=f"ru_s{len(stories) + 1:04d}",
                title=metadata.get("title") or title_from_filename(path),
                author=metadata.get("author", ""),
                category=metadata.get("category") or path.parent.name,
                source_url=metadata.get("source_url", ""),
                fetched_at=metadata.get("fetched_at", ""),
                path=path,
                text=text,
                n_chars=len(text),
                n_words=n_words,
                sha256=text_hash,
            )
        )

    stats = {
        "input_files": len(files),
        "skipped_short": skipped_short,
        "skipped_low_words": skipped_low_words,
        "skipped_duplicate": skipped_duplicate,
    }
    return stories, stats


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def split_stories(
    stories: list[Story],
    train_ratio: float,
    seed: int,
) -> tuple[list[Story], list[Story]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    shuffled = stories[:]
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * train_ratio)
    return shuffled[:split_at], shuffled[split_at:]


def split_into_chunks(text: str, min_chars: int, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if len(current) >= min_chars:
                chunks.append(current)
            current = ""
            for start in range(0, len(sentence), max_chars):
                part = sentence[start : start + max_chars].strip()
                if len(part) >= min_chars:
                    chunks.append(part)
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            current = sentence

    if len(current) >= min_chars:
        chunks.append(current)

    return chunks


def keywords(text: str, top_k: int = 8) -> list[str]:
    words = [
        word.lower().replace("ё", "е")
        for word in re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", text)
        if len(word) > 3 and word.lower().replace("ё", "е") not in RU_STOPWORDS
    ]
    return [word for word, _count in Counter(words).most_common(top_k)]


def make_description(text: str, story: Story) -> str:
    motifs = ", ".join(keywords(text)) or "страх, темнота, тишина"
    clean_title = story.title.strip() or "безымянная страшная история"
    category = story.category.strip() or "страшная история"
    return (
        "Напиши короткий литературный фрагмент хоррора на русском языке. "
        "Стиль: естественная тревожная проза, нарастающее напряжение, без прямого объяснения происходящего. "
        f"Контекст/название: {clean_title}. "
        f"Категория: {category}. "
        f"Опорные мотивы: {motifs}."
    )


def build_chunks(stories: list[Story], split: str, min_chars: int, max_chars: int) -> list[ChunkRow]:
    rows: list[ChunkRow] = []
    for story in stories:
        chunks = split_into_chunks(story.text, min_chars, max_chars)
        for chunk_index, chunk in enumerate(chunks):
            rows.append(
                {
                    "split": split,
                    "story_id": story.story_id,
                    "source_file": display_path(story.path),
                    "title": story.title,
                    "author": story.author,
                    "category": story.category,
                    "source_url": story.source_url,
                    "chunk_id": chunk_index,
                    "text": chunk,
                    "description": make_description(chunk, story),
                    "n_chars": len(chunk),
                    "n_words": word_count(chunk),
                    "text_sha256": sha256_text(chunk),
                }
            )
    return rows


def sample_rows(rows: list[ChunkRow], limit: int, seed: int) -> list[ChunkRow]:
    if limit <= 0 or len(rows) <= limit:
        return rows[:]
    sampled = rows[:]
    random.Random(seed).shuffle(sampled)
    return sorted(sampled[:limit], key=lambda row: (row["story_id"], row["chunk_id"]))


def remove_overlapping_eval_chunks(
    train_chunks: list[ChunkRow],
    eval_chunks: list[ChunkRow],
) -> tuple[list[ChunkRow], int]:
    train_hashes = {row["text_sha256"] for row in train_chunks}
    filtered = [row for row in eval_chunks if row["text_sha256"] not in train_hashes]
    return filtered, len(eval_chunks) - len(filtered)


def write_csv(path: Path, rows: list[ChunkRow] | list[SplitRow] | list[SummaryRow], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100) * len(ordered)) - 1))
    return ordered[index]


def summarize_lengths(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "p95": 0, "max": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def write_summary_csv(path: Path, stories: list[Story], train_chunks: list[ChunkRow], eval_chunks: list[ChunkRow], load_stats: dict[str, int]) -> None:
    rows: list[SummaryRow] = [
        {"metric": "input_files", "value": load_stats["input_files"]},
        {"metric": "stories", "value": len(stories)},
        {"metric": "skipped_short", "value": load_stats["skipped_short"]},
        {"metric": "skipped_low_words", "value": load_stats["skipped_low_words"]},
        {"metric": "skipped_duplicate", "value": load_stats["skipped_duplicate"]},
        {"metric": "train_chunks_full", "value": len(train_chunks)},
        {"metric": "eval_chunks_full", "value": len(eval_chunks)},
        {"metric": "story_chars_total", "value": sum(story.n_chars for story in stories)},
        {"metric": "story_words_total", "value": sum(story.n_words for story in stories)},
    ]
    write_csv(path, rows, ["metric", "value"])


def category_counts(stories: list[Story]) -> Counter[str]:
    return Counter(story.category or "unknown" for story in stories)


def write_corpus_report(
    path: Path,
    stories: list[Story],
    train_stories: list[Story],
    eval_stories: list[Story],
    train_chunks_full: list[ChunkRow],
    eval_chunks_full: list[ChunkRow],
    train_chunks_payload: list[ChunkRow],
    eval_chunks_payload: list[ChunkRow],
    leakage: dict[str, int],
    removed_eval_overlap_chunks: int,
    load_stats: dict[str, int],
    payload_zip: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    story_chars = summarize_lengths([story.n_chars for story in stories])
    story_words = summarize_lengths([story.n_words for story in stories])
    chunk_chars = summarize_lengths([row["n_chars"] for row in train_chunks_full + eval_chunks_full])
    try:
        payload_zip_display = payload_zip.relative_to(Path.cwd())
    except ValueError:
        payload_zip_display = payload_zip

    lines = [
        "# Russian Horror Corpus Preparation Report",
        "",
        "## Corpus",
        "",
        f"- Input Markdown files: {load_stats['input_files']}",
        f"- Stories after filtering/deduplication: {len(stories)}",
        f"- Skipped short stories: {load_stats['skipped_short']}",
        f"- Skipped low-word/non-Russian stories: {load_stats['skipped_low_words']}",
        f"- Skipped duplicate stories: {load_stats['skipped_duplicate']}",
        f"- Train stories: {len(train_stories)}",
        f"- Eval stories: {len(eval_stories)}",
        f"- Story chars: min={story_chars['min']}, median={story_chars['median']:.0f}, mean={story_chars['mean']:.0f}, p95={story_chars['p95']}, max={story_chars['max']}",
        f"- Story words: min={story_words['min']}, median={story_words['median']:.0f}, mean={story_words['mean']:.0f}, p95={story_words['p95']}, max={story_words['max']}",
        "",
        "## Categories",
        "",
    ]
    for category, count in category_counts(stories).most_common():
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "## Chunks",
            "",
            f"- Full train chunks: {len(train_chunks_full)}",
            f"- Full eval chunks: {len(eval_chunks_full)}",
            f"- Removed eval chunks due to exact train overlap: {removed_eval_overlap_chunks}",
            f"- Payload train chunks: {len(train_chunks_payload)}",
            f"- Payload eval chunks: {len(eval_chunks_payload)}",
            f"- Chunk chars: min={chunk_chars['min']}, median={chunk_chars['median']:.0f}, mean={chunk_chars['mean']:.0f}, p95={chunk_chars['p95']}, max={chunk_chars['max']}",
            "",
            "## Leakage Checks",
            "",
            f"- Overlapping story hashes between train/eval: {leakage['story_hash_overlap']}",
            f"- Overlapping chunk hashes between train/eval: {leakage['chunk_hash_overlap']}",
            "",
            "## Colab Payload",
            "",
            f"- Zip: `{payload_zip_display}`",
            "",
            "Payload files:",
            "",
            "- `train_with_descriptions.csv`",
            "- `eval_with_descriptions.csv`",
            "- `split_metadata.csv`",
            "- `corpus_summary.csv`",
            "- `russian_corpus_preparation_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_payload(payload_dir: Path, payload_zip: Path, files: list[Path]) -> None:
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_zip.parent.mkdir(parents=True, exist_ok=True)
    if payload_zip.exists():
        payload_zip.unlink()
    with zipfile.ZipFile(payload_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.name)


def copy_for_payload(source: Path, payload_dir: Path) -> Path:
    target = payload_dir / source.name
    target.write_bytes(source.read_bytes())
    return target


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    payload_dir = args.payload_dir.resolve()
    payload_zip = args.payload_zip.resolve()

    stories, load_stats = load_stories(input_dir, args.min_story_chars, args.min_story_words)
    if not stories:
        raise ValueError(f"No usable Markdown stories found in {input_dir}")

    train_stories, eval_stories = split_stories(stories, args.train_ratio, args.seed)
    train_chunks_full = build_chunks(train_stories, "train", args.min_chars, args.max_chars)
    eval_chunks_full = build_chunks(eval_stories, "eval", args.min_chars, args.max_chars)
    eval_chunks_full, removed_eval_overlap_chunks = remove_overlapping_eval_chunks(
        train_chunks_full,
        eval_chunks_full,
    )
    train_chunks_payload = sample_rows(train_chunks_full, args.max_train_chunks, args.seed)
    eval_chunks_payload = sample_rows(eval_chunks_full, args.max_eval_chunks, args.seed)

    story_train_hashes = {story.sha256 for story in train_stories}
    story_eval_hashes = {story.sha256 for story in eval_stories}
    chunk_train_hashes = {row["text_sha256"] for row in train_chunks_full}
    chunk_eval_hashes = {row["text_sha256"] for row in eval_chunks_full}
    leakage = {
        "story_hash_overlap": len(story_train_hashes & story_eval_hashes),
        "chunk_hash_overlap": len(chunk_train_hashes & chunk_eval_hashes),
    }

    chunk_fields = [
        "split",
        "story_id",
        "source_file",
        "title",
        "author",
        "category",
        "source_url",
        "chunk_id",
        "text",
        "description",
        "n_chars",
        "n_words",
        "text_sha256",
    ]
    split_fields = [
        "split",
        "story_id",
        "source_file",
        "title",
        "author",
        "category",
        "source_url",
        "n_chars",
        "n_words",
        "story_sha256",
    ]

    split_rows: list[SplitRow] = []
    for split_name, split_stories_list in [("train", train_stories), ("eval", eval_stories)]:
        for story in split_stories_list:
            split_rows.append(
                {
                    "split": split_name,
                    "story_id": story.story_id,
                    "source_file": display_path(story.path),
                    "title": story.title,
                    "author": story.author,
                    "category": story.category,
                    "source_url": story.source_url,
                    "n_chars": story.n_chars,
                    "n_words": story.n_words,
                    "story_sha256": story.sha256,
                }
            )

    split_metadata_path = processed_dir / "split_metadata.csv"
    train_full_path = processed_dir / "train_chunks_full.csv"
    eval_full_path = processed_dir / "eval_chunks_full.csv"
    train_payload_path = processed_dir / "train_with_descriptions.csv"
    eval_payload_path = processed_dir / "eval_with_descriptions.csv"
    summary_csv_path = processed_dir / "corpus_summary.csv"
    report_path = reports_dir / "russian_corpus_preparation_report.md"

    write_csv(split_metadata_path, split_rows, split_fields)
    write_csv(train_full_path, train_chunks_full, chunk_fields)
    write_csv(eval_full_path, eval_chunks_full, chunk_fields)
    write_csv(train_payload_path, train_chunks_payload, chunk_fields)
    write_csv(eval_payload_path, eval_chunks_payload, chunk_fields)
    write_summary_csv(summary_csv_path, stories, train_chunks_full, eval_chunks_full, load_stats)

    write_corpus_report(
        report_path,
        stories,
        train_stories,
        eval_stories,
        train_chunks_full,
        eval_chunks_full,
        train_chunks_payload,
        eval_chunks_payload,
        leakage,
        removed_eval_overlap_chunks,
        load_stats,
        payload_zip,
    )

    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_files = [
        copy_for_payload(train_payload_path, payload_dir),
        copy_for_payload(eval_payload_path, payload_dir),
        copy_for_payload(split_metadata_path, payload_dir),
        copy_for_payload(summary_csv_path, payload_dir),
        copy_for_payload(report_path, payload_dir),
    ]
    build_payload(payload_dir, payload_zip, payload_files)

    print(f"Input Markdown files: {load_stats['input_files']}")
    print(f"Stories after filtering/deduplication: {len(stories)}")
    print(
        "Skipped short/low-word/duplicate stories: "
        f"{load_stats['skipped_short']}/{load_stats['skipped_low_words']}/{load_stats['skipped_duplicate']}"
    )
    print(f"Train/eval stories: {len(train_stories)}/{len(eval_stories)}")
    print(f"Full train/eval chunks: {len(train_chunks_full)}/{len(eval_chunks_full)}")
    print(f"Removed eval chunks due to exact train overlap: {removed_eval_overlap_chunks}")
    print(f"Payload train/eval chunks: {len(train_chunks_payload)}/{len(eval_chunks_payload)}")
    print(f"Story hash overlap: {leakage['story_hash_overlap']}")
    print(f"Chunk hash overlap: {leakage['chunk_hash_overlap']}")
    print(f"Report: {report_path}")
    print(f"Payload zip: {payload_zip}")


if __name__ == "__main__":
    main()
