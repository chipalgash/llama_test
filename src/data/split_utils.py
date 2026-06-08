"""Utilities for deterministic story-level train/eval splits."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def split_items(items: Sequence[T], train_ratio: float = 0.8, seed: int = 42) -> tuple[list[T], list[T]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * train_ratio)
    return shuffled[:split_at], shuffled[split_at:]


def assert_no_story_leakage(train_story_ids: set[str], eval_story_ids: set[str]) -> None:
    overlap = train_story_ids & eval_story_ids
    if overlap:
        preview = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Story-level leakage detected: {preview}")
