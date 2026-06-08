"""Statistical comparisons for stylometric feature tables."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def rank_biserial(x_values: list[float], y_values: list[float]) -> float:
    if not x_values or not y_values:
        return float("nan")
    wins = 0.0
    for x_value in x_values:
        for y_value in y_values:
            if x_value > y_value:
                wins += 1.0
            elif x_value == y_value:
                wins += 0.5
    return (2 * wins) / (len(x_values) * len(y_values)) - 1


def mann_whitney_p_value(x_values: list[float], y_values: list[float]) -> float:
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x_values, y_values, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0 if values else float("nan")
    return statistics.stdev(values)


def clean_float(value: str | float | int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")
