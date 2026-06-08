#!/usr/bin/env python3
"""Prepare the normalized Russian payload for the thesis experiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.convert_payload_csv_to_jsonl import convert_language


def main() -> None:
    metadata = convert_language("ru", max_train=500, max_eval=100)
    print(metadata)


if __name__ == "__main__":
    main()
