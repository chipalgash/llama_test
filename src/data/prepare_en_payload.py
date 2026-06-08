#!/usr/bin/env python3
"""Prepare the normalized English payload for the thesis experiment.

The current repository already contains the legacy English CSV split. This
wrapper converts that split to the normalized JSONL contract used by the main
Llama 3 pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.convert_payload_csv_to_jsonl import convert_language


def main() -> None:
    metadata = convert_language("en", max_train=500, max_eval=100)
    print(metadata)


if __name__ == "__main__":
    main()
