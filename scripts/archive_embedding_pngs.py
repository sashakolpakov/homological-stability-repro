#!/usr/bin/env python3
"""Copy canonical benchmark embedding PNGs into the reproducibility archive."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


RUN_GROUPS = (
    "article_benchmark_results",
    "article_benchmark_results_mnist",
    "article_benchmark_results_levine",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh_benchmark_results"))
    parser.add_argument("--output", type=Path, default=Path("data/archived/embedding_pngs"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for group in RUN_GROUPS:
        embedding_dir = args.raw_root / group / "embeddings"
        if not embedding_dir.exists():
            continue
        for source in embedding_dir.glob("*.png"):
            shutil.copy2(source, args.output / source.name)


if __name__ == "__main__":
    main()
