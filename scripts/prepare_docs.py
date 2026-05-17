#!/usr/bin/env python3
"""Prepare generated paper artifacts for the Sphinx site."""

from __future__ import annotations

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


DATASETS = ("blobs", "mnist", "disk", "moons", "levine13", "levine32")
METHOD_SUFFIXES = ("dire-rapids", "tsne", "cuml-umap", "umap")
METRICS = ("time", "stress", "neighborhood", "context", "persistence-dim-0", "persistence-dim-1")


def copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if item.is_file():
            target = destination / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def write_gallery(path: Path) -> None:
    lines = [
        "Figure Gallery",
        "==============",
        "",
        "Embedding Figures",
        "-----------------",
        "",
    ]
    for dataset in DATASETS:
        lines.extend([dataset, "~" * len(dataset), ""])
        for suffix in METHOD_SUFFIXES:
            image = f"_static/paper/pics/embeddings/{dataset}-{suffix}.png"
            if (path.parent / image).exists():
                lines.extend([f".. image:: {image}", "   :width: 48%", ""])
        lines.append("")

    lines.extend(["Metric Figures", "--------------", ""])
    for dataset in DATASETS:
        lines.extend([dataset, "~" * len(dataset), ""])
        for metric in METRICS:
            image = f"_static/paper/pics/{dataset}_comparison/{dataset}-{metric}.png"
            if (path.parent / image).exists():
                lines.extend([f".. image:: {image}", "   :width: 32%", ""])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_page() -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/latex_to_sphinx.py",
            "--latex",
            "dire_short.tex",
            "--bib",
            "dire_short.bib",
            "--output",
            "docs/paper.rst",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-static", type=Path, default=Path("docs/_static/paper"))
    args = parser.parse_args()

    if args.site_static.exists():
        shutil.rmtree(args.site_static)
    args.site_static.mkdir(parents=True, exist_ok=True)

    copy_tree_contents(Path("pics"), args.site_static / "pics")
    for filename in ("dire_short.pdf", "dire_short.tex", "dire_short.bib"):
        source = Path(filename)
        if source.exists():
            shutil.copy2(source, args.site_static / filename)
    write_paper_page()
    write_gallery(Path("docs/figures.rst"))


if __name__ == "__main__":
    main()
