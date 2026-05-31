#!/usr/bin/env python3
"""Pack raw benchmark outputs into self-contained JSON logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASET_BASES = {
    "blobs": "article_benchmark_results",
    "disk": "article_benchmark_results",
    "moons": "article_benchmark_results",
    "mnist": "article_benchmark_results_mnist",
    "levine13": "article_benchmark_results_levine",
    "levine32": "article_benchmark_results_levine",
}
METHODS = ("dire", "cuml_tsne", "cuml_umap", "umap")


def load_label_metadata() -> dict:
    metadata_path = Path(__file__).resolve().parents[1] / "data" / "label_metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_ready(value) for k, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(value) for value in obj]
    return obj


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pack_dataset(raw_root: Path, output: Path, dataset: str) -> None:
    group = DATASET_BASES[dataset]
    base = raw_root / group
    dataset_dir = base / dataset
    results = load_json(dataset_dir / "results.json")
    labels_path = dataset_dir / "labels.npy"
    labels = np.load(labels_path, allow_pickle=True) if labels_path.exists() else np.asarray([])
    label_metadata = load_label_metadata().get(dataset, {})
    dataset_info = results.get("dataset", {})
    dataset_info.setdefault("label_metadata", label_metadata)

    payload = {
        "schema_version": 2,
        "dataset_name": dataset,
        "run_group": group,
        "environment": load_json(base / "environment.json"),
        "dataset": dataset_info,
        "label_metadata": label_metadata,
        "labels": json_ready(labels),
        "methods": {},
    }

    for method in METHODS:
        method_result = results.get("methods", {}).get(method)
        if method_result is None:
            continue
        embedding_path = dataset_dir / f"{method}_embedding.npy"
        method_labels_path = dataset_dir / f"{method}_labels.npy"
        method_payload = dict(method_result)
        if embedding_path.exists():
            method_payload["embedding"] = json_ready(np.load(embedding_path))
        if method_labels_path.exists():
            method_payload["labels"] = json_ready(np.load(method_labels_path, allow_pickle=True))
        payload["methods"][method] = json_ready(method_payload)

    output.mkdir(parents=True, exist_ok=True)
    with open(output / f"{dataset}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def write_manifest(output: Path) -> None:
    manifest = {
        "schema_version": 2,
        "description": "Self-contained JSON benchmark logs for regenerating DiRe paper figures.",
        "datasets": list(DATASET_BASES),
        "methods": list(METHODS),
        "repeat_metrics": {
            "description": "When benchmark repeats are available, each method contains a repeats list and an aggregate block with mean, sample standard deviation, min, max, and n for each plotted metric.",
            "canonical_embedding": "The embedding field stores the first seed only and is used for scatter plots.",
        },
        "label_metadata": "Each dataset log contains label_metadata when labels have semantic display names; render_figures.py also reads data/label_metadata.json as a fallback.",
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh_benchmark_results"))
    parser.add_argument("--output", type=Path, default=Path("data/archived/json_logs"))
    args = parser.parse_args()

    for dataset in DATASET_BASES:
        pack_dataset(args.raw_root, args.output, dataset)
    write_manifest(args.output)


if __name__ == "__main__":
    main()
