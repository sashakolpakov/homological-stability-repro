#!/usr/bin/env python3
"""Render manuscript figures from self-contained JSON benchmark logs."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("dire", "cuml_tsne", "cuml_umap", "umap")
DISPLAY = {
    "dire": "DiRe-RAPIDS",
    "cuml_tsne": "cuML tSNE",
    "cuml_umap": "cuML UMAP",
    "umap": "Original UMAP",
}
EMBEDDING_FILENAMES = {
    "dire": "dire-rapids",
    "cuml_tsne": "tsne",
    "cuml_umap": "cuml-umap",
    "umap": "umap",
}
ARCHIVED_EMBEDDING_FILENAMES = {
    "dire": "dire",
    "cuml_tsne": "cuml-tsne",
    "cuml_umap": "cuml-umap",
    "umap": "umap",
}
DATASETS = ("blobs", "disk", "moons", "mnist", "levine13", "levine32")
METRICS = (
    ("time", "Runtime (seconds)"),
    ("stress", "Embedding stress"),
    ("neighborhood", "Neighborhood preservation"),
    ("context", "SVM context score"),
    ("persistence-dim-0", "DTW Betti discrepancy: dimension 0"),
    ("persistence-dim-1", "DTW Betti discrepancy: dimension 1"),
)


def load_label_metadata(data_root: Path) -> dict:
    metadata_path = data_root.parent / "label_metadata.json"
    if not metadata_path.exists():
        metadata_path = Path("data/label_metadata.json")
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_log(data_root: Path, dataset: str) -> dict:
    with open(data_root / f"{dataset}.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_label(label) -> str:
    if isinstance(label, (np.integer, int)):
        return str(int(label))
    if isinstance(label, (np.floating, float)) and float(label).is_integer():
        return str(int(label))
    text = str(label)
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return str(int(value))
    return text


def labels_to_colors(labels: np.ndarray | None, label_map: dict[str, str] | None = None):
    if labels is None or labels.size == 0:
        return None, []
    labels = np.asarray([canonical_label(label) for label in labels])
    unique = np.asarray(sorted(set(labels), key=label_sort_key))
    label_to_index = {label: index for index, label in enumerate(unique)}
    encoded = np.asarray([label_to_index[label] for label in labels], dtype=np.int32)
    label_map = label_map or {}
    display_labels = [label_map.get(str(label), str(label)) for label in unique]
    return encoded, display_labels


def label_sort_key(label: str):
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def legend_color(index: int, count: int):
    if count <= 1:
        return plt.get_cmap("tab20")(0.0)
    return plt.get_cmap("tab20")(index / (count - 1))


def add_label_legend(ax, unique_labels: list[str], legend_title: str) -> None:
    count = len(unique_labels)
    if count == 0:
        return
    ncol = 1
    fontsize = 4.8 if count > 14 else 6.2
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            markerfacecolor=legend_color(index, count),
            markeredgecolor="none",
            label=format_legend_label(label),
        )
        for index, label in enumerate(unique_labels)
    ]
    ax.legend(
        handles=handles,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=fontsize,
        title_fontsize=7.5,
        borderaxespad=0,
        ncol=ncol,
        columnspacing=0.8,
        handletextpad=0.4,
    )


def format_legend_label(label: str) -> str:
    return textwrap.fill(label, width=24, break_long_words=False, break_on_hyphens=False)


def figure_size_for_labels(labels: np.ndarray | None) -> tuple[float, float]:
    if labels is None or labels.size == 0:
        return (7.2, 5.0)
    unique_count = len(set(canonical_label(label) for label in labels))
    if unique_count > 14:
        return (8.8, 5.2)
    if unique_count > 10:
        return (8.8, 5.0)
    if unique_count > 0:
        return (8.0, 5.0)
    return (7.2, 5.0)


def render_embedding(
    path: Path,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    label_map: dict[str, str] | None,
    legend_title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size_for_labels(labels))
    colors, unique_labels = labels_to_colors(labels, label_map)
    if colors is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.65)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="tab20", s=2, alpha=0.65)
        add_label_legend(ax, unique_labels, legend_title)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_embedding_png_with_legend(
    path: Path,
    source_png: Path,
    labels: np.ndarray | None,
    label_map: dict[str, str] | None,
    legend_title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = plt.imread(source_png)
    fig, ax = plt.subplots(figsize=figure_size_for_labels(labels))
    ax.imshow(image)
    ax.axis("off")

    colors, unique_labels = labels_to_colors(labels, label_map)
    if colors is not None:
        add_label_legend(ax, unique_labels, legend_title)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def metric_value(result: dict, key: str):
    aggregate = result.get("aggregate", {})
    if key in aggregate and "mean" in aggregate[key]:
        return aggregate[key]["mean"]
    metrics = result.get("metrics", {})
    if key == "time":
        return result.get("fit_time_sec")
    if key == "stress":
        return metrics.get("local", {}).get("stress")
    if key == "neighborhood":
        local = metrics.get("local", {})
        if "neighbor_score" in local:
            return local["neighbor_score"]
        neighbor = local.get("neighbor")
        if isinstance(neighbor, list) and neighbor:
            return neighbor[0]
        return None
    if key == "context":
        context = metrics.get("context", {})
        if "svm" in context:
            return context["svm"][2]
        if "knn" in context:
            return context["knn"][2]
        return None
    topology = metrics.get("topology", {}).get("metrics", {})
    if key == "persistence-dim-0":
        return topology.get("dtw_beta0")
    if key == "persistence-dim-1":
        return topology.get("dtw_beta1")
    return None


def metric_error(result: dict, key: str):
    aggregate = result.get("aggregate", {})
    if key in aggregate and "std" in aggregate[key]:
        return aggregate[key]["std"]
    return None


def render_metric(path: Path, dataset: str, results: dict, key: str, title: str) -> bool:
    clean = []
    for method in METHODS:
        if method not in results:
            continue
        value = metric_value(results[method], key)
        if value is not None:
            clean.append((DISPLAY[method], value, metric_error(results[method], key)))
    if not clean:
        return False

    names, values, errors = zip(*clean)
    yerr = [0.0 if error is None else error for error in errors]
    has_error = any(error > 0 for error in yerr)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(names, values, yerr=yerr if has_error else None, capsize=4 if has_error else 0)
    plt.title(f"{dataset}: {title}")
    plt.xticks(rotation=25, ha="right")
    if all(abs(value) < 1e-12 for value in values):
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylim(-0.05, 0.05)
        plt.text(
            0.5,
            0.55,
            "all methods: 0.0",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def render_all(data_root: Path, output: Path, embedding_png_root: Path | None) -> None:
    label_metadata = load_label_metadata(data_root)
    pics = output / "pics"
    for dataset in DATASETS:
        log = load_log(data_root, dataset)
        labels = np.asarray(log.get("labels", []))
        metadata = log.get("label_metadata") or log.get("dataset", {}).get("label_metadata") or label_metadata.get(dataset, {})
        label_map = metadata.get("label_map", {})
        legend_title = metadata.get("legend_title", "Label")
        results = log.get("methods", {})

        for method in METHODS:
            if method not in results or "embedding" not in results[method]:
                continue
            suffix = EMBEDDING_FILENAMES[method]
            output_path = pics / "embeddings" / f"{dataset}-{suffix}.png"
            archived_png = None
            if embedding_png_root is not None:
                archived_png = embedding_png_root / f"{dataset}-{ARCHIVED_EMBEDDING_FILENAMES[method]}.png"
            if archived_png is not None and archived_png.exists():
                render_embedding_png_with_legend(output_path, archived_png, labels, label_map, legend_title)
            else:
                embedding = np.asarray(results[method]["embedding"], dtype=np.float32)
                render_embedding(
                    output_path,
                    embedding,
                    labels,
                    f"{dataset}: {DISPLAY[method]}",
                    label_map,
                    legend_title,
                )

        for key, title in METRICS:
            render_metric(
                pics / f"{dataset}_comparison" / f"{dataset}-{key}.png",
                dataset,
                results,
                key,
                title,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/archived/json_logs"))
    parser.add_argument("--embedding-png-root", type=Path, default=Path("data/archived/embedding_pngs"))
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    render_all(args.data_root, args.output, args.embedding_png_root)


if __name__ == "__main__":
    main()
