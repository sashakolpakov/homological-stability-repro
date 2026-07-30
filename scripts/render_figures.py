#!/usr/bin/env python3
"""Render manuscript figures from self-contained JSON benchmark logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import textwrap
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runtime_protocol import summarize_fit_times


METHODS = (
    "dire",
    "cuml_tsne",
    "cuml_umap",
    "opentsne",
    "umap",
)
DISPLAY = {
    "dire": "DiRe-RAPIDS",
    "cuml_tsne": "cuML tSNE",
    "cuml_umap": "cuML UMAP",
    "opentsne": "openTSNE",
    "umap": "Original UMAP",
}
EMBEDDING_FILENAMES = {
    "dire": "dire-rapids",
    "cuml_tsne": "tsne",
    "cuml_umap": "cuml-umap",
    "opentsne": "opentsne",
    "umap": "umap",
}
ARCHIVED_EMBEDDING_FILENAMES = {
    "dire": "dire",
    "cuml_tsne": "cuml-tsne",
    "cuml_umap": "cuml-umap",
    "opentsne": "opentsne",
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
GPU_METHODS = ("dire", "cuml_tsne", "cuml_umap")
CPU_METHODS = ("opentsne", "umap")
METHOD_COLORS = {
    "dire": "#1b9e77",
    "cuml_umap": "#d95f02",
    "cuml_tsne": "#7570b3",
    "opentsne": "#1f78b4",
    "umap": "#e6ab02",
}
QUALITY_METRICS = METRICS[1:]
REVISION3_BUNDLE_PATTERN = re.compile(
    r"^revision3-results-[A-Za-z0-9._-]+$"
)
DETERMINISTIC_PDF_METADATA = {
    "CreationDate": None,
    "ModDate": None,
}


def save_deterministic_pdf(fig: object, path: Path, **kwargs: object) -> None:
    """Write a Matplotlib PDF without wall-clock metadata."""

    fig.savefig(path, metadata=DETERMINISTIC_PDF_METADATA, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pics_root(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace linked picture root: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def source_bundle_name(data_root: Path) -> str | None:
    resolved = data_root.resolve()
    for candidate in (resolved, *resolved.parents):
        if REVISION3_BUNDLE_PATTERN.fullmatch(candidate.name):
            return candidate.name
    return None


def file_records(root: Path, *, exclude: set[str] | None = None) -> list[dict]:
    excluded = exclude or set()
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() not in excluded
    ]


def write_pics_manifest(
    pics: Path,
    data_root: Path,
    embedding_png_root: Path | None,
) -> None:
    bundle_name = source_bundle_name(data_root)
    manifest = {
        "schema_version": 1,
        "source_bundle": bundle_name,
        "source_mode": (
            "verified-revision3-bundle-small-suite"
            if bundle_name
            else "archived-or-custom-small-suite"
        ),
        "source_json": file_records(data_root),
        "source_embedding_pngs": (
            file_records(embedding_png_root)
            if embedding_png_root is not None and embedding_png_root.is_dir()
            else []
        ),
        "outputs": file_records(pics, exclude={"render_manifest.json"}),
    }
    (pics / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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


def steady_runtime_statistics(result: dict) -> dict:
    """Read explicit steady timing or derive it from historical repeat records."""
    timing = result.get("timing", {})
    steady = timing.get("steady", {}) if isinstance(timing, dict) else {}
    if steady.get("n", 0) > 0 and steady.get("mean") is not None:
        return steady

    derived = summarize_fit_times(result.get("repeats", []))
    steady = derived["steady"]
    if steady["n"] > 0:
        return steady

    aggregate = result.get("aggregate", {}).get("time", {})
    if aggregate.get("mean") is not None:
        return aggregate
    fit_time = result.get("fit_time_sec")
    return {
        "mean": fit_time,
        "std": 0.0 if fit_time is not None else None,
        "min": fit_time,
        "max": fit_time,
        "n": 1 if fit_time is not None else 0,
    }


def metric_value(result: dict, key: str):
    if key == "time":
        return steady_runtime_statistics(result).get("mean")
    aggregate = result.get("aggregate", {})
    if key in aggregate and "mean" in aggregate[key]:
        return aggregate[key]["mean"]
    metrics = result.get("metrics", {})
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
    if key == "time":
        return steady_runtime_statistics(result).get("std")
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


def direction_adjusted_log2_ratio(
    key: str,
    method_value: float,
    dire_value: float,
) -> float:
    """Return comparator-versus-DiRe effect size with positive meaning better.

    For a higher-is-better metric, the effect is
    ``log2(method / DiRe)``. For a lower-is-better discrepancy, it is
    ``log2(DiRe / method)``. Context is evaluated by absolute loss. Exact
    zero-versus-zero ties map to zero; a perfect zero loss versus a nonzero
    loss maps to the appropriate signed infinity.
    """

    method = abs(float(method_value)) if key == "context" else float(method_value)
    dire = abs(float(dire_value)) if key == "context" else float(dire_value)
    if method < 0 or dire < 0:
        raise ValueError(f"{key} values must be nonnegative after orientation")
    if np.isclose(method, 0.0, rtol=0.0, atol=1e-15) and np.isclose(
        dire,
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        return 0.0
    numerator, denominator = (
        (method, dire) if key == "neighborhood" else (dire, method)
    )
    if denominator == 0:
        return math.inf
    if numerator == 0:
        return -math.inf
    return float(np.log2(numerator / denominator))


def format_log2_effect(value: float) -> str:
    if np.isposinf(value):
        return "+∞"
    if np.isneginf(value):
        return "−∞"
    return f"{value:+.2f}"


def render_quality_summary(data_root: Path, path: Path) -> None:
    """Render continuous comparator-versus-DiRe effects and exact raw means."""
    logs = {dataset: load_log(data_root, dataset) for dataset in DATASETS}
    available_methods = [
        method
        for method in METHODS
        if any(method in log.get("methods", {}) for log in logs.values())
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.4), constrained_layout=True)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="#f2f2f2")

    for ax, dataset in zip(axes.flat, DATASETS):
        results = logs[dataset].get("methods", {})
        effect_matrix = np.full(
            (len(available_methods), len(QUALITY_METRICS)),
            np.nan,
        )
        raw_matrix: list[list[float | None]] = []
        for method in available_methods:
            raw_matrix.append(
                [
                    metric_value(results[method], key) if method in results else None
                    for key, _title in QUALITY_METRICS
                ]
            )
        dire_index = available_methods.index("dire")
        for metric_index, (key, _title) in enumerate(QUALITY_METRICS):
            dire_value = raw_matrix[dire_index][metric_index]
            if dire_value is None:
                continue
            for method_index, row in enumerate(raw_matrix):
                value = row[metric_index]
                if value is None:
                    continue
                effect_matrix[method_index, metric_index] = (
                    direction_adjusted_log2_ratio(
                        key,
                        value,
                        dire_value,
                    )
                )

        displayed_effects = np.clip(effect_matrix, -1.0, 1.0)
        image = ax.imshow(
            displayed_effects,
            vmin=-1.0,
            vmax=1.0,
            cmap=cmap,
            aspect="auto",
        )
        ax.set_title(dataset)
        ax.set_xticks(range(len(QUALITY_METRICS)))
        ax.set_xticklabels(
            [
                "stress",
                "neighbors",
                "context score",
                r"DTW $\beta_0$",
                r"DTW $\beta_1$",
            ],
            rotation=28,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(range(len(available_methods)))
        ax.set_yticklabels([DISPLAY[method] for method in available_methods], fontsize=8)
        for row_index, row in enumerate(raw_matrix):
            for column_index, value in enumerate(row):
                effect = effect_matrix[row_index, column_index]
                if value is None or np.isnan(effect):
                    label = "--"
                else:
                    label = f"{format_log2_effect(effect)}\nraw {value:.3g}"
                ax.text(column_index, row_index, label, ha="center", va="center", fontsize=6.5)
        ax.set_xticks(np.arange(-0.5, len(QUALITY_METRICS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(available_methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.015)
    colorbar.set_ticks((-1.0, 0.0, 1.0))
    colorbar.set_ticklabels(
        ("DiRe 2× better", "equal", "comparator 2× better")
    )
    colorbar.set_label("direction-adjusted log₂ ratio vs DiRe (clipped at ±1)")
    fig.suptitle(
        "Embedding-quality summary: effect size versus DiRe and raw mean",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def render_runtime_summary(data_root: Path, path: Path) -> None:
    """Keep CPU and GPU timings in separate panels to avoid cross-device claims."""
    logs = {dataset: load_log(data_root, dataset) for dataset in DATASETS}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    x = np.arange(len(DATASETS))

    for ax, methods, title in (
        (axes[0], GPU_METHODS, "GPU implementations"),
        (axes[1], CPU_METHODS, "CPU implementations"),
    ):
        plotted = False
        width = 0.8 / len(methods)
        for method_index, method in enumerate(methods):
            values = []
            errors = []
            for dataset in DATASETS:
                result = logs[dataset].get("methods", {}).get(method)
                values.append(metric_value(result, "time") if result is not None else np.nan)
                error = metric_error(result, "time") if result is not None else None
                errors.append(0.0 if error is None else error)
            if np.all(np.isnan(values)):
                continue
            plotted = True
            positions = x - 0.4 + width / 2.0 + method_index * width
            ax.bar(
                positions,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=METHOD_COLORS[method],
                label=DISPLAY[method],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=30, ha="right")
        ax.set_ylabel("steady fit_transform time (s)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "not present in archived run", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(
        "Steady-state runtime (cold first fit excluded; mean $\\pm$ sample SD)",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def render_all(data_root: Path, output: Path, embedding_png_root: Path | None) -> None:
    label_metadata = load_label_metadata(data_root)
    pics = output / "pics"
    prepare_pics_root(pics)
    for dataset in DATASETS:
        log = load_log(data_root, dataset)
        labels = np.asarray(log.get("labels", []))
        metadata = log.get("label_metadata") or log.get("dataset", {}).get("label_metadata") or label_metadata.get(dataset, {})
        label_map = metadata.get("label_map", {})
        legend_title = metadata.get("legend_title", "Label")
        results = log.get("methods", {})

        for method in METHODS:
            if method not in results:
                continue
            method_labels = np.asarray(results[method].get("labels", labels))
            suffix = EMBEDDING_FILENAMES[method]
            output_path = pics / "embeddings" / f"{dataset}-{suffix}.png"
            archived_png = None
            if embedding_png_root is not None:
                archived_png = embedding_png_root / f"{dataset}-{ARCHIVED_EMBEDDING_FILENAMES[method]}.png"
            if "embedding" in results[method]:
                embedding = np.asarray(results[method]["embedding"], dtype=np.float32)
                if method_labels.size != embedding.shape[0]:
                    method_labels = np.asarray([])
                render_embedding(
                    output_path,
                    embedding,
                    method_labels,
                    f"{dataset}: {DISPLAY[method]}",
                    label_map,
                    legend_title,
                )
            elif archived_png is not None and archived_png.exists():
                render_embedding_png_with_legend(
                    output_path,
                    archived_png,
                    method_labels,
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
    render_quality_summary(data_root, pics / "metrics-summary.pdf")
    render_runtime_summary(data_root, pics / "runtime-summary.pdf")
    write_pics_manifest(pics, data_root, embedding_png_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/archived/json_logs"))
    parser.add_argument("--embedding-png-root", type=Path, default=Path("data/archived/embedding_pngs"))
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    render_all(args.data_root, args.output, args.embedding_png_root)


if __name__ == "__main__":
    main()
