#!/usr/bin/env python3
"""Render manuscript-ready Revision 3 figures, tables, and value macros.

This script needs only the compact fetched result bundle: it never needs the
4.2 GB 10x count matrix or the 1.1 GB arXiv input embeddings.  Every number in
the TeX fragments is read from machine-generated JSON, avoiding transcription.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from scipy.stats import t as student_t


METHODS = ("dire_auto", "dire", "cuml_umap", "cuml_tsne", "pca2")
DETERMINISTIC_PDF_METADATA = {
    "CreationDate": None,
    "ModDate": None,
}
LAYOUT_METHODS = {
    "tenx": (
        "pca2",
        "dire_auto",
        "dire",
        "cuml_umap",
        "cuml_tsne",
        "cellranger_tsne",
    ),
    "arxiv": ("pca2", "dire_auto", "dire", "cuml_umap", "cuml_tsne"),
}
QUALITY_METHODS = {
    "tenx": (
        "pca2",
        "dire_auto",
        "dire",
        "dire_spectral",
        "dire_topology",
        "cuml_umap",
        "cuml_tsne",
        "cellranger_tsne",
    ),
    "arxiv": (
        "pca2",
        "dire_auto",
        "dire",
        "dire_spectral",
        "dire_topology",
        "cuml_umap",
        "cuml_tsne",
    ),
}
SENSITIVITY_METHODS = (
    "dire_auto",
    "dire",
    "dire_spectral",
    "dire_topology",
)
DISPLAY = {
    "dire_auto": "DiRe (auto cuVS)",
    "dire": "DiRe (IVF-Flat control)",
    "dire_spectral": "DiRe (spectral init)",
    "dire_topology": "DiRe (topology preset)",
    "cuml_umap": "cuML UMAP",
    "cuml_tsne": "cuML t-SNE",
    "pca2": "PCA (2D)",
    "cellranger_tsne": "Cell Ranger t-SNE",
    "opentsne": "openTSNE",
    "umap": "umap-learn",
}
COLORS = {
    "dire_auto": "#1b9e77",
    "dire": "#80b1a8",
    "dire_spectral": "#33a89c",
    "dire_topology": "#006d5b",
    "cuml_umap": "#d95f02",
    "cuml_tsne": "#7570b3",
    "pca2": "#666666",
    "cellranger_tsne": "#e7298a",
    "opentsne": "#1f78b4",
    "umap": "#e6ab02",
}
MARKERS = {
    "dire_auto": "o",
    "dire": "h",
    "dire_spectral": "v",
    "dire_topology": "X",
    "cuml_umap": "s",
    "cuml_tsne": "^",
    "pca2": "D",
    "cellranger_tsne": "P",
}


def save_deterministic_pdf(fig: object, path: Path, **kwargs: object) -> None:
    """Write a Matplotlib PDF without wall-clock metadata."""

    fig.savefig(path, metadata=DETERMINISTIC_PDF_METADATA, **kwargs)


FULL_SIZES = {"tenx": 1_306_127, "arxiv": 723_457}
ABLATION_DATASET_DISPLAY = {
    "blobs": "Blobs",
    "disk": "Disk",
    "moons": "Moons",
    "mnist": "MNIST",
    "levine13": "Levine13",
    "levine32": "Levine32",
}
SMALL_DATASET_ORDER = (
    "blobs",
    "disk",
    "moons",
    "mnist",
    "levine13",
    "levine32",
)
SMALL_DATASET_DISPLAY = {
    "blobs": "Blobs",
    "disk": "Disk",
    "moons": "Moons",
    "mnist": "MNIST",
    "levine13": "Levine13",
    "levine32": "Levine32",
}
SMALL_DISPLAY = {
    "dire": "DiRe-RAPIDS",
    "dire_topology": "DiRe (topology preset)",
    "cuml_umap": "cuML UMAP",
    "cuml_tsne": "cuML t-SNE",
    "opentsne": "openTSNE",
    "umap": "umap-learn",
}
SMALL_TOPOLOGY_BACKEND_DETAIL = "direct GPU rank-based local-kNN atlas"
SMALL_COLORS = {
    "dire": "#1b9e77",
    "dire_topology": "#006d5b",
    "cuml_umap": "#d95f02",
    "cuml_tsne": "#7570b3",
    "opentsne": "#1f78b4",
    "umap": "#e6ab02",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_scalar_topology_audit_records(value):
    """Yield scalar comparison records from the nested CPU topology audit."""
    if isinstance(value, dict):
        if "matches" in value and "absolute_delta" in value:
            yield value
        for child in value.values():
            yield from iter_scalar_topology_audit_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_scalar_topology_audit_records(child)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_records(
    root: Path,
    *,
    exclude: set[str] | None = None,
) -> list[dict[str, object]]:
    """Record exact generated-file membership without self-hashing a manifest."""

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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def matching_topology_audit(
    bundle_root: Path,
    output_root: Path,
) -> dict | None:
    """Return a successful audit only when it belongs to this exact bundle."""
    path = output_root / "topology-audit.json"
    if not path.is_file():
        return None
    audit = read_json(path)
    if audit.get("source_bundle") != bundle_root.name:
        return None
    if audit.get("status") != "success":
        raise RuntimeError(
            f"topology audit for {bundle_root.name} is not successful"
        )
    return audit


def prepare_output_root(
    bundle_root: Path,
    output_root: Path,
    *,
    clean: bool,
) -> None:
    """Prepare generated output without carrying stale bundle artifacts."""
    matching_audit = matching_topology_audit(bundle_root, output_root)
    if clean and output_root.exists():
        resolved = output_root.resolve()
        working_directory = Path.cwd().resolve()
        bundle_directory = bundle_root.resolve()
        if (
            resolved == Path("/").resolve()
            or working_directory.is_relative_to(resolved)
            or bundle_directory.is_relative_to(resolved)
        ):
            raise RuntimeError(
                f"refusing to clean unsafe output root: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if clean and matching_audit is not None:
        write_json(output_root / "topology-audit.json", matching_audit)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(text))


def format_seconds(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    if value < 0.1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def result_path(bundle_root: Path, dataset: str, method: str, size: int) -> Path:
    results_directory = (
        "topology_sensitivity_results"
        if method in ("dire_spectral", "dire_topology")
        else "large_results"
    )
    return (
        bundle_root
        / results_directory
        / dataset
        / method
        / f"n_{size:09d}"
        / "result.json"
    )


def load_result(
    bundle_root: Path,
    dataset: str,
    method: str,
    size: int,
) -> dict | None:
    path = result_path(bundle_root, dataset, method, size)
    return read_json(path) if path.exists() else None


def full_embedding_path(
    bundle_root: Path,
    dataset: str,
    method: str,
) -> Path:
    size = FULL_SIZES[dataset]
    if method == "cellranger_tsne":
        return (
            bundle_root
            / "prepared"
            / "tenx_mouse_brain_cellranger_tsne.npy"
        )
    base = result_path(bundle_root, dataset, method, size).parent
    result = read_json(base / "result.json")
    if result.get("status") != "success":
        raise RuntimeError(
            f"cannot render failed {dataset}/{method} run: {result.get('error')}"
        )
    return base / result["embedding_file"]


def shared_stratified_sample(
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if sample_size >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)
    selected, _rest = train_test_split(
        indices,
        train_size=sample_size,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    return np.sort(selected)


def label_colors(n_classes: int) -> np.ndarray:
    if n_classes <= 20:
        cmap = plt.get_cmap("tab20")
        return np.asarray([cmap(index / max(1, n_classes - 1)) for index in range(n_classes)])
    cmap = plt.get_cmap("turbo")
    return np.asarray([cmap(index / max(1, n_classes - 1)) for index in range(n_classes)])


def plot_layout_panel(
    ax,
    embedding: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    title: str,
    palette: np.ndarray,
    annotate_centroids: bool,
    display_percentiles: tuple[float, float] | None = None,
) -> None:
    view = np.asarray(embedding[indices], dtype=np.float32)
    view_labels = labels[indices]
    ax.scatter(
        view[:, 0],
        view[:, 1],
        c=palette[view_labels],
        s=0.18,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    if annotate_centroids:
        for label in np.unique(view_labels):
            mask = view_labels == label
            centroid = np.median(view[mask], axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(int(label) + 1),
                fontsize=5.3,
                ha="center",
                va="center",
                color="black",
                bbox={
                    "boxstyle": "circle,pad=0.13",
                    "facecolor": "white",
                    "edgecolor": palette[int(label)],
                    "linewidth": 0.6,
                    "alpha": 0.82,
                },
            )
    if display_percentiles is not None:
        lower, upper = display_percentiles
        limits = np.percentile(view, (lower, upper), axis=0)
        spans = np.maximum(limits[1] - limits[0], np.finfo(np.float32).eps)
        padding = 0.015 * spans
        ax.set_xlim(limits[0, 0] - padding[0], limits[1, 0] + padding[0])
        ax.set_ylim(limits[0, 1] - padding[1], limits[1, 1] + padding[1])
        outside = (
            (view[:, 0] < limits[0, 0])
            | (view[:, 0] > limits[1, 0])
            | (view[:, 1] < limits[0, 1])
            | (view[:, 1] > limits[1, 1])
        )
        ax.text(
            0.985,
            0.015,
            (
                f"{100 * outside.mean():.2f}% outside "
                f"{lower:g}--{upper:g} percentile axes"
            ),
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=5.5,
            color="#444444",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.0,
            },
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def render_tenx_layouts(bundle_root: Path, output_root: Path, sample_size: int) -> None:
    labels_raw = np.load(
        bundle_root / "prepared" / "tenx_mouse_brain_kmeans20.npy",
        mmap_mode="r",
    )
    unique = np.unique(labels_raw)
    mapping = {int(value): index for index, value in enumerate(unique)}
    labels = np.fromiter(
        (mapping[int(value)] for value in labels_raw),
        dtype=np.int16,
        count=len(labels_raw),
    )
    indices = shared_stratified_sample(labels, sample_size, seed=42)
    palette = label_colors(len(unique))
    methods = LAYOUT_METHODS["tenx"]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.8), constrained_layout=True)
    for ax, method in zip(axes.flat, methods):
        try:
            embedding = np.load(
                full_embedding_path(bundle_root, "tenx", method),
                mmap_mode="r",
            )
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{DISPLAY[method]}\nnot available\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                wrap=True,
            )
            continue
        plot_layout_panel(
            ax,
            embedding,
            labels,
            indices,
            DISPLAY[method],
            palette,
            annotate_centroids=True,
        )
    for ax in axes.flat[len(methods) :]:
        ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            markerfacecolor=palette[index],
            markeredgecolor="none",
            label=f"cluster {int(original)}",
        )
        for original, index in mapping.items()
    ]
    fig.legend(
        handles=handles,
        title="Released Cell Ranger labels",
        loc="outside lower center",
        frameon=False,
        ncol=5,
        fontsize=7,
        title_fontsize=8,
    )
    fig.suptitle(
        "10x embryonic mouse brain: 1,306,127 cells "
        f"(same stratified display sample, "
        rf"$m_{{\mathrm{{display}}}}={len(indices):,}$)",
        fontsize=12,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "revision3-tenx-layouts.pdf"
    save_deterministic_pdf(fig, path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def render_arxiv_layouts(bundle_root: Path, output_root: Path, sample_size: int) -> None:
    evaluation = read_json(
        bundle_root / "evaluation" / "arxiv" / "evaluation.json"
    )
    labels = np.load(
        bundle_root / "evaluation" / "arxiv" / "evaluation_labels.npy",
        mmap_mode="r",
    )
    names = evaluation["label_policy"]["code_to_name"]
    indices = shared_stratified_sample(labels, sample_size, seed=42)
    n_classes = len(np.unique(labels))
    palette = label_colors(n_classes)
    # Reserve neutral gray for the collapsed "other" group.
    other_code = max(int(code) for code in names)
    palette[other_code] = (0.68, 0.68, 0.68, 0.45)
    methods = LAYOUT_METHODS["arxiv"]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.4), constrained_layout=True)
    for ax, method in zip(axes.flat, methods):
        try:
            embedding = np.load(
                full_embedding_path(bundle_root, "arxiv", method),
                mmap_mode="r",
            )
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{DISPLAY[method]}\nnot available\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                wrap=True,
            )
            continue
        plot_layout_panel(
            ax,
            embedding,
            labels,
            indices,
            DISPLAY[method],
            palette,
            annotate_centroids=False,
            # A very small number of extreme coordinates can otherwise make
            # the visible arXiv core unreadable. The clipped fraction is
            # printed in every panel; all points remain in every metric.
            display_percentiles=(0.5, 99.5),
        )
    for ax in axes.flat[len(methods) :]:
        ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            markerfacecolor=palette[int(code)],
            markeredgecolor="none",
            label=name,
        )
        for code, name in sorted(names.items(), key=lambda item: int(item[0]))
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=6.6,
    )
    fig.suptitle(
        "arXiv BGE-small corpus: 723,457 papers "
        f"(same stratified display sample, "
        rf"$m_{{\mathrm{{display}}}}={len(indices):,}$)",
        fontsize=12,
    )
    path = output_root / "revision3-arxiv-layouts.pdf"
    save_deterministic_pdf(fig, path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def render_dire_topology_sensitivity_layouts(
    bundle_root: Path,
    output_root: Path,
    tenx_sample_size: int,
    arxiv_sample_size: int,
) -> None:
    tenx_raw = np.load(
        bundle_root / "prepared" / "tenx_mouse_brain_kmeans20.npy",
        mmap_mode="r",
    )
    tenx_unique = np.unique(tenx_raw)
    tenx_mapping = {
        int(value): index
        for index, value in enumerate(tenx_unique)
    }
    tenx_labels = np.fromiter(
        (tenx_mapping[int(value)] for value in tenx_raw),
        dtype=np.int16,
        count=len(tenx_raw),
    )
    tenx_indices = shared_stratified_sample(
        tenx_labels,
        tenx_sample_size,
        seed=42,
    )
    tenx_palette = label_colors(len(tenx_unique))

    arxiv_evaluation = read_json(
        bundle_root / "evaluation" / "arxiv" / "evaluation.json"
    )
    arxiv_labels = np.load(
        bundle_root / "evaluation" / "arxiv" / "evaluation_labels.npy",
        mmap_mode="r",
    )
    arxiv_indices = shared_stratified_sample(
        arxiv_labels,
        arxiv_sample_size,
        seed=42,
    )
    arxiv_palette = label_colors(len(np.unique(arxiv_labels)))
    other_code = max(
        int(code)
        for code in arxiv_evaluation["label_policy"]["code_to_name"]
    )
    arxiv_palette[other_code] = (0.68, 0.68, 0.68, 0.45)

    fig, axes = plt.subplots(
        2,
        len(SENSITIVITY_METHODS),
        figsize=(15.2, 7.9),
        constrained_layout=True,
    )
    for column, method in enumerate(SENSITIVITY_METHODS):
        tenx_embedding = np.load(
            full_embedding_path(bundle_root, "tenx", method),
            mmap_mode="r",
        )
        plot_layout_panel(
            axes[0, column],
            tenx_embedding,
            tenx_labels,
            tenx_indices,
            DISPLAY[method],
            tenx_palette,
            annotate_centroids=True,
        )
        arxiv_embedding = np.load(
            full_embedding_path(bundle_root, "arxiv", method),
            mmap_mode="r",
        )
        plot_layout_panel(
            axes[1, column],
            arxiv_embedding,
            arxiv_labels,
            arxiv_indices,
            DISPLAY[method],
            arxiv_palette,
            annotate_centroids=False,
            display_percentiles=(0.1, 99.9),
        )
    axes[0, 0].set_ylabel("10x mouse brain", fontsize=10)
    axes[1, 0].set_ylabel("arXiv corpus", fontsize=10)
    fig.suptitle(
        "DiRe controls: production auto policy, forced IVF-Flat, "
        "spectral initialization, and predeclared topology preset",
        fontsize=12,
    )
    path = output_root / "revision3-dire-topology-sensitivity-layouts.pdf"
    save_deterministic_pdf(fig, path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def all_scaling_results(bundle_root: Path, dataset: str) -> dict[str, list[dict]]:
    root = bundle_root / "large_results" / dataset
    payload: dict[str, list[dict]] = {method: [] for method in METHODS}
    for method in METHODS:
        method_root = root / method
        if not method_root.exists():
            continue
        for result_file in sorted(method_root.glob("n_*/result.json")):
            result = read_json(result_file)
            result["_path"] = str(result_file)
            payload[method].append(result)
    return payload


def render_scaling(bundle_root: Path, output_root: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for column, dataset in enumerate(("tenx", "arxiv")):
        results = all_scaling_results(bundle_root, dataset)
        for method in METHODS:
            successful = [
                result
                for result in results[method]
                if result.get("status") == "success"
            ]
            if not successful:
                continue
            successful.sort(key=lambda result: result["n_samples"])
            x = np.asarray([result["n_samples"] for result in successful])
            times = np.asarray(
                [result["timing"]["steady_mean_sec"] for result in successful]
            )
            time_errors = np.asarray(
                [result["timing"]["steady_std_sec"] for result in successful]
            )
            memory = np.asarray(
                [
                    result["gpu_memory"]["peak_incremental_bytes_max"]
                    / (1024**3)
                    for result in successful
                ]
            )
            axes[0, column].errorbar(
                x,
                times,
                yerr=time_errors,
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=1.5,
                capsize=2,
                label=DISPLAY[method],
            )
            axes[1, column].plot(
                x,
                memory,
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=1.5,
                label=DISPLAY[method],
            )
        title = (
            "10x mouse brain (20-D)"
            if dataset == "tenx"
            else "arXiv papers (384-D)"
        )
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("steady-state fit_transform time (s)")
        axes[1, column].set_ylabel("peak incremental GPU memory (GiB)")
        for row in range(2):
            axes[row, column].set_xscale("log")
            axes[row, column].set_yscale("log")
            axes[row, column].set_xlabel("number of observations")
            axes[row, column].grid(alpha=0.25, which="both")
            axes[row, column].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Same-device large-data scaling (single NVIDIA H100 PCIe)",
        fontsize=12,
    )
    path = output_root / "revision3-large-scaling.pdf"
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)


def make_small_runtime(
    bundle_root: Path,
    output_root: Path,
) -> list[dict]:
    log_root = bundle_root / "small_suite" / "json_logs"
    rows: list[dict] = []
    for path in sorted(log_root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = read_json(path)
        dataset = str(payload["dataset_name"])
        for method, result in payload.get("methods", {}).items():
            if method not in (
                "dire",
                "dire_topology",
                "cuml_umap",
                "cuml_tsne",
                "opentsne",
                "umap",
            ):
                continue
            repeats = [
                record
                for record in result.get("repeats", [])
                if record.get("fit_time_sec") is not None
            ]
            if not repeats:
                continue
            times = np.asarray(
                [record["fit_time_sec"] for record in repeats],
                dtype=np.float64,
            )
            steady = times[1:] if len(times) > 1 else times
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "display": SMALL_DISPLAY[method],
                    "hardware_class": result["hardware_class"],
                    "input_n_samples": int(result["input_n_samples"]),
                    "cold_sec": float(times[0]),
                    "steady_sec": float(steady.mean()),
                    "steady_std_sec": (
                        float(steady.std(ddof=1)) if len(steady) > 1 else 0.0
                    ),
                    "steady_n": int(len(steady)),
                }
            )

    fieldnames = (
        "dataset",
        "method",
        "display",
        "hardware_class",
        "input_n_samples",
        "cold_sec",
        "steady_sec",
        "steady_std_sec",
        "steady_n",
    )
    with (output_root / "revision3-small-runtime.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    dataset_order = SMALL_DATASET_ORDER
    display_dataset = SMALL_DATASET_DISPLAY
    panels = (
        (
            "GPU",
            ("dire", "dire_topology", "cuml_umap", "cuml_tsne"),
        ),
        ("CPU", ("opentsne", "umap")),
    )
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)
    x = np.arange(len(dataset_order), dtype=np.float64)
    for axis, (hardware, methods) in zip(axes, panels):
        width = 0.8 / len(methods)
        for index, method in enumerate(methods):
            positions = x - 0.4 + width / 2 + index * width
            values = [
                lookup.get((dataset, method), {}).get("steady_sec", np.nan)
                for dataset in dataset_order
            ]
            errors = [
                lookup.get((dataset, method), {}).get("steady_std_sec", 0.0)
                for dataset in dataset_order
            ]
            axis.bar(
                positions,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=SMALL_COLORS[method],
                label=SMALL_DISPLAY[method],
            )
        axis.set_yscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [display_dataset[dataset] for dataset in dataset_order],
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("steady-state fit_transform time (s)")
        axis.set_title(
            "Same H100 PCIe accelerator"
            if hardware == "GPU"
            else "CPU references on the same host"
        )
        axis.grid(axis="y", which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Small-suite runtime: hardware classes are reported separately",
        fontsize=12,
    )
    path = output_root / "revision3-small-runtime.pdf"
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)

    lines = [
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        (
            r"Dataset & Method & Hardware & Rows & Cold (s) & "
            r"Steady (s) \\"
        ),
        r"\midrule",
    ]
    previous_hardware = None
    for row in sorted(
        rows,
        key=lambda item: (
            item["hardware_class"],
            dataset_order.index(item["dataset"]),
            item["method"],
        ),
    ):
        if previous_hardware is not None and row["hardware_class"] != previous_hardware:
            lines.append(r"\addlinespace")
        lines.append(
            f"{latex_escape(display_dataset[row['dataset']])} & "
            f"{latex_escape(row['display'])} & "
            f"{latex_escape(row['hardware_class'])} & "
            f"{row['input_n_samples']:,} & "
            f"{format_seconds(row['cold_sec'])} & "
            f"{format_seconds(row['steady_sec'])} \\\\"
        )
        previous_hardware = row["hardware_class"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-small-runtime.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return rows


def make_small_atlas_topology_effects(
    bundle_root: Path,
    output_root: Path,
) -> tuple[list[dict], list[dict]]:
    """Render the repeated, paired atlas-topology audit for the small suite."""

    log_root = bundle_root / "small_suite" / "json_logs"
    metric_keys = (
        ("dtw_beta0", r"$\beta_0$"),
        ("dtw_beta1", r"$\beta_1$"),
    )
    methods = (
        "dire",
        "dire_topology",
        "cuml_umap",
        "cuml_tsne",
        "opentsne",
        "umap",
    )
    summaries: list[dict] = []
    effects: list[dict] = []
    for dataset in SMALL_DATASET_ORDER:
        path = log_root / f"{dataset}.json"
        if not path.exists():
            continue
        payload = read_json(path)
        available = payload.get("methods", {})
        if "dire" not in available or "dire_topology" not in available:
            continue
        records_by_method: dict[str, dict[tuple, dict]] = {}
        identity_hashes: dict[tuple[int, int], str] = {}
        for method in methods:
            result = available.get(method)
            if not isinstance(result, dict):
                continue
            method_records: dict[tuple, dict] = {}
            for record in result.get("repeats", []):
                topology = record.get("metrics", {}).get("topology")
                sample = record.get("topology_sample")
                if not isinstance(topology, dict) or not isinstance(sample, dict):
                    continue
                if topology.get("backend") != "atlas":
                    raise RuntimeError(
                        f"{dataset}/{method} small-suite topology backend is "
                        f"{topology.get('backend')!r}, expected 'atlas'"
                    )
                if sample.get("backend") != "atlas":
                    raise RuntimeError(
                        f"{dataset}/{method} topology metadata does not "
                        "declare the atlas backend"
                    )
                if (
                    topology.get("backend_detail")
                    != SMALL_TOPOLOGY_BACKEND_DETAIL
                    or sample.get("backend_detail")
                    != SMALL_TOPOLOGY_BACKEND_DETAIL
                    or topology.get("prefer_ripser") is not False
                    or sample.get("prefer_ripser") is not False
                ):
                    raise RuntimeError(
                        f"{dataset}/{method} did not execute the direct "
                        "rank-based atlas path"
                    )
                identity = (
                    int(record["seed"]),
                    int(sample["subset_seed"]),
                )
                digest = str(sample["indices_sha256"])
                previous = identity_hashes.setdefault(identity, digest)
                if previous != digest:
                    raise RuntimeError(
                        f"{dataset} paired topology indices differ for "
                        f"layout/subset seeds {identity}"
                    )
                key = (*identity, digest)
                method_records[key] = record
            if method_records:
                records_by_method[method] = method_records
        for method, method_records in records_by_method.items():
            for metric, _metric_display in metric_keys:
                values = np.asarray(
                    [
                        record["metrics"]["topology"]["metrics"][metric]
                        for record in method_records.values()
                    ],
                    dtype=np.float64,
                )
                if not np.isfinite(values).all() or np.any(values < 0.0):
                    raise RuntimeError(
                        f"{dataset}/{method}/{metric} contains invalid values"
                    )
                summaries.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "display": SMALL_DISPLAY[method],
                        "metric": metric,
                        "repeat_count": int(len(values)),
                        "mean": float(values.mean()),
                        "sample_sd": (
                            float(values.std(ddof=1))
                            if len(values) > 1
                            else 0.0
                        ),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )

        reference = records_by_method["dire"]
        for comparator, comparator_records in records_by_method.items():
            if comparator == "dire":
                continue
            paired_keys = sorted(set(reference).intersection(comparator_records))
            if not paired_keys:
                raise RuntimeError(
                    f"{dataset}/{comparator} has no paired atlas records"
                )
            for metric, _metric_display in metric_keys:
                reference_values = np.asarray(
                    [
                        reference[key]["metrics"]["topology"]["metrics"][metric]
                        for key in paired_keys
                    ],
                    dtype=np.float64,
                )
                comparator_values = np.asarray(
                    [
                        comparator_records[key]["metrics"]["topology"][
                            "metrics"
                        ][metric]
                        for key in paired_keys
                    ],
                    dtype=np.float64,
                )
                gaps = comparator_values - reference_values
                gap_mean = float(gaps.mean())
                gap_sd = (
                    float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
                )
                if len(gaps) > 1:
                    critical = float(
                        student_t.ppf(0.975, df=len(gaps) - 1)
                    )
                    half_width = critical * gap_sd / math.sqrt(len(gaps))
                else:
                    half_width = 0.0
                reference_mean = float(reference_values.mean())
                relative_gap = (
                    gap_mean / reference_mean
                    if reference_mean > 0.0
                    else math.nan
                )
                positive = np.maximum(
                    comparator_values,
                    np.finfo(np.float64).tiny,
                )
                reference_positive = np.maximum(
                    reference_values,
                    np.finfo(np.float64).tiny,
                )
                log_ratios = np.log2(positive / reference_positive)
                if math.isfinite(relative_gap) and abs(relative_gap) <= 0.05:
                    conclusion = "practically comparable (within 5%)"
                elif gap_mean - half_width <= 0.0 <= gap_mean + half_width:
                    conclusion = "paired-mean direction unresolved"
                elif gap_mean > 0.0:
                    conclusion = "DiRe has lower discrepancy"
                else:
                    conclusion = f"{SMALL_DISPLAY[comparator]} has lower discrepancy"
                effects.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "reference": "dire",
                        "comparator": comparator,
                        "comparator_display": SMALL_DISPLAY[comparator],
                        "paired_count": int(len(gaps)),
                        "reference_mean": reference_mean,
                        "comparator_mean": float(comparator_values.mean()),
                        "paired_gap_comparator_minus_dire_mean": gap_mean,
                        "paired_gap_sample_sd": gap_sd,
                        "paired_mean_95pct_low": gap_mean - half_width,
                        "paired_mean_95pct_high": gap_mean + half_width,
                        "relative_gap": relative_gap,
                        "paired_log2_ratio_mean": float(log_ratios.mean()),
                        "paired_log2_ratio_sample_sd": (
                            float(log_ratios.std(ddof=1))
                            if len(log_ratios) > 1
                            else 0.0
                        ),
                        "minimum_comparator": float(comparator_values.min()),
                        "minimum_dire": float(reference_values.min()),
                        "paired_gap_values": json.dumps(
                            [float(value) for value in gaps],
                            separators=(",", ":"),
                        ),
                        "paired_layout_subset_ids": json.dumps(
                            [
                                {
                                    "layout_seed": int(key[0]),
                                    "subset_seed": int(key[1]),
                                    "indices_sha256": str(key[2]),
                                }
                                for key in paired_keys
                            ],
                            separators=(",", ":"),
                        ),
                        "interpretation": conclusion,
                    }
                )

    if not summaries:
        return [], []
    if not effects:
        raise RuntimeError(
            "small-suite atlas summaries exist but no paired DiRe effects "
            "could be constructed"
        )

    with (output_root / "revision3-small-atlas-topology-summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summaries[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    with (output_root / "revision3-small-atlas-topology-paired-effects.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(effects[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(effects)

    summary_lookup = {
        (row["dataset"], row["method"], row["metric"]): row
        for row in summaries
    }
    table_methods = ("dire", "dire_topology", "cuml_umap", "cuml_tsne")
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & Metric & DiRe & Topology preset & cuML UMAP & "
            r"cuML t-SNE \\"
        ),
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(SMALL_DATASET_ORDER):
        if dataset_index:
            lines.append(r"\addlinespace")
        for metric, metric_display in metric_keys:
            cells = []
            for method in table_methods:
                row = summary_lookup.get((dataset, method, metric))
                if row is None:
                    cells.append("--")
                else:
                    cells.append(
                        f"{row['mean']:.3f}$\\pm${row['sample_sd']:.3f}"
                        f" [{row['minimum']:.3f}]"
                    )
            lines.append(
                f"{latex_escape(SMALL_DATASET_DISPLAY[dataset])} & "
                f"{metric_display} & "
                + " & ".join(cells)
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-small-atlas-topology-summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    plotted_methods = ("dire_topology", "cuml_umap", "cuml_tsne")
    effect_lookup = {
        (row["dataset"], row["comparator"], row["metric"]): row
        for row in effects
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.6),
        sharex=True,
        constrained_layout=True,
    )
    x = np.arange(len(SMALL_DATASET_ORDER), dtype=np.float64)
    width = 0.78 / len(plotted_methods)
    all_abs = [math.log2(1.05)]
    for axis, (metric, metric_display) in zip(axes, metric_keys):
        for method_index, method in enumerate(plotted_methods):
            positions = (
                x - 0.39 + width / 2.0 + method_index * width
            )
            values = []
            errors = []
            for dataset in SMALL_DATASET_ORDER:
                row = effect_lookup.get((dataset, method, metric))
                values.append(
                    np.nan
                    if row is None
                    else row["paired_log2_ratio_mean"]
                )
                errors.append(
                    0.0
                    if row is None
                    else row["paired_log2_ratio_sample_sd"]
                )
            all_abs.extend(
                abs(value)
                for value in values
                if math.isfinite(value)
            )
            all_abs.extend(
                abs(value) + error
                for value, error in zip(values, errors)
                if math.isfinite(value)
            )
            axis.bar(
                positions,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=SMALL_COLORS[method],
                label=SMALL_DISPLAY[method],
            )
        band = math.log2(1.05)
        axis.axhspan(-band, band, color="#777777", alpha=0.10)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(
            rf"$\log_2$(comparator / DiRe), {metric_display}"
        )
        axis.grid(axis="y", alpha=0.22)
    bound = max(all_abs) * 1.08
    for axis in axes:
        axis.set_ylim(-bound, bound)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [SMALL_DATASET_DISPLAY[name] for name in SMALL_DATASET_ORDER],
        rotation=25,
        ha="right",
    )
    fig.suptitle(
        "Paired atlas-topology effects (positive values favour default DiRe; "
        "whiskers show observed paired SD)",
        fontsize=11,
    )
    path = output_root / "revision3-small-atlas-topology-paired-effects.pdf"
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)
    return summaries, effects


def quality_value(method: dict, metric: str) -> float | None:
    if metric == "knn":
        return method["local"]["neighbor_overlap"]["mean"]
    if metric == "centroid":
        return method["global"]["pairwise_distance_spearman"]["correlation"]
    if metric == "adjacency":
        return method["global"]["centroid_adjacency_recall"]["mean"]
    if metric == "context":
        return method["context_classifier"]["balanced_accuracy"]
    if metric == "dtw_beta0":
        return method["topology"]["aggregate"]["dtw_beta0"]["mean"]
    if metric == "dtw_beta1":
        return method["topology"]["aggregate"]["dtw_beta1"]["mean"]
    if metric == "bottleneck_beta0":
        return method["topology"]["aggregate"]["bottleneck_beta0"]["mean"]
    if metric == "bottleneck_beta1":
        return method["topology"]["aggregate"]["bottleneck_beta1"]["mean"]
    if metric == "q99_dtw_beta0":
        return method["topology"]["aggregate"]["q99_dtw_beta0"]["mean"]
    if metric == "q99_dtw_beta1":
        return method["topology"]["aggregate"]["q99_dtw_beta1"]["mean"]
    if metric == "q99_bottleneck_beta0":
        return method["topology"]["aggregate"]["q99_bottleneck_beta0"]["mean"]
    if metric == "q99_bottleneck_beta1":
        return method["topology"]["aggregate"]["q99_bottleneck_beta1"]["mean"]
    raise ValueError(metric)


def large_quality_log2_effect(
    metric: str,
    comparator_value: float,
    dire_value: float,
) -> float:
    """Return a symmetric comparator-versus-DiRe effect.

    Positive values favour the comparator and negative values favour
    production-policy DiRe.  Correlation is first mapped from [-1, 1] to
    [0, 1]; the remaining higher-is-better metrics already have nonnegative
    ratio scales.  For nonnegative discrepancy metrics the ratio is reversed,
    so a smaller comparator discrepancy gives a positive effect.
    """

    comparator = float(comparator_value)
    dire = float(dire_value)
    if metric == "centroid":
        if not (-1.0 <= comparator <= 1.0 and -1.0 <= dire <= 1.0):
            raise ValueError("centroid correlations must lie in [-1, 1]")
        comparator = (comparator + 1.0) / 2.0
        dire = (dire + 1.0) / 2.0
    elif metric not in {
        "knn",
        "adjacency",
        "context",
        "dtw_beta0",
        "dtw_beta1",
        "bottleneck_beta0",
        "bottleneck_beta1",
        "q99_dtw_beta0",
        "q99_dtw_beta1",
        "q99_bottleneck_beta0",
        "q99_bottleneck_beta1",
    }:
        raise ValueError(f"unsupported large-data quality metric: {metric}")
    if comparator < 0.0 or dire < 0.0:
        raise ValueError(f"{metric} values must be nonnegative")
    if np.isclose(comparator, 0.0, rtol=0.0, atol=1e-15) and np.isclose(
        dire,
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        return 0.0
    higher_is_better = metric in {
        "knn",
        "centroid",
        "adjacency",
        "context",
    }
    numerator, denominator = (
        (comparator, dire)
        if higher_is_better
        else (dire, comparator)
    )
    if denominator == 0.0:
        return math.inf
    if numerator == 0.0:
        return -math.inf
    return float(np.log2(numerator / denominator))


def format_log2_effect(value: float) -> str:
    if np.isposinf(value):
        return "+∞"
    if np.isneginf(value):
        return "−∞"
    return f"{value:+.2f}"


def render_quality(bundle_root: Path, output_root: Path) -> None:
    metrics = (
        ("knn", "kNN overlap", True),
        ("centroid", "centroid $\\rho$", True),
        ("adjacency", "centroid adj.", True),
        ("context", "SVM bal. acc.", True),
        ("bottleneck_beta0", "$d_N$ $\\beta_0$", False),
        ("bottleneck_beta1", "$d_N$ $\\beta_1$", False),
    )
    evaluations = {
        dataset: read_json(
            bundle_root / "evaluation" / dataset / "evaluation.json"
        )
        for dataset in ("tenx", "arxiv")
    }
    available_by_dataset = {
        dataset: [
            method
            for method in QUALITY_METHODS[dataset]
            if method in evaluations[dataset]["methods"]
        ]
        for dataset in ("tenx", "arxiv")
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)
    last_image = None
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="#f2f2f2")
    for ax, dataset in zip(axes, ("tenx", "arxiv")):
        evaluation = evaluations[dataset]
        available = available_by_dataset[dataset]
        raw = np.asarray(
            [
                [quality_value(evaluation["methods"][method], key) for key, _label, _hib in metrics]
                for method in available
            ],
            dtype=np.float64,
        )
        dire_index = available.index("dire_auto")
        effect_matrix = np.full_like(raw, np.nan)
        for column, (key, _label, _higher) in enumerate(metrics):
            dire_value = raw[dire_index, column]
            for row, value in enumerate(raw[:, column]):
                if math.isfinite(value) and math.isfinite(dire_value):
                    effect_matrix[row, column] = large_quality_log2_effect(
                        key,
                        value,
                        dire_value,
                    )
        last_image = ax.imshow(
            np.clip(effect_matrix, -1.0, 1.0),
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_title("10x mouse brain" if dataset == "tenx" else "arXiv corpus")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(
            [label for _key, label, _higher in metrics],
            rotation=28,
            ha="right",
        )
        ax.set_yticks(range(len(available)))
        ax.set_yticklabels([DISPLAY[method] for method in available])
        for row in range(len(available)):
            for column in range(len(metrics)):
                ax.text(
                    column,
                    row,
                    (
                        f"{format_log2_effect(effect_matrix[row, column])}\n"
                        f"raw {raw[row, column]:.3f}"
                    ),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(available), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes, shrink=0.78)
        colorbar.set_ticks((-1.0, 0.0, 1.0))
        colorbar.set_ticklabels(
            ("DiRe 2× better", "equal", "comparator 2× better")
        )
        colorbar.set_label(
            "direction-adjusted log₂ ratio vs production DiRe "
            "(clipped at ±1)"
        )
    fig.suptitle(
        "Large-data fidelity versus production DiRe "
        "(centroid effect uses (1+ρ)/2; cells retain raw ρ)",
        fontsize=11,
    )
    path = output_root / "revision3-large-quality.pdf"
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)


def make_dataset_stats_table(bundle_root: Path, output_root: Path) -> None:
    tenx = read_json(bundle_root / "prepared" / "tenx_manifest.json")
    arxiv = read_json(bundle_root / "prepared" / "arxiv_manifest.json")
    rows = [
        {
            "dataset": "10x embryonic mouse brain",
            "n": tenx["n_samples"],
            "d": tenx["n_features"],
            "input": "official Cell Ranger PCA scores",
            "labels": "official k-means-20 and graph clusters",
            "license": tenx["license"],
        },
        {
            "dataset": "arXiv BGE-small corpus",
            "n": arxiv["n_samples"],
            "d": arxiv["n_features"],
            "input": "mean-pooled BGE-small embeddings",
            "labels": "primary arXiv category",
            "license": "CC BY 4.0 / metadata CC0",
        },
    ]
    with (output_root / "revision3-dataset-stats.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        r"\begin{tabular}{@{}lrrlll@{}}",
        r"\toprule",
        r"Dataset & $n$ & $D$ & Common input & Reference labels & License \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['dataset'])} & {row['n']:,} & {row['d']} & "
            f"{latex_escape(row['input'])} & {latex_escape(row['labels'])} & "
            f"{latex_escape(row['license'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-dataset-stats.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def make_large_metrics_table(bundle_root: Path, output_root: Path) -> list[dict]:
    rows = []
    for dataset in ("tenx", "arxiv"):
        evaluation = read_json(
            bundle_root / "evaluation" / dataset / "evaluation.json"
        )
        for method in QUALITY_METHODS[dataset]:
            if method not in evaluation["methods"]:
                continue
            metrics = evaluation["methods"][method]
            run = metrics.get("run") or {}
            timing = run.get("timing", {})
            memory = run.get("gpu_memory", {})
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "display": DISPLAY[method],
                    "cold_sec": timing.get("cold_start_sec"),
                    "steady_sec": timing.get("steady_mean_sec"),
                    "gpu_gib": (
                        memory.get("peak_incremental_bytes_max") / (1024**3)
                        if memory.get("peak_incremental_bytes_max") is not None
                        else None
                    ),
                    "knn_overlap": quality_value(metrics, "knn"),
                    "centroid_spearman": quality_value(metrics, "centroid"),
                    "centroid_adjacency_recall": quality_value(
                        metrics,
                        "adjacency",
                    ),
                    "context_balanced_accuracy": quality_value(metrics, "context"),
                    "bottleneck_beta0": quality_value(
                        metrics,
                        "bottleneck_beta0",
                    ),
                    "bottleneck_beta1": quality_value(
                        metrics,
                        "bottleneck_beta1",
                    ),
                    "dtw_beta0": quality_value(metrics, "dtw_beta0"),
                    "dtw_beta1": quality_value(metrics, "dtw_beta1"),
                    "q99_bottleneck_beta0": quality_value(
                        metrics,
                        "q99_bottleneck_beta0",
                    ),
                    "q99_bottleneck_beta1": quality_value(
                        metrics,
                        "q99_bottleneck_beta1",
                    ),
                    "q99_dtw_beta0": quality_value(
                        metrics,
                        "q99_dtw_beta0",
                    ),
                    "q99_dtw_beta1": quality_value(
                        metrics,
                        "q99_dtw_beta1",
                    ),
                }
            )
    fieldnames = [
        "dataset",
        "method",
        "display",
        "cold_sec",
        "steady_sec",
        "gpu_gib",
        "knn_overlap",
        "centroid_spearman",
        "centroid_adjacency_recall",
        "context_balanced_accuracy",
        "bottleneck_beta0",
        "bottleneck_beta1",
        "dtw_beta0",
        "dtw_beta1",
        "q99_bottleneck_beta0",
        "q99_bottleneck_beta1",
        "q99_dtw_beta0",
        "q99_dtw_beta1",
    ]
    with (output_root / "revision3-large-metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{tabular}{@{}llrrrrrrrrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & Method & Cold (s) & Steady (s) & GPU (GiB) & "
            r"$k$NN & Centroid $\rho$ & Adj. & SVM & "
            r"$d_N(\beta_0)$ & $d_N(\beta_1)$ & "
            r"DTW $\beta_0$ & DTW $\beta_1$ \\"
        ),
        r"\midrule",
    ]
    previous_dataset = None
    for row in rows:
        if previous_dataset is not None and row["dataset"] != previous_dataset:
            lines.append(r"\addlinespace")
        dataset_name = "10x" if row["dataset"] == "tenx" else "arXiv"
        lines.append(
            f"{dataset_name} & {latex_escape(row['display'])} & "
            f"{format_seconds(row['cold_sec'])} & "
            f"{format_seconds(row['steady_sec'])} & "
            f"{format_float(row['gpu_gib'], 2)} & "
            f"{format_float(row['knn_overlap'])} & "
            f"{format_float(row['centroid_spearman'])} & "
            f"{format_float(row['centroid_adjacency_recall'])} & "
            f"{format_float(row['context_balanced_accuracy'])} & "
            f"{format_float(row['bottleneck_beta0'])} & "
            f"{format_float(row['bottleneck_beta1'])} & "
            f"{format_float(row['dtw_beta0'])} & "
            f"{format_float(row['dtw_beta1'])} \\\\"
        )
        previous_dataset = row["dataset"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-large-metrics.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return rows


def make_backend_policy_audit(
    bundle_root: Path,
    output_root: Path,
) -> list[dict]:
    rows = []
    quality_rows = []
    for dataset in ("tenx", "arxiv"):
        evaluation = read_json(
            bundle_root / "evaluation" / dataset / "evaluation.json"
        )
        audit = evaluation.get("backend_policy_audit")
        if not audit:
            continue
        auto = audit["production_auto"]
        control = audit["forced_ivf_flat_control"]

        def stage(record: dict, name: str) -> float | None:
            payload = record.get("stage_timings", {}).get(name)
            return (
                None
                if not payload
                else float(payload["steady_mean_sec"])
            )

        effective = sorted(
            {
                str(value)
                for value in auto["effective_cuvs_index_types"]
                if value is not None
            }
        )
        fallback_calls = sum(auto["chunked_force_fallback_calls"])
        row = {
            "dataset": dataset,
            "N": int(audit["N_full_dataset"]),
            "R_fit_total": int(audit["R_fit_total"]),
            "R_steady": int(audit["R_steady"]),
            "m_graph_queries": int(audit["m_graph_queries"]),
            "auto_requested_index": auto["requested_cuvs_index_type"],
            "auto_effective_index": ",".join(effective),
            "auto_cold_total_sec": float(auto["cold_total_sec"]),
            "auto_steady_total_sec": float(auto["steady_total_mean_sec"]),
            "auto_steady_total_std_sec": float(
                auto["steady_total_std_sec"]
            ),
            "auto_steady_knn_graph_sec": stage(auto, "knn_graph"),
            "auto_steady_initialization_sec": stage(auto, "initialization"),
            "auto_steady_layout_sec": stage(auto, "layout"),
            "auto_chunked_fallback_calls": int(fallback_calls),
            "control_steady_total_sec": float(
                control["steady_total_mean_sec"]
            ),
            "steady_speedup_control_over_auto": float(
                audit["steady_speedup_control_over_auto"]
            ),
            "graph_overlap_mean": float(
                audit["knn_graph_overlap"]["mean"]
            ),
            "graph_overlap_median": float(
                audit["knn_graph_overlap"]["median"]
            ),
            "graph_overlap_q05": float(
                audit["knn_graph_overlap"]["q05"]
            ),
            "graph_overlap_q95": float(
                audit["knn_graph_overlap"]["q95"]
            ),
            "graph_zero_overlap_fraction": float(
                audit["knn_graph_overlap"]["zero_overlap_fraction"]
            ),
            "graph_exact_set_match_fraction": float(
                audit["knn_graph_overlap"]["exact_set_match_fraction"]
            ),
        }
        rows.append(row)
        for metric, comparison in audit["full_embedding_quality_gate"][
            "metrics"
        ].items():
            direction = comparison["direction"]
            raw_delta = float(comparison["auto_minus_control"])
            quality_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "direction": direction,
                    "production_auto": float(
                        comparison["production_auto"]
                    ),
                    "forced_ivf_flat": float(
                        comparison["forced_ivf_flat"]
                    ),
                    "auto_minus_control": raw_delta,
                    "directional_difference_favoring_auto": (
                        raw_delta if direction == "higher" else -raw_delta
                    ),
                    "relative_auto_minus_control": comparison[
                        "relative_auto_minus_control"
                    ],
                }
            )

    if not rows:
        return []
    with (output_root / "revision3-backend-policy-audit.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "revision3-backend-policy-quality-gate.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(quality_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(quality_rows)

    lines = [
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & Effective auto index & Cold (s) & Steady (s) & "
            r"Graph (s) & PCA (s) & Layout (s) & IVF-Flat (s) & "
            r"Speed-up & Graph overlap \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        dataset_name = "10x" if row["dataset"] == "tenx" else "arXiv"
        lines.append(
            f"{dataset_name} & {latex_escape(row['auto_effective_index'])} & "
            f"{format_seconds(row['auto_cold_total_sec'])} & "
            f"{format_seconds(row['auto_steady_total_sec'])} & "
            f"{format_seconds(row['auto_steady_knn_graph_sec'])} & "
            f"{format_seconds(row['auto_steady_initialization_sec'])} & "
            f"{format_seconds(row['auto_steady_layout_sec'])} & "
            f"{format_seconds(row['control_steady_total_sec'])} & "
            f"{row['steady_speedup_control_over_auto']:.2f}$\\times$ & "
            f"{row['graph_overlap_mean']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-backend-policy-audit.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        1,
        len(rows),
        figsize=(5.6 * len(rows), 4.6),
        constrained_layout=True,
        squeeze=False,
    )
    stage_specs = (
        ("auto_steady_knn_graph_sec", "kNN graph", "#4daf4a"),
        ("auto_steady_initialization_sec", "initialization", "#377eb8"),
        ("auto_steady_layout_sec", "layout", "#984ea3"),
    )
    for ax, row in zip(axes.flat, rows):
        auto_parts = [
            max(0.0, float(row[key] or 0.0))
            for key, _label, _color in stage_specs
        ]
        auto_overhead = max(
            0.0,
            row["auto_steady_total_sec"] - sum(auto_parts),
        )
        auto_parts.append(auto_overhead)
        labels = [label for _key, label, _color in stage_specs] + ["other"]
        colors = [color for _key, _label, color in stage_specs] + ["#bdbdbd"]
        bottom = 0.0
        for value, label, color in zip(auto_parts, labels, colors):
            ax.bar(
                0,
                value,
                bottom=bottom,
                color=color,
                width=0.62,
                label=label,
            )
            bottom += value
        ax.bar(
            1,
            row["control_steady_total_sec"],
            color="#80b1a8",
            width=0.62,
            label="IVF-Flat total",
        )
        ax.text(
            0,
            row["auto_steady_total_sec"],
            f"{row['auto_steady_total_sec']:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            1,
            row["control_steady_total_sec"],
            f"{row['control_steady_total_sec']:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.set_xticks((0, 1), ("production auto", "forced IVF-Flat"))
        ax.set_ylabel("steady fit_transform time (s)")
        ax.set_title(
            "10x mouse brain" if row["dataset"] == "tenx" else "arXiv corpus"
        )
        ax.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "DiRe cuVS policy audit: synchronized stage timings and explicit control",
        fontsize=12,
    )
    path = output_root / "revision3-backend-policy-audit.pdf"
    save_deterministic_pdf(fig, path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return rows


def make_topology_sensitivity_table(
    bundle_root: Path,
    output_root: Path,
    rows: list[dict],
) -> None:
    output_rows = []
    for row in rows:
        evaluation = read_json(
            bundle_root
            / "evaluation"
            / row["dataset"]
            / "evaluation.json"
        )
        records = evaluation["methods"][row["method"]]["topology"]["records"]
        layout_seed_sensitivity = evaluation["methods"][row["method"]][
            "topology"
        ].get("layout_seed_sensitivity")
        ratios = [
            float(record["embedding_scales"]["diameter_to_q99_ratio"])
            for record in records
        ]
        output_row = {
                "dataset": row["dataset"],
                "method": row["method"],
                "display": row["display"],
                "diameter_bottleneck_beta0": row["bottleneck_beta0"],
                "diameter_bottleneck_beta1": row["bottleneck_beta1"],
                "q99_bottleneck_beta0": row["q99_bottleneck_beta0"],
                "q99_bottleneck_beta1": row["q99_bottleneck_beta1"],
                "diameter_dtw_beta0": row["dtw_beta0"],
                "diameter_dtw_beta1": row["dtw_beta1"],
                "q99_dtw_beta0": row["q99_dtw_beta0"],
                "q99_dtw_beta1": row["q99_dtw_beta1"],
                "diameter_to_q99_ratio_mean": float(np.mean(ratios)),
        }
        for key in (
            "bottleneck_beta0",
            "bottleneck_beta1",
            "dtw_beta0",
            "dtw_beta1",
            "q99_bottleneck_beta0",
            "q99_bottleneck_beta1",
            "q99_dtw_beta0",
            "q99_dtw_beta1",
        ):
            aggregate = (
                layout_seed_sensitivity["aggregate"][key]
                if layout_seed_sensitivity
                else None
            )
            output_row[f"layout_seed_{key}_mean"] = (
                aggregate["mean"] if aggregate else None
            )
            output_row[f"layout_seed_{key}_std"] = (
                aggregate["std"] if aggregate else None
            )
        output_rows.append(output_row)
    fields = list(output_rows[0])
    with (output_root / "revision3-topology-sensitivity.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & Method & $d_N(\beta_0)$ & $d_N(\beta_1)$ & "
            r"Q99 $d_B(\beta_0)$ & Q99 $d_B(\beta_1)$ & Diam./Q99 \\"
        ),
        r"\midrule",
    ]
    previous_dataset = None
    for row in output_rows:
        if previous_dataset is not None and row["dataset"] != previous_dataset:
            lines.append(r"\addlinespace")
        dataset_name = "10x" if row["dataset"] == "tenx" else "arXiv"
        lines.append(
            f"{dataset_name} & {latex_escape(row['display'])} & "
            f"{format_float(row['diameter_bottleneck_beta0'], 4)} & "
            f"{format_float(row['diameter_bottleneck_beta1'], 4)} & "
            f"{format_float(row['q99_bottleneck_beta0'], 4)} & "
            f"{format_float(row['q99_bottleneck_beta1'], 4)} & "
            f"{format_float(row['diameter_to_q99_ratio_mean'], 2)} \\\\"
        )
        previous_dataset = row["dataset"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-topology-sensitivity.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    seed_lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & Method & $d_N(\beta_0)$ & $d_N(\beta_1)$ & "
            r"DTW $\beta_0$ & DTW $\beta_1$ \\"
        ),
        r"\midrule",
    ]
    previous_dataset = None
    for row in output_rows:
        if row["layout_seed_bottleneck_beta0_mean"] is None:
            continue
        if previous_dataset is not None and row["dataset"] != previous_dataset:
            seed_lines.append(r"\addlinespace")
        dataset_name = "10x" if row["dataset"] == "tenx" else "arXiv"
        seed_lines.append(
            f"{dataset_name} & {latex_escape(row['display'])} & "
            f"{format_float(row['layout_seed_bottleneck_beta0_mean'], 4)}"
            f"$\\pm${format_float(row['layout_seed_bottleneck_beta0_std'], 4)} & "
            f"{format_float(row['layout_seed_bottleneck_beta1_mean'], 4)}"
            f"$\\pm${format_float(row['layout_seed_bottleneck_beta1_std'], 4)} & "
            f"{format_float(row['layout_seed_dtw_beta0_mean'], 3)}"
            f"$\\pm${format_float(row['layout_seed_dtw_beta0_std'], 3)} & "
            f"{format_float(row['layout_seed_dtw_beta1_mean'], 3)}"
            f"$\\pm${format_float(row['layout_seed_dtw_beta1_std'], 3)} \\\\"
        )
        previous_dataset = row["dataset"]
    seed_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (
        output_root / "revision3-topology-layout-seed-sensitivity.tex"
    ).write_text(
        "\n".join(seed_lines) + "\n",
        encoding="utf-8",
    )


def make_topology_subset_size_sensitivity(
    bundle_root: Path,
    output_root: Path,
) -> list[dict]:
    rows = []
    for dataset in ("tenx", "arxiv"):
        evaluation = read_json(
            bundle_root / "evaluation" / dataset / "evaluation.json"
        )
        for method in QUALITY_METHODS[dataset]:
            method_payload = evaluation["methods"].get(method)
            if method_payload is None:
                continue
            sensitivity = method_payload["topology"].get(
                "subset_size_sensitivity"
            )
            if not sensitivity:
                continue
            for record in sensitivity["records"]:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "display": DISPLAY[method],
                        "subset_size": int(
                            record["sensitivity_subset_size"]
                        ),
                        **{
                            key: float(record[key])
                            for key in (
                                "bottleneck_beta0",
                                "bottleneck_beta1",
                                "q99_bottleneck_beta0",
                                "q99_bottleneck_beta1",
                                "dtw_beta0",
                                "dtw_beta1",
                                "q99_dtw_beta0",
                                "q99_dtw_beta1",
                            )
                        },
                    }
                )
    if not rows:
        return []
    rows.sort(
        key=lambda row: (
            ("tenx", "arxiv").index(row["dataset"]),
            QUALITY_METHODS[row["dataset"]].index(row["method"]),
            row["subset_size"],
        )
    )
    fields = list(rows[0])
    with (
        output_root / "revision3-topology-subset-size-sensitivity.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.0),
        sharex="col",
        constrained_layout=True,
    )
    for row_index, dataset in enumerate(("tenx", "arxiv")):
        dataset_rows = [
            row for row in rows if row["dataset"] == dataset
        ]
        for column, dimension in enumerate((0, 1)):
            ax = axes[row_index, column]
            for method in QUALITY_METHODS[dataset]:
                method_rows = [
                    row
                    for row in dataset_rows
                    if row["method"] == method
                ]
                if not method_rows:
                    continue
                ax.plot(
                    [row["subset_size"] for row in method_rows],
                    [
                        row[f"dtw_beta{dimension}"]
                        for row in method_rows
                    ],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=1.5,
                    markersize=5,
                    label=DISPLAY[method],
                )
            dataset_name = (
                "10x mouse brain" if dataset == "tenx" else "arXiv corpus"
            )
            ax.set_title(
                f"{dataset_name}: Betti-DTW $\\beta_{dimension}$"
            )
            ax.set_xscale("log")
            ax.set_xticks(
                sorted(
                    {
                        row["subset_size"]
                        for row in dataset_rows
                    }
                )
            )
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(
                    lambda value, _position: f"{int(value):,}"
                )
            )
            ax.set_xlabel("nested topology subset size")
            if column == 0:
                ax.set_ylabel("Betti--DTW discrepancy (lower better)")
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Betti-DTW subset-size sensitivity on fixed full-data layouts",
        fontsize=12,
    )
    path = output_root / "revision3-topology-subset-size-sensitivity.pdf"
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)

    sizes = sorted({row["subset_size"] for row in rows})
    lines = [
        r"\begin{tabular}{@{}ll" + ("rr" * len(sizes)) + r"@{}}",
        r"\toprule",
        (
            r"Dataset & Method & "
            + " & ".join(
                f"DTW $\\beta_0$, $m={size:,}$ & "
                f"DTW $\\beta_1$, $m={size:,}$"
                for size in sizes
            )
            + r" \\"
        ),
        r"\midrule",
    ]
    previous_dataset = None
    for dataset in ("tenx", "arxiv"):
        for method in QUALITY_METHODS[dataset]:
            method_rows = {
                row["subset_size"]: row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            }
            if not method_rows:
                continue
            if previous_dataset is not None and dataset != previous_dataset:
                lines.append(r"\addlinespace")
            dataset_name = "10x" if dataset == "tenx" else "arXiv"
            values = []
            for size in sizes:
                row = method_rows.get(size)
                values.extend(
                    (
                        format_float(
                            row["dtw_beta0"] if row else None,
                            4,
                        ),
                        format_float(
                            row["dtw_beta1"] if row else None,
                            4,
                        ),
                    )
                )
            lines.append(
                f"{dataset_name} & {latex_escape(DISPLAY[method])} & "
                + " & ".join(values)
                + r" \\"
            )
            previous_dataset = dataset
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (
        output_root / "revision3-topology-subset-size-sensitivity.tex"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return rows


def make_topology_paired_effects(
    bundle_root: Path,
    output_root: Path,
) -> list[dict]:
    """Report effect magnitudes and observed variability, not rank alone."""

    metric_keys = (
        "dtw_beta0",
        "dtw_beta1",
        "bottleneck_beta0",
        "bottleneck_beta1",
        "q99_dtw_beta0",
        "q99_dtw_beta1",
        "q99_bottleneck_beta0",
        "q99_bottleneck_beta1",
    )
    effect_rows = []
    size_rows = []
    topology_design_by_dataset = {}
    for dataset in ("tenx", "arxiv"):
        evaluation = read_json(
            bundle_root / "evaluation" / dataset / "evaluation.json"
        )
        topology_design_by_dataset[dataset] = evaluation["topology_design"]
        methods = evaluation["methods"]
        if "dire_auto" not in methods:
            continue
        reference = methods["dire_auto"]["topology"]

        for basis, record_key, identifier in (
            ("subset", "records", "subset_id"),
            ("layout", "layout_seed_sensitivity", "layout_seed"),
        ):
            if basis == "subset":
                reference_records = reference["records"]
            else:
                sensitivity = reference.get(record_key)
                if not sensitivity:
                    continue
                reference_records = sensitivity["records"]
            reference_by_id = {
                int(record[identifier]): record
                for record in reference_records
            }
            for competitor, competitor_payload in methods.items():
                if competitor == "dire_auto":
                    continue
                competitor_topology = competitor_payload["topology"]
                if basis == "subset":
                    competitor_records = competitor_topology["records"]
                else:
                    sensitivity = competitor_topology.get(record_key)
                    if not sensitivity:
                        continue
                    competitor_records = sensitivity["records"]
                competitor_by_id = {
                    int(record[identifier]): record
                    for record in competitor_records
                }
                paired_ids = sorted(
                    set(reference_by_id).intersection(competitor_by_id)
                )
                if not paired_ids:
                    continue
                for metric in metric_keys:
                    dire_values = np.asarray(
                        [
                            reference_by_id[pair_id][metric]
                            for pair_id in paired_ids
                        ],
                        dtype=np.float64,
                    )
                    competitor_values = np.asarray(
                        [
                            competitor_by_id[pair_id][metric]
                            for pair_id in paired_ids
                        ],
                        dtype=np.float64,
                    )
                    # All topology metrics are discrepancies: lower is better.
                    # competitor - DiRe is therefore positive when DiRe is
                    # better, zero for a tie, and negative when it is worse.
                    gaps = competitor_values - dire_values
                    gap_mean = float(gaps.mean())
                    gap_std = (
                        float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
                    )
                    competitor_mean = float(competitor_values.mean())
                    effect_rows.append(
                        {
                            "dataset": dataset,
                            "basis": basis,
                            "metric": metric,
                            "dire_method": "dire_auto",
                            "competitor": competitor,
                            "competitor_display": DISPLAY[competitor],
                            "pair_id_field": identifier,
                            "paired_record_count": int(len(gaps)),
                            "paired_ids": ",".join(
                                str(value) for value in paired_ids
                            ),
                            "dire_mean": float(dire_values.mean()),
                            "competitor_mean": competitor_mean,
                            "paired_gap_competitor_minus_dire_mean": gap_mean,
                            "paired_gap_std": gap_std,
                            "paired_gap_values": [
                                float(value) for value in gaps
                            ],
                            "relative_gap_vs_competitor": (
                                gap_mean / abs(competitor_mean)
                                if competitor_mean != 0
                                else None
                            ),
                            "descriptive_paired_standardized_gap": (
                                gap_mean / gap_std
                                if gap_std > 0
                                else None
                            ),
                        }
                    )

        reference_sizes = {
            int(record["sensitivity_subset_size"]): record
            for record in reference["subset_size_sensitivity"]["records"]
        }
        for competitor, competitor_payload in methods.items():
            if competitor == "dire_auto":
                continue
            competitor_sizes = {
                int(record["sensitivity_subset_size"]): record
                for record in competitor_payload["topology"][
                    "subset_size_sensitivity"
                ]["records"]
            }
            for m in sorted(set(reference_sizes).intersection(competitor_sizes)):
                for metric in metric_keys:
                    dire_value = float(reference_sizes[m][metric])
                    competitor_value = float(competitor_sizes[m][metric])
                    gap = competitor_value - dire_value
                    size_rows.append(
                        {
                            "dataset": dataset,
                            "m": m,
                            "metric": metric,
                            "dire_method": "dire_auto",
                            "competitor": competitor,
                            "competitor_display": DISPLAY[competitor],
                            "dire_value": dire_value,
                            "competitor_value": competitor_value,
                            "gap_competitor_minus_dire": gap,
                            "winner": (
                                "dire_auto"
                                if gap > 0
                                else competitor
                                if gap < 0
                                else "tie"
                            ),
                        }
                    )

    if not effect_rows:
        return []
    with (output_root / "revision3-topology-paired-effects.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(effect_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(effect_rows)
    with (output_root / "revision3-topology-size-reversals.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(size_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(size_rows)
    write_json(
        output_root / "revision3-topology-effect-audit.json",
        {
            "schema_version": 2,
            "topology_design": topology_design_by_dataset,
            "gap_definition": (
                "competitor discrepancy minus production-auto DiRe "
                "discrepancy; positive favors DiRe because lower is better"
            ),
            "inference_policy": (
                "The topology row-index subsets and full-layout seeds are "
                "separate descriptive robustness audits. The suite "
                "reports every paired gap, their mean, relative gap, and "
                "observed standard deviation; it does not claim conventional "
                "statistical significance or calculate p-values."
            ),
            "paired_effects": effect_rows,
            "subset_size_records": size_rows,
        },
    )

    table_rows = [
        row
        for row in effect_rows
        if row["dataset"] == "arxiv"
        and row["basis"] == "subset"
        and (
            (
                row["metric"] in ("dtw_beta0", "dtw_beta1")
                and row["competitor"] in (
                    "pca2",
                    "dire",
                    "cuml_umap",
                    "cuml_tsne",
                )
            )
            or (
                row["metric"] in ("q99_dtw_beta0", "q99_dtw_beta1")
                and row["competitor"] in ("cuml_umap", "cuml_tsne")
            )
        )
    ]
    subset_count = int(
        topology_design_by_dataset["arxiv"]["topology_subset_count"]
    )
    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        (
            r"Metric & Comparator & DiRe auto & Comparator & Paired gap "
            rf"$\pm$ SD & Relative gap & {subset_count} paired gaps \\"
        ),
        r"\midrule",
    ]
    for row in table_rows:
        beta = {
            "dtw_beta0": r"$\beta_0$ DTW",
            "dtw_beta1": r"$\beta_1$ DTW",
            "q99_dtw_beta0": r"$\beta_0$ Q99--DTW",
            "q99_dtw_beta1": r"$\beta_1$ Q99--DTW",
        }[row["metric"]]
        relative = row["relative_gap_vs_competitor"]
        relative_text = (
            "--" if relative is None else f"{100.0 * relative:.1f}\\%"
        )
        formatted_values = [
            f"{float(value):+.4f}" for value in row["paired_gap_values"]
        ]
        midpoint = (len(formatted_values) + 1) // 2
        paired_values = (
            r"\shortstack[l]{"
            + ", ".join(formatted_values[:midpoint])
            + r"\\"
            + ", ".join(formatted_values[midpoint:])
            + "}"
        )
        lines.append(
            f"{beta} & {latex_escape(row['competitor_display'])} & "
            f"{row['dire_mean']:.4f} & {row['competitor_mean']:.4f} & "
            f"{row['paired_gap_competitor_minus_dire_mean']:.4f}"
            f"$\\pm${row['paired_gap_std']:.4f} & "
            f"{relative_text} & "
            f"{paired_values} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-arxiv-topology-paired-effects.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    plot_competitors = ("pca2", "dire", "cuml_umap", "cuml_tsne")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 7.8),
        constrained_layout=True,
    )
    for row_index, dataset in enumerate(("tenx", "arxiv")):
        for column, metric in enumerate(("dtw_beta0", "dtw_beta1")):
            ax = axes[row_index, column]
            selected = [
                row
                for row in effect_rows
                if row["dataset"] == dataset
                and row["basis"] == "subset"
                and row["metric"] == metric
                and row["competitor"] in plot_competitors
            ]
            selected.sort(
                key=lambda row: plot_competitors.index(row["competitor"])
            )
            positions = np.arange(len(selected))
            ax.barh(
                positions,
                [
                    row["paired_gap_competitor_minus_dire_mean"]
                    for row in selected
                ],
                xerr=[row["paired_gap_std"] for row in selected],
                color=[COLORS[row["competitor"]] for row in selected],
                alpha=0.88,
                capsize=3,
            )
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_yticks(
                positions,
                [DISPLAY[row["competitor"]] for row in selected],
            )
            ax.set_xlabel("paired gap: comparator minus DiRe auto")
            dataset_name = (
                "10x mouse brain" if dataset == "tenx" else "arXiv corpus"
            )
            dimension = "H0" if metric == "dtw_beta0" else "H1"
            ax.set_title(f"{dataset_name}: {dimension} Betti-DTW")
            ax.grid(axis="x", alpha=0.25)
            ax.text(
                0.99,
                0.03,
                "positive favors DiRe",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#555555",
            )
    fig.suptitle(
        f"Topology effect magnitudes over {subset_count} paired row-index subsets "
        "(bars: mean; whiskers: observed paired SD; no p-values)",
        fontsize=11,
    )
    path = output_root / "revision3-topology-paired-effects.pdf"
    save_deterministic_pdf(fig, path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)
    return effect_rows


def make_tenx_marker_table(bundle_root: Path, output_root: Path) -> None:
    marker_payload = read_json(
        bundle_root
        / "prepared"
        / "tenx_mouse_brain_kmeans20_top_markers.json"
    )
    by_cluster: dict[int, list[dict]] = {}
    for record in marker_payload["records"]:
        by_cluster.setdefault(int(record["cluster"]), []).append(record)
    rows = []
    for cluster, records in sorted(by_cluster.items()):
        records.sort(key=lambda record: int(record["rank"]))
        rows.append(
            {
                "cluster": cluster,
                "cells": int(records[0]["cluster_size"]),
                "markers": ", ".join(record["gene_name"] for record in records[:5]),
            }
        )
    with (output_root / "revision3-tenx-markers.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cluster", "cells", "markers"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        r"\begin{longtable}{@{}rrp{0.62\linewidth}@{}}",
        (
            r"\caption{Released 10x cluster sizes and the five strongest "
            r"positive differential-expression markers used to annotate the "
            r"million-cell visual audit.}"
            r"\label{tab:revision3-tenx-markers}\\"
        ),
        r"\toprule",
        r"Official cluster & Cells & Top released positive markers \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            f"{row['cluster']} & {row['cells']:,} & "
            f"{latex_escape(row['markers'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (output_root / "revision3-tenx-markers.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def make_tenx_adjacency_audit(bundle_root: Path, output_root: Path) -> None:
    evaluation = read_json(
        bundle_root / "evaluation" / "tenx" / "evaluation.json"
    )
    marker_payload = read_json(
        bundle_root
        / "prepared"
        / "tenx_mouse_brain_kmeans20_top_markers.json"
    )
    markers: dict[int, list[str]] = {}
    for record in marker_payload["records"]:
        markers.setdefault(int(record["cluster"]), []).append(
            str(record["gene_name"])
        )
    code_to_cluster = {
        int(code): int(str(name).rsplit(" ", 1)[-1])
        for code, name in evaluation["label_policy"]["code_to_name"].items()
    }
    labels = np.load(
        bundle_root / "evaluation" / "tenx" / "evaluation_labels.npy",
        mmap_mode="r",
    )
    cluster_counts = np.bincount(
        np.asarray(labels, dtype=np.int64),
        minlength=len(code_to_cluster),
    )
    bundled_reference_centroids = np.asarray(
        evaluation["reference"]["label_centroids"]["values"],
        dtype=np.float64,
    )

    def nearest_centroid_codes(centroids: np.ndarray, k: int) -> np.ndarray:
        distances = np.linalg.norm(
            centroids[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        np.fill_diagonal(distances, np.inf)
        return np.argsort(distances, axis=1)[:, :k]

    def recompute_embedding_centroids(method: str) -> np.ndarray:
        embedding_path = full_embedding_path(
            bundle_root,
            "tenx",
            method,
        )
        points = np.load(embedding_path, mmap_mode="r")
        if len(points) != len(labels):
            raise RuntimeError(
                f"10x {method} embedding/label row mismatch: "
                f"{len(points)} != {len(labels)}"
            )
        centroids = np.empty(
            (len(code_to_cluster), points.shape[1]),
            dtype=np.float64,
        )
        for dimension in range(points.shape[1]):
            centroids[:, dimension] = np.bincount(
                labels,
                weights=np.asarray(points[:, dimension], dtype=np.float64),
                minlength=len(code_to_cluster),
            )
        centroids /= cluster_counts[:, None]
        return centroids

    pair_rows: list[dict] = []
    source_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    summaries: list[dict] = []
    canonical_reference: np.ndarray | None = None
    recomputation_checks: dict[str, bool] = {}
    reference_recomputation_checks: dict[str, bool] = {}
    for method in QUALITY_METHODS["tenx"]:
        metrics = evaluation["methods"].get(method)
        if metrics is None:
            continue
        adjacency = metrics["global"].get("centroid_adjacencies")
        if adjacency is None:
            continue
        reference = np.asarray(
            adjacency["reference_nearest_codes"],
            dtype=np.int16,
        )
        embedding = np.asarray(
            adjacency["embedding_nearest_codes"],
            dtype=np.int16,
        )
        if canonical_reference is None:
            canonical_reference = reference
        elif not np.array_equal(reference, canonical_reference):
            raise RuntimeError(
                "10x reference nearest-centroid lists differ by method"
            )
        reference_recomputation_checks[method] = bool(
            np.array_equal(
                reference,
                nearest_centroid_codes(
                    bundled_reference_centroids,
                    reference.shape[1],
                ),
            )
        )
        if not reference_recomputation_checks[method]:
            raise RuntimeError(
                f"10x {method} stored input-space centroid adjacency does not "
                "match the bundled full-data input centroids"
            )
        recomputed_centroids = recompute_embedding_centroids(method)
        recomputed_embedding = nearest_centroid_codes(
            recomputed_centroids,
            embedding.shape[1],
        )
        recomputation_checks[method] = bool(
            np.array_equal(embedding, recomputed_embedding)
        )
        if not recomputation_checks[method]:
            raise RuntimeError(
                f"10x {method} stored centroid adjacency does not match "
                "an independent recomputation from the bundled full layout"
            )

        source_recall = np.empty(len(reference), dtype=np.float64)
        preserved_count = 0
        for source_code in range(len(reference)):
            reference_set = set(int(value) for value in reference[source_code])
            embedding_set = set(int(value) for value in embedding[source_code])
            source_preserved = len(reference_set.intersection(embedding_set))
            source_recall[source_code] = source_preserved / reference.shape[1]
            source_rows.append(
                {
                    "method": method,
                    "display": DISPLAY[method],
                    "source_cluster": code_to_cluster[source_code],
                    "source_cells": int(cluster_counts[source_code]),
                    "input_nearest_clusters": ",".join(
                        str(code_to_cluster[int(value)])
                        for value in reference[source_code]
                    ),
                    "embedding_nearest_clusters": ",".join(
                        str(code_to_cluster[int(value)])
                        for value in embedding[source_code]
                    ),
                    "preserved": source_preserved,
                    "recall": float(source_recall[source_code]),
                }
            )
            for target_code in sorted(reference_set | embedding_set):
                if target_code in reference_set and target_code in embedding_set:
                    status = "preserved"
                    preserved_count += 1
                elif target_code in reference_set:
                    status = "lost"
                else:
                    status = "introduced"
                source_cluster = code_to_cluster[source_code]
                pair_rows.append(
                    {
                        "method": method,
                        "display": DISPLAY[method],
                        "source_cluster": source_cluster,
                        "source_top_markers": ", ".join(
                            markers.get(source_cluster, [])[:5]
                        ),
                        "target_cluster": code_to_cluster[target_code],
                        "status": status,
                    }
                )
        total = int(reference.size)
        weighted_recall = float(
            np.average(source_recall, weights=cluster_counts)
        )
        summaries.append(
            {
                "method": method,
                "display": DISPLAY[method],
                "input_directed_edges": total,
                "preserved_directed_edges": preserved_count,
                "introduced_directed_edges": total - preserved_count,
                "mean_recall": preserved_count / total,
                "source_cell_weighted_recall": weighted_recall,
            }
        )
        maximum_sensitivity_k = min(10, len(code_to_cluster) - 1)
        sensitivity_reference = nearest_centroid_codes(
            bundled_reference_centroids,
            maximum_sensitivity_k,
        )
        sensitivity_embedding = nearest_centroid_codes(
            recomputed_centroids,
            maximum_sensitivity_k,
        )
        for k in range(1, maximum_sensitivity_k + 1):
            k_recall = np.asarray(
                [
                    len(
                        set(
                            int(value)
                            for value in sensitivity_reference[
                                source_code,
                                :k,
                            ]
                        )
                        .intersection(
                            int(value)
                            for value in sensitivity_embedding[
                                source_code,
                                :k,
                            ]
                        )
                    )
                    / k
                    for source_code in range(len(reference))
                ],
                dtype=np.float64,
            )
            for minimum_source_cells in (0, 1_000, 10_000):
                included = cluster_counts >= minimum_source_cells
                sensitivity_rows.append(
                    {
                        "method": method,
                        "display": DISPLAY[method],
                        "k": k,
                        "minimum_source_cells": minimum_source_cells,
                        "included_source_clusters": int(included.sum()),
                        "included_source_cells": int(
                            cluster_counts[included].sum()
                        ),
                        "equal_cluster_weight_recall": float(
                            k_recall[included].mean()
                        ),
                        "source_cell_weighted_recall": float(
                            np.average(
                                k_recall[included],
                                weights=cluster_counts[included],
                            )
                        ),
                    }
                )

    pair_fields = (
        "method",
        "display",
        "source_cluster",
        "source_top_markers",
        "target_cluster",
        "status",
    )
    with (output_root / "revision3-tenx-adjacency-pairs.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=pair_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(pair_rows)
    with (output_root / "revision3-tenx-adjacency-summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summaries[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    with (output_root / "revision3-tenx-adjacency-by-source.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(source_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(source_rows)
    with (output_root / "revision3-tenx-adjacency-sensitivity.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(sensitivity_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    write_json(
        output_root / "revision3-tenx-adjacency-audit.json",
        {
            "interpretation": (
                "Directed nearest-centroid relationships in each layout "
                "compared with the same relationships in the common 20-D "
                "Cell Ranger PCA input. Marker genes are released 10x "
                "differential-expression annotations, not inferred from a "
                "layout. The headline table retains the declared k=3 metric; "
                "the sensitivity records independently recompute full-layout "
                "centroids for k=1 through min(10, number of clusters minus 1)."
            ),
            "k": int(
                evaluation["methods"][summaries[0]["method"]]["global"][
                    "centroid_adjacency_recall"
                ]["k"]
            ),
            "code_to_official_cluster": code_to_cluster,
            "cluster_cell_counts": {
                str(code_to_cluster[code]): int(cluster_counts[code])
                for code in range(len(cluster_counts))
            },
            "stored_embedding_adjacency_matches_full_layout_recomputation": (
                recomputation_checks
            ),
            "stored_reference_adjacency_matches_bundled_input_centroids": (
                reference_recomputation_checks
            ),
            "methods": summaries,
            "sensitivity": sensitivity_rows,
        },
    )
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (
            r"Method & Input directed edges & Preserved & Introduced & "
            r"Equal-cluster recall & Cell-weighted recall \\"
        ),
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{latex_escape(row['display'])} & "
            f"{row['input_directed_edges']} & "
            f"{row['preserved_directed_edges']} & "
            f"{row['introduced_directed_edges']} & "
            f"{row['mean_recall']:.3f} & "
            f"{row['source_cell_weighted_recall']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-tenx-adjacency-summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def ablation_value(method: dict, key: str) -> float | None:
    value = method.get("aggregate", {}).get(key, {}).get("mean")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def make_initialization_ablation(
    bundle_root: Path,
    output_root: Path,
) -> list[dict]:
    source = bundle_root / "ablation" / "all_results.json"
    if not source.exists():
        return []
    payload = read_json(source)
    rows: list[dict] = []
    pairings = (
        ("PCA", "dire_pca_init", "dire"),
        ("Spectral", "dire_spectral_init", "dire_spectral"),
    )
    for dataset in ABLATION_DATASET_DISPLAY:
        methods = payload.get(dataset, {}).get("methods", {})
        for initialization, initial_key, final_key in pairings:
            for stage, method_key in (("initial", initial_key), ("refined", final_key)):
                method = methods.get(method_key, {})
                beta0 = ablation_value(method, "persistence-dim-0")
                beta1 = ablation_value(method, "persistence-dim-1")
                rows.append(
                    {
                        "dataset": dataset,
                        "initialization": initialization,
                        "stage": stage,
                        "method": method_key,
                        "stress": ablation_value(method, "stress"),
                        "neighborhood": ablation_value(method, "neighborhood"),
                        "context": ablation_value(method, "context"),
                        "dtw_beta0": beta0,
                        "dtw_beta1": beta1,
                        "dtw_total": (
                            beta0 + beta1
                            if beta0 is not None and beta1 is not None
                            else None
                        ),
                    }
                )

    if not rows:
        return []
    fieldnames = list(rows[0])
    with (output_root / "revision3-initialization-ablation.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    lookup = {
        (row["dataset"], row["initialization"], row["stage"]): row
        for row in rows
    }
    metrics = (
        ("neighborhood", "Neighborhood preservation", True),
        ("context", "Balanced context score", True),
        ("dtw_total", r"Topology discrepancy ($\beta_0+\beta_1$)", False),
    )
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(12.1, 7.0), constrained_layout=True)
    for row_index, (initialization, _initial_key, _final_key) in enumerate(pairings):
        for column_index, (metric, title, higher_is_better) in enumerate(metrics):
            ax = axes[row_index, column_index]
            plotted = False
            for dataset_index, dataset in enumerate(ABLATION_DATASET_DISPLAY):
                initial = lookup[(dataset, initialization, "initial")][metric]
                refined = lookup[(dataset, initialization, "refined")][metric]
                if initial is None or refined is None:
                    continue
                ax.plot(
                    (0, 1),
                    (initial, refined),
                    marker="o",
                    linewidth=1.4,
                    markersize=4,
                    color=colors(dataset_index),
                    label=ABLATION_DATASET_DISPLAY[dataset],
                )
                plotted = True
            ax.set_xticks((0, 1), ("initial", "after 128 iterations"))
            ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(f"{initialization} initialization")
            ax.grid(axis="y", alpha=0.22)
            direction = "higher is better" if higher_is_better else "lower is better"
            ax.text(
                0.98,
                0.03,
                direction,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=7,
                color="#555555",
            )
            if not plotted:
                ax.text(
                    0.5,
                    0.5,
                    "not available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=6,
            frameon=False,
        )
    fig.suptitle(
        "Initialization ablation: the starting layout and DiRe refinement are "
        "reported separately",
        fontsize=12,
    )
    path = output_root / "revision3-initialization-ablation.pdf"
    save_deterministic_pdf(fig, path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    lines = [
        r"\begin{tabular}{@{}lllrrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & Initialization & Stage & Stress & Neighborhood & "
            r"Context & DTW $\beta_0$ & DTW $\beta_1$ \\"
        ),
        r"\midrule",
    ]
    previous_dataset = None
    for row in rows:
        if previous_dataset is not None and row["dataset"] != previous_dataset:
            lines.append(r"\addlinespace")
        lines.append(
            f"{latex_escape(ABLATION_DATASET_DISPLAY[row['dataset']])} & "
            f"{row['initialization']} & {row['stage']} & "
            f"{format_float(row['stress'])} & "
            f"{format_float(row['neighborhood'])} & "
            f"{format_float(row['context'])} & "
            f"{format_float(row['dtw_beta0'])} & "
            f"{format_float(row['dtw_beta1'])} \\\\"
        )
        previous_dataset = row["dataset"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "revision3-initialization-ablation.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return rows


def best_method(
    rows: list[dict],
    dataset: str,
    key: str,
    higher: bool,
    methods: tuple[str, ...] | None = None,
    relative_tolerance: float = 0.05,
) -> dict | None:
    """Return the observed optimum and its predeclared comparability band.

    Exact ordering is retained in ``method`` and ``key``.  A difference of at
    most ``relative_tolerance`` from that optimum is not promoted into a
    categorical method lead.  The relative band is used only when all
    candidate values and the optimum are nonnegative; signed correlations
    retain exact equality semantics because a ratio around zero is undefined.
    """

    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be nonnegative")
    candidates = [
        row
        for row in rows
        if row["dataset"] == dataset
        and (methods is None or row["method"] in methods)
        and row[key] is not None
        and math.isfinite(row[key])
    ]
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: row[key], reverse=higher)
    optimum = float(ordered[0][key])
    exact_optima = [
        row
        for row in ordered
        if math.isclose(
            float(row[key]),
            optimum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ]
    use_relative_band = optimum > 0.0 and all(
        float(row[key]) >= 0.0 for row in candidates
    )
    if use_relative_band:
        if higher:
            threshold = optimum * (1.0 - relative_tolerance)
            comparable = [
                row
                for row in ordered
                if float(row[key]) >= threshold
                or math.isclose(
                    float(row[key]),
                    threshold,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]
        else:
            threshold = optimum * (1.0 + relative_tolerance)
            comparable = [
                row
                for row in ordered
                if float(row[key]) <= threshold
                or math.isclose(
                    float(row[key]),
                    threshold,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]
    else:
        comparable = exact_optima
    result = dict(ordered[0])
    result["relative_comparability_tolerance"] = relative_tolerance
    result["exact_optimum_methods"] = [
        row["method"] for row in exact_optima
    ]
    result["exact_optimum_displays"] = [
        row.get("display", row["method"]) for row in exact_optima
    ]
    result["practically_comparable_methods"] = [
        row["method"] for row in comparable
    ]
    result["practically_comparable_displays"] = [
        row.get("display", row["method"]) for row in comparable
    ]
    result["unique_exact_optimum"] = len(exact_optima) == 1
    result["unique_within_tolerance"] = len(comparable) == 1
    result["all_candidates_exactly_equal"] = (
        len(exact_optima) == len(candidates)
    )
    result["all_candidates_within_tolerance"] = (
        len(comparable) == len(candidates)
    )
    return result


def append_topology_effect_macros(
    macros: list[str],
    effect_rows: list[dict],
) -> None:
    """Append descriptive paired-effect macros used by the paper and response."""
    dataset_prefix = {"tenx": "Tenx", "arxiv": "Arxiv"}
    competitor_prefix = {
        "pca2": "Pca",
        "dire": "DireIvfFlat",
        "cuml_umap": "Umap",
        "cuml_tsne": "Tsne",
    }
    metric_prefix = {
        "dtw_beta0": "BetaZeroDtw",
        "dtw_beta1": "BetaOneDtw",
        "q99_dtw_beta0": "BetaZeroQNinetyNineDtw",
        "q99_dtw_beta1": "BetaOneQNinetyNineDtw",
    }
    selected = [
        row
        for row in effect_rows
        if row["basis"] == "subset"
        and row["dataset"] in dataset_prefix
        and row["competitor"] in competitor_prefix
        and row["metric"] in metric_prefix
    ]
    selected.sort(
        key=lambda row: (
            row["dataset"],
            row["competitor"],
            row["metric"],
        )
    )
    for row in selected:
        stem = (
            f"{dataset_prefix[row['dataset']]}"
            f"{competitor_prefix[row['competitor']]}MinusDire"
            f"{metric_prefix[row['metric']]}"
        )
        gap = float(row["paired_gap_competitor_minus_dire_mean"])
        gap_std = float(row["paired_gap_std"])
        gap_in_sd = gap / gap_std if gap_std > 0 else None
        macros.extend(
            [
                f"\\newcommand{{\\{stem}Gap}}{{{gap:.4f}}}",
                f"\\newcommand{{\\{stem}GapSd}}{{{gap_std:.4f}}}",
                f"\\newcommand{{\\{stem}GapInObservedSd}}"
                f"{{{'--' if gap_in_sd is None else f'{gap_in_sd:.2f}'}}}",
            ]
        )


def append_small_atlas_effect_macros(
    macros: list[str],
    effect_rows: list[dict],
) -> None:
    """Append CSV-derived small-suite comparison counts."""

    for comparator, comparator_name in (
        ("cuml_umap", "Umap"),
        ("cuml_tsne", "Tsne"),
    ):
        comparison_rows = [
            row
            for row in effect_rows
            if row["comparator"] == comparator
        ]
        dire_lower_rows = [
            row
            for row in comparison_rows
            if float(row["paired_gap_comparator_minus_dire_mean"]) > 0.0
        ]
        comparator_lower_rows = [
            row
            for row in comparison_rows
            if float(row["paired_gap_comparator_minus_dire_mean"]) < 0.0
        ]
        clearly_dire_lower_rows = [
            row
            for row in comparison_rows
            if float(row["paired_mean_95pct_low"]) > 0.0
        ]
        more_than_five_percent_rows = [
            row
            for row in dire_lower_rows
            if float(row["relative_gap"]) > 0.05
        ]
        macros.extend(
            [
                f"\\newcommand{{\\SmallAtlas{comparator_name}ComparisonCount}}"
                f"{{{len(comparison_rows)}}}",
                f"\\newcommand{{\\SmallAtlasDireLowerVs{comparator_name}Count}}"
                f"{{{len(dire_lower_rows)}}}",
                f"\\newcommand{{\\SmallAtlas{comparator_name}LowerVsDireCount}}"
                f"{{{len(comparator_lower_rows)}}}",
                (
                    f"\\newcommand{{\\SmallAtlasDireIntervalLowerVs"
                    f"{comparator_name}Count}}"
                    f"{{{len(clearly_dire_lower_rows)}}}"
                ),
                (
                    f"\\newcommand{{\\SmallAtlasDireFivePctLowerVs"
                    f"{comparator_name}Count}}"
                    f"{{{len(more_than_five_percent_rows)}}}"
                ),
            ]
        )


def make_macros_and_summary(
    bundle_root: Path,
    output_root: Path,
    rows: list[dict],
) -> None:
    row_lookup = {
        (row["dataset"], row["method"]): row
        for row in rows
    }
    macros = [
        r"% Generated by scripts/render_revision3_artifacts.py; do not edit.",
        r"\newcommand{\TenxN}{1,306,127}",
        r"\newcommand{\ArxivN}{723,457}",
    ]
    tenx_evaluation = read_json(
        bundle_root / "evaluation" / "tenx" / "evaluation.json"
    )
    macros.append(
        "\\newcommand{\\TopologySubsetCount}"
        f"{{{int(tenx_evaluation['topology_design']['topology_subset_count'])}}}"
    )
    topology_audit = matching_topology_audit(bundle_root, output_root)
    if topology_audit is not None:
        audit_records = list(
            iter_scalar_topology_audit_records(
                topology_audit
            )
        )
        maximum_delta = max(
            (
                float(record["absolute_delta"])
                for record in audit_records
            ),
            default=0.0,
        )
        macros.extend(
            [
                "\\newcommand{\\TopologyAuditComparisonCount}"
                f"{{{len(audit_records):,}}}",
                "\\newcommand{\\TopologyAuditMaximumAbsoluteDelta}"
                f"{{{maximum_delta:.3g}}}",
            ]
        )
    for dataset in ("tenx", "arxiv"):
        evaluation = read_json(
            bundle_root / "evaluation" / dataset / "evaluation.json"
        )
        audit = evaluation.get("backend_policy_audit")
        if not audit:
            continue
        prefix = "Tenx" if dataset == "tenx" else "Arxiv"
        auto = audit["production_auto"]
        control = audit["forced_ivf_flat_control"]
        stages = auto["stage_timings"]
        fallback_calls = sum(auto["chunked_force_fallback_calls"])
        macros.extend(
            [
                f"\\newcommand{{\\{prefix}BackendFitReplicates}}"
                f"{{{audit['R_fit_total']}}}",
                f"\\newcommand{{\\{prefix}BackendSteadyReplicates}}"
                f"{{{audit['R_steady']}}}",
                f"\\newcommand{{\\{prefix}BackendGraphQuerySample}}"
                f"{{{audit['m_graph_queries']:,}}}",
                f"\\newcommand{{\\{prefix}DireAutoKnnGraphSeconds}}"
                f"{{{format_seconds(stages['knn_graph']['steady_mean_sec'])}}}",
                f"\\newcommand{{\\{prefix}DireAutoProfileSteadySeconds}}"
                f"{{{format_seconds(auto['steady_total_mean_sec'])}}}",
                f"\\newcommand{{\\{prefix}DireIvfFlatProfileSteadySeconds}}"
                f"{{{format_seconds(control['steady_total_mean_sec'])}}}",
                f"\\newcommand{{\\{prefix}DireAutoInitializationSeconds}}"
                f"{{{format_seconds(stages['initialization']['steady_mean_sec'])}}}",
                f"\\newcommand{{\\{prefix}DireAutoLayoutSeconds}}"
                f"{{{format_seconds(stages['layout']['steady_mean_sec'])}}}",
                f"\\newcommand{{\\{prefix}DireAutoChunkedFallbackCalls}}"
                f"{{{fallback_calls}}}",
                f"\\newcommand{{\\{prefix}DireAutoSpeedupVsIvfFlat}}"
                f"{{{audit['steady_speedup_control_over_auto']:.2f}}}",
                f"\\newcommand{{\\{prefix}DireBackendGraphOverlap}}"
                f"{{{audit['knn_graph_overlap']['mean']:.3f}}}",
            ]
        )
    macro_fields = {
        "steady_sec": ("SteadySeconds", format_seconds),
        "gpu_gib": ("PeakGiB", lambda value: format_float(value, 2)),
        "knn_overlap": ("KnnOverlap", lambda value: format_float(value, 3)),
        "centroid_spearman": (
            "CentroidRho",
            lambda value: format_float(value, 3),
        ),
        "centroid_adjacency_recall": (
            "CentroidAdjacency",
            lambda value: format_float(value, 3),
        ),
        "context_balanced_accuracy": (
            "ContextAccuracy",
            lambda value: format_float(value, 3),
        ),
        "bottleneck_beta0": (
            "BetaZeroBottleneck",
            lambda value: format_float(value, 4),
        ),
        "bottleneck_beta1": (
            "BetaOneBottleneck",
            lambda value: format_float(value, 4),
        ),
        "dtw_beta0": ("BetaZeroDtw", lambda value: format_float(value, 3)),
        "dtw_beta1": ("BetaOneDtw", lambda value: format_float(value, 3)),
        "q99_bottleneck_beta0": (
            "BetaZeroQNinetyNineBottleneck",
            lambda value: format_float(value, 4),
        ),
        "q99_bottleneck_beta1": (
            "BetaOneQNinetyNineBottleneck",
            lambda value: format_float(value, 4),
        ),
        "q99_dtw_beta0": (
            "BetaZeroQNinetyNineDtw",
            lambda value: format_float(value, 3),
        ),
        "q99_dtw_beta1": (
            "BetaOneQNinetyNineDtw",
            lambda value: format_float(value, 3),
        ),
    }
    dataset_prefix = {"tenx": "Tenx", "arxiv": "Arxiv"}
    method_prefix = {
        "dire_auto": "DireAuto",
        "dire": "DireIvfFlat",
        "dire_spectral": "DireSpectral",
        "dire_topology": "DireTopology",
        "cuml_umap": "CumlUmap",
        "cuml_tsne": "CumlTsne",
        "pca2": "Pca",
        "cellranger_tsne": "CellrangerTsne",
    }
    for (dataset, method), row in sorted(row_lookup.items()):
        for field, (suffix, formatter) in macro_fields.items():
            macros.append(
                f"\\newcommand{{\\{dataset_prefix[dataset]}{method_prefix[method]}{suffix}}}"
                f"{{{formatter(row[field])}}}"
            )
    adjacency_summary_path = (
        output_root / "revision3-tenx-adjacency-summary.csv"
    )
    adjacency_sensitivity_path = (
        output_root / "revision3-tenx-adjacency-sensitivity.csv"
    )
    if adjacency_summary_path.is_file():
        with adjacency_summary_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            adjacency_rows = {
                row["method"]: row for row in csv.DictReader(handle)
            }
        for method, prefix in (
            ("dire_auto", "TenxDireAuto"),
            ("dire_spectral", "TenxDireSpectral"),
            ("dire_topology", "TenxDireTopology"),
            ("cuml_umap", "TenxCumlUmap"),
        ):
            row = adjacency_rows[method]
            macros.extend(
                [
                    f"\\newcommand{{\\{prefix}AdjacencyPreserved}}"
                    f"{{{int(row['preserved_directed_edges'])}}}",
                    f"\\newcommand{{\\{prefix}AdjacencyIntroduced}}"
                    f"{{{int(row['introduced_directed_edges'])}}}",
                    f"\\newcommand{{\\{prefix}AdjacencyCellWeighted}}"
                    f"{{{float(row['source_cell_weighted_recall']):.3f}}}",
                ]
            )
    if adjacency_sensitivity_path.is_file():
        with adjacency_sensitivity_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            sensitivity_rows = {
                (
                    row["method"],
                    int(row["k"]),
                    int(row["minimum_source_cells"]),
                ): row
                for row in csv.DictReader(handle)
            }
        for method, prefix in (
            ("dire_auto", "TenxDireAuto"),
            ("cuml_umap", "TenxCumlUmap"),
        ):
            filtered = sensitivity_rows[(method, 3, 1_000)]
            macros.append(
                f"\\newcommand{{\\{prefix}AdjacencyFilteredEqual}}"
                f"{{{float(filtered['equal_cluster_weight_recall']):.3f}}}"
            )
            for k, number_word in ((1, "One"), (2, "Two"), (3, "Three")):
                row = sensitivity_rows[(method, k, 0)]
                macros.append(
                    f"\\newcommand{{\\{prefix}AdjacencyK{number_word}}}"
                    f"{{{float(row['equal_cluster_weight_recall']):.3f}}}"
                )
    small_effect_path = (
        output_root / "revision3-small-atlas-topology-paired-effects.csv"
    )
    if small_effect_path.is_file():
        with small_effect_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            small_effect_rows = list(csv.DictReader(handle))
        append_small_atlas_effect_macros(macros, small_effect_rows)
    arxiv_dire = row_lookup[("arxiv", "dire_auto")]
    for field, metric_name in (
        ("bottleneck_beta0", "BetaZeroBottleneck"),
        ("bottleneck_beta1", "BetaOneBottleneck"),
        ("dtw_beta0", "BetaZeroDtw"),
        ("dtw_beta1", "BetaOneDtw"),
    ):
        for baseline, baseline_name in (
            ("cuml_umap", "Umap"),
            ("cuml_tsne", "Tsne"),
        ):
            baseline_value = row_lookup[("arxiv", baseline)][field]
            reduction = 100.0 * (
                baseline_value - arxiv_dire[field]
            ) / baseline_value
            macros.append(
                f"\\newcommand{{\\ArxivDire{metric_name}ReductionVs{baseline_name}Pct}}"
                f"{{{reduction:.1f}}}"
            )
    effect_path = output_root / "revision3-topology-paired-effects.csv"
    if effect_path.is_file():
        with effect_path.open(newline="", encoding="utf-8") as handle:
            append_topology_effect_macros(
                macros,
                list(csv.DictReader(handle)),
            )
    for line in macros:
        if not line.startswith(r"\newcommand"):
            continue
        match = re.match(r"\\newcommand\{\\([^}]+)\}", line)
        if match is None or re.fullmatch(r"[A-Za-z]+", match.group(1)) is None:
            raise ValueError(f"invalid LaTeX control-word name: {line}")
    (output_root / "revision3-results-macros.tex").write_text(
        "\n".join(macros) + "\n",
        encoding="utf-8",
    )

    metric_directions = (
        ("steady_sec", False),
        ("knn_overlap", True),
        ("centroid_spearman", True),
        ("centroid_adjacency_recall", True),
        ("context_balanced_accuracy", True),
        ("bottleneck_beta0", False),
        ("bottleneck_beta1", False),
        ("dtw_beta0", False),
        ("dtw_beta1", False),
    )
    groups = {
        "all_evaluated": None,
        "nonlinear_gpu_defaults": (
            "dire_auto",
            "cuml_umap",
            "cuml_tsne",
        ),
        "nonlinear_gpu_with_sensitivity": (
            "dire_auto",
            "dire",
            "dire_spectral",
            "dire_topology",
            "cuml_umap",
            "cuml_tsne",
        ),
    }
    summary = {
        dataset: {
            group: {
                key: best_method(rows, dataset, key, higher, methods)
                for key, higher in metric_directions
            }
            for group, methods in groups.items()
        }
        for dataset in ("tenx", "arxiv")
    }
    write_json(output_root / "revision3-results-summary.json", summary)

    lines = [
        "# Generated large-data result summary",
        "",
        (
            "This file is generated from the fetched JSON logs. It reports "
            "exact best-observed values together with a predeclared 5% "
            "relative comparability band; a small numerical difference is "
            "not converted into a categorical method lead."
        ),
        (
            "The reported Betti-DTW values use the suite's recorded FastDTW "
            "approximation; H0 bottleneck distances use the exact sorted "
            "zero-birth-bar specialization and H1 uses exact compiled GUDHI."
        ),
        "",
    ]
    for dataset in ("tenx", "arxiv"):
        lines.append(f"## {'10x mouse brain' if dataset == 'tenx' else 'arXiv corpus'}")
        lines.append("")
        for group, group_title in (
            ("all_evaluated", "All evaluated methods and references"),
            (
                "nonlinear_gpu_defaults",
                "Fresh production-policy nonlinear methods on the same GPU",
            ),
            (
                "nonlinear_gpu_with_sensitivity",
                "Same-GPU nonlinear methods including forced-index and predeclared DiRe sensitivity controls",
            ),
        ):
            lines.append(f"### {group_title}")
            lines.append("")
            for key, label in (
                ("steady_sec", "fastest steady-state runtime"),
                ("knn_overlap", "highest local kNN overlap"),
                ("centroid_spearman", "highest centroid-distance correlation"),
                ("centroid_adjacency_recall", "highest centroid-adjacency recall"),
                ("context_balanced_accuracy", "highest balanced context accuracy"),
                (
                    "bottleneck_beta0",
                    "lowest diameter-normalized beta-0 bottleneck distance",
                ),
                (
                    "bottleneck_beta1",
                    "lowest diameter-normalized beta-1 bottleneck distance",
                ),
                ("dtw_beta0", "lowest beta-0 DTW"),
                ("dtw_beta1", "lowest beta-1 DTW"),
            ):
                winner = summary[dataset][group][key]
                if winner is None:
                    lines.append(f"- {label}: unavailable")
                elif winner["all_candidates_exactly_equal"]:
                    lines.append(
                        f"- {label}: non-discriminating equality at "
                        f"{winner[key]:.6g} across "
                        + ", ".join(winner["exact_optimum_displays"])
                    )
                elif not winner["unique_within_tolerance"]:
                    lines.append(
                        f"- {label}: best observed value {winner[key]:.6g}; "
                        "practically comparable within 5%: "
                        + ", ".join(
                            winner["practically_comparable_displays"]
                        )
                    )
                else:
                    lines.append(
                        f"- {label}: {winner['display']} ({winner[key]:.6g})"
                    )
            lines.append("")
    (output_root / "revision3-results-summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("data/revision3/fetched/current"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated/revision3"),
    )
    parser.add_argument("--tenx-display-sample", type=int, default=120_000)
    parser.add_argument("--arxiv-display-sample", type=int, default=100_000)
    parser.add_argument(
        "--clean-output-root",
        action="store_true",
        help=(
            "replace the generated-artifact directory before rendering; a "
            "successful topology audit is retained only when its recorded "
            "source bundle matches --bundle-root"
        ),
    )
    args = parser.parse_args()
    prepare_output_root(
        args.bundle_root,
        args.output_root,
        clean=args.clean_output_root,
    )

    render_scaling(args.bundle_root, args.output_root)
    make_small_runtime(args.bundle_root, args.output_root)
    make_small_atlas_topology_effects(
        args.bundle_root,
        args.output_root,
    )
    render_tenx_layouts(
        args.bundle_root,
        args.output_root,
        args.tenx_display_sample,
    )
    render_arxiv_layouts(
        args.bundle_root,
        args.output_root,
        args.arxiv_display_sample,
    )
    render_dire_topology_sensitivity_layouts(
        args.bundle_root,
        args.output_root,
        args.tenx_display_sample,
        args.arxiv_display_sample,
    )
    render_quality(args.bundle_root, args.output_root)
    make_dataset_stats_table(args.bundle_root, args.output_root)
    rows = make_large_metrics_table(args.bundle_root, args.output_root)
    make_backend_policy_audit(args.bundle_root, args.output_root)
    make_topology_sensitivity_table(
        args.bundle_root,
        args.output_root,
        rows,
    )
    make_topology_subset_size_sensitivity(
        args.bundle_root,
        args.output_root,
    )
    make_topology_paired_effects(args.bundle_root, args.output_root)
    make_tenx_marker_table(args.bundle_root, args.output_root)
    make_tenx_adjacency_audit(args.bundle_root, args.output_root)
    make_initialization_ablation(args.bundle_root, args.output_root)
    make_macros_and_summary(args.bundle_root, args.output_root, rows)
    bundle_manifest_path = args.bundle_root / "bundle-manifest.json"
    bundle_manifest = read_json(bundle_manifest_path)
    topology_audit_path = args.output_root / "topology-audit.json"
    topology_audit = matching_topology_audit(
        args.bundle_root,
        args.output_root,
    )
    generated_records = artifact_records(
        args.output_root,
        exclude={"render_manifest.json"},
    )
    write_json(
        args.output_root / "render_manifest.json",
        {
            "schema_version": 2,
            "source_bundle": {
                "directory_name": args.bundle_root.name,
                "run_id": bundle_manifest.get("run_id"),
                "manifest_sha256": sha256_file(bundle_manifest_path),
            },
            "renderer": {
                "script": Path(__file__).name,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            **(
                {
                    "topology_audit": {
                        "file": topology_audit_path.name,
                        "sha256": sha256_file(topology_audit_path),
                        "status": topology_audit.get("status"),
                    }
                }
                if topology_audit is not None
                else {}
            ),
            "generated": [record["path"] for record in generated_records],
            "outputs": generated_records,
        },
    )


if __name__ == "__main__":
    main()
