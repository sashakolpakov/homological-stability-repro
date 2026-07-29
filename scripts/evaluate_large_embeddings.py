#!/usr/bin/env python3
"""Evaluate full-scale 10x and arXiv layouts with fixed row-index subsets.

The metrics deliberately separate three questions:

* local fidelity: high-dimensional kNN overlap on a common row-index subset;
* labelled/global organization: released-reference-label classification and
  inter-centroid distance/adjacency preservation;
* topology: diameter-normalized bottleneck distance, Betti-curve DTW, and H0
  persistence diagnostics on fixed stratified row-index subsets, with a separately
  labelled 99th-percentile pairwise-scale sensitivity analysis.

Visual cluster separation alone is not treated as evidence of fidelity: a
method can improve separability by inventing gaps.  The generated JSON states
the direction and interpretation of every metric.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DATASETS = ("tenx", "arxiv")
TOPOLOGY_Q = 0.99
TOPOLOGY_METRIC_KEYS = (
    "dtw_beta0",
    "dtw_beta1",
    "bottleneck_beta0",
    "bottleneck_beta1",
    "q99_dtw_beta0",
    "q99_dtw_beta1",
    "q99_bottleneck_beta0",
    "q99_bottleneck_beta1",
)
TOPOLOGY_SEED_METHODS = (
    "dire_auto",
    "dire",
    "dire_spectral",
    "dire_topology",
    "cuml_umap",
    "cuml_tsne",
)
METHOD_DISPLAY = {
    "dire_auto": "DiRe-RAPIDS (production auto policy)",
    "dire": "DiRe-RAPIDS (forced IVF-Flat control)",
    "dire_spectral": "DiRe-RAPIDS (spectral-init sensitivity)",
    "dire_topology": "DiRe-RAPIDS (topology preset)",
    "cuml_umap": "cuML UMAP",
    "cuml_tsne": "cuML t-SNE",
    "pca2": "cuML PCA (2D reference)",
    "cellranger_tsne": "Cell Ranger t-SNE (released reference)",
}
PREPARED = {
    "tenx": {
        "input": "tenx_mouse_brain_pca20.npy",
        "labels": "tenx_mouse_brain_kmeans20.npy",
        "manifest": "tenx_manifest.json",
        "official_layout": "tenx_mouse_brain_cellranger_tsne.npy",
        "full_size": 1_306_127,
    },
    "arxiv": {
        "input": "arxiv_bge_small_384_l2_centered.npy",
        "labels": "arxiv_primary_category_codes.npy",
        "manifest": "arxiv_manifest.json",
        "label_names": "arxiv_primary_category_names.json",
        "full_size": 723_457,
    },
}


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def remap_contiguous(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    unique = np.unique(labels)
    mapping = {int(value): index for index, value in enumerate(unique)}
    remapped = np.fromiter(
        (mapping[int(value)] for value in labels),
        dtype=np.int16,
        count=len(labels),
    )
    return remapped, mapping


def evaluation_labels(
    dataset: str,
    labels: np.ndarray,
    prepared_root: Path,
    arxiv_top_categories: int,
) -> tuple[np.ndarray, dict]:
    if dataset == "tenx":
        collapsed, original_to_code = remap_contiguous(labels)
        code_to_name = {
            str(code): f"10x k-means cluster {original}"
            for original, code in original_to_code.items()
        }
        return collapsed, {
            "policy": "all 20 official Cell Ranger k-means clusters",
            "code_to_name": code_to_name,
        }

    names_payload = read_json(prepared_root / PREPARED["arxiv"]["label_names"])
    code_to_category = {
        int(code): name
        for code, name in names_payload["code_to_primary_category"].items()
    }
    unique, counts = np.unique(labels, return_counts=True)
    ranked = sorted(
        zip(unique, counts),
        key=lambda item: (-int(item[1]), code_to_category[int(item[0])]),
    )
    kept_original = [int(code) for code, _count in ranked[:arxiv_top_categories]]
    kept_to_code = {original: index for index, original in enumerate(kept_original)}
    other_code = len(kept_original)
    collapsed = np.fromiter(
        (kept_to_code.get(int(value), other_code) for value in labels),
        dtype=np.int16,
        count=len(labels),
    )
    code_to_name = {
        str(code): code_to_category[original]
        for original, code in kept_to_code.items()
    }
    code_to_name[str(other_code)] = "other primary categories"
    return collapsed, {
        "policy": (
            f"top {arxiv_top_categories} primary arXiv categories by corpus "
            "frequency; all remaining categories collapsed to 'other'"
        ),
        "code_to_name": code_to_name,
        "kept_original_codes": kept_original,
    }


def stratified_indices(
    labels: np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    """Return an exact-size, deterministic, approximately stratified subset.

    ``sklearn.model_selection.train_test_split(..., stratify=...)`` rejects a
    class represented by a single row.  Such a singleton can legitimately
    occur after drawing the primary topology row-index subset, and it must
    remain in every sufficiently large nested subset rather than making the
    evaluator fail.  Integer quotas below use proportional allocation with a
    one-row lower bound whenever the requested subset can represent every
    class.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if size < 1:
        raise ValueError("sample size must be positive")
    if size >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    classes, inverse, counts = np.unique(
        labels,
        return_inverse=True,
        return_counts=True,
    )
    class_count = len(classes)
    ideal = counts.astype(np.float64) * (float(size) / len(labels))
    lower = (
        np.ones(class_count, dtype=np.int64)
        if size >= class_count
        else np.zeros(class_count, dtype=np.int64)
    )
    quotas = np.floor(ideal).astype(np.int64)
    quotas = np.minimum(counts, np.maximum(lower, quotas))

    # Fixed random tie priorities avoid a systematic label-order preference
    # while keeping every selected row reproducible from ``seed``.
    rng = np.random.default_rng(seed)
    tie_priority = rng.random(class_count)
    while int(quotas.sum()) < size:
        eligible = quotas < counts
        deficits = np.where(eligible, ideal - quotas, -np.inf)
        best = float(np.max(deficits))
        tied = np.flatnonzero(
            eligible & np.isclose(deficits, best, rtol=0.0, atol=1e-12)
        )
        chosen = int(tied[np.argmax(tie_priority[tied])])
        quotas[chosen] += 1
    while int(quotas.sum()) > size:
        eligible = quotas > lower
        excesses = np.where(eligible, quotas - ideal, -np.inf)
        best = float(np.max(excesses))
        tied = np.flatnonzero(
            eligible & np.isclose(excesses, best, rtol=0.0, atol=1e-12)
        )
        chosen = int(tied[np.argmax(tie_priority[tied])])
        quotas[chosen] -= 1

    selected = []
    for class_index, quota in enumerate(quotas):
        if quota == 0:
            continue
        candidates = np.flatnonzero(inverse == class_index)
        if quota == len(candidates):
            selected.append(candidates)
        else:
            selected.append(
                rng.choice(candidates, size=int(quota), replace=False)
            )
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def nested_stratified_indices(
    labels: np.ndarray,
    largest_indices: np.ndarray,
    sizes: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    requested = sorted(
        {
            min(int(size), len(largest_indices))
            for size in sizes
            if int(size) > 0
        },
        reverse=True,
    )
    if len(largest_indices) not in requested:
        requested.insert(0, len(largest_indices))
    current = np.asarray(largest_indices, dtype=np.int64)
    output = {len(current): current}
    for size in requested:
        if size == len(current):
            continue
        local = stratified_indices(
            labels[current],
            size,
            seed + size,
        )
        current = np.sort(current[local])
        output[size] = current
    return output


def remove_self_neighbors(indices: np.ndarray, expected_rows: np.ndarray, k: int) -> np.ndarray:
    result = np.empty((len(indices), k), dtype=np.int64)
    for row, candidates in enumerate(indices):
        keep = candidates[candidates != expected_rows[row]]
        if len(keep) < k:
            raise RuntimeError(
                f"kNN row {row} contains only {len(keep)} non-self neighbors"
            )
        result[row] = keep[:k]
    return result


def compute_knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    from cuml.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
    model.fit(points)
    _distances, indices = model.kneighbors(points)
    indices = to_numpy(indices).astype(np.int64, copy=False)
    rows = np.arange(len(points), dtype=np.int64)
    return remove_self_neighbors(indices, rows, k)


def neighbor_overlap(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"neighbor matrices have different shapes: {reference.shape}, {candidate.shape}"
        )
    k = reference.shape[1]
    overlaps = np.empty(len(reference), dtype=np.float32)
    for row in range(len(reference)):
        overlaps[row] = len(
            np.intersect1d(
                reference[row],
                candidate[row],
                assume_unique=False,
            )
        ) / k
    return {
        "mean": float(overlaps.mean()),
        "std": float(overlaps.std(ddof=1)),
        "median": float(np.median(overlaps)),
        "q05": float(np.quantile(overlaps, 0.05)),
        "q95": float(np.quantile(overlaps, 0.95)),
    }


def neighbor_label_agreement(neighbors: np.ndarray, labels: np.ndarray) -> dict:
    agreement = (labels[neighbors] == labels[:, None]).mean(axis=1)
    return {
        "mean": float(agreement.mean()),
        "std": float(agreement.std(ddof=1)),
        "median": float(np.median(agreement)),
    }


def centroids(points: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    # bincount is substantially faster than repeated boolean slicing for a
    # 723K x 384 memmap and keeps the peak working set small.
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("cannot compute centroid for an empty evaluation class")
    result = np.empty((n_classes, points.shape[1]), dtype=np.float64)
    for dimension in range(points.shape[1]):
        result[:, dimension] = np.bincount(
            labels,
            weights=np.asarray(points[:, dimension], dtype=np.float64),
            minlength=n_classes,
        )
    result /= counts[:, None]
    return result


def centroid_metrics(
    reference_centroids: np.ndarray,
    embedding_centroids: np.ndarray,
    adjacency_k: int,
) -> dict:
    reference_distances = pdist(reference_centroids)
    embedding_distances = pdist(embedding_centroids)
    correlation = spearmanr(reference_distances, embedding_distances)
    n_classes = len(reference_centroids)
    k = min(adjacency_k, n_classes - 1)

    def nearest(matrix):
        distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return np.argsort(distances, axis=1)[:, :k]

    reference_neighbors = nearest(reference_centroids)
    embedding_neighbors = nearest(embedding_centroids)
    recall = neighbor_overlap(reference_neighbors, embedding_neighbors)
    return {
        "pairwise_distance_spearman": {
            "correlation": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "n_pairs": int(len(reference_distances)),
        },
        "centroid_adjacency_recall": {
            "k": k,
            **recall,
        },
        "centroid_adjacencies": {
            "interpretation": (
                "For each evaluation-label code, the k nearest centroid codes "
                "in the common input and in this layout. These directed lists "
                "support pair-level visual audit; the aggregate recall above "
                "is their mean set overlap."
            ),
            "reference_nearest_codes": reference_neighbors.tolist(),
            "embedding_nearest_codes": embedding_neighbors.tolist(),
        },
    }


def classification_metrics(
    points: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict:
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(
            dual="auto",
            class_weight="balanced",
            max_iter=20_000,
            random_state=42,
        ),
    )
    started = time.perf_counter()
    classifier.fit(points[train_indices], labels[train_indices])
    predicted = classifier.predict(points[test_indices])
    elapsed = time.perf_counter() - started
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(labels[test_indices], predicted)
        ),
        "macro_f1": float(
            f1_score(labels[test_indices], predicted, average="macro")
        ),
        "fit_and_predict_wall_time_sec": elapsed,
        "classifier": (
            "StandardScaler + LinearSVC(dual='auto', class_weight='balanced')"
        ),
    }


def point_cloud_scales(points: np.ndarray) -> dict[str, float]:
    array = np.asarray(points, dtype=np.float32)
    if len(array) < 2:
        raise RuntimeError("topology subsets require at least two points")
    distances = pdist(array)
    diameter = float(distances.max())
    pairwise_q99 = float(np.quantile(distances, TOPOLOGY_Q))
    if diameter <= 0 or not math.isfinite(diameter):
        raise RuntimeError(f"invalid topology-subset diameter: {diameter}")
    if pairwise_q99 <= 0 or not math.isfinite(pairwise_q99):
        raise RuntimeError(
            "invalid topology-subset pairwise-distance "
            f"q{TOPOLOGY_Q:g}: {pairwise_q99}"
        )
    return {
        "diameter": diameter,
        "pairwise_distance_q99": pairwise_q99,
        "diameter_to_q99_ratio": diameter / pairwise_q99,
    }


def scale_to_unit_diameter(
    points: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(points, dtype=np.float32)
    scales = point_cloud_scales(array)
    return array / scales["diameter"], scales


def rescale_diagrams(
    diagrams: list[np.ndarray],
    factor: float,
) -> list[np.ndarray]:
    return [
        np.asarray(diagram, dtype=np.float64) * factor
        for diagram in diagrams
    ]


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    array = np.asarray(diagram, dtype=np.float64)
    if len(array) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return array[np.isfinite(array).all(axis=1)]


def bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_finite = finite_diagram(left)
    right_finite = finite_diagram(right)
    if len(left_finite) == 0 and len(right_finite) == 0:
        return 0.0
    if len(left_finite) == 0:
        return float(np.max(right_finite[:, 1] - right_finite[:, 0]) / 2.0)
    if len(right_finite) == 0:
        return float(np.max(left_finite[:, 1] - left_finite[:, 0]) / 2.0)
    from gudhi import bottleneck_distance as gudhi_bottleneck_distance

    # GUDHI's compiled exact implementation avoids the quadratic Python
    # matching matrices used by common convenience wrappers at 4,000 bars.
    return float(
        gudhi_bottleneck_distance(
            left_finite,
            right_finite,
            e=0.0,
        )
    )


def h0_bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the exact bottleneck distance for ordinary finite H0 bars.

    Ripser's finite H0 intervals all start at zero.  When the two diagrams
    have the same number of finite bars, sort their death times.  For each
    possible number ``u`` of bars sent to the diagonal, the optimal remaining
    one-dimensional matching pairs the two sorted tails in order.  The exact
    distance is the minimum, over ``u``, of the largest tail-pair difference
    and half the largest discarded death time.  A suffix maximum evaluates
    all choices in O(n log n) time and O(n) memory (the sort dominates).

    Any diagram outside that H0 contract falls back to the general compiled
    GUDHI matcher.
    """

    left_finite = finite_diagram(left)
    right_finite = finite_diagram(right)
    if (
        len(left_finite) != len(right_finite)
        or not np.all(left_finite[:, 0] == 0.0)
        or not np.all(right_finite[:, 0] == 0.0)
    ):
        return bottleneck_distance(left_finite, right_finite)
    count = len(left_finite)
    if count == 0:
        return 0.0
    left_deaths = np.sort(left_finite[:, 1])
    right_deaths = np.sort(right_finite[:, 1])
    suffix_pair_cost = np.zeros(count + 1, dtype=np.float64)
    for index in range(count - 1, -1, -1):
        suffix_pair_cost[index] = max(
            suffix_pair_cost[index + 1],
            abs(left_deaths[index] - right_deaths[index]),
        )
    best = math.inf
    for unmatched in range(count + 1):
        diagonal_cost = (
            0.0
            if unmatched == 0
            else max(
                left_deaths[unmatched - 1],
                right_deaths[unmatched - 1],
            )
            / 2.0
        )
        best = min(
            best,
            max(diagonal_cost, suffix_pair_cost[unmatched]),
        )
    return float(best)


def betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(diagram) == 0:
        return np.zeros(len(grid), dtype=np.int64)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    return np.asarray(
        [((births <= value) & (deaths > value)).sum() for value in grid],
        dtype=np.int64,
    )


def diagram_summary(diagrams: list[np.ndarray]) -> dict:
    h0 = diagrams[0]
    finite = h0[np.isfinite(h0[:, 1]), 1] - h0[np.isfinite(h0[:, 1]), 0]
    positive = finite[finite > 0]
    if len(positive) == 0:
        return {
            "h0_longest_finite_bar": 0.0,
            "h0_top5_to_median_ratio": None,
            "h0_finite_bars": 0,
        }
    top = np.sort(positive)[-min(5, len(positive)) :]
    median = float(np.median(positive))
    return {
        "h0_longest_finite_bar": float(positive.max()),
        "h0_top5_to_median_ratio": (
            float(top.mean() / median) if median > 0 else None
        ),
        "h0_finite_bars": int(len(positive)),
    }


def topology_reference(reference_points: np.ndarray) -> dict:
    from ripser import ripser

    reference_scaled, scales = scale_to_unit_diameter(reference_points)
    started = time.perf_counter()
    reference_diagrams = ripser(reference_scaled, maxdim=1)["dgms"]
    reference_time = time.perf_counter() - started
    q99_factor = scales["diameter"] / scales["pairwise_distance_q99"]
    diagrams_by_normalization = {
        "diameter": reference_diagrams,
        "pairwise_q99": rescale_diagrams(reference_diagrams, q99_factor),
    }
    return {
        "diagrams_by_normalization": diagrams_by_normalization,
        "diagrams": reference_diagrams,
        "scales": scales,
        "diameter": scales["diameter"],
        "wall_time_sec": reference_time,
        "summary": diagram_summary(reference_diagrams),
        "summaries": {
            name: diagram_summary(diagrams)
            for name, diagrams in diagrams_by_normalization.items()
        },
    }


def topology_diagram_metrics(
    reference_diagrams: list[np.ndarray],
    embedding_diagrams: list[np.ndarray],
    n_steps: int,
    subset_size: int,
) -> dict[str, float]:
    from fastdtw import fastdtw

    finite_deaths = []
    for diagrams in (reference_diagrams, embedding_diagrams):
        for diagram in diagrams:
            if len(diagram):
                finite = diagram[np.isfinite(diagram[:, 1]), 1]
                if len(finite):
                    finite_deaths.append(float(finite.max()))
    grid_max = 1.05 * max(finite_deaths) if finite_deaths else 1.0
    grid = np.linspace(0.0, grid_max, n_steps)
    metrics = {}
    for dimension in (0, 1):
        reference_curve = betti_curve(reference_diagrams[dimension], grid)
        embedding_curve = betti_curve(embedding_diagrams[dimension], grid)
        distance, _path = fastdtw(
            reference_curve.astype(np.float64),
            embedding_curve.astype(np.float64),
        )
        metrics[f"dtw_beta{dimension}"] = float(distance / subset_size)
        distance_function = (
            h0_bottleneck_distance
            if dimension == 0
            else bottleneck_distance
        )
        metrics[f"bottleneck_beta{dimension}"] = distance_function(
            reference_diagrams[dimension],
            embedding_diagrams[dimension],
        )
    return metrics


def topology_pair(
    reference: dict,
    embedding_points: np.ndarray,
    n_steps: int,
) -> dict:
    from ripser import ripser

    embedding_scaled, embedding_scales = scale_to_unit_diameter(embedding_points)
    started = time.perf_counter()
    embedding_diagrams = ripser(embedding_scaled, maxdim=1)["dgms"]
    embedding_time = time.perf_counter() - started
    q99_factor = (
        embedding_scales["diameter"]
        / embedding_scales["pairwise_distance_q99"]
    )
    embedding_by_normalization = {
        "diameter": embedding_diagrams,
        "pairwise_q99": rescale_diagrams(embedding_diagrams, q99_factor),
    }
    normalization_results = {}
    for name in ("diameter", "pairwise_q99"):
        metrics = topology_diagram_metrics(
            reference["diagrams_by_normalization"][name],
            embedding_by_normalization[name],
            n_steps,
            len(embedding_points),
        )
        normalization_results[name] = {
            **metrics,
            "reference_diagram": reference["summaries"][name],
            "embedding_diagram": diagram_summary(
                embedding_by_normalization[name]
            ),
        }
    diameter_metrics = normalization_results["diameter"]
    q99_metrics = normalization_results["pairwise_q99"]
    return {
        # The top-level metric names denote the declared unit-diameter
        # protocol. The explicit normalization blocks below record that
        # protocol together with the separate Q99 sensitivity calculation.
        "dtw_beta0": diameter_metrics["dtw_beta0"],
        "dtw_beta1": diameter_metrics["dtw_beta1"],
        "bottleneck_beta0": diameter_metrics["bottleneck_beta0"],
        "bottleneck_beta1": diameter_metrics["bottleneck_beta1"],
        "q99_dtw_beta0": q99_metrics["dtw_beta0"],
        "q99_dtw_beta1": q99_metrics["dtw_beta1"],
        "q99_bottleneck_beta0": q99_metrics["bottleneck_beta0"],
        "q99_bottleneck_beta1": q99_metrics["bottleneck_beta1"],
        "sample_size": int(len(embedding_points)),
        "n_steps": n_steps,
        "reference_diameter_before_rescaling": reference["diameter"],
        "embedding_diameter_before_rescaling": embedding_scales["diameter"],
        "reference_scales": reference["scales"],
        "embedding_scales": embedding_scales,
        "reference_ripser_wall_time_sec": reference["wall_time_sec"],
        "embedding_ripser_wall_time_sec": embedding_time,
        "reference_diagram": diameter_metrics["reference_diagram"],
        "embedding_diagram": diameter_metrics["embedding_diagram"],
        "normalizations": normalization_results,
    }


def compute_topology_jobs(
    jobs: list[tuple[tuple, dict, np.ndarray]],
    n_steps: int,
    workers: int,
) -> dict[tuple, dict]:
    """Evaluate independent layout/reference pairs in deterministic order."""
    if workers < 1:
        raise ValueError("topology workers must be positive")
    if not jobs:
        return {}
    worker_count = min(workers, len(jobs))
    if worker_count == 1:
        records = [
            topology_pair(reference, embedding_points, n_steps)
            for _key, reference, embedding_points in jobs
        ]
    else:
        from joblib import Parallel, delayed

        records = Parallel(
            n_jobs=worker_count,
            backend="loky",
        )(
            delayed(topology_pair)(
                reference,
                embedding_points,
                n_steps,
            )
            for _key, reference, embedding_points in jobs
        )
    return {
        key: record
        for (key, _reference, _embedding_points), record in zip(jobs, records)
    }


def aggregate_records(records: list[dict], keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "replicate_count": int(len(values)),
            "values": values.tolist(),
        }
    return result


def discover_full_layouts(
    dataset: str,
    large_results_root: Path,
    topology_sensitivity_results_root: Path,
    prepared_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    full_size = PREPARED[dataset]["full_size"]
    layouts: dict[str, np.ndarray] = {}
    run_metadata: dict[str, dict] = {}
    roots = {
        "dire_auto": large_results_root,
        "dire": large_results_root,
        "cuml_umap": large_results_root,
        "cuml_tsne": large_results_root,
        "pca2": large_results_root,
        "dire_spectral": topology_sensitivity_results_root,
        "dire_topology": topology_sensitivity_results_root,
    }
    for method, results_root in roots.items():
        base = (
            results_root
            / dataset
            / method
            / f"n_{full_size:09d}"
        )
        result_path = base / "result.json"
        if not result_path.exists():
            run_metadata[method] = {
                "status": "missing",
                "expected_result": str(result_path),
            }
            continue
        result = read_json(result_path)
        run_metadata[method] = result
        if result.get("status") != "success":
            continue
        embedding_path = base / result["embedding_file"]
        embedding = np.load(embedding_path, mmap_mode="r")
        if embedding.shape != (full_size, 2):
            raise RuntimeError(
                f"{dataset}/{method} embedding has shape {embedding.shape}"
            )
        layouts[method] = embedding
    if dataset == "tenx":
        official = np.load(
            prepared_root / PREPARED["tenx"]["official_layout"],
            mmap_mode="r",
        )
        if official.shape != (full_size, 2):
            raise RuntimeError(f"official 10x t-SNE has shape {official.shape}")
        layouts["cellranger_tsne"] = official
        run_metadata["cellranger_tsne"] = {
            "status": "released_reference",
            "runtime_comparable": False,
            "reason": (
                "Coordinates are from the official Cell Ranger secondary-analysis "
                "archive; original runtime/hardware are not part of this rerun."
            ),
        }
    return layouts, run_metadata


def discover_topology_seed_layouts(
    dataset: str,
    topology_sensitivity_results_root: Path,
) -> dict[str, list[dict]]:
    full_size = PREPARED[dataset]["full_size"]
    layouts: dict[str, list[dict]] = {}
    for method in TOPOLOGY_SEED_METHODS:
        run_root = (
            topology_sensitivity_results_root
            / dataset
            / method
            / f"n_{full_size:09d}"
        )
        result_path = run_root / "result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        if result.get("status") != "success":
            continue
        entries = []
        for record in result.get("records", []):
            embedding_name = record.get("embedding_file")
            if not isinstance(embedding_name, str):
                continue
            embedding = np.load(run_root / embedding_name, mmap_mode="r")
            if embedding.shape != (full_size, 2):
                raise RuntimeError(
                    f"{dataset}/{method} seed layout has shape {embedding.shape}"
                )
            entries.append(
                {
                    "repeat": int(record["repeat"]),
                    "seed": int(record["seed"]),
                    "embedding_file": embedding_name,
                    "embedding": embedding,
                }
            )
        if entries:
            layouts[method] = entries
    return layouts


def summarize_stage_timings(result: dict) -> dict:
    records = result.get("records", [])
    stage_names = ("knn_graph", "initialization", "layout")
    output = {}
    for stage in stage_names:
        values = []
        for record in records:
            diagnostics = record.get("reducer_diagnostics", {})
            stage_timings = diagnostics.get("stage_timings_sec", {})
            value = stage_timings.get(stage)
            if value is not None:
                values.append(float(value))
        if not values:
            continue
        steady = values[1:] if len(values) > 1 else values
        output[stage] = {
            "cold_sec": values[0],
            "steady_mean_sec": float(np.mean(steady)),
            "steady_std_sec": (
                float(np.std(steady, ddof=1)) if len(steady) > 1 else 0.0
            ),
            "steady_replicate_count": int(len(steady)),
            "all_values_sec": values,
        }
    return output


def load_knn_graph_audit(
    large_results_root: Path,
    dataset: str,
    method: str,
    full_size: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    run_root = (
        large_results_root
        / dataset
        / method
        / f"n_{full_size:09d}"
    )
    result = read_json(run_root / "result.json")
    audit = result.get("knn_graph_audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{dataset}/{method} has no kNN graph audit record")
    query = np.load(run_root / audit["query_indices_file"])
    neighbors = np.load(run_root / audit["neighbor_indices_file"])
    if query.ndim != 1 or neighbors.shape != (len(query), int(audit["k"])):
        raise RuntimeError(
            f"{dataset}/{method} kNN audit shapes disagree: "
            f"query={query.shape}, neighbors={neighbors.shape}"
        )
    return result, query, neighbors


def fixed_query_graph_overlap(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        raise RuntimeError(
            f"kNN audit graph shapes disagree: {left.shape} != {right.shape}"
        )
    k = left.shape[1]
    fractions = np.fromiter(
        (
            len(set(map(int, left_row)).intersection(map(int, right_row))) / k
            for left_row, right_row in zip(left, right)
        ),
        dtype=np.float64,
        count=len(left),
    )
    return {
        "definition": (
            "Per-query intersection size divided by k for equally sized "
            "neighbor sets; symmetric because both graphs use the same k."
        ),
        "mean": float(fractions.mean()),
        "std": float(fractions.std(ddof=1)) if len(fractions) > 1 else 0.0,
        "median": float(np.median(fractions)),
        "q05": float(np.quantile(fractions, 0.05)),
        "q95": float(np.quantile(fractions, 0.95)),
        "zero_overlap_fraction": float(np.mean(fractions == 0.0)),
        "exact_set_match_fraction": float(np.mean(fractions == 1.0)),
        "query_count": int(len(fractions)),
        "k": int(k),
    }


def quality_gate_comparison(
    production: dict,
    control: dict,
) -> dict:
    metric_specs = {
        "local_neighbor_overlap": (
            lambda item: item["local"]["neighbor_overlap"]["mean"],
            "higher",
        ),
        "centroid_distance_spearman": (
            lambda item: item["global"]["pairwise_distance_spearman"][
                "correlation"
            ],
            "higher",
        ),
        "centroid_adjacency_recall": (
            lambda item: item["global"]["centroid_adjacency_recall"]["mean"],
            "higher",
        ),
        "context_balanced_accuracy": (
            lambda item: item["context_classifier"]["balanced_accuracy"],
            "higher",
        ),
    }
    for key in TOPOLOGY_METRIC_KEYS:
        metric_specs[f"topology_{key}"] = (
            lambda item, metric=key: item["topology"]["aggregate"][metric][
                "mean"
            ],
            "lower",
        )
    comparisons = {}
    for key, (getter, direction) in metric_specs.items():
        production_value = float(getter(production))
        control_value = float(getter(control))
        comparisons[key] = {
            "direction": direction,
            "production_auto": production_value,
            "forced_ivf_flat": control_value,
            "auto_minus_control": production_value - control_value,
            "relative_auto_minus_control": (
                (production_value - control_value) / abs(control_value)
                if control_value != 0
                else None
            ),
        }
    return comparisons


def build_backend_policy_audit(
    dataset: str,
    full_size: int,
    backend_policy_results_root: Path,
    methods: dict[str, dict],
) -> dict | None:
    if "dire_auto" not in methods or "dire" not in methods:
        return None
    auto_result, auto_query, auto_graph = load_knn_graph_audit(
        backend_policy_results_root,
        dataset,
        "dire_auto",
        full_size,
    )
    flat_result, flat_query, flat_graph = load_knn_graph_audit(
        backend_policy_results_root,
        dataset,
        "dire_ivf_flat_control",
        full_size,
    )
    if not np.array_equal(auto_query, flat_query):
        raise RuntimeError(
            f"{dataset} production/control kNN audit queries are not identical"
        )

    def method_record(result: dict) -> dict:
        diagnostics = [
            record.get("reducer_diagnostics", {})
            for record in result.get("records", [])
        ]
        return {
            "requested_cuvs_index_type": result.get(
                "method_parameters",
                {},
            ).get("cuvs_index_type"),
            "effective_cuvs_index_types": [
                item.get("effective_cuvs_index_type")
                for item in diagnostics
            ],
            "cold_total_sec": result["timing"]["cold_start_sec"],
            "steady_total_mean_sec": result["timing"]["steady_mean_sec"],
            "steady_total_std_sec": result["timing"]["steady_std_sec"],
            "steady_replicate_count": result["timing"]["steady_n"],
            "stage_timings": summarize_stage_timings(result),
            "chunked_force_fallback_calls": [
                int(item.get("chunked_force_fallback_calls", 0))
                for item in diagnostics
            ],
            "torch_compile_failed": [
                item.get("torch_compile_failed")
                for item in diagnostics
            ],
        }

    auto_record = method_record(auto_result)
    flat_record = method_record(flat_result)
    return {
        "policy": (
            "The dedicated production-auto and fresh forced-IVF-Flat profiles "
            "use identical data, DiRe parameters, seeds, repeat count, and "
            "hardware. Forced-IVF-Flat control benchmark files are not "
            "overwritten. The full quality gate compares the production-auto "
            "layout with that historical configuration. No runtime claim is "
            "interpreted without the graph-overlap record and the complete "
            "embedding-quality comparison below."
        ),
        "upstream_diagnostics_issue": (
            "https://github.com/sashakolpakov/dire-rapids/issues/13"
        ),
        "N_full_dataset": int(full_size),
        "R_fit_total": int(len(auto_result.get("records", []))),
        "R_steady": int(auto_result["timing"]["steady_n"]),
        "m_graph_queries": int(len(auto_query)),
        "production_auto": auto_record,
        "forced_ivf_flat_control": flat_record,
        "steady_speedup_control_over_auto": (
            flat_record["steady_total_mean_sec"]
            / auto_record["steady_total_mean_sec"]
        ),
        "knn_graph_overlap": fixed_query_graph_overlap(
            auto_graph,
            flat_graph,
        ),
        "full_embedding_quality_gate": {
            "status": "evaluated",
            "interpretation": (
                "Descriptive auto-versus-control differences across every "
                "predeclared local, global, context, and topology metric; no "
                "equivalence threshold or post-hoc pass criterion is imposed."
            ),
            "metrics": quality_gate_comparison(
                methods["dire_auto"],
                methods["dire"],
            ),
        },
    }


def evaluate_dataset(dataset: str, args) -> dict:
    config = PREPARED[dataset]
    input_path = args.prepared_root / config["input"]
    label_path = args.prepared_root / config["labels"]
    manifest_path = args.prepared_root / config["manifest"]
    X = np.load(input_path, mmap_mode="r")
    raw_labels = np.load(label_path, mmap_mode="r")
    full_size = config["full_size"]
    if len(X) != full_size or len(raw_labels) != full_size:
        raise RuntimeError(
            f"{dataset} prepared shapes disagree with expected full size {full_size:,}"
        )
    labels, label_policy = evaluation_labels(
        dataset,
        np.asarray(raw_labels),
        args.prepared_root,
        args.arxiv_top_categories,
    )
    n_classes = len(np.unique(labels))
    layouts, run_metadata = discover_full_layouts(
        dataset,
        args.large_results_root,
        args.topology_sensitivity_results_root,
        args.prepared_root,
    )
    topology_seed_layouts = discover_topology_seed_layouts(
        dataset,
        args.topology_sensitivity_results_root,
    )
    if not layouts:
        raise RuntimeError(f"no full-scale layouts available for {dataset}")

    output_dir = args.output_root / dataset
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "evaluation_labels.npy", labels)

    print(f"{dataset}: building shared kNN evaluation sample", flush=True)
    knn_indices = stratified_indices(labels, args.knn_sample_size, args.seed)
    np.save(output_dir / "knn_sample_indices.npy", knn_indices)
    X_knn = np.ascontiguousarray(X[knn_indices], dtype=np.float32)
    labels_knn = labels[knn_indices]
    high_knn = compute_knn_indices(X_knn, args.knn_k)
    high_label_agreement = neighbor_label_agreement(high_knn, labels_knn)

    print(f"{dataset}: computing full-data reference centroids", flush=True)
    reference_centroids = centroids(X, labels, n_classes)

    classifier_sample = stratified_indices(
        labels,
        args.classifier_sample_size,
        args.seed + 10_000,
    )
    sample_positions = np.arange(len(classifier_sample))
    train_local, test_local = train_test_split(
        sample_positions,
        test_size=0.3,
        random_state=args.seed,
        shuffle=True,
        stratify=labels[classifier_sample],
    )
    train_global = classifier_sample[train_local]
    test_global = classifier_sample[test_local]
    print(f"{dataset}: fitting high-dimensional context classifier", flush=True)
    reference_classifier = classification_metrics(
        X,
        labels,
        train_global,
        test_global,
    )

    topology_subsets = [
        stratified_indices(
            labels,
            args.topology_subset_size,
            args.seed + 20_000 + subset_id,
        )
        for subset_id in range(args.topology_subset_count)
    ]
    for subset_id, indices in enumerate(topology_subsets):
        np.save(output_dir / f"topology_subset_{subset_id}.npy", indices)
    print(
        f"{dataset}: computing {len(topology_subsets)} shared "
        "high-dimensional topology references",
        flush=True,
    )
    topology_reference_points = [
        np.ascontiguousarray(X[indices], dtype=np.float32)
        for indices in topology_subsets
    ]
    topology_references = [
        topology_reference(points)
        for points in topology_reference_points
    ]
    topology_subset_metadata = []
    for subset_id, (indices, points, reference) in enumerate(
        zip(topology_subsets, topology_reference_points, topology_references)
    ):
        codes, counts = np.unique(labels[indices], return_counts=True)
        topology_subset_metadata.append(
            {
                "subset_id": subset_id,
                "subset_seed": args.seed + 20_000 + subset_id,
                "subset_size": int(len(indices)),
                "label_counts": {
                    str(int(code)): int(count)
                    for code, count in zip(codes, counts)
                },
                "reference_scales": reference["scales"],
                "audit_file": f"topology_reference_{subset_id}.npz",
            }
        )
        np.savez_compressed(
            output_dir / f"topology_reference_{subset_id}.npz",
            points=points,
            indices=indices,
            diameter_h0=reference["diagrams_by_normalization"]["diameter"][0],
            diameter_h1=reference["diagrams_by_normalization"]["diameter"][1],
            pairwise_q99_h0=reference["diagrams_by_normalization"][
                "pairwise_q99"
            ][0],
            pairwise_q99_h1=reference["diagrams_by_normalization"][
                "pairwise_q99"
            ][1],
            diameter=np.asarray(reference["scales"]["diameter"]),
            pairwise_distance_q99=np.asarray(
                reference["scales"]["pairwise_distance_q99"]
            ),
        )

    topology_size_indices = nested_stratified_indices(
        labels,
        topology_subsets[0],
        [
            *args.topology_sensitivity_sizes,
            args.topology_subset_size,
        ],
        args.seed + 30_000,
    )
    topology_size_references = {}
    topology_size_metadata = []
    for size, indices in sorted(topology_size_indices.items()):
        if size == len(topology_subsets[0]):
            points = topology_reference_points[0]
            reference = topology_references[0]
        else:
            points = np.ascontiguousarray(X[indices], dtype=np.float32)
            reference = topology_reference(points)
        topology_size_references[size] = reference
        subset_name = f"topology_size_subset_{size}.npy"
        audit_name = f"topology_size_reference_{size}.npz"
        np.save(output_dir / subset_name, indices)
        np.savez_compressed(
            output_dir / audit_name,
            points=points,
            indices=indices,
            diameter_h0=reference["diagrams_by_normalization"]["diameter"][0],
            diameter_h1=reference["diagrams_by_normalization"]["diameter"][1],
            pairwise_q99_h0=reference["diagrams_by_normalization"][
                "pairwise_q99"
            ][0],
            pairwise_q99_h1=reference["diagrams_by_normalization"][
                "pairwise_q99"
            ][1],
            diameter=np.asarray(reference["scales"]["diameter"]),
            pairwise_distance_q99=np.asarray(
                reference["scales"]["pairwise_distance_q99"]
            ),
        )
        topology_size_metadata.append(
            {
                "subset_size": size,
                "subset_file": subset_name,
                "audit_file": audit_name,
                "reference_scales": reference["scales"],
            }
        )

    topology_jobs = []
    for method, layout in layouts.items():
        for subset_id, (indices, reference) in enumerate(
            zip(topology_subsets, topology_references)
        ):
            topology_jobs.append(
                (
                    ("subset", method, subset_id),
                    reference,
                    np.ascontiguousarray(layout[indices], dtype=np.float32),
                )
            )
        for size, indices in sorted(topology_size_indices.items()):
            if size == len(topology_subsets[0]):
                continue
            topology_jobs.append(
                (
                    ("size", method, size),
                    topology_size_references[size],
                    np.ascontiguousarray(layout[indices], dtype=np.float32),
                )
            )
        for seed_layout in topology_seed_layouts.get(method, []):
            topology_jobs.append(
                (
                    ("seed", method, int(seed_layout["repeat"])),
                    topology_references[0],
                    np.ascontiguousarray(
                        seed_layout["embedding"][topology_subsets[0]],
                        dtype=np.float32,
                    ),
                )
            )
    print(
        f"{dataset}: computing {len(topology_jobs)} layout topology "
        f"comparisons with up to {args.topology_workers} worker(s)",
        flush=True,
    )
    topology_job_results = compute_topology_jobs(
        topology_jobs,
        args.topology_steps,
        args.topology_workers,
    )

    method_metrics = {}
    for method, layout in layouts.items():
        print(f"{dataset}: evaluating {METHOD_DISPLAY[method]}", flush=True)
        layout_knn = np.ascontiguousarray(layout[knn_indices], dtype=np.float32)
        low_knn = compute_knn_indices(layout_knn, args.knn_k)
        layout_centroids = centroids(layout, labels, n_classes)
        classifier = classification_metrics(
            layout,
            labels,
            train_global,
            test_global,
        )
        topology_records = []
        for subset_id, (indices, reference) in enumerate(
            zip(topology_subsets, topology_references)
        ):
            print(
                f"  topology subset {subset_id + 1}/{len(topology_subsets)} "
                f"(m={len(indices):,})",
                flush=True,
            )
            record = topology_job_results[("subset", method, subset_id)]
            record["subset_id"] = subset_id
            record["subset_seed"] = args.seed + 20_000 + subset_id
            topology_records.append(record)
        topology_size_records = []
        for size, indices in sorted(topology_size_indices.items()):
            if size == len(topology_subsets[0]):
                size_record = dict(topology_records[0])
            else:
                print(
                    f"  topology nested subset-size sensitivity m={size:,}",
                    flush=True,
                )
                size_record = topology_job_results[("size", method, size)]
            size_record["sensitivity_subset_size"] = size
            size_record["nested_in_subset_size"] = len(topology_subsets[0])
            topology_size_records.append(size_record)
        topology_aggregate = aggregate_records(
            topology_records,
            TOPOLOGY_METRIC_KEYS,
        )
        for diagram_key in (
            "h0_longest_finite_bar",
            "h0_top5_to_median_ratio",
        ):
            usable = [
                record["embedding_diagram"][diagram_key]
                for record in topology_records
                if record["embedding_diagram"][diagram_key] is not None
            ]
            if usable:
                values = np.asarray(usable, dtype=np.float64)
                topology_aggregate[diagram_key] = {
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "replicate_count": int(len(values)),
                    "values": values.tolist(),
                }
        layout_seed_records = []
        for seed_layout in topology_seed_layouts.get(method, []):
            print(
                f"  topology layout-seed sensitivity seed="
                f"{seed_layout['seed']} on fixed topology subset 1",
                flush=True,
            )
            seed_record = topology_job_results[
                ("seed", method, int(seed_layout["repeat"]))
            ]
            seed_record.update(
                {
                    "layout_repeat": seed_layout["repeat"],
                    "layout_seed": seed_layout["seed"],
                    "layout_embedding_file": seed_layout["embedding_file"],
                    "topology_subset_id": 0,
                    "topology_subset_seed": args.seed + 20_000,
                }
            )
            layout_seed_records.append(seed_record)
        layout_seed_sensitivity = None
        if layout_seed_records:
            layout_seed_sensitivity = {
                "policy": (
                    f"{len(layout_seed_records)} independently seeded full "
                    "layouts evaluated on the same fixed topology row-index "
                    "subset; reported separately from canonical-layout subset "
                    "variation"
                ),
                "records": layout_seed_records,
                "aggregate": aggregate_records(
                    layout_seed_records,
                    TOPOLOGY_METRIC_KEYS,
                ),
            }
        method_metrics[method] = {
            "display": METHOD_DISPLAY[method],
            "local": {
                "knn_k": args.knn_k,
                "sample_size": int(len(knn_indices)),
                "neighbor_overlap": neighbor_overlap(high_knn, low_knn),
                "embedding_neighbor_label_agreement": neighbor_label_agreement(
                    low_knn,
                    labels_knn,
                ),
                "reference_neighbor_label_agreement": high_label_agreement,
            },
            "global": centroid_metrics(
                reference_centroids,
                layout_centroids,
                args.centroid_adjacency_k,
            ),
            "context_classifier": classifier,
            "topology": {
                "subset_policy": (
                    "independently seeded, fixed stratified row-index subsets; "
                    "each subset is applied identically to the high-dimensional "
                    "reference and every layout"
                ),
                "subset_metadata": topology_subset_metadata,
                "records": topology_records,
                "aggregate": topology_aggregate,
                "layout_seed_sensitivity": layout_seed_sensitivity,
                "subset_size_sensitivity": {
                    "policy": (
                        "nested stratified row-index subsets drawn within "
                        "topology subset 0; canonical layout held fixed"
                    ),
                    "subset_metadata": topology_size_metadata,
                    "records": topology_size_records,
                },
            },
            "run": run_metadata.get(method),
        }

    backend_policy_audit = build_backend_policy_audit(
        dataset,
        full_size,
        args.backend_policy_results_root,
        method_metrics,
    )
    payload = {
        "schema_version": 2,
        "dataset": dataset,
        "dataset_manifest": read_json(manifest_path),
        "full_n_samples": full_size,
        "n_input_features": int(X.shape[1]),
        "topology_design": {
            "full_dataset_size": int(full_size),
            "topology_subset_size": int(args.topology_subset_size),
            "topology_subset_count": int(args.topology_subset_count),
            "nested_subset_sizes": sorted(topology_size_indices),
            "layout_seed_count_by_method": {
                method: int(len(entries))
                for method, entries in topology_seed_layouts.items()
            },
            "pairing": (
                "Each fixed row-index subset is applied identically to the "
                "high-dimensional reference and every embedding."
            ),
        },
        "label_policy": label_policy,
        "evaluation_configuration": {
            "seed": args.seed,
            "knn_sample_size": args.knn_sample_size,
            "classifier_sample_size": args.classifier_sample_size,
            "topology_subset_size": args.topology_subset_size,
            "topology_sensitivity_sizes": args.topology_sensitivity_sizes,
            "topology_subset_count": args.topology_subset_count,
            "topology_steps": args.topology_steps,
            "topology_workers": args.topology_workers,
        },
        "metric_protocol": {
            "local": (
                f"Exact brute-force GPU {args.knn_k}-NN on one fixed, "
                f"stratified m={len(knn_indices):,} sample; mean set overlap "
                "between input-space and layout neighbors."
            ),
            "global": (
                "Spearman correlation of all reference-label centroid "
                "distances, plus nearest-centroid adjacency recall."
            ),
            "context": (
                "The same fixed stratified rows and train/test split for every "
                "method; balanced linear SVM avoids majority-class inflation."
            ),
            "topology": (
                f"Ripser H0/H1 on {args.topology_subset_count} "
                "independently seeded, fixed stratified row-index subsets of "
                f"m={args.topology_subset_size:,}; each subset is applied "
                "identically to the reference and every embedding. "
                "Finite H0 bottleneck distances use the exact sorted "
                "zero-birth-bar specialization; H1 uses GUDHI's compiled "
                "exact matcher. "
                "Primary scale "
                "invariance divides each point cloud by its diameter and reports "
                "normalized bottleneck distances plus historical Betti-curve "
                "DTW. A separately labelled sensitivity analysis divides by the "
                f"{100 * TOPOLOGY_Q:.0f}th percentile of pairwise distances; "
                "Betti curves share a filtration grid within each comparison. "
                "Reference subset coordinates and diagrams are bundled for audit. "
                "Nested "
                + "/".join(
                    f"{item['subset_size']:,}"
                    for item in topology_size_metadata
                )
                + "-point nested subsets test whether the canonical-layout "
                "ranking is stable to topology subset size. "
                "For every stochastic nonlinear GPU method, "
                f"{len(topology_seed_layouts.get('dire_auto', []))} "
                "independently seeded full layouts are "
                "additionally compared on one fixed topology row-index subset; "
                "layout-seed and subset-seed variation are reported separately. "
                f"Independent layout/reference pairs use at most "
                f"{args.topology_workers} CPU worker processes; reference "
                "subsets, indices, and output order are unchanged by this "
                "execution-only parallelism."
            ),
        },
        "reference": {
            "neighbor_label_agreement": high_label_agreement,
            "context_classifier": reference_classifier,
            "label_centroids": {
                "policy": (
                    "Full-data centroids in the common prepared input, ordered "
                    "by evaluation-label code and bundled for independent "
                    "centroid-adjacency recomputation."
                ),
                "values": reference_centroids.tolist(),
            },
        },
        "run_metadata": run_metadata,
        "methods": method_metrics,
        "backend_policy_audit": backend_policy_audit,
    }
    write_json(output_dir / "evaluation.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/revision3/prepared"),
    )
    parser.add_argument(
        "--large-results-root",
        type=Path,
        default=Path("data/revision3/large_results"),
    )
    parser.add_argument(
        "--backend-policy-results-root",
        type=Path,
        default=Path("data/revision3/backend_policy_results"),
    )
    parser.add_argument(
        "--topology-sensitivity-results-root",
        type=Path,
        default=Path("data/revision3/topology_sensitivity_results"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/revision3/evaluation"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-sample-size", type=int, default=20_000)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--classifier-sample-size", type=int, default=100_000)
    parser.add_argument("--centroid-adjacency-k", type=int, default=3)
    parser.add_argument("--topology-subset-size", type=int, default=4_000)
    parser.add_argument(
        "--topology-sensitivity-sizes",
        type=int,
        nargs="+",
        default=[1_000, 2_000, 4_000],
    )
    parser.add_argument("--topology-subset-count", type=int, default=10)
    parser.add_argument("--topology-steps", type=int, default=100)
    parser.add_argument(
        "--topology-workers",
        type=int,
        default=1,
        help=(
            "bounded process parallelism for independent layout topology "
            "comparisons; high-dimensional references remain sequential"
        ),
    )
    parser.add_argument("--arxiv-top-categories", type=int, default=15)
    args = parser.parse_args()
    for name in (
        "knn_sample_size",
        "knn_k",
        "classifier_sample_size",
        "centroid_adjacency_k",
        "topology_subset_size",
        "topology_subset_count",
        "topology_steps",
        "topology_workers",
        "arxiv_top_categories",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if any(size < 2 for size in args.topology_sensitivity_sizes):
        raise ValueError("topology sensitivity subset sizes must be at least 2")
    if any(
        size > args.topology_subset_size
        for size in args.topology_sensitivity_sizes
    ):
        raise ValueError(
            "topology sensitivity subset sizes cannot exceed the primary "
            "topology subset size"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": {},
    }
    for dataset in args.datasets:
        payload["datasets"][dataset] = evaluate_dataset(dataset, args)
        write_json(args.output_root / "evaluation_manifest.json", payload)


if __name__ == "__main__":
    main()
