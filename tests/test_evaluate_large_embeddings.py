from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_large_embeddings import (  # noqa: E402
    aggregate_records,
    build_backend_policy_audit,
    diagram_summary,
    fixed_query_graph_overlap,
    h0_bottleneck_distance,
    nested_stratified_indices,
    point_cloud_scales,
    topology_pair,
)


def test_topology_pair_normalizes_by_shared_subset_size(monkeypatch) -> None:
    diagrams = [
        np.asarray([[0.0, 0.4], [0.0, np.inf]], dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
    ]

    ripser_module = ModuleType("ripser")
    ripser_module.ripser = lambda _points, maxdim: {"dgms": diagrams}
    fastdtw_module = ModuleType("fastdtw")
    fastdtw_module.fastdtw = lambda _left, _right: (8.0, [])
    gudhi_module = ModuleType("gudhi")
    gudhi_module.bottleneck_distance = lambda _left, _right, e: 0.125
    monkeypatch.setitem(sys.modules, "ripser", ripser_module)
    monkeypatch.setitem(sys.modules, "fastdtw", fastdtw_module)
    monkeypatch.setitem(sys.modules, "gudhi", gudhi_module)

    reference = {
        "diagrams_by_normalization": {
            "diameter": diagrams,
            "pairwise_q99": diagrams,
        },
        "diagrams": diagrams,
        "scales": {
            "diameter": 1.0,
            "pairwise_distance_q99": 1.0,
            "diameter_to_q99_ratio": 1.0,
        },
        "diameter": 1.0,
        "wall_time_sec": 0.0,
        "summary": diagram_summary(diagrams),
        "summaries": {
            "diameter": diagram_summary(diagrams),
            "pairwise_q99": diagram_summary(diagrams),
        },
    }
    embedding = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )

    result = topology_pair(reference, embedding, n_steps=8)

    assert result["sample_size"] == 4
    assert result["dtw_beta0"] == 2.0
    assert result["dtw_beta1"] == 2.0
    assert result["bottleneck_beta0"] == 0.0
    assert result["q99_dtw_beta0"] == 2.0
    assert set(result["normalizations"]) == {"diameter", "pairwise_q99"}


def test_exact_h0_bottleneck_handles_diagonal_matching() -> None:
    left = np.column_stack(
        (
            np.zeros(3),
            np.asarray([0.711, 0.429, 0.192]),
        )
    )
    right = np.column_stack(
        (
            np.zeros(3),
            np.asarray([4.471, 4.511, 5.604]),
        )
    )

    assert h0_bottleneck_distance(left, right) == 2.802
    assert h0_bottleneck_distance(left, left) == 0.0


def test_h0_bottleneck_falls_back_outside_zero_birth_contract(
    monkeypatch,
) -> None:
    gudhi_module = ModuleType("gudhi")
    gudhi_module.bottleneck_distance = (
        lambda _left, _right, e: 0.125 if e == 0.0 else -1.0
    )
    monkeypatch.setitem(sys.modules, "gudhi", gudhi_module)
    left = np.asarray([[0.1, 0.4]], dtype=np.float64)
    right = np.asarray([[0.0, 0.5]], dtype=np.float64)

    assert h0_bottleneck_distance(left, right) == 0.125


def test_point_cloud_scales_exposes_outlier_sensitivity() -> None:
    compact = np.asarray(
        [[float(index), 0.0] for index in range(100)] + [[10_000.0, 0.0]],
        dtype=np.float32,
    )
    scales = point_cloud_scales(compact)

    assert scales["diameter"] == 10_000.0
    assert scales["pairwise_distance_q99"] < scales["diameter"]
    assert scales["diameter_to_q99_ratio"] > 1.0


def test_nested_topology_samples_are_stratified_subsets() -> None:
    labels = np.repeat(np.arange(4), 25)
    largest = np.arange(100, dtype=np.int64)

    samples = nested_stratified_indices(
        labels,
        largest,
        [20, 40, 80, 100],
        seed=42,
    )

    assert set(samples) == {20, 40, 80, 100}
    assert set(samples[20]).issubset(samples[40])
    assert set(samples[40]).issubset(samples[80])
    assert set(samples[80]).issubset(samples[100])
    for size, indices in samples.items():
        assert len(indices) == size
        assert set(np.unique(labels[indices])) == {0, 1, 2, 3}


def test_nested_topology_samples_retain_singleton_class() -> None:
    labels = np.concatenate(
        [
            np.repeat(0, 50),
            np.repeat(1, 30),
            np.repeat(2, 19),
            np.asarray([3]),
        ]
    )
    largest = np.arange(len(labels), dtype=np.int64)

    first = nested_stratified_indices(
        labels,
        largest,
        [20, 60, 100],
        seed=91,
    )
    second = nested_stratified_indices(
        labels,
        largest,
        [20, 60, 100],
        seed=91,
    )

    assert set(first) == {20, 60, 100}
    assert set(first[20]).issubset(first[60])
    assert set(first[60]).issubset(first[100])
    for size, indices in first.items():
        assert len(indices) == size
        assert 99 in indices
        assert np.array_equal(indices, second[size])


def test_nested_topology_subset_smaller_than_class_count_is_exact() -> None:
    labels = np.arange(10)
    samples = nested_stratified_indices(
        labels,
        np.arange(10, dtype=np.int64),
        [3, 10],
        seed=7,
    )

    assert len(samples[3]) == 3
    assert set(samples[3]).issubset(samples[10])


def test_fixed_query_graph_overlap_reports_distribution() -> None:
    left = np.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        dtype=np.int32,
    )
    right = np.asarray(
        [[1, 2, 3, 4], [5, 6, 20, 21], [30, 31, 32, 33]],
        dtype=np.int32,
    )

    result = fixed_query_graph_overlap(left, right)

    assert result["query_count"] == 3
    assert result["k"] == 4
    assert result["mean"] == 0.5
    assert result["median"] == 0.5
    assert result["zero_overlap_fraction"] == 1 / 3
    assert result["exact_set_match_fraction"] == 1 / 3


def test_aggregate_records_names_replicates_unambiguously() -> None:
    result = aggregate_records(
        [{"score": 1.0}, {"score": 2.0}, {"score": 3.0}],
        ("score",),
    )

    assert result["score"]["replicate_count"] == 3
    assert "n" not in result["score"]


def topology_method_payload(base: float) -> dict:
    aggregate = {
        key: {"mean": base}
        for key in (
            "dtw_beta0",
            "dtw_beta1",
            "bottleneck_beta0",
            "bottleneck_beta1",
            "q99_dtw_beta0",
            "q99_dtw_beta1",
            "q99_bottleneck_beta0",
            "q99_bottleneck_beta1",
        )
    }
    return {
        "local": {"neighbor_overlap": {"mean": 0.5 + base}},
        "global": {
            "pairwise_distance_spearman": {"correlation": 0.6 + base},
            "centroid_adjacency_recall": {"mean": 0.7 + base},
        },
        "context_classifier": {"balanced_accuracy": 0.8 + base},
        "topology": {"aggregate": aggregate},
    }


def write_profile_result(
    root: Path,
    dataset: str,
    method: str,
    full_size: int,
    requested_index: str,
    effective_index: str,
    total: float,
    graph: np.ndarray,
) -> None:
    run_root = root / dataset / method / f"n_{full_size:09d}"
    run_root.mkdir(parents=True)
    query = np.asarray([0, 1, 2], dtype=np.int32)
    np.save(run_root / "queries.npy", query)
    np.save(run_root / "neighbors.npy", graph)
    records = []
    for repeat in range(3):
        records.append(
            {
                "reducer_diagnostics": {
                    "effective_cuvs_index_type": effective_index,
                    "stage_timings_sec": {
                        "knn_graph": total - 2.0,
                        "initialization": 0.5,
                        "layout": 1.0,
                    },
                    "chunked_force_fallback_calls": 0,
                    "torch_compile_failed": False,
                }
            }
        )
    (run_root / "result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "method_parameters": {
                    "cuvs_index_type": requested_index,
                },
                "records": records,
                "timing": {
                    "cold_start_sec": total + 1.0,
                    "steady_mean_sec": total,
                    "steady_std_sec": 0.1,
                    "steady_n": 2,
                },
                "knn_graph_audit": {
                    "query_indices_file": "queries.npy",
                    "neighbor_indices_file": "neighbors.npy",
                    "k": int(graph.shape[1]),
                },
            }
        ),
        encoding="utf-8",
    )


def test_backend_policy_audit_uses_separate_fresh_control(tmp_path: Path) -> None:
    full_size = 3
    auto_graph = np.asarray(
        [[1, 2], [0, 2], [0, 1]],
        dtype=np.int32,
    )
    flat_graph = np.asarray(
        [[1, 2], [0, 2], [1, 0]],
        dtype=np.int32,
    )
    write_profile_result(
        tmp_path,
        "arxiv",
        "dire_auto",
        full_size,
        "auto",
        "ivf_pq",
        13.0,
        auto_graph,
    )
    write_profile_result(
        tmp_path,
        "arxiv",
        "dire_ivf_flat_control",
        full_size,
        "ivf_flat",
        "ivf_flat",
        43.0,
        flat_graph,
    )

    audit = build_backend_policy_audit(
        "arxiv",
        full_size,
        tmp_path,
        {
            "dire_auto": topology_method_payload(0.1),
            "dire": topology_method_payload(0.2),
        },
    )

    assert audit is not None
    assert audit["production_auto"]["effective_cuvs_index_types"] == [
        "ivf_pq",
        "ivf_pq",
        "ivf_pq",
    ]
    assert audit["forced_ivf_flat_control"][
        "requested_cuvs_index_type"
    ] == "ivf_flat"
    assert audit["steady_speedup_control_over_auto"] == 43.0 / 13.0
    assert audit["knn_graph_overlap"]["mean"] == 1.0
    assert audit["full_embedding_quality_gate"]["status"] == "evaluated"
