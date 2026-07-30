from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from large_scale_benchmarks import (  # noqa: E402
    build_reducer,
    dire_runtime_diagnostics,
)
from evaluate_large_embeddings import build_cuvs_knn_ab_audit  # noqa: E402
from summarize_cuvs_knn_ab import summarize_pair  # noqa: E402


def test_build_reducer_isolates_only_cuvs_graph_method(monkeypatch) -> None:
    captured = []
    module = ModuleType("dire_rapids")

    def create_dire(**parameters):
        captured.append(parameters)
        return SimpleNamespace(parameters=parameters)

    module.create_dire = create_dire
    monkeypatch.setitem(sys.modules, "dire_rapids", module)

    _, index_parameters = build_reducer(
        "dire_index_search",
        seed=42,
        n_neighbors=15,
        dire_iterations=128,
    )
    _, all_parameters = build_reducer(
        "dire_all_neighbors",
        seed=42,
        n_neighbors=15,
        dire_iterations=128,
    )

    graph_keys = {
        "cuvs_knn_method",
        "all_neighbors_algo",
        "all_neighbors_n_clusters",
        "all_neighbors_overlap_factor",
    }
    assert {
        key: value
        for key, value in index_parameters.items()
        if key not in graph_keys
    } == {
        key: value
        for key, value in all_parameters.items()
        if key not in graph_keys
    }
    assert index_parameters["cuvs_knn_method"] == "index_search"
    assert index_parameters["cuvs_index_type"] == "auto"
    assert all_parameters["cuvs_knn_method"] == "all_neighbors"
    assert all_parameters["all_neighbors_algo"] == "nn_descent"
    assert all_parameters["all_neighbors_n_clusters"] == 1
    assert all_parameters["all_neighbors_overlap_factor"] == 0
    assert captured == [index_parameters, all_parameters]


def test_runtime_diagnostics_prefers_public_contract() -> None:
    class Reducer:
        cuvs_knn_method = "all_neighbors"

        @staticmethod
        def get_diagnostics():
            return {
                "stage_timings_seconds": {
                    "graph_construction": 1.5,
                    "initialization": 0.5,
                    "layout": 2.5,
                    "total": 4.5,
                },
                "force_chunked_fallback_used": False,
                "force_chunked_fallback_calls": 0,
                "cuvs": {
                    "requested_knn_method": "all_neighbors",
                    "effective_knn_method": "all_neighbors",
                    "requested_index_type": "auto",
                    "effective_index_type": None,
                    "effective_all_neighbors_algo": "nn_descent",
                },
            }

    result = dire_runtime_diagnostics(
        Reducer(),
        {
            "stage_timings_sec": {"knn_graph": 999.0},
            "chunked_force_fallback_calls": 99,
        },
    )

    assert result["stage_timings_sec"]["knn_graph"] == 1.5
    assert result["chunked_force_fallback_calls"] == 0
    assert result["effective_cuvs_knn_method"] == "all_neighbors"
    assert result["effective_all_neighbors_algo"] == "nn_descent"


def write_arm_result(
    root: Path,
    method: str,
    effective_method: str,
    graph: np.ndarray,
    steady_time: float,
) -> None:
    run_root = root / "tenx" / method / "n_000000003"
    run_root.mkdir(parents=True)
    np.save(run_root / "queries.npy", np.arange(3, dtype=np.int32))
    np.save(run_root / "neighbors.npy", graph)
    parameters = {
        "backend": "cuvs",
        "knn_backend": "cuvs",
        "cuvs_index_type": "auto",
        "cuvs_knn_method": effective_method,
        "n_components": 2,
        "n_neighbors": 2,
        "init": "pca",
        "max_iter_layout": 128,
        "random_state": 42,
        "normalize": False,
        "verbose": False,
    }
    if effective_method == "all_neighbors":
        parameters.update(
            {
                "all_neighbors_algo": "nn_descent",
                "all_neighbors_n_clusters": 1,
                "all_neighbors_overlap_factor": 0,
            }
        )
    records = []
    for repeat in range(3):
        effective_index = "ivf_flat" if effective_method == "index_search" else None
        effective_algo = (
            "nn_descent" if effective_method == "all_neighbors" else None
        )
        records.append(
            {
                "seed": 42 + repeat,
                "reducer_diagnostics": {
                    "public_diagnostics": {"contract": "present"},
                    "effective_cuvs_knn_method": effective_method,
                    "effective_cuvs_index_type": effective_index,
                    "effective_all_neighbors_algo": effective_algo,
                    "chunked_force_fallback_calls": 0,
                    "stage_timings_sec": {
                        "knn_graph": steady_time - 1.0,
                        "initialization": 0.25,
                        "layout": 0.75,
                    },
                },
            }
        )
    payload = {
        "status": "success",
        "method_parameters": parameters,
        "records": records,
        "timing": {
            "cold_start_sec": steady_time + 1.0,
            "steady_mean_sec": steady_time,
            "steady_std_sec": 0.1,
            "steady_n": 2,
        },
        "gpu_memory": {
            "peak_used_bytes_max": 100,
            "peak_incremental_bytes_max": 80,
        },
        "knn_graph_audit": {
            "query_indices_file": "queries.npy",
            "neighbor_indices_file": "neighbors.npy",
            "k": int(graph.shape[1]),
        },
    }
    (run_root / "result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_summarize_pair_validates_contract_and_graph_overlap(tmp_path) -> None:
    index_graph = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
    all_graph = np.asarray([[1, 2], [2, 0], [1, 0]], dtype=np.int32)
    write_arm_result(
        tmp_path,
        "dire_index_search",
        "index_search",
        index_graph,
        12.0,
    )
    write_arm_result(
        tmp_path,
        "dire_all_neighbors",
        "all_neighbors",
        all_graph,
        4.0,
    )

    result = summarize_pair(tmp_path, "tenx", 3)

    assert result["status"] == "validated"
    assert result["steady_speedup_index_search_over_all_neighbors"] == 3.0
    assert result["knn_graph_overlap"]["mean"] == 1.0
    assert result["fit_seeds"] == [42, 43, 44]


def quality_payload(offset: float) -> dict:
    topology = {
        key: {"mean": offset}
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
        "local": {"neighbor_overlap": {"mean": 0.8 + offset}},
        "global": {
            "pairwise_distance_spearman": {"correlation": 0.7 + offset},
            "centroid_adjacency_recall": {"mean": 0.6 + offset},
        },
        "context_classifier": {"balanced_accuracy": 0.5 + offset},
        "topology": {"aggregate": topology},
    }


def test_full_quality_audit_uses_explicit_arm_names(tmp_path) -> None:
    graph = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
    write_arm_result(
        tmp_path,
        "dire_index_search",
        "index_search",
        graph,
        12.0,
    )
    write_arm_result(
        tmp_path,
        "dire_all_neighbors",
        "all_neighbors",
        graph,
        4.0,
    )

    audit = build_cuvs_knn_ab_audit(
        "tenx",
        3,
        tmp_path,
        {
            "dire_index_search": quality_payload(0.0),
            "dire_all_neighbors": quality_payload(0.1),
        },
    )

    assert audit is not None
    assert audit["index_search"]["requested_cuvs_knn_method"] == "index_search"
    assert audit["all_neighbors"]["requested_cuvs_knn_method"] == "all_neighbors"
    assert audit["steady_speedup_index_search_over_all_neighbors"] == 3.0
    local = audit["full_embedding_quality_gate"]["metrics"][
        "local_neighbor_overlap"
    ]
    assert local["all_neighbors_minus_index_search"] == pytest.approx(0.1)
