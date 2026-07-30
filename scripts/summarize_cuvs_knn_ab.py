#!/usr/bin/env python3
"""Validate and summarize the cuVS graph-construction A/B experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evaluate_large_embeddings import (
    fixed_query_graph_overlap,
    summarize_stage_timings,
)


ARMS = ("dire_index_search", "dire_all_neighbors")
EXPECTED_EFFECTIVE_METHOD = {
    "dire_index_search": "index_search",
    "dire_all_neighbors": "all_neighbors",
}
GRAPH_PARAMETERS = {
    "cuvs_knn_method",
    "all_neighbors_algo",
    "all_neighbors_n_clusters",
    "all_neighbors_overlap_factor",
    "all_neighbors_device_ids",
    "all_neighbors_algo_params",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def invariant_parameters(result: dict) -> dict:
    return {
        key: value
        for key, value in result.get("method_parameters", {}).items()
        if key not in GRAPH_PARAMETERS
    }


def load_graph(run_root: Path, result: dict) -> tuple[np.ndarray, np.ndarray]:
    audit = result.get("knn_graph_audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{run_root} is missing the k-NN graph audit")
    query = np.load(run_root / audit["query_indices_file"])
    graph = np.load(run_root / audit["neighbor_indices_file"])
    if query.ndim != 1 or graph.shape != (len(query), int(audit["k"])):
        raise RuntimeError(
            f"{run_root} graph audit has inconsistent shapes "
            f"{query.shape} and {graph.shape}"
        )
    return query, graph


def validate_arm(method: str, result: dict) -> dict:
    parameters = result.get("method_parameters", {})
    expected = EXPECTED_EFFECTIVE_METHOD[method]
    if parameters.get("cuvs_knn_method") != expected:
        raise RuntimeError(
            f"{method} requested {parameters.get('cuvs_knn_method')!r}, "
            f"expected {expected!r}"
        )
    records = result.get("records", [])
    if not records:
        raise RuntimeError(f"{method} has no repeat records")
    diagnostics = [
        record.get("reducer_diagnostics", {}) for record in records
    ]
    for repeat, item in enumerate(diagnostics):
        if not isinstance(item.get("public_diagnostics"), dict):
            raise RuntimeError(
                f"{method} repeat {repeat} lacks public DiRe diagnostics"
            )
        if item.get("effective_cuvs_knn_method") != expected:
            raise RuntimeError(
                f"{method} repeat {repeat} used "
                f"{item.get('effective_cuvs_knn_method')!r}, expected "
                f"{expected!r}"
            )
        if int(item.get("chunked_force_fallback_calls", 0)) != 0:
            raise RuntimeError(
                f"{method} repeat {repeat} used the chunked-force fallback"
            )
        if method == "dire_index_search":
            if not item.get("effective_cuvs_index_type"):
                raise RuntimeError(
                    f"{method} repeat {repeat} did not report an effective index"
                )
        elif item.get("effective_all_neighbors_algo") != "nn_descent":
            raise RuntimeError(
                f"{method} repeat {repeat} did not use NN-descent"
            )
    return {
        "requested_parameters": parameters,
        "effective_cuvs_knn_methods": [
            item["effective_cuvs_knn_method"] for item in diagnostics
        ],
        "effective_cuvs_index_types": [
            item.get("effective_cuvs_index_type") for item in diagnostics
        ],
        "effective_all_neighbors_algos": [
            item.get("effective_all_neighbors_algo") for item in diagnostics
        ],
        "fit_seeds": [int(record["seed"]) for record in records],
        "cold_total_sec": float(result["timing"]["cold_start_sec"]),
        "steady_total_mean_sec": float(result["timing"]["steady_mean_sec"]),
        "steady_total_std_sec": float(result["timing"]["steady_std_sec"]),
        "steady_replicate_count": int(result["timing"]["steady_n"]),
        "stage_timings": summarize_stage_timings(result),
        "peak_used_bytes_max": result["gpu_memory"]["peak_used_bytes_max"],
        "peak_incremental_bytes_max": result["gpu_memory"][
            "peak_incremental_bytes_max"
        ],
    }


def summarize_pair(
    results_root: Path,
    dataset: str,
    size: int,
) -> dict:
    results = {}
    roots = {}
    for method in ARMS:
        run_root = results_root / dataset / method / f"n_{size:09d}"
        result_path = run_root / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"missing A/B result: {result_path}")
        result = read_json(result_path)
        if result.get("status") != "success":
            raise RuntimeError(
                f"{dataset}/{method}/n={size} status is "
                f"{result.get('status')!r}: {result.get('error')}"
            )
        results[method] = result
        roots[method] = run_root

    index_result = results["dire_index_search"]
    all_result = results["dire_all_neighbors"]
    index_invariants = invariant_parameters(index_result)
    all_invariants = invariant_parameters(all_result)
    if index_invariants != all_invariants:
        raise RuntimeError(
            f"{dataset}/n={size} changed non-graph parameters between arms"
        )
    index_seeds = [
        int(record["seed"]) for record in index_result["records"]
    ]
    all_seeds = [int(record["seed"]) for record in all_result["records"]]
    if index_seeds != all_seeds:
        raise RuntimeError(f"{dataset}/n={size} repeat seeds differ")

    index_query, index_graph = load_graph(
        roots["dire_index_search"],
        index_result,
    )
    all_query, all_graph = load_graph(
        roots["dire_all_neighbors"],
        all_result,
    )
    if not np.array_equal(index_query, all_query):
        raise RuntimeError(f"{dataset}/n={size} graph-audit queries differ")

    index_arm = validate_arm("dire_index_search", index_result)
    all_arm = validate_arm("dire_all_neighbors", all_result)
    return {
        "dataset": dataset,
        "n_samples": int(size),
        "status": "validated",
        "held_fixed_parameters": index_invariants,
        "fit_seeds": index_seeds,
        "index_search": index_arm,
        "all_neighbors": all_arm,
        "steady_speedup_index_search_over_all_neighbors": (
            index_arm["steady_total_mean_sec"]
            / all_arm["steady_total_mean_sec"]
        ),
        "knn_graph_overlap": fixed_query_graph_overlap(
            index_graph,
            all_graph,
        ),
    }


def evaluation_audits(
    evaluation_root: Path,
    datasets: list[str],
    required: bool,
) -> dict:
    audits = {}
    for dataset in datasets:
        path = evaluation_root / dataset / "evaluation.json"
        if not path.is_file():
            if required:
                raise RuntimeError(f"missing full quality evaluation: {path}")
            continue
        audit = read_json(path).get("cuvs_knn_ab_audit")
        if not isinstance(audit, dict):
            raise RuntimeError(f"{path} has no cuVS k-NN A/B quality audit")
        if audit.get("full_embedding_quality_gate", {}).get("status") != "evaluated":
            raise RuntimeError(f"{path} has an incomplete A/B quality gate")
        audits[dataset] = audit
    return audits


def render_markdown(payload: dict) -> str:
    lines = [
        "# cuVS k-NN A/B summary",
        "",
        "The speedup is index-search steady time divided by all-neighbors "
        "steady time; values above 1 favor all-neighbors.",
        "",
        "| Dataset | Rows | Index-search (s) | All-neighbors (s) | Speedup | "
        "Mean graph overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in payload["profiles"]:
        lines.append(
            f"| {record['dataset']} | {record['n_samples']:,} | "
            f"{record['index_search']['steady_total_mean_sec']:.3f} | "
            f"{record['all_neighbors']['steady_total_mean_sec']:.3f} | "
            f"{record['steady_speedup_index_search_over_all_neighbors']:.3f} | "
            f"{record['knn_graph_overlap']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Full quality audits present: "
            f"{', '.join(sorted(payload['full_quality_audits'])) or 'none'}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/cuvs-knn-ab/results"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("data/cuvs-knn-ab/evaluation"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cuvs-knn-ab/summary"),
    )
    parser.add_argument("--datasets", nargs="+", default=["tenx", "arxiv"])
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="Every requested size must exist for every selected dataset.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        help=(
            "Explicit dataset:size pairs, for example "
            "tenx:100000 arxiv:723457."
        ),
    )
    parser.add_argument("--require-evaluation", action="store_true")
    args = parser.parse_args()

    if args.profiles:
        requested_profiles = []
        for value in args.profiles:
            try:
                dataset, size_text = value.split(":", 1)
                size = int(size_text)
            except ValueError as exc:
                raise ValueError(
                    f"invalid --profiles value {value!r}; use dataset:size"
                ) from exc
            if dataset not in args.datasets or size < 1:
                raise ValueError(f"invalid --profiles value {value!r}")
            requested_profiles.append((dataset, size))
    elif args.sizes:
        requested_profiles = [
            (dataset, size)
            for dataset in args.datasets
            for size in args.sizes
        ]
    else:
        parser.error("one of --sizes or --profiles is required")
    profiles = [
        summarize_pair(args.results_root, dataset, size)
        for dataset, size in requested_profiles
    ]
    payload = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparison": (
            "Explicit cuVS index-search versus in-core single-cluster cuVS "
            "all-neighbors NN-descent"
        ),
        "profiles": profiles,
        "full_quality_audits": evaluation_audits(
            args.evaluation_root,
            args.datasets,
            args.require_evaluation,
        ),
    }
    output_json = args.output_root / "summary.json"
    write_json(output_json, payload)
    output_markdown = args.output_root / "summary.md"
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_markdown.with_suffix(".md.tmp")
    temporary.write_text(render_markdown(payload), encoding="utf-8")
    temporary.replace(output_markdown)
    print(f"validated {len(profiles)} paired profiles")
    print(f"wrote {output_json}")
    print(f"wrote {output_markdown}")


if __name__ == "__main__":
    main()
