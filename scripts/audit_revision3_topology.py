#!/usr/bin/env python3
"""Recompute every bundled large-data topology score without the raw datasets.

The compact Revision 3 bundle contains the fixed high-dimensional reference
subsets, their row indices, and every full layout from the declared d=2
benchmark profile. The profile dimension is not a restriction of DiRe's
user-specified target dimension. This script re-runs Ripser, exact GUDHI
H1 bottleneck distance, the exact sorted zero-birth-bar H0 specialization,
and Betti-curve DTW from those inputs and checks the results against the
bundled evaluation JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from evaluate_large_embeddings import topology_pair, topology_reference


FULL_SIZES = {"tenx": 1_306_127, "arxiv": 723_457}
METRIC_KEYS = (
    "dtw_beta0",
    "dtw_beta1",
    "bottleneck_beta0",
    "bottleneck_beta1",
    "q99_dtw_beta0",
    "q99_dtw_beta1",
    "q99_bottleneck_beta0",
    "q99_bottleneck_beta1",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("numpy", "ripser", "gudhi", "fastdtw", "joblib"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def layout_path(bundle_root: Path, dataset: str, method: str) -> Path:
    if method == "cellranger_tsne":
        return (
            bundle_root
            / "prepared"
            / "tenx_mouse_brain_cellranger_tsne.npy"
        )
    result_directory = (
        "topology_sensitivity_results"
        if method in ("dire_spectral", "dire_topology")
        else "large_results"
    )
    run_root = (
        bundle_root
        / result_directory
        / dataset
        / method
        / f"n_{FULL_SIZES[dataset]:09d}"
    )
    result = read_json(run_root / "result.json")
    if result.get("status") != "success":
        raise RuntimeError(
            f"bundled layout run is not successful: {dataset}/{method}"
        )
    return run_root / result["embedding_file"]


def close_enough(
    actual: float,
    expected: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def compare_metrics(
    actual_record: dict,
    expected_record: dict,
    relative_tolerance: float,
    absolute_tolerance: float,
    mismatch_context: dict,
    mismatches: list[dict],
) -> dict:
    comparisons = {}
    for key in METRIC_KEYS:
        actual = float(actual_record[key])
        expected = float(expected_record[key])
        matches = close_enough(
            actual,
            expected,
            relative_tolerance,
            absolute_tolerance,
        )
        comparisons[key] = {
            "actual": actual,
            "expected": expected,
            "absolute_delta": abs(actual - expected),
            "matches": matches,
        }
        if not matches:
            mismatches.append(
                {
                    **mismatch_context,
                    "metric": key,
                    **comparisons[key],
                }
            )
    return comparisons


def compute_topology_pairs(
    jobs: list[tuple[dict, np.ndarray, int]],
    workers: int,
) -> list[dict]:
    """Run independent topology comparisons with deterministic result order."""
    if workers < 1:
        raise ValueError("topology audit workers must be positive")
    if not jobs:
        return []
    worker_count = min(workers, len(jobs))
    if worker_count == 1:
        return [
            topology_pair(reference, points, n_steps)
            for reference, points, n_steps in jobs
        ]
    from joblib import Parallel, delayed

    return Parallel(
        n_jobs=worker_count,
        backend="loky",
    )(
        delayed(topology_pair)(reference, points, n_steps)
        for reference, points, n_steps in jobs
    )


def audit_dataset(
    bundle_root: Path,
    dataset: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    workers: int,
) -> dict:
    evaluation_root = bundle_root / "evaluation" / dataset
    evaluation = read_json(evaluation_root / "evaluation.json")
    methods = evaluation["methods"]
    method_layouts = {
        method: np.load(
            layout_path(bundle_root, dataset, method),
            mmap_mode="r",
        )
        for method in methods
    }
    output = {
        "dataset": dataset,
        "methods": {},
        "mismatches": [],
    }
    first_method = next(iter(methods.values()))
    subset_metadata = first_method["topology"]["subset_metadata"]
    reference_by_subset = {}
    primary_jobs = []
    primary_job_metadata = []
    for metadata in subset_metadata:
        subset_id = int(metadata["subset_id"])
        audit_file = evaluation_root / metadata["audit_file"]
        with np.load(audit_file) as archive:
            points = np.asarray(archive["points"], dtype=np.float32)
            indices = np.asarray(archive["indices"])
        separate_indices = np.load(
            evaluation_root / f"topology_subset_{subset_id}.npy"
        )
        if not np.array_equal(indices, separate_indices):
            raise RuntimeError(
                f"{dataset} topology subset {subset_id} index files disagree"
            )
        reference = topology_reference(points)
        reference_by_subset[subset_id] = {
            "reference": reference,
            "indices": indices,
        }
        for method, method_payload in methods.items():
            records = method_payload["topology"]["records"]
            expected_record = next(
                record
                for record in records
                if int(record["subset_id"]) == subset_id
            )
            primary_jobs.append(
                (
                    reference,
                    np.ascontiguousarray(
                        method_layouts[method][indices],
                        dtype=np.float32,
                    ),
                    int(expected_record["n_steps"]),
                )
            )
            primary_job_metadata.append(
                (method, subset_id, expected_record)
            )
    actual_records = compute_topology_pairs(primary_jobs, workers)
    for (method, subset_id, expected_record), actual_record in zip(
        primary_job_metadata,
        actual_records,
    ):
        comparisons = compare_metrics(
            actual_record,
            expected_record,
            relative_tolerance,
            absolute_tolerance,
            {"method": method, "topology_subset_id": subset_id},
            output["mismatches"],
        )
        method_output = output["methods"].setdefault(
            method,
            {
                "topology_subset_sensitivity": {},
                "subset_size_sensitivity": {},
                "layout_seed_sensitivity": {},
            },
        )
        method_output["topology_subset_sensitivity"][
            str(subset_id)
        ] = comparisons

    fixed = reference_by_subset[0]
    size_metadata = first_method["topology"]["subset_size_sensitivity"][
        "subset_metadata"
    ]
    size_references = {}
    previous_indices = None
    previous_size = None
    for metadata in size_metadata:
        size = int(metadata["subset_size"])
        audit_file = evaluation_root / metadata["audit_file"]
        with np.load(audit_file) as archive:
            points = np.asarray(archive["points"], dtype=np.float32)
            indices = np.asarray(archive["indices"])
        separate_indices = np.load(
            evaluation_root / metadata["subset_file"]
        )
        if not np.array_equal(indices, separate_indices):
            raise RuntimeError(
                f"{dataset} topology size {size} index files disagree"
            )
        if len(indices) != size:
            raise RuntimeError(
                f"{dataset} topology size metadata says {size}, "
                f"but the bundled sample contains {len(indices)} rows"
            )
        if previous_indices is not None and not np.isin(
            previous_indices,
            indices,
        ).all():
            raise RuntimeError(
                f"{dataset} topology subset m={previous_size} is not nested "
                f"inside m={size}"
            )
        size_references[size] = {
            "reference": topology_reference(points),
            "indices": indices,
        }
        previous_indices = indices
        previous_size = size
    largest_size = max(size_references)
    if not np.array_equal(
        size_references[largest_size]["indices"],
        fixed["indices"],
    ):
        raise RuntimeError(
            f"{dataset} largest nested topology subset is not primary subset 0"
        )

    jobs = []
    job_metadata = []
    for method, method_payload in methods.items():
        sensitivity = method_payload["topology"]["subset_size_sensitivity"]
        for expected_record in sensitivity["records"]:
            size = int(expected_record["sensitivity_subset_size"])
            subset = size_references[size]
            jobs.append(
                (
                    subset["reference"],
                    np.ascontiguousarray(
                        method_layouts[method][subset["indices"]],
                        dtype=np.float32,
                    ),
                    int(expected_record["n_steps"]),
                )
            )
            job_metadata.append((method, size, expected_record))
    actual_records = compute_topology_pairs(jobs, workers)
    for (method, size, expected_record), actual_record in zip(
        job_metadata,
        actual_records,
    ):
        comparisons = compare_metrics(
            actual_record,
            expected_record,
            relative_tolerance,
            absolute_tolerance,
            {"method": method, "topology_subset_size": size},
            output["mismatches"],
        )
        output["methods"][method]["subset_size_sensitivity"][
            str(size)
        ] = comparisons

    jobs = []
    job_metadata = []
    for method, method_payload in methods.items():
        sensitivity = method_payload["topology"].get(
            "layout_seed_sensitivity"
        )
        if not sensitivity:
            continue
        for expected_record in sensitivity["records"]:
            run_root = (
                bundle_root
                / "topology_sensitivity_results"
                / dataset
                / method
                / f"n_{FULL_SIZES[dataset]:09d}"
            )
            embedding = np.load(
                run_root / expected_record["layout_embedding_file"],
                mmap_mode="r",
            )
            jobs.append(
                (
                    fixed["reference"],
                    np.ascontiguousarray(
                        embedding[fixed["indices"]],
                        dtype=np.float32,
                    ),
                    int(expected_record["n_steps"]),
                )
            )
            job_metadata.append((method, expected_record))
    actual_records = compute_topology_pairs(jobs, workers)
    for (method, expected_record), actual_record in zip(
        job_metadata,
        actual_records,
    ):
        layout_seed = int(expected_record["layout_seed"])
        comparisons = compare_metrics(
            actual_record,
            expected_record,
            relative_tolerance,
            absolute_tolerance,
            {"method": method, "layout_seed": layout_seed},
            output["mismatches"],
        )
        output["methods"][method]["layout_seed_sensitivity"][
            str(layout_seed)
        ] = comparisons
    output["status"] = "success" if not output["mismatches"] else "mismatch"
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(FULL_SIZES),
        default=list(FULL_SIZES),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/revision3/topology-audit.json"),
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=(
            "independent topology comparisons to run concurrently; result "
            "ordering and comparisons remain deterministic"
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    started = time.perf_counter()
    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 2,
        "source_bundle": args.bundle_root.name,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "audit_environment": {
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "workers": args.workers,
            "dependency_versions": dependency_versions(),
        },
        "datasets": {},
    }
    for dataset in args.datasets:
        print(f"auditing bundled topology: {dataset}", flush=True)
        payload["datasets"][dataset] = audit_dataset(
            args.bundle_root,
            dataset,
            args.relative_tolerance,
            args.absolute_tolerance,
            args.workers,
        )
    payload["wall_time_sec"] = time.perf_counter() - started
    payload["status"] = (
        "success"
        if all(
            result["status"] == "success"
            for result in payload["datasets"].values()
        )
        else "mismatch"
    )
    write_json(args.output, payload)
    print(f"topology audit: {payload['status']} ({args.output})")
    if payload["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
