#!/usr/bin/env python3
"""Run a small blobs-only sweep over topology sample fractions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


METHODS = ("dire", "cuml_tsne", "cuml_umap", "umap")


def fraction_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def metric(record: dict, method: str, key: str):
    method_record = record.get("methods", {}).get(method, {})
    aggregate = method_record.get("aggregate", {})
    if key in aggregate:
        return aggregate[key].get("mean")
    return None


def sample_size(record: dict, method: str):
    sample = topology_sample(record, method)
    if sample:
        return sample.get("sample_size")
    return None


def topology_sample(record: dict, method: str):
    repeats = record.get("methods", {}).get(method, {}).get("repeats", [])
    for repeat in repeats:
        sample = repeat.get("topology_sample")
        if sample:
            return sample
    return None


def run_fraction(args: argparse.Namespace, fraction: float) -> dict:
    output = args.output / f"blobs_topology_{fraction_tag(fraction)}"
    command = [
        sys.executable,
        "scripts/article_benchmarks.py",
        "--output",
        str(output),
        "--datasets",
        "blobs",
        "--max-points",
        str(args.max_points),
        "--metric-subsample",
        str(args.metric_subsample),
        "--topology-sample-fraction",
        str(fraction),
        "--topology-steps",
        str(args.topology_steps),
        "--seed",
        str(args.seed),
        "--repeats",
        str(args.repeats),
        "--topology-repeats",
        str(args.topology_repeats),
    ]
    print(f"\n=== blobs topology sample fraction {fraction:g} ===", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    wall_time = time.perf_counter() - started
    with open(output / "blobs" / "results.json", "r", encoding="utf-8") as handle:
        record = json.load(handle)
    row = {
        "fraction": fraction,
        "output": str(output),
        "wall_time_sec": wall_time,
        "methods": {},
    }
    for method in METHODS:
        sample = topology_sample(record, method) or {}
        row["methods"][method] = {
            "sample_size": sample.get("sample_size"),
            "topology_wall_time_sec": sample.get("wall_time_sec"),
            "runtime_sec": metric(record, method, "time"),
            "dtw_beta0": metric(record, method, "persistence-dim-0"),
            "dtw_beta1": metric(record, method, "persistence-dim-1"),
        }
    return row


def projection(summary: list[dict], datasets: int, methods: int, topology_repeats: int) -> list[dict]:
    rows = []
    calls = datasets * methods * topology_repeats
    for row in summary:
        times = [
            values.get("topology_wall_time_sec")
            for values in row["methods"].values()
            if values.get("topology_wall_time_sec") is not None
        ]
        if not times:
            rows.append({"fraction": row["fraction"], "projected_topology_hours": None})
            continue
        mean_time = sum(times) / len(times)
        rows.append(
            {
                "fraction": row["fraction"],
                "mean_blobs_topology_call_sec": mean_time,
                "projected_topology_calls": calls,
                "projected_topology_hours": mean_time * calls / 3600.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/topology_sampling_sweep"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--max-points", type=int, default=10_000)
    parser.add_argument("--metric-subsample", type=float, default=0.1)
    parser.add_argument("--topology-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--topology-repeats", type=int, default=1)
    parser.add_argument("--project-datasets", type=int, default=6)
    parser.add_argument("--project-methods", type=int, default=4)
    parser.add_argument("--project-topology-repeats", type=int, default=1)
    parser.add_argument("--warn-topology-hours", type=float, default=1.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = [run_fraction(args, fraction) for fraction in args.fractions]
    projected = projection(
        summary,
        args.project_datasets,
        args.project_methods,
        args.project_topology_repeats,
    )
    summary_path = args.output / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"sweep": summary, "projection": projected}, handle, indent=2)

    print("\nSummary")
    print("fraction method sample_size fit_runtime_sec topology_sec dtw_beta0 dtw_beta1")
    for row in summary:
        for method, values in row["methods"].items():
            print(
                row["fraction"],
                method,
                values["sample_size"],
                values["runtime_sec"],
                values["topology_wall_time_sec"],
                values["dtw_beta0"],
                values["dtw_beta1"],
            )
    print("\nTopology runtime projection")
    print("fraction mean_blobs_topology_call_sec projected_topology_calls projected_topology_hours")
    for row in projected:
        print(
            row["fraction"],
            row.get("mean_blobs_topology_call_sec"),
            row.get("projected_topology_calls"),
            row.get("projected_topology_hours"),
        )
        hours = row.get("projected_topology_hours")
        if hours is not None and hours > args.warn_topology_hours:
            print(
                "WARNING: projected topology runtime exceeds "
                f"{args.warn_topology_hours:g} hours for fraction {row['fraction']}. "
                "Use a smaller topology sample fraction or fewer topology repeats on this device."
            )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
