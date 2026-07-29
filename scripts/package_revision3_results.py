#!/usr/bin/env python3
"""Create a compact, checksummed Revision 3 result bundle.

The bundle contains everything needed to render figures and tables locally, but
deliberately excludes the raw 10x archive and the high-dimensional prepared
matrices.  Those public inputs are re-fetched only when rerunning reducers.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

from revision3_source_manifest import build_source_manifest


FULL_SIZES = {"tenx": 1_306_127, "arxiv": 723_457}
METHODS = ("dire_auto", "dire", "cuml_umap", "cuml_tsne", "pca2")
PREPARED_FILES = (
    "manifest.json",
    "tenx_manifest.json",
    "tenx_mouse_brain_graph_clusters.npy",
    "tenx_mouse_brain_kmeans20.npy",
    "tenx_mouse_brain_cellranger_tsne.npy",
    "tenx_mouse_brain_kmeans20_top_markers.json",
    "arxiv_manifest.json",
    "arxiv_primary_category_codes.npy",
    "arxiv_primary_category_names.json",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required result artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_files(source_root: Path, destination_root: Path) -> None:
    if not source_root.exists():
        return
    for source in sorted(source_root.rglob("*")):
        if source.is_file():
            copy_file(source, destination_root / source.relative_to(source_root))


def include_exact_source_snapshot(
    environment: dict,
    bundle_root: Path,
) -> None:
    recorded = environment.get("revision3_source_files")
    current = build_source_manifest(Path("."))
    if not isinstance(recorded, list):
        raise RuntimeError(
            "benchmark environment does not contain a source-file manifest"
        )
    recorded_by_path = {
        str(item["path"]): item
        for item in recorded
        if isinstance(item, dict) and "path" in item
    }
    current_by_path = {item["path"]: item for item in current}
    if recorded_by_path != current_by_path:
        changed = sorted(
            path
            for path in set(recorded_by_path) | set(current_by_path)
            if recorded_by_path.get(path) != current_by_path.get(path)
        )
        raise RuntimeError(
            "Revision 3 source changed after the benchmark environment was "
            f"recorded: {changed}"
        )
    snapshot_root = bundle_root / "source"
    for record in current:
        copy_file(
            Path(record["path"]),
            snapshot_root / record["path"],
        )
    write_json(
        snapshot_root / "source-manifest.json",
        {
            "schema_version": 1,
            "policy": (
                "Exact source files used by the benchmark; each hash was "
                "recorded before reducer execution and rechecked at packaging."
            ),
            "git_head": environment.get("git_head"),
            "git_status_short": environment.get("git_status_short"),
            "git_remote_origin": environment.get("git_remote_origin"),
            "files": current,
        },
    )


def include_large_results(
    large_root: Path,
    bundle_root: Path,
) -> None:
    for name in ("environment.json", "manifest.json"):
        copy_file(large_root / name, bundle_root / "large_results" / name)
    for dataset, full_size in FULL_SIZES.items():
        dataset_root = large_root / dataset
        for result_file in sorted(dataset_root.glob("*/n_*/result.json")):
            relative = result_file.relative_to(large_root)
            copy_file(result_file, bundle_root / "large_results" / relative)
            worker_log = result_file.parent / "worker.log"
            if worker_log.exists():
                copy_file(
                    worker_log,
                    bundle_root
                    / "large_results"
                    / worker_log.relative_to(large_root),
                )
        for method in METHODS:
            full_root = dataset_root / method / f"n_{full_size:09d}"
            result_path = full_root / "result.json"
            if not result_path.exists():
                continue
            result = read_json(result_path)
            if result.get("status") != "success":
                continue
            embedding_path = full_root / result["embedding_file"]
            copy_file(
                embedding_path,
                bundle_root
                / "large_results"
                / embedding_path.relative_to(large_root),
            )
            audit = result.get("knn_graph_audit")
            if isinstance(audit, dict):
                for key in ("query_indices_file", "neighbor_indices_file"):
                    filename = audit.get(key)
                    if not isinstance(filename, str):
                        raise RuntimeError(
                            f"{dataset}/{method} has invalid {key}"
                        )
                    audit_path = full_root / filename
                    copy_file(
                        audit_path,
                        bundle_root
                        / "large_results"
                        / audit_path.relative_to(large_root),
                    )


def include_topology_sensitivity_results(
    sensitivity_root: Path,
    bundle_root: Path,
) -> None:
    destination_root = bundle_root / "topology_sensitivity_results"
    for name in ("environment.json", "manifest.json"):
        copy_file(sensitivity_root / name, destination_root / name)
    for dataset, full_size in FULL_SIZES.items():
        dataset_root = sensitivity_root / dataset
        for result_file in sorted(dataset_root.glob("*/n_*/result.json")):
            copy_file(
                result_file,
                destination_root / result_file.relative_to(sensitivity_root),
            )
            worker_log = result_file.parent / "worker.log"
            if worker_log.exists():
                copy_file(
                    worker_log,
                    destination_root
                    / worker_log.relative_to(sensitivity_root),
                )
        for method in (
            "dire_auto",
            "dire",
            "dire_spectral",
            "dire_topology",
            "cuml_umap",
            "cuml_tsne",
        ):
            full_root = dataset_root / method / f"n_{full_size:09d}"
            result_path = full_root / "result.json"
            if not result_path.exists():
                continue
            result = read_json(result_path)
            if result.get("status") != "success":
                continue
            embedding_names = {result["embedding_file"]}
            embedding_names.update(
                record["embedding_file"]
                for record in result.get("records", [])
                if isinstance(record.get("embedding_file"), str)
            )
            for embedding_name in sorted(embedding_names):
                embedding_path = full_root / embedding_name
                copy_file(
                    embedding_path,
                    destination_root
                    / embedding_path.relative_to(sensitivity_root),
                )
            audit = result.get("knn_graph_audit")
            if isinstance(audit, dict):
                for key in ("query_indices_file", "neighbor_indices_file"):
                    filename = audit.get(key)
                    if not isinstance(filename, str):
                        raise RuntimeError(
                            f"{dataset}/{method} has invalid {key}"
                        )
                    audit_path = full_root / filename
                    copy_file(
                        audit_path,
                        destination_root
                        / audit_path.relative_to(sensitivity_root),
                    )


def include_backend_policy_results(
    profile_root: Path,
    bundle_root: Path,
) -> None:
    destination_root = bundle_root / "backend_policy_results"
    for name in ("environment.json", "manifest.json"):
        copy_file(profile_root / name, destination_root / name)
    for dataset, full_size in FULL_SIZES.items():
        for method in ("dire_auto", "dire_ivf_flat_control"):
            run_root = (
                profile_root
                / dataset
                / method
                / f"n_{full_size:09d}"
            )
            result_path = run_root / "result.json"
            copy_file(
                result_path,
                destination_root / result_path.relative_to(profile_root),
            )
            worker_log = run_root / "worker.log"
            if worker_log.exists():
                copy_file(
                    worker_log,
                    destination_root / worker_log.relative_to(profile_root),
                )
            result = read_json(result_path)
            if result.get("status") != "success":
                continue
            filenames = {result["embedding_file"]}
            audit = result.get("knn_graph_audit")
            if isinstance(audit, dict):
                filenames.update(
                    audit[key]
                    for key in (
                        "query_indices_file",
                        "neighbor_indices_file",
                    )
                )
            for filename in sorted(filenames):
                source = run_root / filename
                copy_file(
                    source,
                    destination_root / source.relative_to(profile_root),
                )


def build_file_manifest(bundle_root: Path) -> list[dict]:
    records = []
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file() and path.name != "bundle-manifest.json":
            records.append(
                {
                    "path": path.relative_to(bundle_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def deterministic_tar(bundle_root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    # tarfile's ``w:gz`` mode writes the current time into the gzip header.
    # Construct that layer explicitly so identical inputs produce identical
    # archive bytes, not merely identical tar members.
    with archive.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as output:
                for path in sorted(bundle_root.rglob("*")):
                    arcname = Path(bundle_root.name) / path.relative_to(bundle_root)
                    info = output.gettarinfo(str(path), arcname=str(arcname))
                    # Stable ownership makes archives comparable across hosts.
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            output.addfile(info, handle)
                    else:
                        output.addfile(info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision3-root",
        type=Path,
        default=Path("data/revision3"),
    )
    parser.add_argument(
        "--small-json-root",
        type=Path,
        default=Path("data/revision3/small_json_logs"),
    )
    parser.add_argument(
        "--small-embedding-root",
        type=Path,
        default=Path("data/revision3/small_embedding_pngs"),
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("data/revision3/ablation_results"),
    )
    parser.add_argument(
        "--bundles-root",
        type=Path,
        default=Path("data/revision3/bundles"),
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    environment = read_json(args.revision3_root / "large_results" / "environment.json")
    timestamp = environment.get("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    default_run_id = (
        timestamp.replace(":", "").replace("-", "").replace("T", "-").replace("Z", "")
    )
    run_id = args.run_id or default_run_id
    staging_parent = args.revision3_root / "bundle_staging"
    bundle_root = staging_parent / f"revision3-results-{run_id}"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)
    include_exact_source_snapshot(environment, bundle_root)

    prepared_root = args.revision3_root / "prepared"
    for name in PREPARED_FILES:
        copy_file(prepared_root / name, bundle_root / "prepared" / name)
    include_large_results(
        args.revision3_root / "large_results",
        bundle_root,
    )
    include_backend_policy_results(
        args.revision3_root / "backend_policy_results",
        bundle_root,
    )
    include_topology_sensitivity_results(
        args.revision3_root / "topology_sensitivity_results",
        bundle_root,
    )
    copy_tree_files(
        args.revision3_root / "evaluation",
        bundle_root / "evaluation",
    )
    copy_tree_files(args.small_json_root, bundle_root / "small_suite" / "json_logs")
    copy_tree_files(
        args.small_embedding_root,
        bundle_root / "small_suite" / "embedding_pngs",
    )
    copy_tree_files(args.ablation_root, bundle_root / "ablation")

    readme = """# Revision 3 benchmark result bundle

This compact bundle contains machine-readable run logs, full layouts from the
declared d=2 benchmark profile, released reference labels, auditable topology
reference subsets and diagrams, topology/context/local evaluation records, a
separately named production-auto versus forced-IVF-Flat backend-policy profile,
small-suite artifacts, and the exact checksummed runner source used on the
benchmark host. It intentionally excludes raw public downloads and full
high-dimensional input matrices.

Render figures and TeX tables from the repository root:

    python3 scripts/render_revision3_artifacts.py \\
      --bundle-root /path/to/this/directory \\
      --output-root generated/revision3

Every included file is covered by `bundle-manifest.json`; the outer archive has
an additional SHA-256 sidecar.
"""
    (bundle_root / "README.md").write_text(readme, encoding="utf-8")
    file_records = build_file_manifest(bundle_root)
    bundle_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "Compact result bundle for the Scientific Reports Revision 3 "
            "reproduction and local manuscript build."
        ),
        "exclusions": [
            "raw 10x archive",
            "prepared 1.3M x 20 10x input matrix",
            "prepared 723457 x 384 arXiv input matrix",
            "Hugging Face cache",
        ],
        "files": file_records,
        "total_bytes": sum(record["bytes"] for record in file_records),
    }
    write_json(bundle_root / "bundle-manifest.json", bundle_manifest)

    args.bundles_root.mkdir(parents=True, exist_ok=True)
    archive = args.bundles_root / f"{bundle_root.name}.tar.gz"
    deterministic_tar(bundle_root, archive)
    archive_hash = sha256_file(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    (args.bundles_root / "LATEST").write_text(
        f"{archive.name}\n",
        encoding="utf-8",
    )
    print(f"bundle: {archive}")
    print(f"sha256: {archive_hash}")
    print(f"uncompressed declared bytes: {bundle_manifest['total_bytes']:,}")


if __name__ == "__main__":
    main()
