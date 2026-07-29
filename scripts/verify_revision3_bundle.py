#!/usr/bin/env python3
"""Safely extract and verify a compact Revision 3 result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import tarfile
from pathlib import Path, PurePosixPath


MAX_ARCHIVE_FILES = 100_000
MAX_UNCOMPRESSED_BYTES = 3 * 1024**3
CURRENT_EVALUATION_SCHEMA = 2
CURRENT_TOPOLOGY_SUBSET_COUNT = 10
CURRENT_TOPOLOGY_SUBSET_SIZE = 4_000
CURRENT_NESTED_SUBSET_SIZES = (1_000, 2_000, 4_000)
TOPOLOGY_METRICS = (
    "dtw_beta0",
    "dtw_beta1",
    "bottleneck_beta0",
    "bottleneck_beta1",
    "q99_dtw_beta0",
    "q99_dtw_beta1",
    "q99_bottleneck_beta0",
    "q99_bottleneck_beta1",
)
EXPECTED_METHODS = {
    "tenx": {
        "pca2",
        "dire_auto",
        "dire",
        "dire_spectral",
        "dire_topology",
        "cuml_umap",
        "cuml_tsne",
        "cellranger_tsne",
    },
    "arxiv": {
        "pca2",
        "dire_auto",
        "dire",
        "dire_spectral",
        "dire_topology",
        "cuml_umap",
        "cuml_tsne",
    },
}
FORBIDDEN_CURRENT_SCHEMA_TERMS = (
    "R_" + "sample",
    "R_" + "layout",
    "dire_" + "better_count",
    "competitor_" + "better_count",
    "tie_" + "count",
)
SMALL_DATASETS = (
    "blobs",
    "disk",
    "moons",
    "mnist",
    "levine13",
    "levine32",
)
SMALL_GPU_METHODS = (
    "dire",
    "dire_topology",
    "cuml_tsne",
    "cuml_umap",
)
SMALL_CPU_METHODS = ("opentsne", "umap")
SMALL_METHODS = SMALL_GPU_METHODS + SMALL_CPU_METHODS
SMALL_GPU_REPEATS = 20
SMALL_CPU_REPEATS = 10
SMALL_TOPOLOGY_SUBSET_SIZE = 1_000
SMALL_TOPOLOGY_BACKEND_DETAIL = "direct GPU rank-based local-kNN atlas"
SMALL_TOPOLOGY_PRESET_COMMIT = (
    "9117dc45a3e130fa1d636dfd181f3e97960c5b3b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_path(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts:
        raise RuntimeError(f"unsafe absolute/empty archive path: {name!r}")
    if any(part in ("", ".", "..") for part in value.parts):
        raise RuntimeError(f"unsafe archive path component: {name!r}")
    return value


def read_manifest(bundle_root: Path) -> dict:
    path = bundle_root / "bundle-manifest.json"
    if not path.is_file():
        raise RuntimeError(f"bundle manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise RuntimeError(
            f"unsupported bundle schema: {payload.get('schema_version')!r}"
        )
    return payload


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"required JSON root is not an object: {path}")
    return payload


def forbidden_schema_hits(value, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            for term in FORBIDDEN_CURRENT_SCHEMA_TERMS:
                if term in key_text:
                    hits.append(f"{path}.{key_text}: {term}")
            hits.extend(forbidden_schema_hits(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(forbidden_schema_hits(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        for term in FORBIDDEN_CURRENT_SCHEMA_TERMS:
            if term in value:
                hits.append(f"{path}: {term}")
    return hits


def topology_indices_sha256(indices: list[int]) -> str:
    payload = struct.pack(
        f"<{len(indices)}q",
        *(int(value) for value in indices),
    )
    return hashlib.sha256(payload).hexdigest()


def verify_small_atlas_suite(bundle_root: Path) -> dict:
    """Require the fresh paired atlas protocol and reject stale small paths."""

    log_root = bundle_root / "small_suite" / "json_logs"
    suite_manifest = read_json(log_root / "manifest.json")
    if suite_manifest.get("schema_version") != 2:
        raise RuntimeError("small-suite manifest does not use schema 2")
    if tuple(suite_manifest.get("datasets", ())) != SMALL_DATASETS:
        raise RuntimeError("small-suite manifest has the wrong dataset set/order")
    if tuple(suite_manifest.get("methods", ())) != SMALL_METHODS:
        raise RuntimeError("small-suite manifest has the wrong method set/order")

    dataset_summary = {}
    for dataset in SMALL_DATASETS:
        payload = read_json(log_root / f"{dataset}.json")
        if payload.get("schema_version") != 2:
            raise RuntimeError(f"{dataset} small-suite log does not use schema 2")
        if payload.get("dataset_name") != dataset:
            raise RuntimeError(f"{dataset} small-suite log names another dataset")
        methods = payload.get("methods")
        if not isinstance(methods, dict) or set(methods) != set(SMALL_METHODS):
            raise RuntimeError(
                f"{dataset} small-suite methods differ from the declared set"
            )

        paired_identity: dict[tuple[int, int], tuple[str, tuple[int, ...]]] = {}
        method_counts = {}
        for method in SMALL_METHODS:
            result = methods[method]
            expected_repeats = (
                SMALL_GPU_REPEATS
                if method in SMALL_GPU_METHODS
                else SMALL_CPU_REPEATS
            )
            expected_hardware = (
                "GPU" if method in SMALL_GPU_METHODS else "CPU"
            )
            if result.get("error"):
                raise RuntimeError(
                    f"{dataset}/{method} small-suite run contains an error"
                )
            if result.get("hardware_class") != expected_hardware:
                raise RuntimeError(
                    f"{dataset}/{method} has the wrong hardware class"
                )
            if int(result.get("repeats_requested", -1)) != expected_repeats:
                raise RuntimeError(
                    f"{dataset}/{method} has the wrong requested repeat count"
                )
            if (
                int(result.get("topology_repeats_requested", -1))
                != expected_repeats
            ):
                raise RuntimeError(
                    f"{dataset}/{method} does not compute topology on every run"
                )
            if result.get("topology_backend") != "atlas":
                raise RuntimeError(
                    f"{dataset}/{method} does not declare the atlas backend"
                )
            if (
                result.get("topology_backend_detail")
                != SMALL_TOPOLOGY_BACKEND_DETAIL
                or result.get("topology_prefer_ripser") is not False
            ):
                raise RuntimeError(
                    f"{dataset}/{method} does not prove the direct rank-based "
                    "atlas path"
                )
            if (
                int(result.get("topology_sample_size", -1))
                != SMALL_TOPOLOGY_SUBSET_SIZE
            ):
                raise RuntimeError(
                    f"{dataset}/{method} has the wrong topology subset size"
                )
            if method == "dire_topology":
                origin = result.get("configuration_origin", {})
                if origin.get("name") != "dire_rapids.TOPOLOGY_TUNED":
                    raise RuntimeError(
                        f"{dataset}/dire_topology has no preset provenance"
                    )
                if origin.get("source_commit") != SMALL_TOPOLOGY_PRESET_COMMIT:
                    raise RuntimeError(
                        f"{dataset}/dire_topology has the wrong preset commit"
                    )

            records = result.get("repeats")
            if not isinstance(records, list) or len(records) != expected_repeats:
                raise RuntimeError(
                    f"{dataset}/{method} has an incomplete repeat list"
                )
            seen_seeds: set[int] = set()
            for record in records:
                if record.get("error") or record.get("topology_computed") is not True:
                    raise RuntimeError(
                        f"{dataset}/{method} has a failed/non-topology repeat"
                    )
                seed = int(record["seed"])
                if seed in seen_seeds:
                    raise RuntimeError(
                        f"{dataset}/{method} repeats layout seed {seed}"
                    )
                seen_seeds.add(seed)
                sample = record.get("topology_sample")
                topology = record.get("metrics", {}).get("topology")
                if not isinstance(sample, dict) or not isinstance(topology, dict):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} lacks topology metadata"
                    )
                if (
                    sample.get("backend") != "atlas"
                    or topology.get("backend") != "atlas"
                ):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} retained a non-atlas backend"
                    )
                if (
                    sample.get("backend_detail")
                    != SMALL_TOPOLOGY_BACKEND_DETAIL
                    or topology.get("backend_detail")
                    != SMALL_TOPOLOGY_BACKEND_DETAIL
                    or sample.get("prefer_ripser") is not False
                    or topology.get("prefer_ripser") is not False
                ):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} did not execute the direct "
                        "rank-based atlas path"
                    )
                if sample.get("sample_policy") != (
                    "fixed-size uniform without replacement"
                ):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} has the wrong subset policy"
                    )
                indices = sample.get("indices")
                if (
                    not isinstance(indices, list)
                    or len(indices) != SMALL_TOPOLOGY_SUBSET_SIZE
                    or len(set(int(value) for value in indices))
                    != SMALL_TOPOLOGY_SUBSET_SIZE
                ):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} has invalid subset indices"
                    )
                if int(sample.get("sample_size", -1)) != len(indices):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} has inconsistent subset size"
                    )
                digest = str(sample.get("indices_sha256", ""))
                if digest != topology_indices_sha256(indices):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} subset hash is invalid"
                    )
                subset_seed = int(sample["subset_seed"])
                identity = (seed, subset_seed)
                signature = (digest, tuple(int(value) for value in indices))
                prior = paired_identity.setdefault(identity, signature)
                if prior != signature:
                    raise RuntimeError(
                        f"{dataset} paired subset identity differs across methods"
                    )
                metrics = topology.get("metrics")
                if not isinstance(metrics, dict):
                    raise RuntimeError(
                        f"{dataset}/{method}/{seed} has no topology metrics"
                    )
                for metric in ("dtw_beta0", "dtw_beta1"):
                    value = metrics.get(metric)
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                    ):
                        raise RuntimeError(
                            f"{dataset}/{method}/{seed} has invalid {metric}"
                        )
            method_counts[method] = len(records)
        dataset_summary[dataset] = method_counts
    return {
        "backend": "atlas",
        "topology_subset_size": SMALL_TOPOLOGY_SUBSET_SIZE,
        "datasets": dataset_summary,
    }


def verify_current_revision3_contract(bundle_root: Path) -> dict:
    """Verify the semantic contract required by the pending revision."""
    manifest = read_manifest(bundle_root)
    stale_result_paths = sorted(
        str(record["path"])
        for record in manifest["files"]
        if str(record["path"]).startswith(
            (
                "remote_json_" + "logs/",
                "remote_json_" + "logs_full/",
                "remote_embedding_" + "pngs/",
                "remote_embedding_" + "pngs_full/",
                "topology_sampling_" + "sweep/",
                "reviewer" + "3/",
            )
        )
    )
    if stale_result_paths:
        raise RuntimeError(
            f"bundle contains stale result paths: {stale_result_paths[:5]}"
        )
    obsolete_paths = sorted(
        str(record["path"])
        for record in manifest["files"]
        if "topology_sample_" in str(record["path"])
        or "topology_size_sample_" in str(record["path"])
    )
    if obsolete_paths:
        raise RuntimeError(
            "bundle contains obsolete topology-sample paths: "
            f"{obsolete_paths[:5]}"
        )

    evaluation_manifest = read_json(
        bundle_root / "evaluation" / "evaluation_manifest.json"
    )
    if evaluation_manifest.get("schema_version") != CURRENT_EVALUATION_SCHEMA:
        raise RuntimeError(
            "evaluation manifest does not use the current schema: "
            f"{evaluation_manifest.get('schema_version')!r}"
        )
    manifest_hits = forbidden_schema_hits(evaluation_manifest)
    if manifest_hits:
        raise RuntimeError(
            f"evaluation manifest contains forbidden schema terms: "
            f"{manifest_hits[:5]}"
        )

    dataset_results = {}
    for dataset, expected_methods in EXPECTED_METHODS.items():
        evaluation_root = bundle_root / "evaluation" / dataset
        evaluation = read_json(evaluation_root / "evaluation.json")
        if evaluation.get("schema_version") != CURRENT_EVALUATION_SCHEMA:
            raise RuntimeError(
                f"{dataset} evaluation schema is not current: "
                f"{evaluation.get('schema_version')!r}"
            )
        hits = forbidden_schema_hits(evaluation)
        if hits:
            raise RuntimeError(
                f"{dataset} evaluation contains forbidden schema terms: "
                f"{hits[:5]}"
            )

        topology_design = evaluation.get("topology_design", {})
        subset_count = int(topology_design.get("topology_subset_count", -1))
        subset_size = int(topology_design.get("topology_subset_size", -1))
        nested_sizes = tuple(
            sorted(int(value) for value in topology_design.get(
                "nested_subset_sizes",
                [],
            ))
        )
        if subset_count != CURRENT_TOPOLOGY_SUBSET_COUNT:
            raise RuntimeError(
                f"{dataset} topology subset count is {subset_count}, "
                f"expected {CURRENT_TOPOLOGY_SUBSET_COUNT}"
            )
        if subset_size != CURRENT_TOPOLOGY_SUBSET_SIZE:
            raise RuntimeError(
                f"{dataset} topology subset size is {subset_size}, "
                f"expected {CURRENT_TOPOLOGY_SUBSET_SIZE}"
            )
        if nested_sizes != CURRENT_NESTED_SUBSET_SIZES:
            raise RuntimeError(
                f"{dataset} nested subset sizes are {nested_sizes}, "
                f"expected {CURRENT_NESTED_SUBSET_SIZES}"
            )

        methods = evaluation.get("methods")
        if not isinstance(methods, dict):
            raise RuntimeError(f"{dataset} evaluation has no method records")
        missing_methods = sorted(expected_methods - set(methods))
        if missing_methods:
            raise RuntimeError(
                f"{dataset} evaluation is missing methods: {missing_methods}"
            )
        expected_subset_ids = list(range(CURRENT_TOPOLOGY_SUBSET_COUNT))
        for method in sorted(expected_methods):
            topology = methods[method].get("topology")
            if not isinstance(topology, dict):
                raise RuntimeError(
                    f"{dataset}/{method} has no topology record"
                )
            metadata = topology.get("subset_metadata")
            records = topology.get("records")
            if not isinstance(metadata, list) or not isinstance(records, list):
                raise RuntimeError(
                    f"{dataset}/{method} has incomplete topology subsets"
                )
            metadata_ids = sorted(int(row["subset_id"]) for row in metadata)
            record_ids = sorted(int(row["subset_id"]) for row in records)
            if (
                metadata_ids != expected_subset_ids
                or record_ids != expected_subset_ids
            ):
                raise RuntimeError(
                    f"{dataset}/{method} topology subset IDs are incomplete"
                )
            for row in metadata:
                if int(row.get("subset_size", -1)) != CURRENT_TOPOLOGY_SUBSET_SIZE:
                    raise RuntimeError(
                        f"{dataset}/{method} topology subset "
                        f"{row['subset_id']} has the wrong size"
                    )
                audit_file = evaluation_root / str(row.get("audit_file", ""))
                if not audit_file.is_file():
                    raise RuntimeError(
                        f"{dataset}/{method} topology reference audit is "
                        f"missing: {audit_file.name}"
                    )
            for record in records:
                missing_metrics = [
                    metric for metric in TOPOLOGY_METRICS
                    if metric not in record
                ]
                if missing_metrics:
                    raise RuntimeError(
                        f"{dataset}/{method} topology subset "
                        f"{record['subset_id']} is missing {missing_metrics}"
                    )
            sensitivity = topology.get("subset_size_sensitivity", {})
            sensitivity_sizes = tuple(
                sorted(
                    int(row["sensitivity_subset_size"])
                    for row in sensitivity.get("records", [])
                )
            )
            if sensitivity_sizes != CURRENT_NESTED_SUBSET_SIZES:
                raise RuntimeError(
                    f"{dataset}/{method} subset-size sensitivity is "
                    f"{sensitivity_sizes}, expected "
                    f"{CURRENT_NESTED_SUBSET_SIZES}"
                )
            sensitivity_metadata = sensitivity.get("subset_metadata", [])
            sensitivity_metadata_sizes = tuple(
                sorted(int(row["subset_size"]) for row in sensitivity_metadata)
            )
            if sensitivity_metadata_sizes != CURRENT_NESTED_SUBSET_SIZES:
                raise RuntimeError(
                    f"{dataset}/{method} subset-size audit metadata is "
                    f"{sensitivity_metadata_sizes}, expected "
                    f"{CURRENT_NESTED_SUBSET_SIZES}"
                )
            for row in sensitivity_metadata:
                for key in ("subset_file", "audit_file"):
                    path = evaluation_root / str(row.get(key, ""))
                    if not path.is_file():
                        raise RuntimeError(
                            f"{dataset}/{method} subset-size audit file is "
                            f"missing: {path.name}"
                        )

        code_to_name = evaluation.get("label_policy", {}).get("code_to_name")
        centroids = (
            evaluation.get("reference", {})
            .get("label_centroids", {})
            .get("values")
        )
        if not isinstance(code_to_name, dict) or not code_to_name:
            raise RuntimeError(f"{dataset} evaluation has no label mapping")
        if (
            not isinstance(centroids, list)
            or len(centroids) != len(code_to_name)
            or any(not isinstance(row, list) or not row for row in centroids)
        ):
            raise RuntimeError(
                f"{dataset} bundled input centroids do not match label codes"
            )
        for subset_id in expected_subset_ids:
            path = evaluation_root / f"topology_subset_{subset_id}.npy"
            if not path.is_file():
                raise RuntimeError(
                    f"{dataset} topology subset file is missing: {path.name}"
                )
        dataset_results[dataset] = {
            "methods": len(expected_methods),
            "topology_subset_count": subset_count,
            "topology_subset_size": subset_size,
            "label_centroids": len(centroids),
        }
    result_files = sorted(bundle_root.rglob("result.json"))
    non_success_results = []
    for path in result_files:
        payload = read_json(path)
        if payload.get("status") != "success":
            non_success_results.append(
                {
                    "path": path.relative_to(bundle_root).as_posix(),
                    "status": payload.get("status"),
                }
            )
    if non_success_results:
        raise RuntimeError(
            "bundle contains non-success reducer runs: "
            f"{non_success_results[:5]}"
        )
    return {
        "evaluation_schema": CURRENT_EVALUATION_SCHEMA,
        "datasets": dataset_results,
        "small_suite": verify_small_atlas_suite(bundle_root),
        "successful_result_files": len(result_files),
    }


def verify_directory(
    bundle_root: Path,
    *,
    require_current_contract: bool = False,
) -> dict:
    bundle_root = bundle_root.absolute()
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise RuntimeError(f"bundle root must be a real directory: {bundle_root}")
    symlinks = [path for path in bundle_root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"bundle directory contains symbolic links: {symlinks[:5]}")
    manifest = read_manifest(bundle_root)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("bundle manifest has no file records")

    declared: set[str] = set()
    declared_bytes = 0
    for record in records:
        relative = safe_member_path(str(record["path"]))
        relative_text = relative.as_posix()
        if relative_text in declared:
            raise RuntimeError(f"duplicate manifest record: {relative_text}")
        declared.add(relative_text)
        path = bundle_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"declared regular file is missing: {relative_text}")
        expected_bytes = int(record["bytes"])
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"size mismatch for {relative_text}: "
                f"expected {expected_bytes}, found {actual_bytes}"
            )
        expected_hash = str(record["sha256"])
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative_text}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
        declared_bytes += actual_bytes

    actual = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    expected = declared | {"bundle-manifest.json"}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"bundle file set mismatch; missing={missing}, unexpected={unexpected}"
        )
    if declared_bytes != int(manifest.get("total_bytes", -1)):
        raise RuntimeError(
            "manifest total_bytes does not equal the sum of its file records"
        )
    result = {
        "bundle_root": str(bundle_root),
        "run_id": manifest.get("run_id"),
        "verified_files": len(records),
        "verified_bytes": declared_bytes,
    }
    if require_current_contract:
        result["current_revision3_contract"] = (
            verify_current_revision3_contract(bundle_root)
        )
    return result


def inspect_archive(archive: Path) -> tuple[list[tarfile.TarInfo], str]:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    if not members:
        raise RuntimeError("result archive is empty")
    if len(members) > MAX_ARCHIVE_FILES:
        raise RuntimeError(
            f"archive contains {len(members):,} entries; limit is {MAX_ARCHIVE_FILES:,}"
        )
    names: set[str] = set()
    roots: set[str] = set()
    total_bytes = 0
    for member in members:
        relative = safe_member_path(member.name)
        normalized = relative.as_posix()
        if normalized in names:
            raise RuntimeError(f"duplicate archive member: {normalized}")
        names.add(normalized)
        roots.add(relative.parts[0])
        if not (member.isdir() or member.isreg()):
            raise RuntimeError(
                f"archive member is not a regular file/directory: {normalized}"
            )
        if member.isreg():
            if member.size < 0:
                raise RuntimeError(f"negative member size: {normalized}")
            total_bytes += member.size
    if len(roots) != 1:
        raise RuntimeError(f"archive must contain one root directory, found {roots}")
    if total_bytes > MAX_UNCOMPRESSED_BYTES:
        raise RuntimeError(
            f"archive declares {total_bytes:,} bytes; "
            f"limit is {MAX_UNCOMPRESSED_BYTES:,}"
        )
    return members, next(iter(roots))


def extract_and_verify(
    archive: Path,
    destination_root: Path,
    *,
    require_current_contract: bool = False,
) -> dict:
    archive = archive.resolve()
    destination_root = destination_root.resolve()
    members, root_name = inspect_archive(archive)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / root_name
    if destination.exists():
        if destination.is_symlink():
            raise RuntimeError(f"existing bundle destination is a symlink: {destination}")
        result = verify_directory(
            destination,
            require_current_contract=require_current_contract,
        )
        result["reused_existing_directory"] = True
        return result

    temporary = destination_root / f".{root_name}.extracting-{os.getpid()}"
    if temporary.exists():
        raise RuntimeError(f"temporary extraction path already exists: {temporary}")
    temporary.mkdir()
    try:
        with tarfile.open(archive, "r:gz") as handle:
            indexed = {member.name: member for member in handle.getmembers()}
            for original in members:
                member = indexed[original.name]
                relative = safe_member_path(member.name)
                tail = relative.parts[1:]
                if not tail:
                    if not member.isdir():
                        raise RuntimeError("archive root must be a directory")
                    continue
                target = temporary.joinpath(*tail)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read archive member: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        result = verify_directory(
            temporary,
            require_current_contract=require_current_contract,
        )
        temporary.rename(destination)
        result["bundle_root"] = str(destination.resolve())
        result["reused_existing_directory"] = False
        return result
    except Exception as error:
        try:
            if temporary.is_symlink():
                temporary.unlink()
            elif temporary.exists():
                shutil.rmtree(temporary)
        except OSError as cleanup_error:
            raise RuntimeError(
                "bundle verification failed and its temporary extraction "
                f"could not be removed: {temporary}: {cleanup_error}"
            ) from error
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--bundle-root", type=Path)
    parser.add_argument("--destination-root", type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument(
        "--require-current-contract",
        action="store_true",
        help=(
            "also require the ten-subset Revision 3 evaluation schema, "
            "complete paired topology records, and bundled input centroids"
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.archive is not None:
        if args.destination_root is None:
            parser.error("--destination-root is required with --archive")
        if args.expected_archive_sha256:
            actual = sha256_file(args.archive)
            if actual != args.expected_archive_sha256:
                raise RuntimeError(
                    f"outer archive SHA-256 mismatch: expected "
                    f"{args.expected_archive_sha256}, found {actual}"
                )
        result = extract_and_verify(
            args.archive,
            args.destination_root,
            require_current_contract=args.require_current_contract,
        )
    else:
        if args.destination_root is not None:
            parser.error("--destination-root is only valid with --archive")
        result = verify_directory(
            args.bundle_root,
            require_current_contract=args.require_current_contract,
        )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"verified {result['verified_files']:,} files "
            f"({result['verified_bytes']:,} bytes)"
        )
        print(result["bundle_root"])


if __name__ == "__main__":
    main()
