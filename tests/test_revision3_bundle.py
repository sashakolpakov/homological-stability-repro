from __future__ import annotations

import json
import hashlib
import struct
import sys
import tarfile
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from package_revision3_results import deterministic_tar  # noqa: E402
from revision3_source_manifest import (  # noqa: E402
    REVISION3_SOURCE_FILES,
    build_source_manifest,
)
from verify_revision3_bundle import (  # noqa: E402
    EXPECTED_METHODS,
    SMALL_CPU_REPEATS,
    SMALL_DATASETS,
    SMALL_GPU_METHODS,
    SMALL_GPU_REPEATS,
    SMALL_METHODS,
    TOPOLOGY_METRICS,
    extract_and_verify,
    sha256_file,
    verify_current_revision3_contract,
    verify_directory,
)


def make_bundle(root: Path) -> Path:
    bundle = root / "revision3-results-fixture"
    payload = bundle / "evaluation" / "result.json"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"status":"success"}\n', encoding="utf-8")
    readme = bundle / "README.md"
    readme.write_text("# fixture\n", encoding="utf-8")
    records = [
        {
            "path": path.relative_to(bundle).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (readme, payload)
    ]
    manifest = {
        "schema_version": 1,
        "run_id": "fixture",
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def write_bundle_manifest(bundle: Path, run_id: str) -> None:
    files = sorted(
        path
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    )
    records = [
        {
            "path": path.relative_to(bundle).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def make_current_contract_bundle(root: Path) -> Path:
    bundle = root / "revision3-results-current-fixture"
    evaluation_root = bundle / "evaluation"
    datasets = {}
    for dataset, method_names in EXPECTED_METHODS.items():
        dataset_root = evaluation_root / dataset
        dataset_root.mkdir(parents=True)
        subset_metadata = [
            {
                "subset_id": subset_id,
                "subset_seed": 20_042 + subset_id,
                "subset_size": 4_000,
                "audit_file": f"topology_reference_{subset_id}.npz",
            }
            for subset_id in range(10)
        ]
        for subset_id in range(10):
            (dataset_root / f"topology_subset_{subset_id}.npy").write_bytes(
                b"fixture"
            )
            (dataset_root / f"topology_reference_{subset_id}.npz").write_bytes(
                b"fixture"
            )
        size_metadata = [
            {
                "subset_size": size,
                "subset_file": f"topology_size_subset_{size}.npy",
                "audit_file": f"topology_size_reference_{size}.npz",
            }
            for size in (1_000, 2_000, 4_000)
        ]
        for row in size_metadata:
            (dataset_root / row["subset_file"]).write_bytes(b"fixture")
            (dataset_root / row["audit_file"]).write_bytes(b"fixture")
        methods = {}
        for method in sorted(method_names):
            methods[method] = {
                "topology": {
                    "subset_metadata": subset_metadata,
                    "records": [
                        {
                            "subset_id": subset_id,
                            **{
                                metric: float(subset_id)
                                for metric in TOPOLOGY_METRICS
                            },
                        }
                        for subset_id in range(10)
                    ],
                    "subset_size_sensitivity": {
                        "subset_metadata": size_metadata,
                        "records": [
                            {
                                "sensitivity_subset_size": size,
                                **{
                                    metric: float(size)
                                    for metric in TOPOLOGY_METRICS
                                },
                            }
                            for size in (1_000, 2_000, 4_000)
                        ]
                    },
                }
            }
        evaluation = {
            "schema_version": 2,
            "dataset": dataset,
            "topology_design": {
                "topology_subset_count": 10,
                "topology_subset_size": 4_000,
                "nested_subset_sizes": [1_000, 2_000, 4_000],
            },
            "label_policy": {
                "code_to_name": {
                    "0": "label 0",
                    "1": "label 1",
                }
            },
            "reference": {
                "label_centroids": {
                    "values": [[0.0, 1.0], [1.0, 0.0]],
                }
            },
            "methods": methods,
        }
        (dataset_root / "evaluation.json").write_text(
            json.dumps(evaluation),
            encoding="utf-8",
        )
        datasets[dataset] = {"schema_version": 2}
    (evaluation_root / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )
    small_root = bundle / "small_suite" / "json_logs"
    small_root.mkdir(parents=True)
    (small_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "datasets": list(SMALL_DATASETS),
                "methods": list(SMALL_METHODS),
            }
        ),
        encoding="utf-8",
    )
    for dataset in SMALL_DATASETS:
        methods = {}
        for method in SMALL_METHODS:
            repeat_count = (
                SMALL_GPU_REPEATS
                if method in SMALL_GPU_METHODS
                else SMALL_CPU_REPEATS
            )
            records = []
            for repeat in range(repeat_count):
                indices = list(range(repeat, repeat + 1_000))
                digest = hashlib.sha256(
                    struct.pack(
                        f"<{len(indices)}q",
                        *indices,
                    )
                ).hexdigest()
                records.append(
                    {
                        "seed": 42 + repeat,
                        "fit_time_sec": 1.0 + repeat,
                        "topology_computed": True,
                        "topology_sample": {
                            "sample_size": 1_000,
                            "sample_fraction": 0.1,
                            "sample_policy": (
                                "fixed-size uniform without replacement"
                            ),
                            "subset_seed": 1042 + repeat,
                            "indices_sha256": digest,
                            "indices": indices,
                            "backend": "atlas",
                            "backend_detail": (
                                "direct GPU rank-based local-kNN atlas"
                            ),
                            "prefer_ripser": False,
                        },
                        "metrics": {
                            "topology": {
                                "backend": "atlas",
                                "backend_detail": (
                                    "direct GPU rank-based local-kNN atlas"
                                ),
                                "prefer_ripser": False,
                                "metrics": {
                                    "dtw_beta0": 0.1 + 0.01 * repeat,
                                    "dtw_beta1": 0.2 + 0.01 * repeat,
                                },
                            }
                        },
                    }
                )
            method_payload = {
                "hardware_class": (
                    "GPU" if method in SMALL_GPU_METHODS else "CPU"
                ),
                "repeats_requested": repeat_count,
                "topology_repeats_requested": repeat_count,
                "topology_backend": "atlas",
                "topology_backend_detail": (
                    "direct GPU rank-based local-kNN atlas"
                ),
                "topology_prefer_ripser": False,
                "topology_sample_size": 1_000,
                "repeats": records,
            }
            methods[method] = method_payload
        (small_root / f"{dataset}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "dataset_name": dataset,
                    "methods": methods,
                }
            ),
            encoding="utf-8",
        )
    (bundle / "README.md").write_text("# current fixture\n", encoding="utf-8")
    write_bundle_manifest(bundle, "current-fixture")
    return bundle


def test_bundle_archive_round_trip(tmp_path: Path) -> None:
    source = make_bundle(tmp_path / "source")
    archive = tmp_path / "fixture.tar.gz"
    deterministic_tar(source, archive)

    result = extract_and_verify(archive, tmp_path / "fetched")

    assert result["run_id"] == "fixture"
    assert result["verified_files"] == 2
    assert result["reused_existing_directory"] is False
    assert Path(result["bundle_root"]).name == source.name
    assert verify_directory(Path(result["bundle_root"]))["verified_files"] == 2


def test_bundle_archive_is_byte_deterministic(tmp_path: Path) -> None:
    source = make_bundle(tmp_path / "source")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    deterministic_tar(source, first)
    deterministic_tar(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first) == sha256_file(second)


def test_bundle_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "README.md").write_text("# modified\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="mismatch"):
        verify_directory(bundle)


def test_failed_archive_extraction_is_removed(tmp_path: Path) -> None:
    source = make_bundle(tmp_path / "source")
    (source / "README.md").write_text("# invalidated manifest\n", encoding="utf-8")
    archive = tmp_path / "invalid.tar.gz"
    deterministic_tar(source, archive)
    fetched = tmp_path / "fetched"

    with pytest.raises(RuntimeError, match="mismatch"):
        extract_and_verify(archive, fetched)

    assert list(fetched.iterdir()) == []


def test_archive_traversal_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("unsafe\n", encoding="utf-8")
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="../payload.txt")

    with pytest.raises(RuntimeError, match="unsafe archive path"):
        extract_and_verify(archive, tmp_path / "fetched")


def test_revision3_source_manifest_is_complete_and_hashed() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    records = build_source_manifest(repository_root)

    assert tuple(record["path"] for record in records) == REVISION3_SOURCE_FILES
    assert all(len(record["sha256"]) == 64 for record in records)
    assert all(record["bytes"] > 0 for record in records)


def test_current_revision3_contract_accepts_ten_subset_schema(
    tmp_path: Path,
) -> None:
    bundle = make_current_contract_bundle(tmp_path)

    result = verify_directory(bundle, require_current_contract=True)

    assert result["current_revision3_contract"]["evaluation_schema"] == 2
    assert result["current_revision3_contract"]["datasets"]["tenx"][
        "topology_subset_count"
    ] == 10


def test_current_revision3_contract_rejects_forbidden_replication_term(
    tmp_path: Path,
) -> None:
    bundle = make_current_contract_bundle(tmp_path)
    path = bundle / "evaluation" / "tenx" / "evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluation_configuration"] = {"R_" + "sample": 10}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="forbidden schema terms"):
        verify_current_revision3_contract(bundle)


def test_current_revision3_contract_rejects_removed_preset_metadata(
    tmp_path: Path,
) -> None:
    bundle = make_current_contract_bundle(tmp_path)
    readme = bundle / "README.md"
    readme.write_text(
        "# fixture\nremoved " + "topology " + "preset\n",
        encoding="utf-8",
    )
    write_bundle_manifest(bundle, "current-fixture")

    with pytest.raises(RuntimeError, match="removed preset"):
        verify_current_revision3_contract(bundle)


def test_current_revision3_contract_rejects_failed_reducer_run(
    tmp_path: Path,
) -> None:
    bundle = make_current_contract_bundle(tmp_path)
    failed = (
        bundle
        / "large_results"
        / "tenx"
        / "dire_auto"
        / "n_001306127"
        / "result.json"
    )
    failed.parent.mkdir(parents=True)
    failed.write_text(
        json.dumps({"status": "failed", "error": "fixture"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="non-success reducer runs"):
        verify_current_revision3_contract(bundle)


@pytest.mark.parametrize(
    "relative",
    (
        Path("remote_json_" + "logs_full") / "disk.json",
        Path("remote_embedding_" + "pngs_full") / "disk-dire.png",
        Path("topology_sampling_" + "sweep") / "result.json",
        Path("reviewer" + "3") / "result.json",
    ),
)
def test_current_revision3_contract_rejects_stale_result_paths(
    tmp_path: Path,
    relative: Path,
) -> None:
    bundle = make_current_contract_bundle(tmp_path)
    stale = bundle / relative
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n", encoding="utf-8")
    write_bundle_manifest(bundle, "current-fixture")

    with pytest.raises(RuntimeError, match="stale result paths"):
        verify_current_revision3_contract(bundle)


def test_current_revision3_contract_rejects_non_atlas_small_suite(
    tmp_path: Path,
) -> None:
    bundle = make_current_contract_bundle(tmp_path)
    path = bundle / "small_suite" / "json_logs" / "disk.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["methods"]["dire"]["repeats"][0]["metrics"]["topology"][
        "backend"
    ] = "ripser"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-atlas backend"):
        verify_current_revision3_contract(bundle)


def test_old_integrity_valid_bundle_fails_current_contract(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(tmp_path)

    with pytest.raises(RuntimeError, match="evaluation_manifest"):
        verify_directory(bundle, require_current_contract=True)
