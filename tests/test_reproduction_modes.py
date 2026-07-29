from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from package_revision3_remote_source import (  # noqa: E402
    ARCHIVE_MANIFEST,
    FORBIDDEN_TOP_LEVELS,
    MAX_COMPRESSED_BYTES,
    REVISION3_SOURCE_FILES,
    collect_source_files,
    create_source_archive,
    verify_source_archive,
)
from verify_archived_clone_payload import verify_archived_clone  # noqa: E402


def test_remote_source_archive_is_small_deterministic_and_data_free(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_result = create_source_archive(ROOT, first)
    second_result = create_source_archive(ROOT, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["mode"] == "remote-full-rerun-source-only"
    assert first_result["compressed_bytes"] < MAX_COMPRESSED_BYTES
    assert first_result["archive_sha256"] == second_result["archive_sha256"]
    assert verify_source_archive(first)["file_count"] > 0

    with tarfile.open(first, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
    assert ARCHIVE_MANIFEST in names
    assert names == {ARCHIVE_MANIFEST, *REVISION3_SOURCE_FILES}
    assert not {
        name
        for name in names
        if name != ARCHIVE_MANIFEST
        and Path(name).parts[0] in FORBIDDEN_TOP_LEVELS
    }


def test_source_mode_ignores_every_file_outside_exact_allowlist(
    tmp_path: Path,
) -> None:
    for relative_name in REVISION3_SOURCE_FILES:
        path = tmp_path / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    leak = tmp_path / "scripts" / "committed-benchmark.npy"
    leak.write_bytes(b"not source")

    collected = collect_source_files(tmp_path)

    assert {
        path.relative_to(tmp_path).as_posix()
        for path in collected
    } == set(REVISION3_SOURCE_FILES)
    assert leak not in collected


def test_ordinary_clone_payload_is_committed_and_complete() -> None:
    result = verify_archived_clone(ROOT)

    assert result["mode"] == "ordinary-clone-with-committed-archive"
    assert result["archived_json_files"] == 7
    assert result["archived_embedding_pngs"] == 36
    assert result["git_tracking_checked"] is True
    assert result["source_bundle"] == "revision3-results-20260729-202035"


def test_remote_shell_runner_uses_verified_archive_not_repository_tar() -> None:
    source = (ROOT / "scripts/run_remote_revision3.sh").read_text(
        encoding="utf-8"
    )
    assert "package_revision3_remote_source.py" in source
    assert '< "$SOURCE_ARCHIVE"' in source
    assert "-czf - ." not in source
    assert (".super" + "seded-") not in source
    assert (
        "rm -rf -- $REMOTE_DIR/docker $REMOTE_DIR/requirements "
        "$REMOTE_DIR/scripts"
    ) in source
    assert (
        "rm -f -- $REMOTE_DIR/.dockerignore $REMOTE_DIR/Makefile "
        "$REMOTE_DIR/README.md"
    ) in source
    assert "then rm -rf -- $REMOTE_DIR" in source


def test_docker_context_is_allowlist_only() -> None:
    patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "**" in patterns
    assert "!docker/Dockerfile" in patterns
    assert "!requirements/bench.txt" in patterns
    assert not any(line.startswith("!data") for line in patterns)
    assert not any(line.startswith("!generated") for line in patterns)


def test_remote_pipeline_runs_and_fetches_matching_cpu_only_audit() -> None:
    reproduce = (ROOT / "scripts/reproduce_revision3.sh").read_text(
        encoding="utf-8"
    )
    inside = (ROOT / "scripts/run_revision3_inside_container.sh").read_text(
        encoding="utf-8"
    )
    fetch = (ROOT / "scripts/fetch_revision3_results.sh").read_text(
        encoding="utf-8"
    )
    local_build = (ROOT / "scripts/build_revision3_local.sh").read_text(
        encoding="utf-8"
    )
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert (
        'EVALUATION_TOPOLOGY_WORKERS="${EVALUATION_TOPOLOGY_WORKERS:-24}"'
        in reproduce
    )
    assert (
        'EVALUATION_TOPOLOGY_WORKERS="${EVALUATION_TOPOLOGY_WORKERS:-24}"'
        in inside
    )
    assert 'PIPELINE_VERSION="${REVISION3_PIPELINE_VERSION:-v10}"' in reproduce
    assert '-e REVISION3_PIPELINE_VERSION="$PIPELINE_VERSION"' in reproduce
    assert 'PIPELINE_VERSION="${REVISION3_PIPELINE_VERSION:-v10}"' in inside
    assert 'SMALL_GPU_REPEATS="${SMALL_GPU_REPEATS:-20}"' in reproduce
    assert 'SMALL_CPU_REPEATS="${SMALL_CPU_REPEATS:-10}"' in reproduce
    assert (
        'SMALL_TOPOLOGY_REPEATS="${SMALL_TOPOLOGY_REPEATS:-20}"'
        in reproduce
    )
    assert (
        'TOPOLOGY_REPEATS="${SMALL_TOPOLOGY_REPEATS:-20}"'
        in inside
    )
    assert 'TOPOLOGY_SAMPLE_SIZE="${SMALL_TOPOLOGY_SAMPLE_SIZE:-1000}"' in inside
    assert "TOPOLOGY_BACKEND" not in inside
    assert (
        'METHODS="dire dire_topology cuml_tsne cuml_umap opentsne umap"'
        in inside
    )
    assert "python3 scripts/audit_revision3_topology.py" in reproduce
    assert "NVIDIA_VISIBLE_DEVICES=void" in reproduce
    assert '--workers "${CPU_AUDIT_WORKERS:-24}"' in reproduce
    assert "cpu_audit/LATEST" in fetch
    assert 'payload.get("source_bundle") != bundle_name' in fetch
    assert 'payload.get("status") != "success"' in fetch
    assert "prune_revision3_entries" in fetch
    assert "--require-clean-local-artifacts" in fetch
    assert "Removing superseded artifact" in fetch
    assert "prune_children_except" in reproduce
    assert '"$REVISION3_ROOT/bundles"' in reproduce
    assert '"$REVISION3_ROOT/bundle_staging"' in reproduce
    assert '"$REVISION3_ROOT/cpu_audit"' in reproduce
    assert '"$REVISION3_ROOT/stages"' in reproduce
    assert (
        '--data-root "$BUNDLE_ROOT/small_suite/json_logs"'
        in local_build
    )
    assert (
        '--embedding-png-root "$BUNDLE_ROOT/small_suite/embedding_pngs"'
        in local_build
    )
    assert (
        'DATA_ROOT="data/revision3/fetched/$$bundle/small_suite/json_logs"'
        in makefile
    )
    submission_recipe = makefile.split("revision3-submission:", 1)[1].split(
        "\nbenchmarks:",
        1,
    )[0]
    assert "--require-clean-local-artifacts" in submission_recipe
    assert submission_recipe.index("revision3-render") < submission_recipe.index(
        "--require-clean-local-artifacts"
    )
    assert "revision3-topology-audit" not in submission_recipe
