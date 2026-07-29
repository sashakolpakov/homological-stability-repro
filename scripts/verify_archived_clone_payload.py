#!/usr/bin/env python3
"""Verify that an ordinary clone contains the complete GPU-free payload."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DATASETS = ("blobs", "disk", "moons", "mnist", "levine13", "levine32")
METHODS = (
    "dire",
    "dire-topology",
    "cuml-tsne",
    "cuml-umap",
    "opentsne",
    "umap",
)
EXPECTED_LOG_METHODS = (
    "dire",
    "dire_topology",
    "cuml_tsne",
    "cuml_umap",
    "opentsne",
    "umap",
)
EXPECTED_JSON = {f"{dataset}.json" for dataset in DATASETS} | {"manifest.json"}
EXPECTED_PNG = {
    f"{dataset}-{method}.png" for dataset in DATASETS for method in METHODS
}
REQUIRED_REVISION3_ARTIFACTS = {
    "render_manifest.json",
    "revision3-results-summary.json",
    "revision3-results-summary.md",
    "revision3-large-scaling.pdf",
    "revision3-large-scaling.png",
    "revision3-large-quality.pdf",
    "revision3-large-quality.png",
    "revision3-small-runtime.pdf",
    "revision3-small-runtime.png",
    "revision3-tenx-layouts.pdf",
    "revision3-tenx-layouts.png",
    "revision3-arxiv-layouts.pdf",
    "revision3-arxiv-layouts.png",
}
MIN_ARCHIVED_BYTES = 1 * 1024 * 1024
MIN_REVISION3_BYTES = 1 * 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RuntimeError(f"required committed payload directory is absent: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"payload file must not be a symlink: {path}")
    return files


def _tracked_files(repository_root: Path) -> set[str] | None:
    if not (repository_root / ".git").exists():
        return None
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "data/archived",
            "generated/revision3",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return {
        name
        for name in result.stdout.decode("utf-8").split("\0")
        if name
    }


def verify_archived_clone(repository_root: Path) -> dict:
    repository_root = repository_root.resolve()
    json_root = repository_root / "data" / "archived" / "json_logs"
    png_root = repository_root / "data" / "archived" / "embedding_pngs"
    revision3_root = repository_root / "generated" / "revision3"

    json_files = _regular_files(json_root)
    png_files = _regular_files(png_root)
    revision3_files = _regular_files(revision3_root)
    json_names = {path.name for path in json_files}
    png_names = {path.name for path in png_files}
    revision3_names = {path.name for path in revision3_files}
    if json_names != EXPECTED_JSON:
        raise RuntimeError(
            "archived JSON payload differs from the declared six-dataset "
            f"contract: missing={sorted(EXPECTED_JSON - json_names)}, "
            f"extra={sorted(json_names - EXPECTED_JSON)}"
        )
    if png_names != EXPECTED_PNG:
        raise RuntimeError(
            "archived embedding payload differs from the 6x6 contract: "
            f"missing={sorted(EXPECTED_PNG - png_names)}, "
            f"extra={sorted(png_names - EXPECTED_PNG)}"
        )
    missing_revision3 = sorted(REQUIRED_REVISION3_ARTIFACTS - revision3_names)
    if missing_revision3:
        raise RuntimeError(
            "committed Revision 3 publication payload is incomplete: "
            + ", ".join(missing_revision3)
        )

    for path in json_files:
        payload = path.read_bytes()
        if payload.startswith(LFS_MAGIC):
            raise RuntimeError(f"Git LFS pointer found instead of JSON data: {path}")
        try:
            json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid committed JSON payload: {path}") from exc

    archived_manifest = json.loads(
        (json_root / "manifest.json").read_text(encoding="utf-8")
    )
    if tuple(archived_manifest.get("methods", ())) != EXPECTED_LOG_METHODS:
        raise RuntimeError(
            "archived JSON payload does not declare the final six-method "
            f"protocol: {archived_manifest.get('methods')!r}"
        )

    publication_manifest = json.loads(
        (revision3_root / "render_manifest.json").read_text(encoding="utf-8")
    )
    source_bundle = publication_manifest.get("source_bundle", {})
    expected_bundle = source_bundle.get("directory_name")
    if archived_manifest.get("source_bundle") != expected_bundle:
        raise RuntimeError(
            "archived small-suite provenance does not match the Revision 3 "
            f"publication bundle: {archived_manifest.get('source_bundle')!r} "
            f"!= {expected_bundle!r}"
        )
    expected_bundle_manifest_sha = source_bundle.get("manifest_sha256")
    if (
        archived_manifest.get("source_bundle_manifest_sha256")
        != expected_bundle_manifest_sha
    ):
        raise RuntimeError(
            "archived small-suite bundle-manifest hash does not match the "
            "Revision 3 publication manifest"
        )
    expected_audit_sha = publication_manifest.get("topology_audit", {}).get(
        "sha256"
    )
    if (
        archived_manifest.get("source_topology_audit_sha256")
        != expected_audit_sha
    ):
        raise RuntimeError(
            "archived small-suite topology-audit hash does not match the "
            "Revision 3 publication manifest"
        )
    archive_sha = archived_manifest.get("source_bundle_archive_sha256")
    if (
        not isinstance(archive_sha, str)
        or len(archive_sha) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha)
    ):
        raise RuntimeError(
            "archived small-suite source bundle SHA-256 is missing or invalid"
        )

    for path in png_files:
        payload = path.read_bytes()
        if payload.startswith(LFS_MAGIC) or not payload.startswith(PNG_MAGIC):
            raise RuntimeError(f"invalid committed PNG payload: {path}")

    archived_files = json_files + png_files
    archived_bytes = sum(path.stat().st_size for path in archived_files)
    revision3_bytes = sum(path.stat().st_size for path in revision3_files)
    if archived_bytes < MIN_ARCHIVED_BYTES:
        raise RuntimeError(
            "committed archived benchmark payload is unexpectedly small: "
            f"{archived_bytes} < {MIN_ARCHIVED_BYTES} bytes"
        )
    if revision3_bytes < MIN_REVISION3_BYTES:
        raise RuntimeError(
            "committed Revision 3 publication payload is unexpectedly small: "
            f"{revision3_bytes} < {MIN_REVISION3_BYTES} bytes"
        )

    tracked = _tracked_files(repository_root)
    if tracked is not None:
        required_tracked = {
            path.relative_to(repository_root).as_posix()
            for path in archived_files + revision3_files
        }
        untracked = sorted(required_tracked - tracked)
        if untracked:
            raise RuntimeError(
                "ordinary-clone payload contains files that are not committed: "
                + ", ".join(untracked)
            )

    return {
        "mode": "ordinary-clone-with-committed-archive",
        "archived_json_files": len(json_files),
        "archived_embedding_pngs": len(png_files),
        "archived_bytes": archived_bytes,
        "revision3_files": len(revision3_files),
        "revision3_bytes": revision3_bytes,
        "git_tracking_checked": tracked is not None,
        "source_bundle": expected_bundle,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_archived_clone(args.repository_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
