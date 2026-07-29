#!/usr/bin/env python3
"""Define and hash the exact source surface used by the Revision 3 profile."""

from __future__ import annotations

import hashlib
from pathlib import Path


REVISION3_SOURCE_FILES = (
    ".dockerignore",
    "Makefile",
    "README.md",
    "docker/Dockerfile",
    "requirements/bench.in",
    "requirements/bench.txt",
    "requirements/docs.txt",
    "requirements/render.txt",
    "requirements/topology-audit.txt",
    "scripts/archive_embedding_pngs.py",
    "scripts/article_benchmarks.py",
    "scripts/audit_revision3_topology.py",
    "scripts/build_revision3_local.sh",
    "scripts/evaluate_large_embeddings.py",
    "scripts/fetch_revision3_results.sh",
    "scripts/large_scale_benchmarks.py",
    "scripts/package_json_logs.py",
    "scripts/package_revision3_remote_source.py",
    "scripts/package_revision3_results.py",
    "scripts/prepare_large_datasets.py",
    "scripts/render_figures.py",
    "scripts/render_revision3_artifacts.py",
    "scripts/reproduce_revision3.sh",
    "scripts/revision3_source_manifest.py",
    "scripts/runtime_protocol.py",
    "scripts/run_benchmarks.sh",
    "scripts/run_initialization_ablation.sh",
    "scripts/run_remote_revision3.sh",
    "scripts/run_revision3_inside_container.sh",
    "scripts/verify_revision3_bundle.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(repository_root: Path) -> list[dict]:
    records = []
    for relative_name in REVISION3_SOURCE_FILES:
        path = repository_root / relative_name
        if not path.is_file():
            raise FileNotFoundError(
                f"required Revision 3 source file is missing: {path}"
            )
        records.append(
            {
                "path": relative_name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records
