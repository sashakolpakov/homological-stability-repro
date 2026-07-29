#!/usr/bin/env python3
"""Create and verify the source-only archive used by remote GPU runners.

The remote rerun and the archived-data reproduction path are deliberately
different modes.  This module implements the former: it packages executable
source and lock files, never committed benchmark data or derived publication
artifacts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath

from revision3_source_manifest import REVISION3_SOURCE_FILES


ARCHIVE_MANIFEST = "REMOTE_SOURCE_MANIFEST.json"
FORBIDDEN_TOP_LEVELS = {
    ".git",
    ".matplotlib-cache",
    ".venv",
    "data",
    "docs",
    "generated",
    "pics",
}
FORBIDDEN_DATA_SUFFIXES = {
    ".feather",
    ".h5",
    ".h5ad",
    ".jpeg",
    ".jpg",
    ".mtx",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".tsv",
}
IGNORED_NAMES = {".DS_Store", "__pycache__"}
MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
MAX_COMPRESSED_BYTES = 8 * 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_relative_name(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError(f"unsafe source-archive path: {name!r}")
    if relative.parts[0] in FORBIDDEN_TOP_LEVELS:
        raise RuntimeError(
            "source-only archive contains forbidden top-level path: "
            f"{relative.parts[0]}"
        )
    if relative.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
        raise RuntimeError(
            "source-only archive contains a data/derived-artifact suffix: "
            f"{name}"
        )
    return relative


def collect_source_files(repository_root: Path) -> list[Path]:
    """Return exactly the allowlisted regular Revision 3 source files."""

    repository_root = repository_root.resolve()
    files: list[Path] = []
    for relative_name in REVISION3_SOURCE_FILES:
        relative = _validate_relative_name(relative_name)
        if any(part in IGNORED_NAMES for part in relative.parts):
            raise RuntimeError(
                f"source allowlist contains an ignored path: {relative_name}"
            )
        path = repository_root.joinpath(*relative.parts)
        if not path.exists():
            raise FileNotFoundError(
                f"required remote source file is missing: {path}"
            )
        if path.is_symlink():
            raise RuntimeError(
                f"links are not permitted in the remote source archive: {relative}"
            )
        if not path.is_file():
            raise RuntimeError(
                f"remote source allowlist entry is not a regular file: {relative}"
            )
        files.append(path)
    return files


def _tar_info(name: str, payload: bytes, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def create_source_archive(repository_root: Path, output: Path) -> dict:
    """Build a deterministic archive and verify it before returning."""

    repository_root = repository_root.resolve()
    files = collect_source_files(repository_root)
    records = []
    payloads: list[tuple[str, bytes, int]] = []
    for path in files:
        name = path.relative_to(repository_root).as_posix()
        payload = path.read_bytes()
        mode = stat.S_IMODE(path.stat().st_mode)
        records.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        payloads.append((name, payload, mode))

    total_bytes = sum(record["bytes"] for record in records)
    if total_bytes > MAX_UNCOMPRESSED_BYTES:
        raise RuntimeError(
            "remote source payload exceeds the uncompressed size gate: "
            f"{total_bytes} > {MAX_UNCOMPRESSED_BYTES} bytes"
        )
    manifest = {
        "schema_version": 1,
        "mode": "remote-full-rerun-source-only",
        "source_allowlist": list(REVISION3_SOURCE_FILES),
        "forbidden_top_levels": sorted(FORBIDDEN_TOP_LEVELS),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files": records,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with tarfile.open(
                    fileobj=gzip_handle,
                    mode="w",
                    format=tarfile.GNU_FORMAT,
                ) as archive:
                    archive.addfile(
                        _tar_info(
                            ARCHIVE_MANIFEST,
                            manifest_payload,
                            0o644,
                        ),
                        io.BytesIO(manifest_payload),
                    )
                    for name, payload, mode in payloads:
                        archive.addfile(
                            _tar_info(name, payload, mode),
                            io.BytesIO(payload),
                        )
        if temporary.stat().st_size > MAX_COMPRESSED_BYTES:
            raise RuntimeError(
                "remote source archive exceeds the compressed size gate: "
                f"{temporary.stat().st_size} > {MAX_COMPRESSED_BYTES} bytes"
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    result = verify_source_archive(output)
    result["archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return result


def verify_source_archive(archive_path: Path) -> dict:
    """Verify content, hashes, paths, types, and both size gates."""

    archive_path = archive_path.resolve()
    compressed_bytes = archive_path.stat().st_size
    if compressed_bytes > MAX_COMPRESSED_BYTES:
        raise RuntimeError(
            "remote source archive exceeds the compressed size gate: "
            f"{compressed_bytes} > {MAX_COMPRESSED_BYTES} bytes"
        )

    payload_by_name: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise RuntimeError(
                    "remote source archive permits regular files only: "
                    f"{member.name}"
                )
            if member.name in payload_by_name:
                raise RuntimeError(
                    f"duplicate remote source archive member: {member.name}"
                )
            if member.name != ARCHIVE_MANIFEST:
                _validate_relative_name(member.name)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(
                    f"could not read remote source archive member: {member.name}"
                )
            payload_by_name[member.name] = extracted.read()

    try:
        manifest = json.loads(payload_by_name[ARCHIVE_MANIFEST])
    except KeyError as exc:
        raise RuntimeError("remote source archive has no manifest") from exc
    if manifest.get("mode") != "remote-full-rerun-source-only":
        raise RuntimeError("remote source archive declares the wrong mode")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("remote source archive manifest has no file list")

    declared_names = set()
    declared_total = 0
    for record in records:
        name = record.get("path")
        if not isinstance(name, str):
            raise RuntimeError("remote source archive manifest has an invalid path")
        _validate_relative_name(name)
        if name in declared_names:
            raise RuntimeError(f"duplicate path in source manifest: {name}")
        declared_names.add(name)
        try:
            payload = payload_by_name[name]
        except KeyError as exc:
            raise RuntimeError(
                f"declared remote source file is absent: {name}"
            ) from exc
        if len(payload) != record.get("bytes"):
            raise RuntimeError(f"byte-count mismatch for remote source file: {name}")
        if sha256_bytes(payload) != record.get("sha256"):
            raise RuntimeError(f"SHA-256 mismatch for remote source file: {name}")
        declared_total += len(payload)

    actual_names = set(payload_by_name) - {ARCHIVE_MANIFEST}
    if actual_names != declared_names:
        extra = sorted(actual_names - declared_names)
        missing = sorted(declared_names - actual_names)
        raise RuntimeError(
            "source archive and manifest member sets differ: "
            f"extra={extra}, missing={missing}"
        )
    if declared_total != manifest.get("total_bytes"):
        raise RuntimeError("remote source archive total-byte count is inconsistent")
    if declared_total > MAX_UNCOMPRESSED_BYTES:
        raise RuntimeError(
            "remote source payload exceeds the uncompressed size gate: "
            f"{declared_total} > {MAX_UNCOMPRESSED_BYTES} bytes"
        )
    required_names = set(REVISION3_SOURCE_FILES)
    if actual_names != required_names:
        extra = sorted(actual_names - required_names)
        missing = sorted(required_names - actual_names)
        raise RuntimeError(
            "remote source archive differs from the exact Revision 3 "
            f"allowlist: extra={extra}, missing={missing}"
        )
    if manifest.get("source_allowlist") != list(REVISION3_SOURCE_FILES):
        raise RuntimeError(
            "remote source archive records a different source allowlist"
        )

    return {
        "mode": manifest["mode"],
        "file_count": len(actual_names),
        "uncompressed_bytes": declared_total,
        "compressed_bytes": compressed_bytes,
        "archive": str(archive_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.verify is None):
        parser.error("choose exactly one of --output or --verify")
    if args.output is not None:
        result = create_source_archive(args.repository_root, args.output)
    else:
        result = verify_source_archive(args.verify)
        result["archive_sha256"] = hashlib.sha256(
            args.verify.read_bytes()
        ).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
