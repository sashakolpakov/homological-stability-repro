#!/usr/bin/env python3
"""Reject the withdrawn title and any visualization-only framing of DiRe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_TITLE = (
    "Dimensionality reduction for homological stability and global structure "
    "preservation"
)
EXPECTED_TITLE_DECLARATION = (
    r"\title{Dimensionality reduction for homological stability \\ "
    r"and global structure preservation}"
)
PRE_REVISION3_ABSTRACT = (
    "We propose DiRe, a force--directed dimensionality reduction framework "
    "designed to preserve global structure and homological features while "
    "remaining practical on modern hardware."
)
PRE_REVISION3_NONVISUAL_SCOPE = (
    "Other uses include dimensionality reduction to other, possibly higher "
    "and thus non--visual dimensions, for the subsequent use of classifiers "
    "such as SVMs."
)
PRE_REVISION3_CONCLUSION = (
    "A new dimensionality reduction framework, called "
    r"\texttt{DiRe}, has been developed with an explicit emphasis on "
    "preserving and measuring the dataset's homological structure."
)
WITHDRAWN_TITLE_FRAGMENT = "DiRe: scalable " + "two"
WITHDRAWN_SCOPE_FRAGMENTS = (
    "We study DiRe as a method for " + "two--dimensional visualization",
    "The empirical scope of this study is " + "two--dimensional visualization",
    "experiments evaluate DiRe as a " + "two--dimensional visualization method",
    "manuscript now concerns " + "two--dimensional visualization",
    "visualization framing and removed the " + "general--purpose claim",
    "retitled to identify DiRe as a " + "two--dimensional",
    "with d << D (usually d=2 or 3) specified by " + r"\texttt{dimension}",
    "construct a weighted Laplacian, optionally with a similarity kernel",
    "pca (classical or kernel--based)",
    "initial embedding and optimized layout are exposed by the software",
    "DiRe is fastest or effectively tied on Disk",
    "total NVML memory",
    "total GPU memory",
    "comparative conclusions are limited to the tested settings",
    "further by narrowing the scope",
    "code, narrowed claims, and measured evidence",
    "limits comparative conclusions to the tested",
)
WITHDRAWN_SCOPE_PATTERNS = (
    (
        "DiRe defined as dimension-fixed",
        re.compile(
            r"\bDiRe\s+(?:itself\s+)?"
            r"(?:is|is\s+primarily|is\s+merely|is\s+only|"
            r"should\s+be\s+framed\s+as|is\s+best\s+understood\s+as)\s+"
            r"(?!not\b)(?:an?\s+)?"
            r"(?:two[-–—\s]*dimensional|2[-\s]?D)\s+"
            r"(?:embedder|embedding(?:\s+(?:method|algorithm))?|"
            r"method|algorithm|tool|framework)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "DiRe defined as a graphical or visualization tool",
        re.compile(
            r"\bDiRe\s+(?:itself\s+)?"
            r"(?:is|is\s+primarily|is\s+merely|is\s+only|"
            r"should\s+be\s+framed\s+as|is\s+best\s+understood\s+as)\s+"
            r"(?!not\b)(?:an?\s+)?"
            r"(?:(?:purely|solely|merely|just)\s+)?"
            r"(?:visuali[sz]ation|graphical|plotting)"
            r"(?:[-–—\s]*only)?\s+"
            r"(?:method|algorithm|tool|framework|technique)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "manuscript reframes DiRe as visualization-only",
        re.compile(
            r"\b(?:we|this\s+manuscript|the\s+paper)\s+"
            r"(?:now\s+)?(?:frame|frames|framed|treat|treats|treated|"
            r"study|studies|studied|present|presents|presented|"
            r"describe|describes|described)\s+DiRe\s+as\s+"
            r"(?:an?\s+)?(?:(?:purely|solely|merely|just)\s+)?"
            r"(?:(?:two[-–—\s]*dimensional|2[-\s]?D)\s+)?"
            r"(?:visuali[sz]ation|graphical|plotting)"
            r"(?:[-–—\s]*only)?\s+"
            r"(?:method|algorithm|tool|framework|technique)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
)
TEXT_SUFFIXES = {
    ".cff",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".git",
    ".matplotlib-cache",
    ".pytest_cache",
    ".venv",
    "_build",
    "_static",
    "__pycache__",
}
REQUIRED_REVISION3_GENERATED_FILES = (
    "render_manifest.json",
    "topology-audit.json",
    "revision3-results-macros.tex",
    "revision3-topology-effect-audit.json",
    "revision3-topology-paired-effects.csv",
    "revision3-arxiv-topology-paired-effects.tex",
    "revision3-topology-subset-size-sensitivity.csv",
    "revision3-topology-subset-size-sensitivity.tex",
    "revision3-topology-subset-size-sensitivity.png",
    "revision3-small-atlas-topology-summary.csv",
    "revision3-small-atlas-topology-summary.tex",
    "revision3-small-atlas-topology-paired-effects.csv",
    "revision3-small-atlas-topology-paired-effects.pdf",
    "revision3-small-atlas-topology-paired-effects.png",
    "revision3-tenx-adjacency-audit.json",
    "revision3-tenx-adjacency-summary.csv",
    "revision3-tenx-adjacency-summary.tex",
    "revision3-tenx-adjacency-by-source.csv",
    "revision3-tenx-adjacency-sensitivity.csv",
)
RETIRED_REVISION3_TERMS = (
    "R_" + "sample",
    "R_" + "layout",
    "paired_" + "subset_count",
    "paired_" + "subset_ids",
    "dire_" + "better_count",
    "competitor_" + "better_count",
    "tie_" + "count",
)
RETIRED_REVISION3_PATTERNS = (
    (
        "retired sample-count notation",
        re.compile(
            r"(?<![A-Za-z0-9])R\s*_\s*(?:sample|"
            r"\{\s*(?:\\(?:rm|mathrm|text)\s*\{?)?\s*sample\s*\}?\s*\}|"
            r"\\(?:rm|mathrm|text)\s*\{?\s*sample\s*\}?)",
            flags=re.IGNORECASE,
        ),
    ),
)
REFEREE_PROSE_FILES = {
    Path("editor_cover_letter_round2.tex"),
    Path("response_to_referees_round2.tex"),
}
INTERNAL_NAMESPACE_PATTERN = re.compile(
    r"reviewer(?:3|_3|-3)",
    flags=re.IGNORECASE,
)
REFEREE_LABEL_PATTERN = re.compile(
    r"reviewer(?:\s|~|\\~|-)+3",
    flags=re.IGNORECASE,
)
REVISION3_BUNDLE_NAME_PATTERN = re.compile(
    r"^revision3-results-[A-Za-z0-9._-]+$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
STALE_REPRO_PATHS = (
    Path("scripts") / ("run_remote_" + "gpu.sh"),
    Path("scripts") / ("run_lambda_" + "gpu.sh"),
    Path("scripts") / ("sweep_topology_" + "sampling.py"),
    Path("scripts") / ("audit_reviewer" + "3_topology.py"),
    Path("scripts") / ("build_reviewer" + "3_local.sh"),
    Path("scripts") / ("fetch_reviewer" + "3_results.sh"),
    Path("scripts") / ("package_reviewer" + "3_results.py"),
    Path("scripts") / ("render_reviewer" + "3_artifacts.py"),
    Path("scripts") / ("reproduce_reviewer" + "3.sh"),
    Path("scripts") / ("reviewer" + "3_source_manifest.py"),
    Path("scripts") / ("run_remote_reviewer" + "3.sh"),
    Path("scripts") / ("run_reviewer" + "3_inside_container.sh"),
    Path("scripts") / ("verify_reviewer" + "3_bundle.py"),
    Path("data") / ("remote_json_" + "logs"),
    Path("data") / ("remote_json_" + "logs_full"),
    Path("data") / ("remote_embedding_" + "pngs"),
    Path("data") / ("remote_embedding_" + "pngs_full"),
    Path("data") / ("topology_sampling_" + "sweep"),
    Path("data") / ("reviewer" + "3"),
    Path("generated") / ("reviewer" + "3"),
)


def normalize_latex_title(value: str) -> str:
    value = value.replace(r"\\", " ")
    return " ".join(value.split())


def manuscript_title(source: str) -> str:
    match = re.search(r"\\title\{([^{}]*)\}", source, flags=re.DOTALL)
    if not match:
        raise ValueError("dire_short.tex has no simple \\title{...} declaration")
    return normalize_latex_title(match.group(1))


def matching_brace(text: str, start: int) -> int:
    """Return the close brace paired with ``text[start]``."""

    depth = 0
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unmatched brace")


def remove_latex_command_arguments(text: str, command: str) -> str:
    """Remove nested command arguments while retaining surrounding prose."""

    needle = re.compile(rf"\\{re.escape(command)}\s*\{{")
    position = 0
    while True:
        match = needle.search(text, position)
        if not match:
            return text
        brace = match.end() - 1
        try:
            end = matching_brace(text, brace)
        except ValueError:
            return text
        text = text[: match.start()] + " " + text[end + 1 :]
        position = match.start() + 1


def scope_concession_errors(relative: Path, text: str) -> list[str]:
    """Detect semantic variants of the withdrawn method-level framing."""

    # The response must reproduce the referee's false premise accurately.
    # Audit the authors' prose while excluding text inside \reviewcomment.
    if relative == Path("response_to_referees_round2.tex"):
        text = remove_latex_command_arguments(text, "reviewcomment")
    return [
        f"{relative}: contains withdrawn semantic scope framing ({label})"
        for label, pattern in WITHDRAWN_SCOPE_PATTERNS
        if pattern.search(text)
    ]


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative == Path("scripts/check_authorial_invariants.py"):
            continue
        if relative.parts[:2] == ("data", "revision3"):
            continue
        yield path


def read_json_object(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"{path}: missing JSON artifact")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{path}: invalid JSON ({error})")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path}: JSON root is not an object")
        return None
    return payload


def tree_hashes(root: Path) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}
    if not root.is_dir() or root.is_symlink():
        return records
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        records[path.relative_to(root).as_posix()] = (
            path.stat().st_size,
            digest.hexdigest(),
        )
    return records


def check_revision3_namespace(root: Path) -> list[str]:
    """Keep referee identity separate from the Revision 3 artifact namespace."""

    errors: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if INTERNAL_NAMESPACE_PATTERN.search(relative.as_posix()):
            errors.append(f"{relative}: obsolete internal artifact namespace")
        if (
            not path.is_file()
            or path.suffix.lower() not in TEXT_SUFFIXES
            or relative == Path("scripts/check_authorial_invariants.py")
        ):
            continue
        text = path.read_text(encoding="utf-8")
        if INTERNAL_NAMESPACE_PATTERN.search(text):
            errors.append(f"{relative}: obsolete internal namespace in source")
        if (
            relative not in REFEREE_PROSE_FILES
            and REFEREE_LABEL_PATTERN.search(text)
        ):
            errors.append(
                f"{relative}: referee label used for a Revision 3 artifact"
            )
    return errors


def check_stale_repro_surfaces(root: Path) -> list[str]:
    """Reject retired runner paths and references to their output trees."""

    errors: list[str] = []
    stale_fragments = tuple(path.as_posix() for path in STALE_REPRO_PATHS)
    for relative in STALE_REPRO_PATHS:
        if (root / relative).exists():
            errors.append(f"{relative}: stale reproduction path still exists")
    for path in iter_text_files(root):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root)
        for fragment in stale_fragments:
            if fragment in text:
                errors.append(
                    f"{relative}: references stale reproduction path "
                    f"{fragment!r}"
                )
    return errors


def check_revision3_generated_contract(root: Path) -> list[str]:
    """Prove committed Revision 3 artifacts belong to the current rerun."""
    errors: list[str] = []
    generated = root / "generated" / "revision3"
    for filename in REQUIRED_REVISION3_GENERATED_FILES:
        path = generated / filename
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(
                f"generated/revision3/{filename}: missing or empty"
            )

    retired_filename_fragments = (
        "topology-sample-size",
        "topology_sample_",
        "topology_size_sample_",
    )
    scan_roots = (
        generated,
        root / "docs" / "_static" / "paper" / "generated" / "revision3",
        root
        / "docs"
        / "_build"
        / "html"
        / "_static"
        / "paper"
        / "generated"
        / "revision3",
    )
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            errors.append(
                f"{scan_root.relative_to(root)}: generated copy is missing"
            )
            continue
        for filename in REQUIRED_REVISION3_GENERATED_FILES:
            path = scan_root / filename
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(
                    f"{path.relative_to(root)}: missing or empty generated copy"
                )
                continue
            canonical = generated / filename
            if (
                scan_root != generated
                and canonical.is_file()
                and path.read_bytes() != canonical.read_bytes()
            ):
                errors.append(
                    f"{path.relative_to(root)}: stale copy differs from "
                    f"generated/revision3/{filename}"
                )
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(fragment in path.name for fragment in retired_filename_fragments):
                errors.append(f"{relative}: retired artifact filename")
            if path.suffix.lower() not in {
                ".csv",
                ".json",
                ".md",
                ".rst",
                ".tex",
                ".txt",
            }:
                continue
            text = path.read_text(encoding="utf-8")
            for term in RETIRED_REVISION3_TERMS:
                if term in text:
                    errors.append(
                        f"{relative}: contains retired Revision 3 term {term!r}"
                    )

    canonical_trees = (
        (
            generated,
            root / "docs" / "_static" / "paper" / "generated" / "revision3",
        ),
        (
            generated,
            root
            / "docs"
            / "_build"
            / "html"
            / "_static"
            / "paper"
            / "generated"
            / "revision3",
        ),
        (
            root / "pics",
            root / "docs" / "_static" / "paper" / "pics",
        ),
        (
            root / "pics",
            root
            / "docs"
            / "_build"
            / "html"
            / "_static"
            / "paper"
            / "pics",
        ),
    )
    for canonical, downstream in canonical_trees:
        for tree in (canonical, downstream):
            if tree.is_dir() and not tree.is_symlink():
                linked_entries = sorted(
                    path.relative_to(root).as_posix()
                    for path in tree.rglob("*")
                    if path.is_symlink()
                )
                if linked_entries:
                    errors.append(
                        f"{tree.relative_to(root)}: linked artifact entries "
                        f"are not permitted: {linked_entries[:5]}"
                    )
        canonical_records = tree_hashes(canonical)
        downstream_records = tree_hashes(downstream)
        if not canonical_records:
            errors.append(
                f"{canonical.relative_to(root)}: canonical artifact tree is "
                "missing or empty"
            )
            continue
        if canonical_records != downstream_records:
            missing = sorted(set(canonical_records) - set(downstream_records))
            extra = sorted(set(downstream_records) - set(canonical_records))
            changed = sorted(
                name
                for name in set(canonical_records) & set(downstream_records)
                if canonical_records[name] != downstream_records[name]
            )
            errors.append(
                f"{downstream.relative_to(root)}: downstream artifact tree "
                "differs from canonical "
                f"(missing={missing[:5]}, extra={extra[:5]}, "
                f"changed={changed[:5]})"
            )

    for relative in (
        Path("docs/paper.rst"),
        Path("docs/revision3.rst"),
        Path("docs/_build/html/paper.html"),
        Path("docs/_build/html/revision3.html"),
    ):
        path = root / relative
        if not path.is_file():
            errors.append(f"{relative}: generated downstream file is missing")
            continue
        text = path.read_text(encoding="utf-8")
        for term in RETIRED_REVISION3_TERMS:
            if term in text:
                errors.append(
                    f"{relative}: contains retired Revision 3 term {term!r}"
                )

    render_manifest = read_json_object(
        generated / "render_manifest.json",
        errors,
    )
    topology_audit = read_json_object(
        generated / "topology-audit.json",
        errors,
    )
    effect_audit = read_json_object(
        generated / "revision3-topology-effect-audit.json",
        errors,
    )
    centroid_audit = read_json_object(
        generated / "revision3-tenx-adjacency-audit.json",
        errors,
    )
    if render_manifest is not None and topology_audit is not None:
        if render_manifest.get("schema_version") != 2:
            errors.append("render_manifest.json: expected schema 2")
        source_bundle = (
            render_manifest.get("source_bundle", {})
            .get("directory_name")
        )
        if not source_bundle:
            errors.append("render_manifest.json: source bundle is missing")
        if topology_audit.get("source_bundle") != source_bundle:
            errors.append(
                "render/topology audit source-bundle identifiers disagree"
            )
        if topology_audit.get("status") != "success":
            errors.append("topology-audit.json: status is not success")
        if (
            render_manifest.get("topology_audit", {}).get("status")
            != "success"
        ):
            errors.append(
                "render_manifest.json: successful topology audit is not recorded"
            )

    if effect_audit is not None:
        if effect_audit.get("schema_version") != 2:
            errors.append(
                "revision3-topology-effect-audit.json: expected schema 2"
            )
        designs = effect_audit.get("topology_design", {})
        for dataset in ("tenx", "arxiv"):
            if (
                designs.get(dataset, {}).get("topology_subset_count")
                != 10
            ):
                errors.append(
                    f"topology effect audit: {dataset} does not record 10 subsets"
                )
        paired_effects = effect_audit.get("paired_effects")
        if not isinstance(paired_effects, list) or not paired_effects:
            errors.append("topology effect audit: paired effects are missing")
        else:
            subset_rows = [
                row for row in paired_effects
                if row.get("basis") == "subset"
            ]
            if not subset_rows:
                errors.append("topology effect audit: subset rows are missing")
            for row in subset_rows:
                if row.get("pair_id_field") != "subset_id":
                    errors.append(
                        "topology effect audit: subset pair ID field is wrong"
                    )
                if row.get("paired_record_count") != 10:
                    errors.append(
                        "topology effect audit: subset pair count is not 10"
                    )
                gaps = row.get("paired_gap_values")
                if not isinstance(gaps, list) or len(gaps) != 10:
                    errors.append(
                        "topology effect audit: not every subset gap is retained"
                    )
            for row in (
                row for row in paired_effects
                if row.get("basis") == "layout"
            ):
                if row.get("pair_id_field") != "layout_seed":
                    errors.append(
                        "topology effect audit: layout pair ID field is wrong"
                    )
                gaps = row.get("paired_gap_values")
                if (
                    not isinstance(gaps, list)
                    or row.get("paired_record_count") != len(gaps)
                ):
                    errors.append(
                        "topology effect audit: layout pair count is inconsistent"
                    )

    if centroid_audit is not None:
        for field in (
            "stored_reference_adjacency_matches_bundled_input_centroids",
            "stored_embedding_adjacency_matches_full_layout_recomputation",
        ):
            checks = centroid_audit.get(field)
            if (
                not isinstance(checks, dict)
                or not checks
                or not all(value is True for value in checks.values())
            ):
                errors.append(
                    f"revision3-tenx-adjacency-audit.json: {field} failed"
                )

    macros_path = generated / "revision3-results-macros.tex"
    if macros_path.is_file():
        macros = macros_path.read_text(encoding="utf-8")
        for command in (
            r"\newcommand{\TopologySubsetCount}{10}",
            r"\newcommand{\TopologyAuditComparisonCount}",
            r"\newcommand{\TopologyAuditMaximumAbsoluteDelta}",
            r"\newcommand{\TenxDireAutoAdjacencyPreserved}",
            r"\newcommand{\TenxCumlUmapAdjacencyPreserved}",
            r"\newcommand{\TenxDireAutoAdjacencyCellWeighted}",
            r"\newcommand{\TenxCumlUmapAdjacencyCellWeighted}",
            r"\newcommand{\TenxDireAutoAdjacencyFilteredEqual}",
            r"\newcommand{\TenxCumlUmapAdjacencyFilteredEqual}",
            r"\newcommand{\TenxDireAutoAdjacencyKOne}",
            r"\newcommand{\TenxCumlUmapAdjacencyKOne}",
        ):
            if command not in macros:
                errors.append(
                    f"revision3-results-macros.tex: missing {command}"
                )
    return errors


def check_revision3_artifact_cleanliness(root: Path) -> list[str]:
    """Require one verified local run and reject stale/failed fetch remnants."""

    errors: list[str] = []
    data_root = root / "data" / "revision3"
    downloads = data_root / "downloads"
    fetched = data_root / "fetched"
    current_path = fetched / "CURRENT"

    if not downloads.is_dir() or downloads.is_symlink():
        errors.append("data/revision3/downloads: final download directory is missing")
    if not fetched.is_dir() or fetched.is_symlink():
        errors.append("data/revision3/fetched: final extraction directory is missing")
        return errors
    if not current_path.is_file() or current_path.is_symlink():
        errors.append("data/revision3/fetched/CURRENT: final pointer is missing")
        return errors

    current = current_path.read_text(encoding="utf-8").strip()
    if not REVISION3_BUNDLE_NAME_PATTERN.fullmatch(current):
        errors.append(
            "data/revision3/fetched/CURRENT: unsafe or invalid bundle name "
            f"{current!r}"
        )
        return errors

    fetched_entries = sorted(
        path
        for path in fetched.iterdir()
        if path.name != "CURRENT"
    )
    expected_bundle = fetched / current
    if not expected_bundle.is_dir() or expected_bundle.is_symlink():
        errors.append(
            f"data/revision3/fetched/{current}: current bundle directory is missing"
        )
    unexpected_fetched = [
        path.relative_to(root).as_posix()
        for path in fetched_entries
        if path != expected_bundle
    ]
    if unexpected_fetched:
        errors.append(
            "data/revision3/fetched: stale or failed extraction entries remain: "
            f"{unexpected_fetched}"
        )

    if downloads.is_dir():
        expected_download_names = {
            f"{current}.tar.gz",
            f"{current}.tar.gz.sha256",
            f"{current}-topology-audit.json",
            f"{current}-topology-audit.json.sha256",
        }
        actual_download_names = {
            path.name for path in downloads.iterdir()
        }
        missing = sorted(expected_download_names - actual_download_names)
        unexpected = sorted(actual_download_names - expected_download_names)
        if missing:
            errors.append(
                "data/revision3/downloads: final bundle files are missing: "
                f"{missing}"
            )
        if unexpected:
            errors.append(
                "data/revision3/downloads: stale or superseded files remain: "
                f"{unexpected}"
            )
        for name in sorted(expected_download_names):
            path = downloads / name
            if path.exists() and (not path.is_file() or path.is_symlink()):
                errors.append(
                    f"data/revision3/downloads/{name}: expected a regular file"
                )

        hash_pairs = (
            (
                downloads / f"{current}.tar.gz",
                downloads / f"{current}.tar.gz.sha256",
            ),
            (
                downloads / f"{current}-topology-audit.json",
                downloads / f"{current}-topology-audit.json.sha256",
            ),
        )
        for payload_path, sidecar_path in hash_pairs:
            if not (
                payload_path.is_file()
                and not payload_path.is_symlink()
                and sidecar_path.is_file()
                and not sidecar_path.is_symlink()
            ):
                continue
            sidecar_tokens = sidecar_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).split()
            expected_hash = sidecar_tokens[0] if sidecar_tokens else ""
            if not SHA256_PATTERN.fullmatch(expected_hash):
                errors.append(
                    f"{sidecar_path.relative_to(root)}: invalid SHA-256 sidecar"
                )
                continue
            digest = hashlib.sha256()
            with payload_path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                errors.append(
                    f"{payload_path.relative_to(root)}: SHA-256 does not match "
                    f"{sidecar_path.name}"
                )

        audit_download = downloads / f"{current}-topology-audit.json"
        if audit_download.is_file() and not audit_download.is_symlink():
            try:
                audit = json.loads(audit_download.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                errors.append(
                    f"{audit_download.relative_to(root)}: invalid JSON ({error})"
                )
            else:
                if audit.get("status") != "success":
                    errors.append(
                        f"{audit_download.relative_to(root)}: CPU audit is not "
                        "successful"
                    )
                if audit.get("source_bundle") != current:
                    errors.append(
                        f"{audit_download.relative_to(root)}: CPU audit source "
                        "does not match CURRENT"
                    )

    for path in data_root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        lowered = path.name.casefold()
        if (
            ".extracting-" in lowered
            or lowered.endswith(".partial")
            or lowered.endswith(".failed")
            or lowered.startswith("failed-")
            or lowered.startswith("interrupted-")
        ):
            errors.append(f"{relative}: stale/failed run remnant")
        if path.is_file() and path.name == "pipeline.status":
            status = path.read_text(encoding="utf-8", errors="replace")
            if "state=success" not in status:
                errors.append(f"{relative}: pipeline status is not successful")
        if path.is_file() and path.name == "result.json":
            try:
                result_payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                errors.append(f"{relative}: invalid result JSON ({error})")
            else:
                if result_payload.get("status") != "success":
                    errors.append(
                        f"{relative}: retained reducer run is not successful"
                    )

    render_manifest_path = (
        root / "generated" / "revision3" / "render_manifest.json"
    )
    if not render_manifest_path.is_file() or render_manifest_path.is_symlink():
        errors.append(
            "generated/revision3/render_manifest.json: final render manifest "
            "is missing"
        )
    else:
        try:
            render_manifest = json.loads(
                render_manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            errors.append(
                "generated/revision3/render_manifest.json: "
                f"invalid JSON ({error})"
            )
        else:
            if render_manifest.get("schema_version") != 2:
                errors.append(
                    "generated/revision3/render_manifest.json: unsupported "
                    "schema"
                )
            rendered_bundle = (
                render_manifest.get("source_bundle", {})
                .get("directory_name")
            )
            if rendered_bundle != current:
                errors.append(
                    "generated/revision3/render_manifest.json: rendered source "
                    f"{rendered_bundle!r} does not match CURRENT {current!r}"
                )
            records = render_manifest.get("outputs")
            generated_root = render_manifest_path.parent
            if not isinstance(records, list):
                errors.append(
                    "generated/revision3/render_manifest.json: outputs is not "
                    "a list"
                )
            else:
                actual = {
                    path.relative_to(generated_root).as_posix(): path
                    for path in generated_root.rglob("*")
                    if path.is_file()
                    and path.relative_to(generated_root).as_posix()
                    != "render_manifest.json"
                }
                declared: dict[str, dict] = {}
                for record in records:
                    if not isinstance(record, dict):
                        errors.append(
                            "generated/revision3/render_manifest.json: "
                            "non-object output record"
                        )
                        continue
                    relative_name = str(record.get("path", ""))
                    relative_path = Path(relative_name)
                    if (
                        not relative_name
                        or relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or relative_name in declared
                    ):
                        errors.append(
                            "generated/revision3/render_manifest.json: "
                            f"unsafe/duplicate output path {relative_name!r}"
                        )
                        continue
                    declared[relative_name] = record
                if set(declared) != set(actual):
                    errors.append(
                        "generated/revision3/render_manifest.json: output "
                        "membership differs from disk"
                    )
                for relative_name in sorted(set(declared) & set(actual)):
                    path = actual[relative_name]
                    record = declared[relative_name]
                    if path.is_symlink() or not path.is_file():
                        errors.append(
                            f"{path.relative_to(root)}: expected a regular file"
                        )
                        continue
                    if path.stat().st_size != record.get("bytes"):
                        errors.append(
                            f"{path.relative_to(root)}: byte count differs from "
                            "generated-artifact manifest"
                        )
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest != record.get("sha256"):
                        errors.append(
                            f"{path.relative_to(root)}: SHA-256 differs from "
                            "generated-artifact manifest"
                        )

    generated_audit = root / "generated" / "revision3" / "topology-audit.json"
    downloaded_audit = downloads / f"{current}-topology-audit.json"
    if not generated_audit.is_file() or generated_audit.is_symlink():
        errors.append(
            "generated/revision3/topology-audit.json: final CPU audit is missing"
        )
    elif downloaded_audit.is_file() and not downloaded_audit.is_symlink():
        if generated_audit.read_bytes() != downloaded_audit.read_bytes():
            errors.append(
                "generated/revision3/topology-audit.json: installed audit "
                "differs from the verified download"
            )

    pics_root = root / "pics"
    pics_manifest_path = pics_root / "render_manifest.json"
    if not pics_manifest_path.is_file() or pics_manifest_path.is_symlink():
        errors.append("pics/render_manifest.json: final picture manifest is missing")
    else:
        try:
            pics_manifest = json.loads(
                pics_manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            errors.append(
                f"pics/render_manifest.json: invalid JSON ({error})"
            )
        else:
            if pics_manifest.get("schema_version") != 1:
                errors.append("pics/render_manifest.json: unsupported schema")
            if pics_manifest.get("source_bundle") != current:
                errors.append(
                    "pics/render_manifest.json: source bundle does not match "
                    "CURRENT"
                )
            if (
                pics_manifest.get("source_mode")
                != "verified-revision3-bundle-small-suite"
            ):
                errors.append(
                    "pics/render_manifest.json: final pictures do not record "
                    "the verified bundle small suite"
                )

            def verify_records(
                label: str,
                records,
                directory: Path,
                *,
                excluded_names: set[str] | None = None,
            ) -> None:
                exclusions = excluded_names or set()
                if not isinstance(records, list):
                    errors.append(f"pics/render_manifest.json: {label} is not a list")
                    return
                actual = {
                    path.relative_to(directory).as_posix(): path
                    for path in directory.rglob("*")
                    if path.is_file()
                    and path.relative_to(directory).as_posix() not in exclusions
                } if directory.is_dir() and not directory.is_symlink() else {}
                declared: dict[str, dict] = {}
                for record in records:
                    if not isinstance(record, dict):
                        errors.append(
                            f"pics/render_manifest.json: {label} has a "
                            "non-object record"
                        )
                        continue
                    relative_name = str(record.get("path", ""))
                    relative_path = Path(relative_name)
                    if (
                        not relative_name
                        or relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or relative_name in declared
                    ):
                        errors.append(
                            f"pics/render_manifest.json: unsafe/duplicate "
                            f"{label} path {relative_name!r}"
                        )
                        continue
                    declared[relative_name] = record
                if set(declared) != set(actual):
                    errors.append(
                        f"pics/render_manifest.json: {label} membership "
                        "differs from disk"
                    )
                for relative_name in sorted(set(declared) & set(actual)):
                    path = actual[relative_name]
                    record = declared[relative_name]
                    if path.is_symlink() or not path.is_file():
                        errors.append(
                            f"{path.relative_to(root)}: expected a regular file"
                        )
                        continue
                    if path.stat().st_size != record.get("bytes"):
                        errors.append(
                            f"{path.relative_to(root)}: byte count differs from "
                            "picture manifest"
                        )
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest != record.get("sha256"):
                        errors.append(
                            f"{path.relative_to(root)}: SHA-256 differs from "
                            "picture manifest"
                        )

            verify_records(
                "source_json",
                pics_manifest.get("source_json"),
                expected_bundle / "small_suite" / "json_logs",
            )
            verify_records(
                "source_embedding_pngs",
                pics_manifest.get("source_embedding_pngs"),
                expected_bundle / "small_suite" / "embedding_pngs",
            )
            verify_records(
                "outputs",
                pics_manifest.get("outputs"),
                pics_root,
                excluded_names={"render_manifest.json"},
            )
    return errors


def check(
    root: Path,
    require_generated: bool = False,
    require_clean_local_artifacts: bool = False,
) -> list[str]:
    errors: list[str] = check_revision3_namespace(root)
    errors.extend(check_stale_repro_surfaces(root))
    manuscript_path = root / "dire_short.tex"
    manuscript = manuscript_path.read_text(encoding="utf-8")

    if EXPECTED_TITLE_DECLARATION not in manuscript:
        errors.append(
            "dire_short.tex: exact pre-Revision-3 title declaration not found"
        )

    try:
        title = manuscript_title(manuscript)
    except ValueError as error:
        errors.append(str(error))
    else:
        if title != EXPECTED_TITLE:
            errors.append(
                f"manuscript title is {title!r}, expected {EXPECTED_TITLE!r}"
            )

    for path in iter_text_files(root):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root)
        errors.extend(scope_concession_errors(relative, text))
        if WITHDRAWN_TITLE_FRAGMENT.casefold() in text.casefold():
            errors.append(f"{relative}: contains withdrawn title")
        for fragment in WITHDRAWN_SCOPE_FRAGMENTS:
            if fragment.casefold() in text.casefold():
                errors.append(
                    f"{relative}: contains withdrawn scope wording {fragment!r}"
                )
        for term in RETIRED_REVISION3_TERMS:
            if term in text:
                errors.append(
                    f"{relative}: contains retired Revision 3 term {term!r}"
                )
        for label, pattern in RETIRED_REVISION3_PATTERNS:
            if pattern.search(text):
                errors.append(
                    f"{relative}: contains retired Revision 3 {label}"
                )

    required_source_phrases = {
        Path("dire_short.tex"): "user--specified target dimension",
        Path("response_to_referees_round2.tex"): (
            r"outright \emph{incompetent}"
        ),
        Path("README.md"): "user-specified target dimension",
        Path("data/README.md"): "user-specified target dimension",
        Path("docs/index.rst"): "user-specified target dimension",
        Path("scripts/audit_revision3_topology.py"): (
            "user-specified target dimension"
        ),
    }
    for relative, phrase in required_source_phrases.items():
        text = (root / relative).read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        if phrase not in normalized_text:
            errors.append(f"{relative}: missing required wording {phrase!r}")

    manuscript_requirements = (
        PRE_REVISION3_ABSTRACT,
        PRE_REVISION3_NONVISUAL_SCOPE,
        PRE_REVISION3_CONCLUSION,
        r"\texttt{n\_components}",
        "symmetrised unweighted kNN graph",
        "cuML TruncatedSVD",
        r"\texttt{normalize=True}",
        "peak incremental device--memory use",
    )
    for phrase in manuscript_requirements:
        if phrase not in manuscript:
            errors.append(f"dire_short.tex: missing code-grounded wording {phrase!r}")

    response_path = root / "response_to_referees_round2.tex"
    response = response_path.read_text(encoding="utf-8")
    response_requirements = (
        "experiments only cover 2D visualization use cases",
        r"\texttt{n\_components}",
        "inference that DiRe is therefore a two--dimensional method is outright",
        r"\emph{hallucinated implementation premise}",
        r"\emph{hallucinated characterization}",
        "The exact submitted title and the pre--Reviewer~3 general framing are",
        "No scope concession was made.",
        "We did not adopt the requested",
        "visualization--only reframing.",
    )
    for phrase in response_requirements:
        if phrase not in response:
            errors.append(
                "response_to_referees_round2.tex: "
                f"missing reviewer-scope wording {phrase!r}"
            )
    for relative, text in (
        (Path("dire_short.tex"), manuscript),
        (response_path, response),
    ):
        if "A10" in text:
            errors.append(
                f"{relative}: stale A10 claim remains in the fresh H100 "
                "manuscript record"
            )

    for relative in (
        Path("response_to_referees_round2.tex"),
        Path("editor_cover_letter_round2.tex"),
    ):
        text = (root / relative).read_text(encoding="utf-8")
        normalized = " ".join(text.replace(r"\\", " ").split())
        if EXPECTED_TITLE not in normalized:
            errors.append(f"{relative}: exact submitted title not found")
        if "user-specified target dimension" not in text.replace("--", "-"):
            errors.append(
                f"{relative}: corrected method/experiment scope not found"
            )

    cover = (root / "editor_cover_letter_round2.tex").read_text(encoding="utf-8")
    cover_requirements = (
        "Reviewer~2",
        "``adequately addressed.''",
        "substantially overlapping grounds",
        "shameful practice of moving the goalposts",
        "considerable additional work",
        "matters already addressed and accepted",
        "retaining the exact submitted title",
        "pre--Reviewer~3 general framing",
        "did not accept Reviewer~3's proposed",
        "visualization-only framing",
        r"\input{generated/revision3/revision3-results-macros.tex}",
    )
    normalized_cover = " ".join(cover.split())
    for phrase in cover_requirements:
        if phrase not in normalized_cover:
            errors.append(
                "editor_cover_letter_round2.tex: "
                f"missing review-sequence wording {phrase!r}"
            )

    generated_page = root / "docs" / "paper.rst"
    generated_html = root / "docs" / "_build" / "html" / "paper.html"
    if require_generated:
        for path in (generated_page, generated_html):
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"{path.relative_to(root)}: missing generated output")
        errors.extend(check_revision3_generated_contract(root))
    if require_clean_local_artifacts:
        errors.extend(check_revision3_artifact_cleanliness(root))
    if generated_page.is_file():
        generated_text = generated_page.read_text(encoding="utf-8")
        first_line = generated_text.splitlines()[0]
        if first_line != EXPECTED_TITLE:
            errors.append(
                f"docs/paper.rst title is {first_line!r}, "
                f"expected {EXPECTED_TITLE!r}"
            )
        converted_requirements = (
            PRE_REVISION3_ABSTRACT.replace("--", "-"),
            PRE_REVISION3_NONVISUAL_SCOPE.replace("--", "-"),
            "A new dimensionality reduction framework, called ``DiRe``, "
            "has been developed with an explicit emphasis on preserving and "
            "measuring the dataset's homological structure.",
        )
        for phrase in converted_requirements:
            if phrase not in generated_text:
                errors.append(
                    "docs/paper.rst: missing pre-Revision-3 framing "
                    f"{phrase!r}"
                )
        if "{}" in generated_text:
            errors.append(
                "docs/paper.rst: contains an unexpanded empty LaTeX group"
            )
        if re.search(r"(?<=[A-Za-z0-9`\]])(?=:math:`)", generated_text):
            errors.append(
                "docs/paper.rst: inline math role lacks a left markup boundary"
            )
    if generated_html.is_file():
        html = generated_html.read_text(encoding="utf-8")
        if EXPECTED_TITLE not in html:
            errors.append("docs/_build/html/paper.html: expected title not found")
        if WITHDRAWN_TITLE_FRAGMENT.casefold() in html.casefold():
            errors.append(
                "docs/_build/html/paper.html: contains withdrawn title"
            )
        if ":math:" in html or re.search(
            r"<cite>(?:pm|times|rightarrow)</cite>",
            html,
        ):
            errors.append(
                "docs/_build/html/paper.html: contains leaked "
                "reStructuredText inline-math markup"
            )

    html_root = root / "docs" / "_build" / "html"
    if html_root.is_dir():
        for path in html_root.rglob("*.html"):
            if "_static" in path.relative_to(html_root).parts:
                continue
            html = path.read_text(encoding="utf-8")
            relative = path.relative_to(root)
            if WITHDRAWN_TITLE_FRAGMENT.casefold() in html.casefold():
                errors.append(f"{relative}: contains withdrawn title")
            errors.extend(scope_concession_errors(relative, html))
            for fragment in WITHDRAWN_SCOPE_FRAGMENTS:
                if fragment.casefold() in html.casefold():
                    errors.append(
                        f"{relative}: contains withdrawn scope wording "
                        f"{fragment!r}"
                    )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--require-generated", action="store_true")
    parser.add_argument(
        "--require-clean-local-artifacts",
        action="store_true",
        help=(
            "also require exactly one matching local downloaded/extracted "
            "Revision 3 bundle and reject failed or superseded remnants"
        ),
    )
    args = parser.parse_args()

    errors = check(
        args.root.resolve(),
        require_generated=args.require_generated,
        require_clean_local_artifacts=args.require_clean_local_artifacts,
    )
    if errors:
        raise SystemExit(
            "Authorial invariant check failed:\n- " + "\n- ".join(errors)
        )
    print("Authorial invariants: OK")


if __name__ == "__main__":
    main()
