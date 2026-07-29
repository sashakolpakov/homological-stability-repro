from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_authorial_invariants import (  # noqa: E402
    STALE_REPRO_PATHS,
    check_revision3_artifact_cleanliness,
    check_revision3_generated_contract,
    check_revision3_namespace,
    check_stale_repro_surfaces,
    scope_concession_errors,
)


def write_clean_artifact_fixture(root: Path, bundle_name: str) -> None:
    downloads = root / "data" / "revision3" / "downloads"
    fetched = root / "data" / "revision3" / "fetched"
    generated = root / "generated" / "revision3"
    downloads.mkdir(parents=True)
    fetched.mkdir(parents=True)
    generated.mkdir(parents=True)
    (fetched / "CURRENT").write_text(f"{bundle_name}\n", encoding="utf-8")
    bundle = fetched / bundle_name
    json_root = bundle / "small_suite" / "json_logs"
    embedding_root = bundle / "small_suite" / "embedding_pngs"
    json_root.mkdir(parents=True)
    embedding_root.mkdir(parents=True)
    (json_root / "blobs.json").write_text(
        '{"status":"success"}\n',
        encoding="utf-8",
    )
    (embedding_root / "blobs-dire.png").write_bytes(b"png fixture")

    archive = downloads / f"{bundle_name}.tar.gz"
    archive.write_bytes(b"verified bundle fixture")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (downloads / f"{bundle_name}.tar.gz.sha256").write_text(
        f"{archive_hash}  {archive.name}\n",
        encoding="utf-8",
    )

    audit_payload = {
        "source_bundle": bundle_name,
        "status": "success",
    }
    audit_bytes = (json.dumps(audit_payload, sort_keys=True) + "\n").encode()
    audit = downloads / f"{bundle_name}-topology-audit.json"
    audit.write_bytes(audit_bytes)
    audit_hash = hashlib.sha256(audit_bytes).hexdigest()
    (downloads / f"{bundle_name}-topology-audit.json.sha256").write_text(
        f"{audit_hash}  {audit.name}\n",
        encoding="utf-8",
    )
    installed_audit = generated / "topology-audit.json"
    installed_audit.write_bytes(audit_bytes)
    pics = root / "pics"
    pics.mkdir()
    (pics / "runtime-summary.png").write_bytes(b"rendered")

    def record(path: Path, directory: Path) -> dict:
        return {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    (generated / "render_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_bundle": {"directory_name": bundle_name},
                "outputs": [record(installed_audit, generated)],
            }
        ),
        encoding="utf-8",
    )
    (pics / "render_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_bundle": bundle_name,
                "source_mode": "verified-revision3-bundle-small-suite",
                "source_json": [record(json_root / "blobs.json", json_root)],
                "source_embedding_pngs": [
                    record(
                        embedding_root / "blobs-dire.png",
                        embedding_root,
                    )
                ],
                "outputs": [
                    record(pics / "runtime-summary.png", pics)
                ],
            }
        ),
        encoding="utf-8",
    )


class AuthorialInvariantTests(unittest.TestCase):
    def test_semantic_scope_audit_rejects_dimension_and_graphics_narrowing(
        self,
    ) -> None:
        text = (
            "DiRe is a " + "two-dimensional embedding method. "
            "DiRe is a " + "graphical tool."
        )

        errors = scope_concession_errors(Path("paper.md"), text)

        self.assertEqual(len(errors), 2, errors)

    def test_semantic_scope_audit_accepts_explicit_negation_and_benchmark_scope(
        self,
    ) -> None:
        text = (
            "DiRe is not a " + "visualization-only method. "
            "This benchmark configures DiRe with d=2."
        )

        errors = scope_concession_errors(Path("paper.md"), text)

        self.assertEqual(errors, [])

    def test_semantic_scope_audit_excludes_reviewer_quote_but_checks_response(
        self,
    ) -> None:
        quoted = (
            "\\reviewcomment{If DiRe is a "
            + "visualization method, it should be framed as such.}\n"
            "\\response DiRe accepts a user-selected output dimension."
        )
        conceded = quoted.replace(
            "DiRe accepts a user-selected output dimension.",
            "DiRe is a " + "plotting tool.",
        )

        self.assertEqual(
            scope_concession_errors(
                Path("response_to_referees_round2.tex"),
                quoted,
            ),
            [],
        )
        self.assertTrue(
            scope_concession_errors(
                Path("response_to_referees_round2.tex"),
                conceded,
            )
        )

    def test_submission_sources_preserve_authorial_invariants(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_authorial_invariants.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generated_contract_rejects_retired_schema_terms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated" / "revision3"
            static = (
                root
                / "docs"
                / "_static"
                / "paper"
                / "generated"
                / "revision3"
            )
            built_static = (
                root
                / "docs"
                / "_build"
                / "html"
                / "_static"
                / "paper"
                / "generated"
                / "revision3"
            )
            for directory in (generated, static, built_static):
                directory.mkdir(parents=True)
            (generated / "obsolete.json").write_text(
                json.dumps({"R_" + "sample": 3}),
                encoding="utf-8",
            )
            for relative in (
                Path("docs/paper.rst"),
                Path("docs/revision3.rst"),
                Path("docs/_build/html/paper.html"),
                Path("docs/_build/html/revision3.html"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")

            errors = check_revision3_generated_contract(root)

            self.assertTrue(
                any("retired Revision 3 term" in error for error in errors),
                errors,
            )

    def test_generated_contract_rejects_stale_downstream_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated" / "revision3"
            static = (
                root
                / "docs"
                / "_static"
                / "paper"
                / "generated"
                / "revision3"
            )
            built_static = (
                root
                / "docs"
                / "_build"
                / "html"
                / "_static"
                / "paper"
                / "generated"
                / "revision3"
            )
            for directory in (generated, static, built_static):
                directory.mkdir(parents=True)
            filename = "revision3-results-macros.tex"
            (generated / filename).write_text("current", encoding="utf-8")
            (static / filename).write_text("stale", encoding="utf-8")
            (built_static / filename).write_text("current", encoding="utf-8")
            for relative in (
                Path("docs/paper.rst"),
                Path("docs/revision3.rst"),
                Path("docs/_build/html/paper.html"),
                Path("docs/_build/html/revision3.html"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")

            errors = check_revision3_generated_contract(root)

            self.assertTrue(
                any("stale copy differs" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any(
                    "downstream artifact tree differs from canonical" in error
                    for error in errors
                ),
                errors,
            )

    def test_internal_namespace_rejects_referee_named_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_token = "review" + "er3"
            path = root / "scripts" / f"build_{bad_token}.sh"
            path.parent.mkdir(parents=True)
            path.write_text("#!/bin/sh\n", encoding="utf-8")

            errors = check_revision3_namespace(root)

            self.assertTrue(
                any("obsolete internal artifact namespace" in error for error in errors),
                errors,
            )

    def test_stale_reproduction_path_is_rejected(self) -> None:
        for stale_relative in STALE_REPRO_PATHS:
            with self.subTest(path=stale_relative):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    stale = root / stale_relative
                    stale.parent.mkdir(parents=True, exist_ok=True)
                    if stale.suffix:
                        stale.write_text("fixture\n", encoding="utf-8")
                    else:
                        stale.mkdir()

                    errors = check_stale_repro_surfaces(root)

                    self.assertTrue(
                        any(
                            "stale reproduction path still exists" in error
                            for error in errors
                        ),
                        errors,
                    )

    def test_artifact_cleanliness_rejects_failed_and_superseded_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_name = "revision3-results-final"
            downloads = root / "data" / "revision3" / "downloads"
            fetched = root / "data" / "revision3" / "fetched"
            generated = root / "generated" / "revision3"
            downloads.mkdir(parents=True)
            fetched.mkdir(parents=True)
            generated.mkdir(parents=True)
            (fetched / "CURRENT").write_text(
                f"{bundle_name}\n",
                encoding="utf-8",
            )
            (fetched / bundle_name).mkdir()
            (fetched / ".revision3-results-old.extracting-123").mkdir()
            (fetched / "revision3-results-superseded").mkdir()
            for name in (
                f"{bundle_name}.tar.gz",
                f"{bundle_name}.tar.gz.sha256",
                f"{bundle_name}-topology-audit.json",
                f"{bundle_name}-topology-audit.json.sha256",
                "revision3-results-old.tar.gz",
            ):
                (downloads / name).write_text("fixture", encoding="utf-8")
            (generated / "render_manifest.json").write_text(
                json.dumps(
                    {"source_bundle": {"directory_name": bundle_name}}
                ),
                encoding="utf-8",
            )

            errors = check_revision3_artifact_cleanliness(root)

            self.assertTrue(
                any("stale or failed extraction entries" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("stale or superseded files remain" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("stale/failed run remnant" in error for error in errors),
                errors,
            )

    def test_artifact_cleanliness_accepts_one_matching_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_name = "revision3-results-final"
            write_clean_artifact_fixture(root, bundle_name)

            errors = check_revision3_artifact_cleanliness(root)

            self.assertEqual(errors, [])

    def test_artifact_cleanliness_rejects_hash_and_installed_audit_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_name = "revision3-results-final"
            write_clean_artifact_fixture(root, bundle_name)
            downloads = root / "data" / "revision3" / "downloads"
            generated = root / "generated" / "revision3"
            (downloads / f"{bundle_name}.tar.gz").write_bytes(b"tampered")
            (generated / "topology-audit.json").write_text(
                '{"status":"failed"}\n',
                encoding="utf-8",
            )

            errors = check_revision3_artifact_cleanliness(root)

            self.assertTrue(
                any("SHA-256 does not match" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("installed audit differs" in error for error in errors),
                errors,
            )


if __name__ == "__main__":
    unittest.main()
