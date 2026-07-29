from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_figures import (  # noqa: E402
    direction_adjusted_log2_ratio,
    prepare_pics_root,
    save_deterministic_pdf,
    write_pics_manifest,
)
from render_revision3_artifacts import large_quality_log2_effect  # noqa: E402


def test_higher_is_better_ratio_is_positive_for_better_comparator() -> None:
    assert direction_adjusted_log2_ratio("neighborhood", 0.8, 0.4) == 1.0
    assert direction_adjusted_log2_ratio("neighborhood", 0.2, 0.4) == -1.0


def test_lower_is_better_ratio_is_positive_for_better_comparator() -> None:
    assert direction_adjusted_log2_ratio("stress", 0.5, 1.0) == 1.0
    assert direction_adjusted_log2_ratio("stress", 2.0, 1.0) == -1.0


def test_context_uses_absolute_loss_and_preserves_exact_zero() -> None:
    assert direction_adjusted_log2_ratio("context", 0.0, 0.0) == 0.0
    assert math.isinf(direction_adjusted_log2_ratio("context", -0.1, 0.0))
    assert direction_adjusted_log2_ratio("context", -0.1, 0.0) < 0
    assert math.isinf(direction_adjusted_log2_ratio("context", 0.0, -0.1))
    assert direction_adjusted_log2_ratio("context", 0.0, -0.1) > 0


def test_negative_noncontext_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        direction_adjusted_log2_ratio("stress", -0.1, 0.2)


def test_large_quality_effect_uses_production_dire_as_zero_reference() -> None:
    assert large_quality_log2_effect("knn", 0.4, 0.4) == 0.0
    assert large_quality_log2_effect("knn", 0.8, 0.4) == 1.0
    assert large_quality_log2_effect("bottleneck_beta0", 0.5, 1.0) == 1.0


def test_large_quality_correlation_uses_documented_unit_interval_map() -> None:
    assert large_quality_log2_effect("centroid", 1.0, 0.0) == 1.0
    assert large_quality_log2_effect("centroid", 0.0, 1.0) == -1.0
    with pytest.raises(ValueError):
        large_quality_log2_effect("centroid", 1.1, 0.0)


def test_picture_manifest_replaces_stale_files_and_records_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "revision3-results-final"
    data_root = bundle / "small_suite" / "json_logs"
    embedding_root = bundle / "small_suite" / "embedding_pngs"
    pics = tmp_path / "output" / "pics"
    data_root.mkdir(parents=True)
    embedding_root.mkdir(parents=True)
    pics.mkdir(parents=True)
    (pics / "stale.png").write_bytes(b"stale")

    prepare_pics_root(pics)
    (data_root / "blobs.json").write_text(
        '{"status":"success"}\n',
        encoding="utf-8",
    )
    (embedding_root / "blobs-dire.png").write_bytes(b"png fixture")
    (pics / "runtime-summary.png").write_bytes(b"rendered")
    write_pics_manifest(pics, data_root, embedding_root)

    assert not (pics / "stale.png").exists()
    manifest = json.loads(
        (pics / "render_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_bundle"] == "revision3-results-final"
    assert manifest["source_mode"] == "verified-revision3-bundle-small-suite"
    assert manifest["source_json"][0]["path"] == "blobs.json"
    assert manifest["outputs"][0]["path"] == "runtime-summary.png"


def test_canonical_picture_pdf_is_byte_stable(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    axis.bar([0, 1], [1.0, 2.0])
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"

    save_deterministic_pdf(figure, first)
    save_deterministic_pdf(figure, second)
    plt.close(figure)

    first_bytes = first.read_bytes()
    assert first_bytes == second.read_bytes()
    assert b"/CreationDate" not in first_bytes
    assert b"/ModDate" not in first_bytes
