from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_revision3_artifacts import (  # noqa: E402
    append_small_atlas_effect_macros,
    append_topology_effect_macros,
    artifact_records,
    best_method,
    make_small_atlas_topology_effects,
    make_small_runtime,
    make_tenx_adjacency_audit,
    make_topology_paired_effects,
    make_topology_subset_size_sensitivity,
    matching_topology_audit,
    prepare_output_root,
    result_path,
    save_deterministic_pdf,
)


def test_artifact_records_are_exact_hashed_and_exclude_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "table.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "figure.png").write_bytes(b"png")
    (tmp_path / "render_manifest.json").write_text(
        "old manifest",
        encoding="utf-8",
    )

    records = artifact_records(
        tmp_path,
        exclude={"render_manifest.json"},
    )

    assert [record["path"] for record in records] == [
        "figure.png",
        "nested/table.csv",
    ]
    assert [record["bytes"] for record in records] == [3, 8]
    assert all(len(str(record["sha256"])) == 64 for record in records)


def test_generated_pdf_omits_wall_clock_metadata_and_is_byte_stable(
    tmp_path: Path,
) -> None:
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 0.0])
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"

    save_deterministic_pdf(figure, first)
    save_deterministic_pdf(figure, second)
    plt.close(figure)

    first_bytes = first.read_bytes()
    assert first_bytes == second.read_bytes()
    assert b"/CreationDate" not in first_bytes
    assert b"/ModDate" not in first_bytes


def test_metric_leaders_can_be_reported_for_all_and_nonlinear_methods() -> None:
    rows = [
        {"dataset": "tenx", "method": "pca2", "score": 0.9},
        {"dataset": "tenx", "method": "dire_auto", "score": 0.85},
        {"dataset": "tenx", "method": "dire", "score": 0.7},
        {"dataset": "tenx", "method": "cuml_umap", "score": 0.8},
        {"dataset": "tenx", "method": "cuml_tsne", "score": 0.6},
    ]

    assert best_method(rows, "tenx", "score", True)["method"] == "pca2"
    assert (
        best_method(
            rows,
            "tenx",
            "score",
            True,
            ("dire_auto", "cuml_umap", "cuml_tsne"),
        )["method"]
        == "dire_auto"
    )


def test_metric_summary_retains_exact_optima_and_comparability_band() -> None:
    rows = [
        {
            "dataset": "tenx",
            "method": "dire_auto",
            "display": "DiRe auto",
            "score": 0.4,
        },
        {
            "dataset": "tenx",
            "method": "cuml_umap",
            "display": "UMAP",
            "score": 0.4,
        },
        {
            "dataset": "tenx",
            "method": "cuml_tsne",
            "display": "t-SNE",
            "score": 0.3,
        },
    ]

    winner = best_method(rows, "tenx", "score", True)

    assert winner["unique_exact_optimum"] is False
    assert winner["all_candidates_exactly_equal"] is False
    assert winner["exact_optimum_methods"] == ["dire_auto", "cuml_umap"]
    assert winner["exact_optimum_displays"] == ["DiRe auto", "UMAP"]
    assert winner["practically_comparable_methods"] == [
        "dire_auto",
        "cuml_umap",
    ]

    all_equal = best_method(rows[:2], "tenx", "score", True)
    assert all_equal["all_candidates_exactly_equal"] is True


def test_metric_summary_does_not_call_a_two_point_five_percent_gap_a_lead() -> None:
    rows = [
        {
            "dataset": "arxiv",
            "method": "dire_auto",
            "display": "DiRe auto",
            "score": 1.0,
        },
        {
            "dataset": "arxiv",
            "method": "cuml_umap",
            "display": "UMAP",
            "score": 1.025,
        },
        {
            "dataset": "arxiv",
            "method": "cuml_tsne",
            "display": "t-SNE",
            "score": 1.2,
        },
    ]

    result = best_method(rows, "arxiv", "score", False)

    assert result["method"] == "dire_auto"
    assert result["unique_exact_optimum"] is True
    assert result["unique_within_tolerance"] is False
    assert result["practically_comparable_methods"] == [
        "dire_auto",
        "cuml_umap",
    ]


def test_topology_effect_macros_report_gap_and_observed_variability() -> None:
    macros: list[str] = []
    append_topology_effect_macros(
        macros,
        [
            {
                "dataset": "arxiv",
                "basis": "subset",
                "metric": "dtw_beta0",
                "competitor": "cuml_umap",
                "paired_gap_competitor_minus_dire_mean": "1.828833",
                "paired_gap_std": "0.105258",
            },
            {
                "dataset": "arxiv",
                "basis": "subset",
                "metric": "q99_dtw_beta1",
                "competitor": "cuml_umap",
                "paired_gap_competitor_minus_dire_mean": "0.028667",
                "paired_gap_std": "0.025642",
            },
        ],
    )
    rendered = "\n".join(macros)

    assert r"\ArxivUmapMinusDireBetaZeroDtwGap}{1.8288}" in rendered
    assert r"\ArxivUmapMinusDireBetaZeroDtwGapSd}{0.1053}" in rendered
    assert r"\ArxivUmapMinusDireBetaZeroDtwGapInObservedSd}{17.37}" in rendered
    assert "DireBetterCount" not in rendered
    assert "ReplicateCount" not in rendered
    assert (
        r"\ArxivUmapMinusDireBetaOneQNinetyNineDtwGap}{0.0287}"
        in rendered
    )
    assert "Q99" not in rendered


def test_small_atlas_effect_macros_are_derived_from_paired_effect_rows() -> None:
    rows = []
    for comparator in ("cuml_umap", "cuml_tsne"):
        for index, gap in enumerate((0.20, 0.04, -0.10)):
            rows.append(
                {
                    "comparator": comparator,
                    "paired_gap_comparator_minus_dire_mean": gap,
                    "paired_mean_95pct_low": (
                        0.10 if gap == 0.20 else -0.02
                    ),
                    "relative_gap": (
                        0.20 if gap == 0.20 else abs(gap)
                    ),
                    "row": index,
                }
            )
    macros: list[str] = []

    append_small_atlas_effect_macros(macros, rows)

    rendered = "\n".join(macros)
    assert r"\SmallAtlasUmapComparisonCount}{3}" in rendered
    assert r"\SmallAtlasDireLowerVsUmapCount}{2}" in rendered
    assert r"\SmallAtlasUmapLowerVsDireCount}{1}" in rendered
    assert r"\SmallAtlasDireIntervalLowerVsUmapCount}{1}" in rendered
    assert r"\SmallAtlasDireFivePctLowerVsUmapCount}{1}" in rendered
    assert r"\SmallAtlasTsneComparisonCount}{3}" in rendered


def test_topology_effect_table_reports_all_ten_paired_gaps(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    output = tmp_path / "generated"
    output.mkdir()
    metric_keys = (
        "dtw_beta0",
        "dtw_beta1",
        "bottleneck_beta0",
        "bottleneck_beta1",
        "q99_dtw_beta0",
        "q99_dtw_beta1",
        "q99_bottleneck_beta0",
        "q99_bottleneck_beta1",
    )
    for dataset in ("tenx", "arxiv"):
        evaluation_root = bundle / "evaluation" / dataset
        evaluation_root.mkdir(parents=True)
        methods = {}
        for method_index, method in enumerate(
            ("dire_auto", "pca2", "dire", "cuml_umap", "cuml_tsne")
        ):
            records = [
                {
                    "subset_id": subset_id,
                    **{
                        key: 1.0 + 0.1 * method_index + 0.01 * subset_id
                        for key in metric_keys
                    },
                }
                for subset_id in range(10)
            ]
            size_records = [
                {
                    "sensitivity_subset_size": size,
                    **{
                        key: 1.0 + 0.1 * method_index + 1.0 / size
                        for key in metric_keys
                    },
                }
                for size in (1_000, 2_000, 4_000)
            ]
            layout_seed_records = [
                {
                    "layout_seed": layout_seed,
                    **{
                        key: 1.0 + 0.1 * method_index + 0.001 * layout_seed
                        for key in metric_keys
                    },
                }
                for layout_seed in (42, 43, 44)
            ]
            methods[method] = {
                "topology": {
                    "records": records,
                    "subset_size_sensitivity": {
                        "records": size_records,
                    },
                    "layout_seed_sensitivity": {
                        "records": layout_seed_records,
                    },
                }
            }
        (evaluation_root / "evaluation.json").write_text(
            json.dumps(
                {
                    "topology_design": {
                        "topology_subset_count": 10,
                    },
                    "methods": methods,
                }
            ),
            encoding="utf-8",
        )

    rows = make_topology_paired_effects(bundle, output)

    subset_rows = [row for row in rows if row["basis"] == "subset"]
    assert subset_rows
    assert all(row["pair_id_field"] == "subset_id" for row in subset_rows)
    assert all(row["paired_record_count"] == 10 for row in subset_rows)
    assert all(row["paired_ids"] == "0,1,2,3,4,5,6,7,8,9" for row in subset_rows)
    assert all(len(row["paired_gap_values"]) == 10 for row in subset_rows)
    layout_rows = [row for row in rows if row["basis"] == "layout"]
    assert layout_rows
    assert all(row["pair_id_field"] == "layout_seed" for row in layout_rows)
    assert all(row["paired_record_count"] == 3 for row in layout_rows)
    assert all(row["paired_ids"] == "42,43,44" for row in layout_rows)
    retired_pair_count = "paired_" + "subset_count"
    retired_pair_ids = "paired_" + "subset_ids"
    assert all(retired_pair_count not in row for row in rows)
    assert all(retired_pair_ids not in row for row in rows)
    rendered = (
        output / "revision3-arxiv-topology-paired-effects.tex"
    ).read_text(encoding="utf-8")
    assert "10 paired gaps" in rendered
    assert "Direction" not in rendered


def test_sensitivity_results_use_the_separate_bundle_tree(tmp_path: Path) -> None:
    path = result_path(
        tmp_path,
        "tenx",
        "dire_spectral",
        1_306_127,
    )

    assert "topology_sensitivity_results" in path.parts
    assert "large_results" not in path.parts


def test_small_runtime_separates_gpu_and_cpu_records(tmp_path: Path) -> None:
    log_root = tmp_path / "bundle" / "small_suite" / "json_logs"
    log_root.mkdir(parents=True)
    methods = {}
    for method, hardware, base in (
        ("dire", "GPU", 1.0),
        ("cuml_umap", "GPU", 2.0),
        ("cuml_tsne", "GPU", 3.0),
        ("opentsne", "CPU", 4.0),
        ("umap", "CPU", 5.0),
    ):
        methods[method] = {
            "hardware_class": hardware,
            "input_n_samples": 10_000,
            "repeats": [
                {"fit_time_sec": base + 1.0},
                {"fit_time_sec": base},
                {"fit_time_sec": base + 0.2},
            ],
        }
    (log_root / "blobs.json").write_text(
        json.dumps({"dataset_name": "blobs", "methods": methods}),
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    output.mkdir()

    rows = make_small_runtime(tmp_path / "bundle", output)

    assert len(rows) == 5
    assert {row["hardware_class"] for row in rows} == {"GPU", "CPU"}
    assert (output / "revision3-small-runtime.csv").is_file()
    assert b"\r" not in (
        output / "revision3-small-runtime.csv"
    ).read_bytes()
    assert (output / "revision3-small-runtime.tex").is_file()
    assert (output / "revision3-small-runtime.png").is_file()
    assert (output / "revision3-small-runtime.pdf").is_file()


def test_small_atlas_effects_retain_paired_values_and_subset_identity(
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "bundle" / "small_suite" / "json_logs"
    log_root.mkdir(parents=True)
    methods = {}
    for method, offset in (
        ("dire", 0.0),
        ("cuml_umap", 0.4),
        ("cuml_tsne", 0.6),
    ):
        repeats = []
        for repeat in range(4):
            subset_seed = 1042 + repeat
            digest = f"{subset_seed:064x}"
            repeats.append(
                {
                    "seed": 42 + repeat,
                    "topology_sample": {
                        "backend": "atlas",
                        "backend_detail": (
                            "direct GPU rank-based local-kNN atlas"
                        ),
                        "prefer_ripser": False,
                        "subset_seed": subset_seed,
                        "indices_sha256": digest,
                    },
                    "metrics": {
                        "topology": {
                            "backend": "atlas",
                            "backend_detail": (
                                "direct GPU rank-based local-kNN atlas"
                            ),
                            "prefer_ripser": False,
                            "metrics": {
                                "dtw_beta0": 1.0 + offset + 0.01 * repeat,
                                "dtw_beta1": 2.0 + offset + 0.02 * repeat,
                            },
                        }
                    },
                }
            )
        methods[method] = {"repeats": repeats}
    (log_root / "blobs.json").write_text(
        json.dumps({"dataset_name": "blobs", "methods": methods}),
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    output.mkdir()

    summaries, effects = make_small_atlas_topology_effects(
        tmp_path / "bundle",
        output,
    )

    assert summaries
    assert effects
    assert all(row["paired_count"] == 4 for row in effects)
    assert all(
        len(json.loads(row["paired_gap_values"])) == 4
        for row in effects
    )
    assert {
        row["comparator"]
        for row in effects
    } == {"cuml_umap", "cuml_tsne"}
    assert (
        output / "revision3-small-atlas-topology-summary.csv"
    ).is_file()
    assert (
        output / "revision3-small-atlas-topology-paired-effects.csv"
    ).is_file()
    assert (
        output / "revision3-small-atlas-topology-paired-effects.png"
    ).is_file()


def test_small_atlas_effects_reject_unpaired_subset_hashes(
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "bundle" / "small_suite" / "json_logs"
    log_root.mkdir(parents=True)
    methods = {}
    for method, digest in (
        ("dire", "a" * 64),
        ("cuml_umap", "b" * 64),
    ):
        methods[method] = {
            "repeats": [
                {
                    "seed": 42,
                    "topology_sample": {
                        "backend": "atlas",
                        "backend_detail": (
                            "direct GPU rank-based local-kNN atlas"
                        ),
                        "prefer_ripser": False,
                        "subset_seed": 1042,
                        "indices_sha256": digest,
                    },
                    "metrics": {
                        "topology": {
                            "backend": "atlas",
                            "backend_detail": (
                                "direct GPU rank-based local-kNN atlas"
                            ),
                            "prefer_ripser": False,
                            "metrics": {
                                "dtw_beta0": 1.0,
                                "dtw_beta1": 2.0,
                            },
                        }
                    },
                }
            ]
        }
    (log_root / "blobs.json").write_text(
        json.dumps({"dataset_name": "blobs", "methods": methods}),
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    output.mkdir()

    with pytest.raises(RuntimeError, match="indices differ"):
        make_small_atlas_topology_effects(
            tmp_path / "bundle",
            output,
        )


def test_topology_subset_size_artifacts_use_all_nested_sizes(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    output = tmp_path / "generated"
    output.mkdir()
    for dataset in ("tenx", "arxiv"):
        evaluation_root = bundle / "evaluation" / dataset
        evaluation_root.mkdir(parents=True)
        methods = {}
        for method in ("dire", "cuml_umap"):
            records = []
            for size in (1_000, 2_000, 4_000):
                records.append(
                    {
                        "sensitivity_subset_size": size,
                        "bottleneck_beta0": 1.0 / size,
                        "bottleneck_beta1": 2.0 / size,
                        "q99_bottleneck_beta0": 3.0 / size,
                        "q99_bottleneck_beta1": 4.0 / size,
                        "dtw_beta0": 5.0 / size,
                        "dtw_beta1": 6.0 / size,
                        "q99_dtw_beta0": 7.0 / size,
                        "q99_dtw_beta1": 8.0 / size,
                    }
                )
            methods[method] = {
                "topology": {
                    "subset_size_sensitivity": {"records": records}
                }
            }
        (evaluation_root / "evaluation.json").write_text(
            json.dumps({"methods": methods}),
            encoding="utf-8",
        )

    rows = make_topology_subset_size_sensitivity(bundle, output)

    assert {row["subset_size"] for row in rows} == {1_000, 2_000, 4_000}
    assert (
        output / "revision3-topology-subset-size-sensitivity.csv"
    ).is_file()
    assert (
        output / "revision3-topology-subset-size-sensitivity.tex"
    ).is_file()
    assert (
        output / "revision3-topology-subset-size-sensitivity.pdf"
    ).is_file()


def make_synthetic_tenx_centroid_bundle(
    root: Path,
    *,
    corrupt_reference: bool = False,
) -> tuple[Path, Path]:
    bundle = root / "bundle"
    output = root / "generated"
    output.mkdir()
    evaluation_root = bundle / "evaluation" / "tenx"
    evaluation_root.mkdir(parents=True)
    prepared_root = bundle / "prepared"
    prepared_root.mkdir()
    run_root = (
        bundle
        / "large_results"
        / "tenx"
        / "dire_auto"
        / "n_001306127"
    )
    run_root.mkdir(parents=True)

    cluster_sizes = np.asarray([500, 1_500, 20_000, 30_000, 40_000])
    labels = np.concatenate(
        [
            np.full(size, code, dtype=np.int16)
            for code, size in enumerate(cluster_sizes)
        ]
    )
    np.save(evaluation_root / "evaluation_labels.npy", labels)
    input_centroids = np.asarray(
        [[0.0], [1.0], [3.0], [10.0], [30.0]],
        dtype=np.float64,
    )
    embedding = np.concatenate(
        [
            np.repeat(input_centroids[[code]], size, axis=0)
            for code, size in enumerate(cluster_sizes)
        ],
        axis=0,
    )
    np.save(run_root / "embedding.npy", embedding)
    (run_root / "result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "embedding_file": "embedding.npy",
            }
        ),
        encoding="utf-8",
    )

    distances = np.abs(input_centroids - input_centroids.T)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :2]
    stored_reference = nearest.copy()
    if corrupt_reference:
        stored_reference[0] = stored_reference[0, ::-1]
    evaluation = {
        "label_policy": {
            "code_to_name": {
                str(code): f"Official cluster {code}"
                for code in range(5)
            }
        },
        "reference": {
            "label_centroids": {
                "values": input_centroids.tolist(),
            }
        },
        "methods": {
            "dire_auto": {
                "global": {
                    "centroid_adjacency_recall": {"k": 2},
                    "centroid_adjacencies": {
                        "reference_nearest_codes": stored_reference.tolist(),
                        "embedding_nearest_codes": nearest.tolist(),
                    },
                }
            }
        },
    }
    (evaluation_root / "evaluation.json").write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )
    (prepared_root / "tenx_mouse_brain_kmeans20_top_markers.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cluster": code,
                        "gene_name": f"Gene{code}",
                    }
                    for code in range(5)
                ]
            }
        ),
        encoding="utf-8",
    )
    return bundle, output


def test_tenx_centroid_audit_recomputes_reference_and_layout_edges(
    tmp_path: Path,
) -> None:
    bundle, output = make_synthetic_tenx_centroid_bundle(tmp_path)

    make_tenx_adjacency_audit(bundle, output)

    audit = json.loads(
        (output / "revision3-tenx-adjacency-audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit[
        "stored_reference_adjacency_matches_bundled_input_centroids"
    ] == {"dire_auto": True}
    assert audit[
        "stored_embedding_adjacency_matches_full_layout_recomputation"
    ] == {"dire_auto": True}
    assert audit["methods"][0]["mean_recall"] == 1.0
    assert audit["methods"][0]["source_cell_weighted_recall"] == 1.0
    assert (
        output / "revision3-tenx-adjacency-sensitivity.csv"
    ).is_file()
    sensitivity_rows = list(
        csv.DictReader(
            (
                output / "revision3-tenx-adjacency-sensitivity.csv"
            ).open(encoding="utf-8")
        )
    )
    assert {int(row["k"]) for row in sensitivity_rows} == {1, 2, 3, 4}


def test_tenx_centroid_audit_rejects_corrupt_reference_edges(
    tmp_path: Path,
) -> None:
    bundle, output = make_synthetic_tenx_centroid_bundle(
        tmp_path,
        corrupt_reference=True,
    )

    with pytest.raises(
        RuntimeError,
        match="input-space centroid adjacency",
    ):
        make_tenx_adjacency_audit(bundle, output)


def test_clean_output_retains_only_matching_successful_audit(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "revision3-results-current"
    bundle.mkdir()
    output = tmp_path / "generated"
    output.mkdir()
    (output / "stale-artifact.txt").write_text("stale", encoding="utf-8")
    audit = {
        "source_bundle": bundle.name,
        "status": "success",
        "datasets": {},
    }
    (output / "topology-audit.json").write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    prepare_output_root(bundle, output, clean=True)

    assert not (output / "stale-artifact.txt").exists()
    assert matching_topology_audit(bundle, output) == audit


def test_clean_output_drops_audit_from_previous_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "revision3-results-new"
    bundle.mkdir()
    output = tmp_path / "generated"
    output.mkdir()
    (output / "topology-audit.json").write_text(
        json.dumps(
            {
                "source_bundle": "revision3-results-old",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )

    prepare_output_root(bundle, output, clean=True)

    assert matching_topology_audit(bundle, output) is None
    assert not (output / "topology-audit.json").exists()


def test_matching_failed_topology_audit_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "revision3-results-current"
    bundle.mkdir()
    output = tmp_path / "generated"
    output.mkdir()
    (output / "topology-audit.json").write_text(
        json.dumps(
            {
                "source_bundle": bundle.name,
                "status": "mismatch",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="is not successful"):
        matching_topology_audit(bundle, output)
