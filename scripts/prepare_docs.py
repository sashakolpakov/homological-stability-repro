#!/usr/bin/env python3
"""Prepare generated paper artifacts for the Sphinx site."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import shutil
import sys
from pathlib import Path


DATASETS = ("blobs", "mnist", "disk", "moons", "levine13", "levine32")
METHOD_SUFFIXES = (
    "dire-rapids",
    "dire-topology",
    "tsne",
    "cuml-umap",
    "umap",
    "opentsne",
)
METRICS = ("time", "stress", "neighborhood", "context", "persistence-dim-0", "persistence-dim-1")
REVISION3_FIGURES = (
    ("revision3-tenx-layouts.png", "10x mouse-brain layouts"),
    ("revision3-arxiv-layouts.png", "arXiv layouts"),
    ("revision3-large-scaling.png", "Same-H100 scaling"),
    (
        "revision3-backend-policy-audit.png",
        "DiRe production-auto versus forced-IVF-Flat backend-policy audit",
    ),
    ("revision3-small-runtime.png", "Separated GPU and CPU runtime references"),
    (
        "revision3-small-atlas-topology-paired-effects.png",
        "Paired six-dataset atlas-topology effects against default DiRe",
    ),
    ("revision3-large-quality.png", "Large-data fidelity summary"),
    (
        "revision3-dire-topology-sensitivity-layouts.png",
        "DiRe default, spectral-only, and predeclared topology-preset layouts",
    ),
    (
        "revision3-topology-subset-size-sensitivity.png",
        "Nested topology subset-size sensitivity",
    ),
    (
        "revision3-topology-paired-effects.png",
        "Paired topology effect magnitudes and observed variability",
    ),
    ("revision3-initialization-ablation.png", "Initialization ablation"),
)


def iter_topology_comparisons(value):
    """Yield scalar audit records from the nested topology-audit payload."""

    if isinstance(value, dict):
        if "matches" in value and "absolute_delta" in value:
            yield value
        for child in value.values():
            yield from iter_topology_comparisons(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_topology_comparisons(child)


def copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if item.is_file():
            target = destination / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def write_gallery(path: Path) -> None:
    lines = [
        "Figure Gallery",
        "==============",
        "",
        "Embedding Figures",
        "-----------------",
        "",
    ]
    for dataset in DATASETS:
        lines.extend([dataset, "~" * len(dataset), ""])
        for suffix in METHOD_SUFFIXES:
            image = f"_static/paper/pics/embeddings/{dataset}-{suffix}.png"
            if (path.parent / image).exists():
                lines.extend([f".. image:: {image}", "   :width: 48%", ""])
        lines.append("")

    lines.extend(["Metric Figures", "--------------", ""])
    for dataset in DATASETS:
        lines.extend([dataset, "~" * len(dataset), ""])
        for metric in METRICS:
            image = f"_static/paper/pics/{dataset}_comparison/{dataset}-{metric}.png"
            if (path.parent / image).exists():
                lines.extend([f".. image:: {image}", "   :width: 32%", ""])
        lines.append("")

    lines.extend(["Million-point and initialization figures", "----------------------------------------", ""])
    for filename, caption in REVISION3_FIGURES:
        image = f"_static/paper/generated/revision3/{filename}"
        if (path.parent / image).exists():
            lines.extend(
                [
                    caption,
                    "~" * len(caption),
                    "",
                    f".. image:: {image}",
                    "   :width: 92%",
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_revision3_page(path: Path, generated_root: Path) -> None:
    lines = [
        "Million-point Reproducibility Profile",
        "=====================================",
        "",
        "The large-data profile is generated from the same checksummed records "
        "used by the LaTeX manuscript. CPU-reference and same-GPU timings are "
        "kept in separate panels. DiRe retains the manuscript's general framing "
        "as a dimensionality-reduction framework, accepts a user-specified target "
        "dimension including non-visual dimensions used in subsequent analysis, "
        "and is not a visualization-only method. This profile evaluates d=2 "
        "outputs; that value is the recorded experimental parameter, not the "
        "scope of DiRe. Topology is recomputed on independently seeded, fixed "
        "row-index subsets "
        "with both the historical Betti-curve DTW score and an established "
        "diameter-normalized bottleneck distance. A separately labelled "
        "99th-percentile pairwise-scale sensitivity check, plus the sampled "
        "reference coordinates and diagrams, makes outlier sensitivity auditable. "
        "The primary topology audit uses ten independently seeded, fixed "
        "4,000-row index subsets; every subset is applied identically to the "
        "high-dimensional reference and every embedding. Variation across "
        "three independently seeded full layouts is evaluated separately on "
        "one fixed subset. Nested "
        "1,000/2,000/4,000-row subsets separately expose whether paired effects "
        "reverse with diagnostic subset size. Every subset-specific paired gap, "
        "its mean, relative effect, and observed standard deviation are "
        "reported rather than rank alone; no p-values are claimed from this "
        "descriptive robustness audit.",
        "",
        "The six-dataset reference suite is a separate, consistent atlas "
        "protocol. It contains 20 GPU fits and 10 CPU-reference fits per "
        "configuration and calls the GPU rank-based local-kNN atlas evaluator "
        "directly on every fit over a paired fixed 1,000-row index subset. "
        "Every record must identify the atlas backend, the direct evaluator, "
        "and prefer_ripser=False, so a wrapper preference cannot silently "
        "change the protocol. Default DiRe and the source-distributed topology "
        "preset are both retained. Raw means, sample standard deviations, "
        "minima, and every paired gap are reported.",
        "",
        "Production DiRe uses the package's automatic cuVS index policy. A "
        "separately named forced-IVF-Flat control records synchronized graph, "
        "initialization, and layout timings, effective index type, force-fallback "
        "status, and fixed-query graph overlap. Runtime is interpreted only "
        "alongside the full local/global/context/topology comparison; no "
        "post-hoc quality-equivalence threshold is imposed.",
        "",
        "The production row uses the recorded ``init='pca'`` pathway. In the "
        "pinned cuVS code this dispatches to cuML PCA for the 20-feature 10x "
        "input and cuML TruncatedSVD for the 384-feature arXiv input.",
        "",
        "After fetching, ``make revision3-topology-audit`` independently re-runs "
        "Ripser, the exact sorted zero-birth-bar H0 bottleneck specialization, "
        "compiled exact GUDHI H1 matching, and Betti-curve DTW from the "
        "compact bundle for the topology-subset, subset-size, and full-layout-seed "
        "analyses; neither full public dataset is required.",
        "",
        "The bundle embeds the exact runner and dependency-lock snapshot. Its "
        "file hashes are recorded before reducer execution and rechecked during "
        "packaging, independently of the checkout's Git cleanliness.",
        "",
        "DiRe and every newly executed baseline are fitted to every observation. "
        "The separately labelled Cell Ranger t-SNE reference supplies released "
        "coordinates for all cells and is not presented as a same-input or "
        "runtime rerun. Fixed row-index subsetting is confined to the "
        "topology diagnostic: "
        "1,306,127 cells already imply about 8.53e11 unordered pairs before "
        "higher Vietoris--Rips simplices, so fixed shared row-index subsets are a "
        "controlled estimator rather than a claim of full-data persistent "
        "homology.",
        "",
    ]
    audit_path = generated_root / "topology-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        comparisons = list(iter_topology_comparisons(audit))
        mismatches = [
            record for record in comparisons if not record["matches"]
        ]
        maximum_delta = max(
            (float(record["absolute_delta"]) for record in comparisons),
            default=0.0,
        )
        environment = audit.get("audit_environment", {})
        lines.extend(
            [
                "CPU-only topology verification",
                "------------------------------",
                "",
                (
                    f"The retained bundle was independently recomputed with no "
                    f"GPU exposed using {environment.get('workers', 'recorded')} "
                    f"CPU workers. The audit checked {len(comparisons):,} scalar "
                    f"topology values, found {len(mismatches)} mismatches, and "
                    f"recorded a maximum absolute delta of {maximum_delta:g}. "
                    f"The full comparison record and audit-script hash are "
                    f"retained in ``generated/revision3/topology-audit.json``."
                ),
                "",
            ]
        )

    stats_path = generated_root / "revision3-dataset-stats.csv"
    if stats_path.exists():
        with stats_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        lines.extend(
            [
                "Datasets",
                "--------",
                "",
                ".. list-table:: Public large-data inputs",
                "   :header-rows: 1",
                "",
                "   * - Dataset",
                "     - Observations",
                "     - Input dimensions",
                "     - Reference labels",
                "     - License",
            ]
        )
        for row in rows:
            lines.extend(
                [
                    f"   * - {row['dataset']}",
                    f"     - {int(row['n']):,}",
                    f"     - {row['d']}",
                    f"     - {row['labels']}",
                    f"     - {row['license']}",
                ]
            )
        lines.append("")

    small_topology_path = (
        generated_root / "revision3-small-atlas-topology-summary.csv"
    )
    if small_topology_path.exists():
        with small_topology_path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["method"]
                in ("dire", "dire_topology", "cuml_umap", "cuml_tsne")
            ]
        small_effect_path = (
            generated_root
            / "revision3-small-atlas-topology-paired-effects.csv"
        )
        small_effect_summary = ""
        if small_effect_path.exists():
            with small_effect_path.open(
                newline="",
                encoding="utf-8",
            ) as handle:
                effect_rows = list(csv.DictReader(handle))
            umap_rows = [
                row
                for row in effect_rows
                if row["comparator"] == "cuml_umap"
            ]
            tsne_rows = [
                row
                for row in effect_rows
                if row["comparator"] == "cuml_tsne"
            ]
            dire_lower_umap = sum(
                float(row["paired_gap_comparator_minus_dire_mean"]) > 0.0
                for row in umap_rows
            )
            dire_five_percent_umap = sum(
                float(row["paired_gap_comparator_minus_dire_mean"]) > 0.0
                and float(row["relative_gap"]) > 0.05
                for row in umap_rows
            )
            dire_interval_umap = sum(
                float(row["paired_mean_95pct_low"]) > 0.0
                for row in umap_rows
            )
            dire_lower_tsne = sum(
                float(row["paired_gap_comparator_minus_dire_mean"]) > 0.0
                for row in tsne_rows
            )
            small_effect_summary = (
                f" Default DiRe has lower mean discrepancy than cuML UMAP "
                f"in {dire_lower_umap}/{len(umap_rows)} dataset-homology "
                f"comparisons; {dire_five_percent_umap} exceed the "
                "predeclared five-percent practical-difference band and "
                f"{dire_interval_umap} paired descriptive intervals exclude "
                "zero in DiRe's direction. Against cuML t-SNE, the split is "
                f"{dire_lower_tsne}/{len(tsne_rows) - dire_lower_tsne}: "
                "DiRe is lower in both dimensions on Disk and Half-moons "
                "and in H0 on both Levine datasets, while t-SNE is lower in "
                "both dimensions on Blobs and MNIST and in H1 on both Levine "
                "datasets."
            )
        lines.extend(
            [
                "Six-dataset atlas audit",
                "-----------------------",
                "",
                "Lower Betti-curve DTW discrepancy is better. Every GPU "
                "configuration has 20 paired records. The table reports the "
                "raw mean, sample standard deviation, and minimum; the "
                "generated paired-effect CSV retains every individual gap and "
                "its subset identity."
                + small_effect_summary,
                "",
                ".. list-table:: Repeated atlas-topology diagnostics",
                "   :header-rows: 1",
                "",
                "   * - Dataset",
                "     - Homology",
                "     - Method",
                "     - Repeats",
                "     - Mean +/- sample SD",
                "     - Minimum",
            ]
        )
        for row in rows:
            dimension = "H0" if row["metric"] == "dtw_beta0" else "H1"
            lines.extend(
                [
                    f"   * - {row['dataset']}",
                    f"     - {dimension}",
                    f"     - {row['display']}",
                    f"     - {row['repeat_count']}",
                    (
                        "     - "
                        f"{float(row['mean']):.4f} +/- "
                        f"{float(row['sample_sd']):.4f}"
                    ),
                    f"     - {float(row['minimum']):.4f}",
                ]
            )
        lines.append("")

    backend_path = generated_root / "revision3-backend-policy-audit.csv"
    if backend_path.exists():
        with backend_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        lines.extend(
            [
                "Backend-policy audit",
                "--------------------",
                "",
                "These are fresh paired profiles. The production-auto and "
                "forced-IVF-Flat rows use the same full input, parameters, "
                "seeds, repeat count, and H100. Graph overlap is reported "
                "because a faster approximate index is not assumed to be "
                "quality-equivalent.",
                "",
                ".. list-table:: DiRe cuVS policy and stage profile",
                "   :header-rows: 1",
                "",
                "   * - Dataset",
                "     - Effective auto index",
                "     - Auto steady (s)",
                "     - Graph (s)",
                "     - Initialization (s)",
                "     - Layout (s)",
                "     - Forced IVF-Flat (s)",
                "     - Speed-up",
                "     - Mean 16-NN overlap",
                "     - Chunked fallbacks",
            ]
        )
        for row in rows:
            dataset = "10x mouse brain" if row["dataset"] == "tenx" else "arXiv"
            lines.extend(
                [
                    f"   * - {dataset}",
                    f"     - {row['auto_effective_index']}",
                    f"     - {float(row['auto_steady_total_sec']):.2f}",
                    f"     - {float(row['auto_steady_knn_graph_sec']):.2f}",
                    f"     - {float(row['auto_steady_initialization_sec']):.2f}",
                    f"     - {float(row['auto_steady_layout_sec']):.2f}",
                    f"     - {float(row['control_steady_total_sec']):.2f}",
                    (
                        "     - "
                        f"{float(row['steady_speedup_control_over_auto']):.2f}x"
                    ),
                    f"     - {float(row['graph_overlap_mean']):.3f}",
                    f"     - {row['auto_chunked_fallback_calls']}",
                ]
            )
        lines.append("")

    effects_path = generated_root / "revision3-topology-paired-effects.csv"
    if effects_path.exists():
        with effects_path.open(newline="", encoding="utf-8") as handle:
            effect_rows = [
                row
                for row in csv.DictReader(handle)
                if row["basis"] == "subset"
                and row["metric"]
                in (
                    "dtw_beta0",
                    "dtw_beta1",
                    "q99_dtw_beta0",
                    "q99_dtw_beta1",
                )
                and row["competitor"]
                in ("pca2", "cuml_umap", "cuml_tsne")
            ]
        subset_count = int(effect_rows[0]["paired_record_count"])
        lines.extend(
            [
                "Paired topology effects",
                "-----------------------",
                "",
                "The gap is comparator discrepancy minus production-auto DiRe "
                "discrepancy. Positive values favour DiRe because lower "
                "Betti-DTW discrepancy is better. The standard deviation is "
                f"the observed standard deviation of the {subset_count} paired "
                "subset-specific gaps. Each topology subset is one independently "
                "seeded set of 4,000 row indices applied identically to the "
                "high-dimensional reference and every embedding. The final "
                f"column gives all {subset_count} gaps in fixed subset-seed "
                "order rather "
                "than reducing their signs to a count. This is a descriptive "
                "robustness audit, not a significance test.",
                "",
                ".. list-table:: Primary paired Betti-DTW effects",
                "   :header-rows: 1",
                "",
                "   * - Dataset",
                "     - Homology",
                "     - Comparator",
                "     - DiRe auto",
                "     - Comparator",
                "     - Gap +/- observed SD",
                f"     - {subset_count} paired gaps",
            ]
        )
        for row in effect_rows:
            dataset = "10x mouse brain" if row["dataset"] == "tenx" else "arXiv"
            dimension = "H0" if "beta0" in row["metric"] else "H1"
            normalization = (
                "unit diameter"
                if not row["metric"].startswith("q99_")
                else "Q99 sensitivity"
            )
            homology = f"{dimension}, {normalization}"
            lines.extend(
                [
                    f"   * - {dataset}",
                    f"     - {homology}",
                    f"     - {row['competitor_display']}",
                    f"     - {float(row['dire_mean']):.4f}",
                    f"     - {float(row['competitor_mean']):.4f}",
                    (
                        "     - "
                        f"{float(row['paired_gap_competitor_minus_dire_mean']):.4f} "
                        f"+/- {float(row['paired_gap_std']):.4f}"
                    ),
                    "     - "
                    + ", ".join(
                        f"{float(value):+.4f}"
                        for value in json.loads(row["paired_gap_values"])
                    ),
                ]
            )
        lines.append("")

    adjacency_path = (
        generated_root / "revision3-tenx-adjacency-summary.csv"
    )
    adjacency_pairs_path = (
        generated_root / "revision3-tenx-adjacency-pairs.csv"
    )
    if adjacency_path.exists() and adjacency_pairs_path.exists():
        with adjacency_path.open(newline="", encoding="utf-8") as handle:
            adjacency_rows = list(csv.DictReader(handle))
        with adjacency_pairs_path.open(newline="", encoding="utf-8") as handle:
            pair_rows = list(csv.DictReader(handle))
        pair_status = {
            (
                row["method"],
                int(row["source_cluster"]),
                int(row["target_cluster"]),
            ): row["status"]
            for row in pair_rows
        }
        by_method = {row["method"]: row for row in adjacency_rows}
        dire_row = by_method["dire_auto"]
        dire_spectral_row = by_method["dire_spectral"]
        dire_topology_row = by_method["dire_topology"]
        umap_row = by_method["cuml_umap"]
        sensitivity_path = (
            generated_root / "revision3-tenx-adjacency-sensitivity.csv"
        )
        sensitivity_by_key = {}
        if sensitivity_path.exists():
            with sensitivity_path.open(
                newline="",
                encoding="utf-8",
            ) as handle:
                sensitivity_by_key = {
                    (
                        row["method"],
                        int(row["k"]),
                        int(row["minimum_source_cells"]),
                    ): row
                    for row in csv.DictReader(handle)
                }
        shared_k_values = sorted(
            {
                key[1]
                for key in sensitivity_by_key
                if key[0] == "dire_auto" and key[2] == 0
            }.intersection(
                {
                    key[1]
                    for key in sensitivity_by_key
                    if key[0] == "cuml_umap" and key[2] == 0
                }
            )
        )
        adjacency_k_text = ""
        if shared_k_values:
            gaps_by_k = {
                k: (
                    float(
                        sensitivity_by_key[
                            ("dire_auto", k, 0)
                        ]["equal_cluster_weight_recall"]
                    )
                    - float(
                        sensitivity_by_key[
                            ("cuml_umap", k, 0)
                        ]["equal_cluster_weight_recall"]
                    )
                )
                for k in shared_k_values
            }
            nonnegative_k = [
                k for k, gap in gaps_by_k.items() if gap >= 0.0
            ]
            adjacency_k_text = (
                f" Across k={shared_k_values[0]}--{shared_k_values[-1]}, "
                "the DiRe-minus-UMAP equal-cluster-recall gap ranges from "
                f"{min(gaps_by_k.values()):+.3f} to "
                f"{max(gaps_by_k.values()):+.3f}; DiRe meets or exceeds "
                "UMAP at "
                + (
                    ", ".join(str(k) for k in nonnegative_k)
                    if nonnegative_k
                    else "none of these k values"
                )
                + "."
            )
        notable = []
        if (
            pair_status.get(("dire_auto", 9, 10)) == "preserved"
            and pair_status.get(("dire_auto", 10, 9)) == "preserved"
        ):
            notable.append(
                "9 <-> 10 (Rrm2/Hmgb2 and Ube2c/Cenpf/Ccnb1 "
                "cell-cycle marker profiles)"
            )
        if pair_status.get(("dire_auto", 20, 19)) == "preserved":
            notable.append("20 -> 19 (both released profiles contain S100a8)")
        if pair_status.get(("dire_auto", 13, 15)) == "preserved":
            notable.append(
                "13 -> 15 (Cldn5/Flt1 to Rgs5/Ndufa4l2 profiles)"
            )
        if pair_status.get(("dire_auto", 19, 16)) == "preserved":
            notable.append("19 -> 16 (myeloid marker profiles)")
        lines.extend(
            [
                "10x centroid-adjacency audit",
                "----------------------------",
                "",
                (
                    "Each layout is compared with the same 60 directed "
                    "three-nearest-centroid relations in the common 20-D input. "
                    f"Production-auto DiRe preserves "
                    f"{dire_row['preserved_directed_edges']}/60 and introduces "
                    f"{dire_row['introduced_directed_edges']}; UMAP preserves "
                    f"{umap_row['preserved_directed_edges']}/60 and introduces "
                    f"{umap_row['introduced_directed_edges']}. Independent "
                    "recomputation from every bundled full layout reproduces "
                    "the stored adjacency lists exactly. Equal-cluster and "
                    "source-cell-weighted recalls are both reported; the latter "
                    f"are {float(dire_row['source_cell_weighted_recall']):.3f} "
                    "for production-auto DiRe and "
                    f"{float(umap_row['source_cell_weighted_recall']):.3f} for "
                    "UMAP. "
                    + (
                        "Excluding source clusters below 1,000 cells does not "
                        "remove the difference: equal-cluster recall becomes "
                        f"{float(sensitivity_by_key[('dire_auto', 3, 1000)]['equal_cluster_weight_recall']):.3f} "
                        "for DiRe and "
                        f"{float(sensitivity_by_key[('cuml_umap', 3, 1000)]['equal_cluster_weight_recall']):.3f} "
                        "for UMAP. "
                        if (
                            ("dire_auto", 3, 1000) in sensitivity_by_key
                            and ("cuml_umap", 3, 1000) in sensitivity_by_key
                        )
                        else ""
                    )
                    + "The following "
                    "marker-compatible examples are therefore interpretive, "
                    "not a claim of aggregate superiority: "
                    + "; ".join(notable)
                    + "."
                    + adjacency_k_text
                    + " The one-factor controls confirm configuration "
                    "sensitivity: spectral-initialised DiRe preserves "
                    f"{dire_spectral_row['preserved_directed_edges']}/60 "
                    "relations with source-cell-weighted recall "
                    f"{float(dire_spectral_row['source_cell_weighted_recall']):.3f}, "
                    "and the predeclared topology preset preserves "
                    f"{dire_topology_row['preserved_directed_edges']}/60 with "
                    "weighted recall "
                    f"{float(dire_topology_row['source_cell_weighted_recall']):.3f}. "
                    "UMAP remains the leader in the predeclared equal-cluster "
                    "comparison; neither DiRe control replaces the production "
                    "row."
                ),
                "",
                ".. list-table:: Directed centroid relations retained from the 20-D input",
                "   :header-rows: 1",
                "",
                "   * - Method",
                "     - Input edges",
                "     - Preserved",
                "     - Introduced",
                "     - Equal-cluster recall",
                "     - Cell-weighted recall",
            ]
        )
        for row in adjacency_rows:
            lines.extend(
                [
                    f"   * - {row['display']}",
                    f"     - {row['input_directed_edges']}",
                    f"     - {row['preserved_directed_edges']}",
                    f"     - {row['introduced_directed_edges']}",
                    f"     - {float(row['mean_recall']):.3f}",
                    (
                        "     - "
                        f"{float(row['source_cell_weighted_recall']):.3f}"
                    ),
                ]
            )
        lines.append("")

    marker_path = generated_root / "revision3-tenx-markers.csv"
    if marker_path.exists():
        with marker_path.open(newline="", encoding="utf-8") as handle:
            marker_rows = list(csv.DictReader(handle))
        lines.extend(
            [
                "Released 10x marker annotation",
                "------------------------------",
                "",
                "These cluster sizes and positive markers come from the pinned "
                "public Cell Ranger analysis; they are not labels inferred "
                "from any d=2 benchmark layout.",
                "",
                ".. list-table:: Official cluster marker reference",
                "   :header-rows: 1",
                "",
                "   * - Cluster",
                "     - Cells",
                "     - Five strongest released positive markers",
            ]
        )
        for row in marker_rows:
            lines.extend(
                [
                    f"   * - {row['cluster']}",
                    f"     - {int(row['cells']):,}",
                    f"     - {row['markers']}",
                ]
            )
        lines.append("")

    summary_path = generated_root / "revision3-results-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        labels = {
            "steady_sec": "fastest steady-state runtime",
            "knn_overlap": "highest local kNN overlap",
            "centroid_spearman": "highest centroid-distance correlation",
            "centroid_adjacency_recall": "highest centroid-adjacency recall",
            "context_balanced_accuracy": "highest balanced context accuracy",
            "bottleneck_beta0": (
                "lowest diameter-normalized beta-0 bottleneck distance"
            ),
            "bottleneck_beta1": (
                "lowest diameter-normalized beta-1 bottleneck distance"
            ),
            "dtw_beta0": "lowest beta-0 DTW",
            "dtw_beta1": "lowest beta-1 DTW",
        }
        lines.extend(["Metric comparisons", "------------------", ""])
        for dataset, display in (("tenx", "10x mouse brain"), ("arxiv", "arXiv corpus")):
            lines.extend([display, "~" * len(display), ""])
            for group, group_display in (
                ("all_evaluated", "All evaluated methods and references"),
                (
                    "nonlinear_gpu_defaults",
                    "Fresh default nonlinear methods on the same GPU",
                ),
                (
                    "nonlinear_gpu_with_sensitivity",
                    "Same-GPU nonlinear methods including the predeclared DiRe sensitivity preset",
                ),
            ):
                lines.extend([group_display, "^" * len(group_display), ""])
                for key, label in labels.items():
                    winner = summary.get(dataset, {}).get(group, {}).get(key)
                    if winner is None:
                        lines.append(f"* {label}: unavailable")
                    elif winner["all_candidates_exactly_equal"]:
                        lines.append(
                            f"* {label}: non-discriminating equality at "
                            f"{winner[key]:.6g} across "
                            + ", ".join(winner["exact_optimum_displays"])
                        )
                    elif not winner["unique_within_tolerance"]:
                        lines.append(
                            f"* {label}: best observed value "
                            f"{winner[key]:.6g}; practically comparable "
                            "within 5%: "
                            + ", ".join(
                                winner[
                                    "practically_comparable_displays"
                                ]
                            )
                        )
                    else:
                        lines.append(
                            f"* {label}: {winner['display']} ({winner[key]:.6g})"
                        )
                lines.append("")

    lines.extend(["Figures", "-------", ""])
    for filename, caption in REVISION3_FIGURES:
        if (generated_root / filename).exists():
            lines.extend(
                [
                    f".. figure:: _static/paper/generated/revision3/{filename}",
                    "   :width: 96%",
                    "",
                    f"   {caption}.",
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_paper_page() -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/latex_to_sphinx.py",
            "--latex",
            "dire_short.tex",
            "--bib",
            "dire_short.bib",
            "--output",
            "docs/paper.rst",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-static", type=Path, default=Path("docs/_static/paper"))
    args = parser.parse_args()

    if args.site_static.exists():
        shutil.rmtree(args.site_static)
    args.site_static.mkdir(parents=True, exist_ok=True)

    copy_tree_contents(Path("pics"), args.site_static / "pics")
    generated_root = Path("generated/revision3")
    copy_tree_contents(
        generated_root,
        args.site_static / "generated" / "revision3",
    )
    for filename in ("dire_short.pdf", "dire_short.tex", "dire_short.bib"):
        source = Path(filename)
        if source.exists():
            shutil.copy2(source, args.site_static / filename)
    write_paper_page()
    write_gallery(Path("docs/figures.rst"))
    write_revision3_page(Path("docs/revision3.rst"), generated_root)


if __name__ == "__main__":
    main()
