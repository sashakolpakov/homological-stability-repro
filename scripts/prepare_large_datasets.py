#!/usr/bin/env python3
"""Prepare the two large, public datasets used in the Revision 3 rerun.

The preparation step is intentionally deterministic and produces compact NumPy
arrays that every reducer receives unchanged:

* 10x Genomics: the official 20-component PCA projection of 1,306,127
  embryonic-mouse-brain cells, plus Cell Ranger cluster labels that are
  external to our rerun, marker statistics, and the historical Cell Ranger
  t-SNE layout.
* arXiv: 723,457 public BGE-small paper embeddings and primary-category
  metadata from the version-pinned Hugging Face release.

Raw downloads and prepared arrays live below an ignored data directory.  The
generated manifest records URLs, revisions, hashes, shapes, and transformations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


TENX_URL = (
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell/"
    "1M_neurons/1M_neurons_analysis.tar.gz"
)
TENX_SHA256 = "12636d6f6d3802d34ff491977479c1d43adcf65c575c57f9a3fd3ed57348ee09"
TENX_ARCHIVE_NAME = "1M_neurons_analysis.tar.gz"
TENX_MEMBERS = {
    "pca": "analysis/pca/20_components/projection.csv",
    "graph_clusters": "analysis/clustering/graphclust/clusters.csv",
    "kmeans20_clusters": "analysis/clustering/kmeans_20_clusters/clusters.csv",
    "kmeans20_diffexp": (
        "analysis/diffexp/kmeans_20_clusters/differential_expression.csv"
    ),
    "official_tsne": "analysis/tsne/2_components/projection.csv",
}

ARXIV_DATASET_ID = "igriv/dire-arxiv-bge-small-embeddings"
ARXIV_REVISION = "9b4ceb24ccd20b5d82956ee26a1d80e678d45a7b"


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def download_streaming(
    url: str,
    destination: Path,
    retries: int = 4,
    timeout_seconds: int = 120,
) -> None:
    """Download through the standard library, resuming a partial file safely."""

    if retries < 1:
        raise ValueError("download retries must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(1, retries + 1):
        resume_offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "homological-stability-repro/1"},
        )
        if resume_offset:
            request.add_header("Range", f"bytes={resume_offset}-")
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                status = response.getcode()
                append = resume_offset > 0 and status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as target:
                    shutil.copyfileobj(
                        response,
                        target,
                        length=8 * 1024 * 1024,
                    )
            partial.replace(destination)
            return
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(
                    f"download failed after {retries} attempts: {url}"
                ) from error
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"download attempt {attempt}/{retries} failed; "
                f"retrying in {delay}s: {error}",
                flush=True,
            )
            time.sleep(delay)


def extract_tenx_members(archive: Path, extract_root: Path) -> dict[str, Path]:
    extract_root.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with tarfile.open(archive, "r:gz") as bundle:
        available = {member.name: member for member in bundle.getmembers()}
        for label, member_name in TENX_MEMBERS.items():
            if member_name not in available:
                raise RuntimeError(f"10x archive is missing required member: {member_name}")
            member = available[member_name]
            if not member.isfile():
                raise RuntimeError(f"10x archive member is not a regular file: {member_name}")
            destination = extract_root / member_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read 10x archive member: {member_name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            extracted[label] = destination
    return extracted


def _require_aligned_barcodes(reference, candidate, candidate_name: str) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"10x barcode count mismatch: PCA={len(reference):,}, "
            f"{candidate_name}={len(candidate):,}"
        )
    if not np.array_equal(reference, candidate):
        mismatch = np.flatnonzero(reference != candidate)
        first = int(mismatch[0]) if mismatch.size else -1
        raise RuntimeError(
            f"10x barcode order mismatch for {candidate_name} at row {first}"
        )


def summarize_tenx_markers(diffexp_path: Path, cluster_sizes: dict[int, int]) -> list[dict]:
    import pandas as pd

    frame = pd.read_csv(diffexp_path)
    records: list[dict] = []
    for cluster in sorted(cluster_sizes):
        mean_key = f"Cluster {cluster} Mean UMI Counts"
        fold_key = f"Cluster {cluster} Log2 fold change"
        p_key = f"Cluster {cluster} Adjusted p value"
        missing = [key for key in (mean_key, fold_key, p_key) if key not in frame]
        if missing:
            raise RuntimeError(f"10x differential-expression file lacks columns: {missing}")
        view = frame[["Gene ID", "Gene Name", mean_key, fold_key, p_key]].copy()
        view.columns = ["gene_id", "gene_name", "mean_umi", "log2_fold_change", "adjusted_p"]
        view = view[
            np.isfinite(view["mean_umi"])
            & np.isfinite(view["log2_fold_change"])
            & np.isfinite(view["adjusted_p"])
            & (view["mean_umi"] > 0)
            & (view["log2_fold_change"] > 0)
            & (view["adjusted_p"] <= 0.05)
        ].copy()
        # Ranking by fold change alone tends to select vanishingly rare features.
        # This deterministic score retains specificity while giving expression
        # magnitude some weight.  Raw values remain in the output.
        view["ranking_score"] = view["log2_fold_change"] * np.log1p(view["mean_umi"])
        view.sort_values(
            ["ranking_score", "log2_fold_change", "mean_umi", "gene_name"],
            ascending=[False, False, False, True],
            inplace=True,
        )
        for rank, row in enumerate(view.head(10).itertuples(index=False), start=1):
            records.append(
                {
                    "cluster": int(cluster),
                    "cluster_size": int(cluster_sizes[cluster]),
                    "rank": rank,
                    "gene_id": str(row.gene_id),
                    "gene_name": str(row.gene_name),
                    "mean_umi": float(row.mean_umi),
                    "log2_fold_change": float(row.log2_fold_change),
                    "adjusted_p": float(row.adjusted_p),
                    "ranking_score": float(row.ranking_score),
                }
            )
    return records


def prepare_tenx(raw_root: Path, prepared_root: Path, force: bool) -> dict:
    import pandas as pd

    output_manifest = prepared_root / "tenx_manifest.json"
    if output_manifest.exists() and not force:
        with output_manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = [
            prepared_root / manifest["files"][key]
            for key in (
                "input",
                "graph_labels",
                "kmeans20_labels",
                "official_tsne",
                "markers",
            )
        ]
        if all(path.exists() for path in expected):
            print("10x prepared artifacts already exist; reusing them", flush=True)
            return manifest

    started = time.time()
    archive = raw_root / "10x" / TENX_ARCHIVE_NAME
    if not archive.exists() or sha256_file(archive) != TENX_SHA256:
        if archive.exists():
            corrupt = archive.with_suffix(archive.suffix + ".sha256-mismatch")
            archive.replace(corrupt)
            print(f"moved hash-mismatched archive to {corrupt}", flush=True)
        print(f"downloading official 10x archive: {TENX_URL}", flush=True)
        download_streaming(TENX_URL, archive)
    actual_hash = sha256_file(archive)
    if actual_hash != TENX_SHA256:
        raise RuntimeError(
            f"10x archive SHA-256 mismatch: expected {TENX_SHA256}, got {actual_hash}"
        )

    extracted = extract_tenx_members(archive, raw_root / "10x" / "extracted")
    pca_columns = [f"PC-{index}" for index in range(1, 21)]
    pca_types = {column: "float32" for column in pca_columns}
    print("reading official 10x 20-component PCA projection", flush=True)
    pca = pd.read_csv(
        extracted["pca"],
        usecols=["Barcode", *pca_columns],
        dtype=pca_types,
    )
    pca_barcodes = pca.pop("Barcode").to_numpy()
    X = np.ascontiguousarray(pca.to_numpy(dtype=np.float32, copy=False))
    del pca
    # Apply DiRe's documented numerical-safety transformation once, here, so
    # every reducer receives byte-identical input. Translation and one global
    # scalar preserve Euclidean neighbor rankings exactly.
    X -= X.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    common_scale = float(np.abs(X).max())
    if common_scale <= 0 or not np.isfinite(common_scale):
        raise RuntimeError(f"invalid 10x common input scale: {common_scale}")
    X /= common_scale

    print("reading official 10x released cluster assignments", flush=True)
    graph = pd.read_csv(
        extracted["graph_clusters"],
        dtype={"Barcode": "string", "Cluster": "int16"},
    )
    _require_aligned_barcodes(
        pca_barcodes,
        graph["Barcode"].astype(str).to_numpy(),
        "graph clustering",
    )
    graph_labels = graph["Cluster"].to_numpy(dtype=np.int16, copy=True)
    del graph

    kmeans20 = pd.read_csv(
        extracted["kmeans20_clusters"],
        dtype={"Barcode": "string", "Cluster": "int16"},
    )
    _require_aligned_barcodes(
        pca_barcodes,
        kmeans20["Barcode"].astype(str).to_numpy(),
        "k-means-20 clustering",
    )
    kmeans20_labels = kmeans20["Cluster"].to_numpy(dtype=np.int16, copy=True)
    del kmeans20

    official_tsne = pd.read_csv(
        extracted["official_tsne"],
        dtype={"TSNE-1": "float32", "TSNE-2": "float32"},
    )
    if "Barcode" in official_tsne:
        _require_aligned_barcodes(
            pca_barcodes,
            official_tsne.pop("Barcode").astype(str).to_numpy(),
            "official t-SNE",
        )
    tsne_columns = [column for column in official_tsne if column.startswith("TSNE-")]
    if len(tsne_columns) != 2:
        raise RuntimeError(f"expected two official t-SNE columns, found {tsne_columns}")
    official_tsne_array = np.ascontiguousarray(
        official_tsne[tsne_columns].to_numpy(dtype=np.float32, copy=False)
    )
    del official_tsne

    if not (
        len(X)
        == len(graph_labels)
        == len(kmeans20_labels)
        == len(official_tsne_array)
        == 1_306_127
    ):
        raise RuntimeError(
            "unexpected 10x row count after alignment: "
            f"X={len(X):,}, graph={len(graph_labels):,}, "
            f"kmeans20={len(kmeans20_labels):,}, tSNE={len(official_tsne_array):,}"
        )
    if not np.isfinite(X).all() or not np.isfinite(official_tsne_array).all():
        raise RuntimeError("10x prepared coordinates contain non-finite values")

    prepared_root.mkdir(parents=True, exist_ok=True)
    input_name = "tenx_mouse_brain_pca20.npy"
    graph_name = "tenx_mouse_brain_graph_clusters.npy"
    kmeans_name = "tenx_mouse_brain_kmeans20.npy"
    tsne_name = "tenx_mouse_brain_cellranger_tsne.npy"
    np.save(prepared_root / input_name, X)
    np.save(prepared_root / graph_name, graph_labels)
    np.save(prepared_root / kmeans_name, kmeans20_labels)
    np.save(prepared_root / tsne_name, official_tsne_array)

    unique_clusters, counts = np.unique(kmeans20_labels, return_counts=True)
    cluster_sizes = {
        int(cluster): int(count) for cluster, count in zip(unique_clusters, counts)
    }
    marker_records = summarize_tenx_markers(
        extracted["kmeans20_diffexp"],
        cluster_sizes,
    )
    markers_name = "tenx_mouse_brain_kmeans20_top_markers.json"
    write_json(
        prepared_root / markers_name,
        {
            "source": TENX_MEMBERS["kmeans20_diffexp"],
            "selection": (
                "adjusted_p <= 0.05, positive log2 fold change and mean UMI; "
                "rank by log2_fold_change * log1p(mean_umi)"
            ),
            "top_n_per_cluster": 10,
            "records": marker_records,
        },
    )

    barcode_digest = hashlib.sha256()
    for barcode in pca_barcodes:
        barcode_digest.update(str(barcode).encode("utf-8"))
        barcode_digest.update(b"\0")

    manifest = {
        "schema_version": 1,
        "dataset": "10x 1.3 Million Brain Cells from E18 Mice",
        "source_url": TENX_URL,
        "source_archive_sha256": actual_hash,
        "license": "CC BY 4.0",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "input": (
            "Official Cell Ranger 1.3.0 20-component PCA projection, "
            "feature-mean-centered and divided by one global maximum absolute "
            "value; every reducer receives the same float32 rows in the same "
            "barcode order. This translation and uniform rescaling preserve "
            "Euclidean neighbor rankings exactly."
        ),
        "common_input_transform": {
            "feature_centering": True,
            "uniform_global_scale": common_scale,
        },
        "labels": {
            "visualization": "Official Cell Ranger k-means clustering with k=20",
            "fine_grained": "Official Cell Ranger graph clustering (60 clusters)",
            "independence": (
                "Labels were released by 10x and are not fitted to any rerun layout."
            ),
            "kmeans20_cluster_sizes": cluster_sizes,
        },
        "official_reference_layout": (
            "Cell Ranger t-SNE coordinates from the released secondary analysis; "
            "included for visual/quality reference only, never in fresh runtime comparisons."
        ),
        "barcode_order_sha256": barcode_digest.hexdigest(),
        "files": {
            "input": input_name,
            "graph_labels": graph_name,
            "kmeans20_labels": kmeans_name,
            "official_tsne": tsne_name,
            "markers": markers_name,
        },
        "preparation_wall_time_sec": time.time() - started,
    }
    write_json(output_manifest, manifest)
    print(
        f"prepared 10x: {X.shape[0]:,} cells x {X.shape[1]} PCs, "
        f"{len(unique_clusters)} readable clusters",
        flush=True,
    )
    return manifest


def _hash_identifier_batch(digest, values) -> None:
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")


def prepare_arxiv(raw_root: Path, prepared_root: Path, force: bool) -> dict:
    del raw_root  # Hugging Face controls its own cache below HF_HOME.
    from datasets import load_dataset
    from numpy.lib.format import open_memmap

    output_manifest = prepared_root / "arxiv_manifest.json"
    if output_manifest.exists() and not force:
        with output_manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = [
            prepared_root / manifest["files"][key]
            for key in ("input", "labels", "label_names")
        ]
        if all(path.exists() for path in expected):
            print("arXiv prepared artifacts already exist; reusing them", flush=True)
            return manifest

    started = time.time()
    print(
        f"loading {ARXIV_DATASET_ID} at revision {ARXIV_REVISION}",
        flush=True,
    )
    embeddings_ds = load_dataset(
        ARXIV_DATASET_ID,
        split="train",
        revision=ARXIV_REVISION,
    )
    n_samples = len(embeddings_ds)
    n_features = 384
    if n_samples != 723_457:
        raise RuntimeError(f"unexpected arXiv embedding row count: {n_samples:,}")

    prepared_root.mkdir(parents=True, exist_ok=True)
    input_name = "arxiv_bge_small_384_l2_centered.npy"
    input_path = prepared_root / input_name
    X = open_memmap(
        input_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_features),
    )
    embedding_id_digest = hashlib.sha256()
    offset = 0
    norm_min = float("inf")
    norm_max = 0.0
    norm_sum = 0.0
    feature_sum = np.zeros(n_features, dtype=np.float64)
    for batch in embeddings_ds.iter(batch_size=4096):
        values = np.asarray(batch["embedding"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != n_features:
            raise RuntimeError(f"unexpected arXiv embedding batch shape: {values.shape}")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise RuntimeError("arXiv source embeddings contain zero/non-finite norms")
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum())
        values /= norms[:, None]
        feature_sum += values.sum(axis=0, dtype=np.float64)
        X[offset : offset + len(values)] = values
        _hash_identifier_batch(embedding_id_digest, batch["arxiv_id"])
        offset += len(values)
        if offset % 100_000 < len(values):
            print(f"  wrote {offset:,}/{n_samples:,} normalized vectors", flush=True)
    if offset != n_samples:
        raise RuntimeError(f"arXiv write stopped at {offset:,}/{n_samples:,}")
    # Apply the same common transformation as for 10x. Sequential memmap
    # passes avoid allocating a second 1.1 GB dense matrix.
    feature_mean = (feature_sum / n_samples).astype(np.float32)
    common_scale = 0.0
    for start in range(0, n_samples, 8192):
        stop = min(start + 8192, n_samples)
        block = np.asarray(X[start:stop])
        block -= feature_mean
        X[start:stop] = block
        common_scale = max(common_scale, float(np.abs(block).max()))
    if common_scale <= 0 or not np.isfinite(common_scale):
        raise RuntimeError(f"invalid arXiv common input scale: {common_scale}")
    for start in range(0, n_samples, 8192):
        stop = min(start + 8192, n_samples)
        X[start:stop] = np.asarray(X[start:stop]) / common_scale
    X.flush()
    del X

    metadata_ds = load_dataset(
        ARXIV_DATASET_ID,
        name="metadata",
        split="train",
        revision=ARXIV_REVISION,
    )
    if len(metadata_ds) != n_samples:
        raise RuntimeError(
            f"arXiv metadata row count {len(metadata_ds):,} != embeddings {n_samples:,}"
        )
    metadata_id_digest = hashlib.sha256()
    categories: list[str] = []
    n_chunks: list[int] = []
    for batch in metadata_ds.iter(batch_size=8192):
        _hash_identifier_batch(metadata_id_digest, batch["arxiv_id"])
        categories.extend(str(value) for value in batch["primary_category"])
        n_chunks.extend(int(value) for value in batch["n_chunks"])
    if embedding_id_digest.hexdigest() != metadata_id_digest.hexdigest():
        raise RuntimeError("arXiv embedding and metadata identifier orders do not match")

    label_names = sorted(set(categories))
    name_to_code = {name: index for index, name in enumerate(label_names)}
    label_codes = np.fromiter(
        (name_to_code[name] for name in categories),
        dtype=np.int16,
        count=n_samples,
    )
    labels_name = "arxiv_primary_category_codes.npy"
    label_names_name = "arxiv_primary_category_names.json"
    np.save(prepared_root / labels_name, label_codes)
    unique_codes, category_counts = np.unique(label_codes, return_counts=True)
    count_map = {
        label_names[int(code)]: int(count)
        for code, count in zip(unique_codes, category_counts)
    }
    write_json(
        prepared_root / label_names_name,
        {
            "code_to_primary_category": {
                str(index): name for index, name in enumerate(label_names)
            },
            "primary_category_counts": count_map,
        },
    )

    chunks = np.asarray(n_chunks, dtype=np.int32)
    manifest = {
        "schema_version": 1,
        "dataset": "DiRe arXiv BGE-small paper-level embeddings",
        "source": ARXIV_DATASET_ID,
        "source_revision": ARXIV_REVISION,
        "canonical_doi": "10.5281/zenodo.19837856",
        "license": {
            "numerical_artifacts": "CC BY 4.0",
            "descriptive_metadata": "CC0",
        },
        "n_samples": n_samples,
        "n_features": n_features,
        "input": (
            "Paper-level mean-pooled BAAI/bge-small-en-v1.5 embeddings, "
            "converted to float32, L2-normalized row-wise, feature-mean-centered, "
            "and divided by one global maximum absolute value. Every reducer "
            "receives exactly the same array; the final translation and uniform "
            "rescaling preserve Euclidean neighbor rankings exactly."
        ),
        "common_input_transform": {
            "row_l2_normalization": True,
            "feature_centering": True,
            "uniform_global_scale": common_scale,
        },
        "source_vector_norm": {
            "minimum": norm_min,
            "mean": norm_sum / n_samples,
            "maximum": norm_max,
        },
        "n_chunks_per_paper": {
            "minimum": int(chunks.min()),
            "median": float(np.median(chunks)),
            "mean": float(chunks.mean()),
            "maximum": int(chunks.max()),
        },
        "identifier_order_sha256": embedding_id_digest.hexdigest(),
        "n_primary_categories": len(label_names),
        "files": {
            "input": input_name,
            "labels": labels_name,
            "label_names": label_names_name,
        },
        "preparation_wall_time_sec": time.time() - started,
    }
    write_json(output_manifest, manifest)
    print(
        f"prepared arXiv: {n_samples:,} papers x {n_features} dimensions, "
        f"{len(label_names)} primary categories",
        flush=True,
    )
    return manifest


def environment_snapshot() -> dict:
    uname = os.uname()
    payload = {
        "python": os.sys.version,
        "platform": {
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        payload["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        payload["git_head_error"] = str(exc)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("tenx", "arxiv"),
        default=("tenx", "arxiv"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/revision3/raw"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/revision3/prepared"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.prepared_root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    if "tenx" in args.datasets:
        manifests["tenx"] = prepare_tenx(
            args.raw_root,
            args.prepared_root,
            args.force,
        )
    if "arxiv" in args.datasets:
        manifests["arxiv"] = prepare_arxiv(
            args.raw_root,
            args.prepared_root,
            args.force,
        )
    write_json(
        args.prepared_root / "manifest.json",
        {
            "schema_version": 1,
            "purpose": (
                "Versioned preparation manifest for the large-data response "
                "to Scientific Reports Revision 3."
            ),
            "environment": environment_snapshot(),
            "datasets": manifests,
        },
    )


if __name__ == "__main__":
    main()
