#!/usr/bin/env python3
"""Generate benchmark artifacts for the DiRe paper."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import time
import urllib.error
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, make_blobs, make_moons
from dire_rapids import create_dire
from dire_rapids.metrics import evaluate_embedding
from dire_rapids.utils import _load_cytof, _load_dire_dataset
from runtime_protocol import summarize_fit_times

warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value .* overridden to 1 by setting random_state.*",
    category=UserWarning,
    module=r"umap\.umap_",
)
warnings.filterwarnings(
    "ignore",
    message=r"Graph is not fully connected, spectral embedding may not work as expected.*",
    category=UserWarning,
    module=r"sklearn\.manifold\._spectral_embedding",
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.linalg\.svd: During SVD computation with the selected cusolver driver.*",
    category=UserWarning,
)

DEFAULT_METHODS = ("dire", "cuml_tsne", "cuml_umap", "opentsne", "umap")
CPU_METHODS = frozenset({"opentsne", "umap"})
KNOWN_METHODS = (
    "dire",
    "dire_topology",
    "cuml_tsne",
    "cuml_umap",
    "opentsne",
    "umap",
    "dire_pca_init",
    "dire_spectral",
    "dire_spectral_init",
)
DISPLAY = {
    "dire": "DiRe-RAPIDS",
    "dire_topology": "DiRe (topology preset)",
    "cuml_tsne": "cuML tSNE",
    "cuml_umap": "cuML UMAP",
    "opentsne": "openTSNE",
    "umap": "Original UMAP",
    "dire_pca_init": "PCA initialization",
    "dire_spectral": "DiRe (spectral init.)",
    "dire_spectral_init": "Spectral initialization",
}
BACKEND = {
    "dire": "GPU: DiRe-RAPIDS/PyTorch/cuVS",
    "dire_topology": "GPU: DiRe-RAPIDS/PyTorch/cuVS",
    "cuml_tsne": "GPU: RAPIDS cuML",
    "cuml_umap": "GPU: RAPIDS cuML",
    "opentsne": "CPU: openTSNE",
    "umap": "CPU: umap-learn reference",
    "dire_pca_init": "GPU: DiRe-RAPIDS initialization only",
    "dire_spectral": "GPU: DiRe-RAPIDS/PyTorch/cuVS",
    "dire_spectral_init": "GPU: DiRe-RAPIDS initialization only",
}
METRIC_KEYS = (
    ("time", "Runtime (seconds)"),
    ("stress", "Embedding stress"),
    ("neighborhood", "Neighborhood preservation"),
    ("context", "SVM context score"),
    ("persistence-dim-0", "DTW Betti discrepancy: dimension 0"),
    ("persistence-dim-1", "DTW Betti discrepancy: dimension 1"),
)


def load_label_metadata():
    metadata_path = Path(__file__).resolve().parents[1] / "data" / "label_metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def environment_info():
    info = {}
    for name in [
        "numpy",
        "torch",
        "cuml",
        "cuvs",
        "cudf",
        "cupy",
        "sklearn",
        "umap",
        "openTSNE",
        "dire_rapids",
    ]:
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            info[name] = f"unavailable: {type(exc).__name__}: {exc}"
    info["platform"] = platform.platform()
    info["python"] = platform.python_version()
    info["cpu_count"] = os.cpu_count()
    try:
        dist = importlib.metadata.distribution("dire-rapids")
        direct_url = dist.read_text("direct_url.json")
        info["dire_rapids_direct_url"] = json.loads(direct_url) if direct_url else None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        info["dire_rapids_direct_url_error"] = str(exc)
    try:
        info["lscpu"] = subprocess.check_output(["lscpu"], text=True).strip()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        info["lscpu_error"] = str(exc)
    try:
        import torch
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        info["torch_cuda_error"] = str(exc)
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        info["nvidia_smi_error"] = str(exc)
    return info


def take_subset(X, y, max_points, seed):
    if max_points is None or len(X) <= max_points:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_points, replace=False)
    return X[idx], None if y is None else y[idx]


def fetch_mnist_openml():
    ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = ds.data.astype(np.float32) / 255.0
    y = ds.target.astype(np.int32)
    return X, y, "openml:mnist_784"


def fetch_mnist_hf():
    from datasets import load_dataset  # lazy import; optional fallback dependency

    last_exc = None
    # `mnist` can fail on some datasets/huggingface_hub combinations; prefer explicit namespace ids.
    for repo in ("ylecun/mnist", "mnist"):
        try:
            splits = [load_dataset(repo, split=split) for split in ("train", "test")]
            X = np.concatenate(
                [np.asarray(split["image"], dtype=np.float32).reshape(-1, 28 * 28) for split in splits],
                axis=0,
            ) / 255.0
            y = np.concatenate([np.asarray(split["label"], dtype=np.int32) for split in splits], axis=0)
            return X, y, f"huggingface:{repo}(train+test)"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
    raise RuntimeError(f"Hugging Face MNIST fetch failed for all known repo ids: {last_exc}") from last_exc


def load_dataset(name, max_points, seed, full_datasets):
    label_metadata = load_label_metadata()
    row_cap = None if full_datasets else max_points
    if name == "blobs":
        X, y = make_blobs(n_samples=10_000, centers=12, n_features=1000, random_state=seed)
        X = X.astype(np.float32)
        return X, y, {
            "source": "sklearn.make_blobs",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "label_metadata": label_metadata.get(name, {}),
        }
    if name == "disk":
        X, y = _load_dire_dataset("disk_uniform", n_samples=10_000, n_features=2, random_state=seed)
        return X.astype(np.float32), y, {"source": "dire:disk_uniform", "n_samples": len(X), "n_features": X.shape[1]}
    if name == "moons":
        X, y = make_moons(n_samples=10_000, noise=0.05, random_state=seed)
        return X.astype(np.float32), y, {
            "source": "sklearn.make_moons",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "label_metadata": label_metadata.get(name, {}),
        }
    if name == "mnist":
        try:
            X, y, source = fetch_mnist_openml()
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            warnings.warn(
                f"OpenML mnist_784 fetch failed ({type(exc).__name__}: {exc}); falling back to Hugging Face.",
                RuntimeWarning,
            )
            X, y, source = fetch_mnist_hf()
        X, y = take_subset(X, y, row_cap, seed)
        return X, y, {
            "source": source,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "preprocessing": "scaled to [0,1]",
            "label_metadata": label_metadata.get(name, {}),
        }
    if name in {"levine13", "levine32"}:
        X, y = _load_cytof(
            name,
            drop_columns=("label", "individual"),
            arcsinh_cofactor=5,
            drop_unassigned=True,
        )
        X, y = take_subset(X.astype(np.float32), y, row_cap, seed)
        return X, y, {
            "source": f"cytof:{name}",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "preprocessing": "drop label/individual metadata columns; arcsinh cofactor 5",
            "label_metadata": label_metadata.get(name, {}),
        }
    raise ValueError(f"unknown dataset: {name}")


class OpenTSNEReducer:
    """Small sklearn-style adapter around openTSNE's ``fit`` API."""

    def __init__(self, seed, n_jobs):
        self.seed = seed
        self.n_jobs = n_jobs

    def fit_transform(self, X):
        from openTSNE import TSNE

        embedding = TSNE(
            n_components=2,
            perplexity=30,
            initialization="pca",
            negative_gradient_method="fft",
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=False,
        ).fit(X)
        return np.asarray(embedding, dtype=np.float32)


def make_reducer(method, seed, args):
    if method == "dire":
        return create_dire(
            n_components=2,
            n_neighbors=16,
            init="pca",
            max_iter_layout=args.dire_iterations,
            random_state=seed,
            verbose=False,
        )
    if method == "dire_topology":
        from dire_rapids import TOPOLOGY_TUNED

        return create_dire(
            n_components=2,
            random_state=seed,
            verbose=False,
            **TOPOLOGY_TUNED,
        )
    if method == "dire_pca_init":
        return create_dire(
            n_components=2,
            n_neighbors=16,
            init="pca",
            max_iter_layout=0,
            random_state=seed,
            verbose=False,
        )
    if method == "dire_spectral":
        return create_dire(
            n_components=2,
            n_neighbors=16,
            init="spectral",
            max_iter_layout=args.dire_iterations,
            random_state=seed,
            verbose=False,
        )
    if method == "dire_spectral_init":
        return create_dire(
            n_components=2,
            n_neighbors=16,
            init="spectral",
            max_iter_layout=0,
            random_state=seed,
            verbose=False,
        )
    if method == "cuml_umap":
        from cuml import UMAP
        return UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed, verbose=False)
    if method == "cuml_tsne":
        from cuml import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=seed, verbose=False)
    if method == "opentsne":
        return OpenTSNEReducer(seed=seed, n_jobs=args.cpu_jobs)
    if method == "umap":
        from umap import UMAP
        return UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            n_jobs=args.cpu_jobs,
            random_state=seed,
        )
    raise ValueError(method)


def to_numpy(embedding):
    if hasattr(embedding, "get"):
        embedding = embedding.get()
    try:
        import torch
        if torch.is_tensor(embedding):
            embedding = embedding.detach().cpu().numpy()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return np.asarray(embedding, dtype=np.float32)


def method_dataset(X, y, method, args):
    cap = args.cpu_max_points if method in CPU_METHODS else None
    if cap is None or len(X) <= cap:
        return X, y, None
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(X), size=cap, replace=False))
    return X[idx], None if y is None else y[idx], {
        "source_n_samples": int(len(X)),
        "n_samples": int(len(idx)),
        "sample_policy": "uniform without replacement",
        "sample_seed": int(args.seed),
        "reason": f"method cap for CPU baseline {method}",
    }


def plot_embedding(path, embedding, labels, title):
    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.65)
    else:
        _, color_labels = np.unique(labels, return_inverse=True)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=color_labels, cmap="tab20", s=2, alpha=0.65)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def metric_value(result, key):
    aggregate = result.get("aggregate", {})
    if key in aggregate and "mean" in aggregate[key]:
        return aggregate[key]["mean"]
    metrics = result.get("metrics", {})
    if key == "time":
        return result.get("fit_time_sec")
    if key == "stress":
        return metrics.get("local", {}).get("stress")
    if key == "neighborhood":
        local = metrics.get("local", {})
        if "neighbor_score" in local:
            return local["neighbor_score"]
        neighbor = local.get("neighbor")
        if isinstance(neighbor, (list, tuple)) and neighbor:
            return neighbor[0]
        return None
    if key == "context":
        ctx = metrics.get("context", {})
        if "svm" in ctx:
            return ctx["svm"][2]
        if "knn" in ctx:
            return ctx["knn"][2]
        return None
    topo = metrics.get("topology", {}).get("metrics", {})
    if key == "persistence-dim-0":
        return topo.get("dtw_beta0")
    if key == "persistence-dim-1":
        return topo.get("dtw_beta1")
    return None


def metric_error(result, key):
    aggregate = result.get("aggregate", {})
    if key in aggregate and "std" in aggregate[key]:
        return aggregate[key]["std"]
    return None


def aggregate_metric_repeats(records):
    aggregate = {}
    for key, _title in METRIC_KEYS:
        values = [metric_value(record, key) for record in records if "error" not in record]
        values = [value for value in values if value is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(len(arr)),
        }
    return aggregate


def plot_metric_bars(path, dataset_name, results, key, title):
    methods = [method for method in results if method in DISPLAY]
    values = [metric_value(results[m], key) for m in methods]
    errors = [metric_error(results[m], key) for m in methods]
    names = [DISPLAY[m] for m in methods]
    clean = [(n, v, e) for n, v, e in zip(names, values, errors) if v is not None]
    if not clean:
        return
    names, values, errors = zip(*clean)
    yerr = [0.0 if err is None else err for err in errors]
    has_error = any(err > 0 for err in yerr)
    plt.figure(figsize=(6, 4))
    plt.bar(names, values, yerr=yerr if has_error else None, capsize=4 if has_error else 0)
    plt.title(f"{dataset_name}: {title}")
    plt.xticks(rotation=25, ha="right")
    if all(abs(value) < 1e-12 for value in values):
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylim(-0.05, 0.05)
        plt.text(
            0.5,
            0.55,
            "all methods: 0.0",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def topology_subset_indices(n_samples, args, seed):
    sample_size = min(args.topology_sample_size, n_samples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False))


def atlas_curve_parameters(args) -> dict:
    return {
        "k_neighbors": args.topology_atlas_neighbors,
        "density_threshold": args.topology_atlas_density_threshold,
        "overlap_factor": args.topology_atlas_overlap_factor,
        "n_steps": args.topology_steps,
    }


def compute_atlas_curve(points, args):
    from dire_rapids.betti_curve import compute_betti_curve_gpu

    return compute_betti_curve_gpu(
        points,
        **atlas_curve_parameters(args),
    )


def compute_atlas_topology_metrics(
    data,
    embedding,
    args,
    seed,
    reference_curve=None,
):
    """Compute the explicit rank-based local-kNN atlas diagnostic.

    ``dire_rapids.metrics.compute_global_metrics`` currently reports
    ``backend="atlas"`` while its internal backend selector defaults to
    ``prefer_ripser=True``.  Calling that wrapper would therefore ignore the
    atlas neighborhood, density, and overlap parameters whenever Ripser is
    installed.  The benchmark invokes the GPU atlas implementation directly
    so the recorded backend is an executed code path, not merely a label.
    """

    del seed  # The fixed row-index subset is already selected by this seed.
    from fastdtw import fastdtw

    curve_parameters = atlas_curve_parameters(args)
    high = (
        compute_atlas_curve(data, args)
        if reference_curve is None
        else reference_curve
    )
    low = compute_atlas_curve(embedding, args)
    metrics = {}
    for dimension in (0, 1):
        high_curve = np.column_stack(
            (
                high["filtration_values"],
                high[f"beta_{dimension}"],
            )
        )
        low_curve = np.column_stack(
            (
                low["filtration_values"],
                low[f"beta_{dimension}"],
            )
        )
        distance, _path = fastdtw(high_curve, low_curve, dist=2)
        metrics[f"dtw_beta{dimension}"] = float(distance / len(data))
    return {
        "metrics": metrics,
        "backend": "atlas",
        "backend_detail": "direct GPU rank-based local-kNN atlas",
        "prefer_ripser": False,
        "parameters": {
            **curve_parameters,
            "implementation": (
                "dire_rapids.betti_curve.compute_betti_curve_gpu"
            ),
        },
    }


def subset_digest(indices):
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes(order="C")
    ).hexdigest()


def run_method_once(
    X,
    y,
    method,
    seed,
    args,
    compute_topology,
    topology_reference_cache,
):
    reducer = make_reducer(method, seed, args)
    t0 = time.perf_counter()
    embedding = to_numpy(reducer.fit_transform(X))
    fit_time = time.perf_counter() - t0
    metrics = evaluate_embedding(
        X,
        embedding,
        y,
        subsample_threshold=args.metric_subsample,
        compute_distortion=True,
        compute_context=y is not None,
        compute_topology=False,
        verbose=False,
    )
    topology_metadata = None
    if compute_topology:
        subset_seed = args.topology_subset_seed + (seed - args.seed)
        idx = topology_subset_indices(len(X), args, subset_seed)
        print(
            f"Computing topology for {method}, seed {seed}: "
            f"{len(idx)}/{len(X)} points",
            flush=True,
        )
        topology_t0 = time.perf_counter()
        digest = subset_digest(idx)
        reference_key = (
            int(subset_seed),
            digest,
            int(len(X)),
            int(X.shape[1]),
        )
        cache_hit = reference_key in topology_reference_cache
        if cache_hit:
            reference_curve = topology_reference_cache[reference_key]
            reference_wall_time = 0.0
        else:
            reference_t0 = time.perf_counter()
            reference_curve = compute_atlas_curve(X[idx], args)
            reference_wall_time = time.perf_counter() - reference_t0
            topology_reference_cache[reference_key] = reference_curve
        embedding_t0 = time.perf_counter()
        metrics["topology"] = compute_atlas_topology_metrics(
            X[idx],
            embedding[idx],
            args,
            subset_seed,
            reference_curve=reference_curve,
        )
        embedding_wall_time = time.perf_counter() - embedding_t0
        topology_time = time.perf_counter() - topology_t0
        topology_metadata = {
            "sample_size": int(len(idx)),
            "sample_fraction": float(len(idx) / len(X)),
            "sample_policy": "fixed-size uniform without replacement",
            "subset_seed": int(subset_seed),
            "indices_sha256": digest,
            "indices": idx.tolist(),
            "backend": "atlas",
            "backend_detail": "direct GPU rank-based local-kNN atlas",
            "prefer_ripser": False,
            "n_steps": int(args.topology_steps),
            "wall_time_sec": float(topology_time),
            "reference_curve_cache_hit": bool(cache_hit),
            "reference_curve_wall_time_sec": float(reference_wall_time),
            "embedding_curve_wall_time_sec": float(embedding_wall_time),
        }
    return embedding, {
        "seed": seed,
        "fit_time_sec": fit_time,
        "metric_subsample": args.metric_subsample,
        "topology_sample": topology_metadata,
        "metrics": json_ready(metrics),
    }


def run_dataset(name, args):
    dataset_dir = args.output / name
    emb_dir = args.output / "embeddings"
    cmp_dir = args.output / f"{name}_comparison"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    X, y, info = load_dataset(name, args.max_points, args.seed, args.full_datasets)
    results = {"dataset": info, "methods": {}}
    topology_reference_cache = {}
    np.save(dataset_dir / "labels.npy", np.asarray([] if y is None else y))

    for method in args.methods:
        X_method, y_method, input_sample = method_dataset(X, y, method, args)
        records = []
        repeats = args.cpu_repeats if method in CPU_METHODS else args.repeats
        method_result = {
            "display": DISPLAY[method],
            "backend": BACKEND[method],
            "hardware_class": "CPU" if method in CPU_METHODS else "GPU",
            "repeats_requested": repeats,
            "topology_repeats_requested": (
                min(args.topology_repeats, repeats) if args.topology else 0
            ),
            "metric_subsample": args.metric_subsample,
            "topology_backend": "atlas" if args.topology else None,
            "topology_backend_detail": (
                "direct GPU rank-based local-kNN atlas"
                if args.topology
                else None
            ),
            "topology_prefer_ripser": False if args.topology else None,
            "topology_sample_size": args.topology_sample_size if args.topology else None,
            "topology_steps": args.topology_steps if args.topology else None,
            "input_n_samples": int(len(X_method)),
            "input_sample": input_sample,
        }
        if method == "dire_topology":
            from dire_rapids import TOPOLOGY_TUNED

            method_result["configuration_origin"] = {
                "name": "dire_rapids.TOPOLOGY_TUNED",
                "source_commit": "9117dc45a3e130fa1d636dfd181f3e97960c5b3b",
                "committed_utc": "2026-04-23T02:01:37Z",
                "selection_independence": (
                    "The named source-distributed preset predates the "
                    "Revision 3 benchmark."
                ),
            }
            method_result["parameters"] = {
                "n_components": 2,
                **json_ready(TOPOLOGY_TUNED),
            }
        if y_method is not None and len(y_method) != len(y):
            np.save(dataset_dir / f"{method}_labels.npy", np.asarray(y_method))
        try:
            print(f"\nDataset {name}, method {method}", flush=True)
            if input_sample is not None:
                print(
                    f"  using {len(X_method)}/{len(X)} input points for {method}",
                    flush=True,
                )
            for repeat_index in range(repeats):
                repeat_seed = args.seed + repeat_index
                compute_topology = args.topology and repeat_index < args.topology_repeats
                print(
                    f"  repeat {repeat_index + 1}/{repeats}, seed {repeat_seed}"
                    f"{' with topology' if compute_topology else ''}",
                    flush=True,
                )
                embedding, record = run_method_once(
                    X_method,
                    y_method,
                    method,
                    repeat_seed,
                    args,
                    compute_topology,
                    topology_reference_cache,
                )
                record["input_n_samples"] = int(len(X_method))
                record["input_sample"] = input_sample
                record["topology_computed"] = bool(compute_topology)
                records.append(record)
                if repeat_index == 0:
                    np.save(dataset_dir / f"{method}_embedding.npy", embedding)
                    plot_embedding(
                        emb_dir / f"{name}-{method.replace('_', '-')}.png",
                        embedding,
                        y_method,
                        f"{name}: {DISPLAY[method]}",
                    )
                    method_result["seed"] = repeat_seed
                    method_result["fit_time_sec"] = record["fit_time_sec"]
                    method_result["metrics"] = record["metrics"]
            method_result["repeats"] = records
            method_result["aggregate"] = aggregate_metric_repeats(records)
            method_result["timing"] = summarize_fit_times(records)
            results["methods"][method] = method_result
        except Exception as exc:  # pylint: disable=broad-exception-caught
            method_result["repeats"] = records
            if records:
                method_result["aggregate"] = aggregate_metric_repeats(records)
                method_result["timing"] = summarize_fit_times(records)
            method_result["error"] = f"{type(exc).__name__}: {exc}"
            results["methods"][method] = method_result

    with open(dataset_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(results), f, indent=2)

    for suffix, title in METRIC_KEYS:
        key = suffix
        plot_metric_bars(cmp_dir / f"{name}-{suffix}.png", name, results["methods"], key, title)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("article_benchmark_results"))
    parser.add_argument("--datasets", nargs="+", default=["blobs", "disk", "moons", "mnist", "levine13", "levine32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=10_000)
    parser.add_argument(
        "--full-datasets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="load all available rows for datasets with fixed external sources; synthetic datasets keep their configured size",
    )
    parser.add_argument(
        "--cpu-max-points",
        "--umap-max-points",
        dest="cpu_max_points",
        type=int,
        default=10_000,
        help="maximum input points for CPU baselines; use 0 to disable this method-specific cap",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=KNOWN_METHODS,
        default=list(DEFAULT_METHODS),
        help="reducers to run, in display order",
    )
    parser.add_argument(
        "--dire-iterations",
        type=int,
        default=128,
        help="layout iterations for full DiRe methods",
    )
    parser.add_argument(
        "--cpu-jobs",
        type=int,
        default=1,
        help="worker count for CPU baselines",
    )
    parser.add_argument("--metric-subsample", type=float, default=0.05)
    parser.add_argument(
        "--topology-sample-size",
        type=int,
        default=1_000,
        help=(
            "fixed number of rows in each paired atlas-topology subset; "
            "datasets with fewer rows use all rows"
        ),
    )
    parser.add_argument("--topology-steps", type=int, default=100)
    parser.add_argument(
        "--topology-subset-seed",
        type=int,
        default=1042,
        help=(
            "seed for the first fixed row-index subset; repeat i uses this "
            "seed plus i, independently of the layout seed"
        ),
    )
    parser.add_argument(
        "--topology-atlas-neighbors",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--topology-atlas-density-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--topology-atlas-overlap-factor",
        type=float,
        default=1.5,
    )
    parser.add_argument("--topology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=1, help="number of seeded runs used for metric means and error bars")
    parser.add_argument(
        "--cpu-repeats",
        type=int,
        default=3,
        help="number of seeded runs for slower CPU baselines",
    )
    parser.add_argument(
        "--topology-repeats",
        type=int,
        default=1,
        help="number of initial seeded runs that compute persistent-homology metrics",
    )
    args = parser.parse_args()
    if args.metric_subsample < 0.05:
        raise ValueError("metric_subsample must be at least 0.05")
    if args.max_points < 1:
        raise ValueError("max_points must be positive")
    if args.cpu_max_points < 0:
        raise ValueError("cpu_max_points must be non-negative")
    if args.cpu_max_points == 0:
        args.cpu_max_points = None
    if args.dire_iterations < 1:
        raise ValueError("dire_iterations must be positive")
    if args.repeats < 1 or args.cpu_repeats < 1:
        raise ValueError("repeat counts must be positive")
    if args.topology_sample_size < 1:
        raise ValueError("topology_sample_size must be positive")
    if args.topology_atlas_neighbors < 2:
        raise ValueError("topology_atlas_neighbors must be at least 2")
    if not 0.0 < args.topology_atlas_density_threshold <= 1.0:
        raise ValueError(
            "topology_atlas_density_threshold must be in (0, 1]"
        )
    if args.topology_atlas_overlap_factor <= 0.0:
        raise ValueError("topology_atlas_overlap_factor must be positive")
    args.topology_repeats = max(0, min(args.topology_repeats, args.repeats))

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "environment.json", "w", encoding="utf-8") as f:
        json.dump(environment_info(), f, indent=2)

    all_results = {}
    for dataset in args.datasets:
        all_results[dataset] = run_dataset(dataset, args)
        with open(args.output / "all_results.json", "w", encoding="utf-8") as f:
            json.dump(json_ready(all_results), f, indent=2)


if __name__ == "__main__":
    main()
