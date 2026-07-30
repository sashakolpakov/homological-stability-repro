#!/usr/bin/env python3
"""Run isolated, same-GPU scaling benchmarks on the prepared large datasets.

Each (dataset, method, size) configuration runs in its own subprocess.  This
prevents allocator state and failures in one reducer from contaminating another.
Within that subprocess the reducer is fitted repeatedly: the first timing is
reported as cold-start/end-to-end, and later timings as steady-state.  Total GPU
memory is sampled through NVML rather than through a framework-specific
allocator, so cuML, cuVS, CuPy, and PyTorch allocations are measured together.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from revision3_source_manifest import build_source_manifest


METHODS = (
    "dire_auto",
    "dire",
    "dire_ivf_flat_control",
    "dire_spectral",
    "cuml_umap",
    "cuml_tsne",
    "pca2",
)
DEFAULT_METHODS = tuple(
    method for method in METHODS if method != "dire_ivf_flat_control"
)
DISPLAY = {
    "dire_auto": "DiRe-RAPIDS (production auto policy)",
    "dire": "DiRe-RAPIDS (forced IVF-Flat control)",
    "dire_ivf_flat_control": "DiRe-RAPIDS (fresh IVF-Flat policy control)",
    "dire_spectral": "DiRe-RAPIDS (spectral-init sensitivity)",
    "cuml_umap": "cuML UMAP",
    "cuml_tsne": "cuML t-SNE",
    "pca2": "cuML PCA (2D reference)",
}
EXPECTED_DIRE_COMMIT = "293b622cc79fa8ea6fd5b54009e0930e3385b22f"
EXPECTED_DIRE_REF = "refs/heads/main"
DIRE_REF_RESOLVED_UTC = "2026-07-28"

DATASET_FILES = {
    "tenx": {
        "input": "tenx_mouse_brain_pca20.npy",
        "labels": "tenx_mouse_brain_kmeans20.npy",
        "manifest": "tenx_manifest.json",
        "default_sizes": (10_000, 50_000, 100_000, 250_000, 500_000, 1_306_127),
    },
    "arxiv": {
        "input": "arxiv_bge_small_384_l2_centered.npy",
        "labels": "arxiv_primary_category_codes.npy",
        "manifest": "arxiv_manifest.json",
        "default_sizes": (10_000, 50_000, 100_000, 250_000, 500_000, 723_457),
    },
}


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"unavailable: {type(exc).__name__}: {exc}"


def environment_snapshot() -> dict:
    direct_url = None
    try:
        distribution = importlib.metadata.distribution("dire-rapids")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        direct_url = {"error": str(exc)}
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "torch",
                "cuml-cu12",
                "cuvs-cu12",
                "cudf-cu12",
                "cupy-cuda12x",
                "dire-rapids",
                "nvidia-ml-py",
                "pandas",
                "pyarrow",
                "datasets",
                "openTSNE",
                "umap-learn",
                "ripser",
                "fastdtw",
                "gudhi",
            )
        },
        "dire_rapids_direct_url": direct_url,
        "expected_dire_commit": EXPECTED_DIRE_COMMIT,
        "expected_dire_ref": EXPECTED_DIRE_REF,
        "dire_ref_resolved_utc": DIRE_REF_RESOLVED_UTC,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "lscpu": command_output(["lscpu"]),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "git_remote_origin": command_output(
            ["git", "remote", "get-url", "origin"]
        ),
        "revision3_source_files": build_source_manifest(Path(".")),
        "pip_freeze_all": command_output(
            [sys.executable, "-m", "pip", "freeze", "--all"]
        ),
    }


def resolve_dire_commit(snapshot: dict) -> str | None:
    direct = snapshot.get("dire_rapids_direct_url")
    if not isinstance(direct, dict):
        return None
    vcs = direct.get("vcs_info")
    if not isinstance(vcs, dict):
        return None
    commit = vcs.get("commit_id")
    return str(commit) if commit else None


class NVMLMonitor:
    """Poll total memory used on one physical GPU."""

    def __init__(self, device_index: int = 0, interval_sec: float = 0.05):
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.total_bytes: int | None = None
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None

    def _sample(self) -> None:
        if self._pynvml is None or self._handle is None:
            return
        info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        used = int(info.used)
        self.total_bytes = int(info.total)
        if self.baseline_bytes is None:
            self.baseline_bytes = used
        self.peak_bytes = used if self.peak_bytes is None else max(self.peak_bytes, used)
        self.samples += 1

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self._sample()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.error = f"{type(exc).__name__}: {exc}"
                return

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._sample()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4 * self.interval_sec))
        try:
            self._sample()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        incremental = None
        if self.peak_bytes is not None and self.baseline_bytes is not None:
            incremental = max(0, self.peak_bytes - self.baseline_bytes)
        return {
            "baseline_bytes": self.baseline_bytes,
            "peak_used_bytes": self.peak_bytes,
            "peak_incremental_bytes": incremental,
            "total_bytes": self.total_bytes,
            "samples": self.samples,
            "poll_interval_sec": self.interval_sec,
            "error": self.error,
        }


def gpu_synchronize() -> None:
    # CuPy synchronizes work submitted by cuML/cuVS.  DiRe also uses PyTorch,
    # so synchronize its default device when torch is already loaded.
    try:
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    if "torch" in sys.modules:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # pylint: disable=broad-exception-caught
            pass


def cleanup_gpu() -> None:
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    if "torch" in sys.modules:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pylint: disable=broad-exception-caught
            pass


def to_numpy(embedding) -> np.ndarray:
    if hasattr(embedding, "to_numpy"):
        embedding = embedding.to_numpy()
    if hasattr(embedding, "get"):
        embedding = embedding.get()
    if "torch" in sys.modules:
        try:
            import torch

            if torch.is_tensor(embedding):
                embedding = embedding.detach().cpu().numpy()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise RuntimeError(f"reducer returned unexpected shape {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("reducer returned non-finite coordinates")
    return np.ascontiguousarray(array)


def build_reducer(method: str, seed: int, n_neighbors: int, dire_iterations: int):
    parameters: dict
    if method in (
        "dire_auto",
        "dire",
        "dire_ivf_flat_control",
        "dire_spectral",
    ):
        # Importing dire_rapids loads RAPIDS before torch to avoid shared-library
        # ordering problems documented by the package.
        from dire_rapids import create_dire

        common = {
            "backend": "cuvs",
            "knn_backend": "cuvs",
            "cuvs_index_type": (
                "auto" if method == "dire_auto" else "ivf_flat"
            ),
            "n_components": 2,
            "random_state": seed,
            "normalize": False,
            "verbose": False,
        }
        parameters = {
            **common,
            "n_neighbors": n_neighbors,
            "init": "spectral" if method == "dire_spectral" else "pca",
            "max_iter_layout": dire_iterations,
        }
        return create_dire(**parameters), parameters
    if method == "cuml_umap":
        from cuml import UMAP

        parameters = {
            "n_components": 2,
            "n_neighbors": n_neighbors,
            "min_dist": 0.1,
            "init": "spectral",
            "random_state": seed,
            "verbose": False,
        }
        return UMAP(**parameters), parameters
    if method == "cuml_tsne":
        from cuml import TSNE

        parameters = {
            "n_components": 2,
            "perplexity": 30.0,
            "method": "fft",
            "random_state": seed,
            "verbose": False,
        }
        return TSNE(**parameters), parameters
    if method == "pca2":
        from cuml.decomposition import PCA

        parameters = {
            "n_components": 2,
            "whiten": False,
        }
        return PCA(**parameters), parameters
    raise ValueError(f"unknown method: {method}")


def reducer_diagnostics(reducer) -> dict:
    diagnostics = {
        "class": f"{type(reducer).__module__}.{type(reducer).__name__}",
    }
    for attribute in (
        "_last_knn_backend",
        "_last_knn_reducer",
        "_last_knn_chunk_size",
        "_last_knn_distance_strategy",
        "cuvs_index_type",
        "n_neighbors",
        "neg_ratio",
        "max_iter_layout",
        "use_exact_repulsion",
        "normalize",
    ):
        if hasattr(reducer, attribute):
            diagnostics[attribute] = json_ready(getattr(reducer, attribute))
    return diagnostics


def instrument_dire_reducer(reducer) -> dict:
    """Instrument one DiRe instance without changing its numerical work.

    The upstream library issue tracking public diagnostics is
    https://github.com/sashakolpakov/dire-rapids/issues/13.  Until that is
    resolved independently, this benchmark records synchronized timings around
    the existing private stage boundaries and observes the index type passed to
    the existing cuVS search call.
    """

    state = {
        "stage_timings_sec": {
            "knn_graph": 0.0,
            "initialization": 0.0,
            "layout": 0.0,
        },
        "stage_call_counts": {
            "knn_graph": 0,
            "initialization": 0,
            "layout": 0,
        },
        "effective_cuvs_index_type": None,
        "chunked_force_fallback_calls": 0,
    }

    def time_method(attribute: str, stage: str) -> None:
        if not hasattr(reducer, attribute):
            return
        original = getattr(reducer, attribute)

        def wrapped(*args, **kwargs):
            gpu_synchronize()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                gpu_synchronize()
                state["stage_timings_sec"][stage] += (
                    time.perf_counter() - started
                )
                state["stage_call_counts"][stage] += 1

        setattr(reducer, attribute, wrapped)

    time_method("_compute_knn", "knn_graph")
    time_method("_initialize_embedding", "initialization")
    time_method("_optimize_layout", "layout")

    if hasattr(reducer, "_search_cuvs"):
        original_search = reducer._search_cuvs

        def observed_search(index, index_type, *args, **kwargs):
            state["effective_cuvs_index_type"] = str(index_type)
            return original_search(index, index_type, *args, **kwargs)

        reducer._search_cuvs = observed_search

    if hasattr(reducer, "_compute_forces_chunked"):
        original_chunked = reducer._compute_forces_chunked

        def counted_chunked(*args, **kwargs):
            state["chunked_force_fallback_calls"] += 1
            return original_chunked(*args, **kwargs)

        reducer._compute_forces_chunked = counted_chunked

    return state


def dire_runtime_diagnostics(reducer, instrumentation: dict | None) -> dict:
    diagnostics = reducer_diagnostics(reducer)
    if instrumentation is None:
        return diagnostics
    diagnostics.update(instrumentation)
    try:
        from dire_rapids import dire_pytorch

        diagnostics["torch_compile_failed"] = bool(
            getattr(dire_pytorch, "_torch_compile_failed", False)
        )
        diagnostics["attraction_compile_failed"] = bool(
            getattr(dire_pytorch, "_attraction_compile_failed", False)
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        diagnostics["compile_diagnostics_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
    return diagnostics


def load_worker_input(input_path: Path, subset_path: Path | None) -> np.ndarray:
    source = np.load(input_path, mmap_mode="r")
    if source.dtype != np.float32 or source.ndim != 2:
        raise RuntimeError(
            f"prepared input must be a float32 matrix, found {source.dtype} {source.shape}"
        )
    if subset_path is None:
        # Most reducers make their own device copy.  Keep the full input
        # memory-mapped and avoid an unnecessary host-side duplicate.
        return source
    indices = np.load(subset_path)
    return np.ascontiguousarray(source[indices], dtype=np.float32)


def worker_main(args) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    payload = {
        "schema_version": 1,
        "status": "running",
        "dataset": args.dataset,
        "method": args.method,
        "display": DISPLAY[args.method],
        "n_samples": args.n_samples,
        "input_path": str(args.input_path),
        "subset_path": None if args.subset_path is None else str(args.subset_path),
        "seed": args.seed,
        "repeats_requested": args.repeats,
        "n_neighbors": args.n_neighbors,
        "dire_iterations": args.dire_iterations,
        "knn_audit_sample_size_requested": args.knn_audit_sample_size,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.method == "dire_auto":
        payload["configuration_origin"] = {
            "name": "production cuVS auto-index policy",
            "control_method": "dire",
            "quality_gate": (
                "Timing is not promoted independently: the fixed-query graph "
                "overlap and full local/global/topology evaluation must also "
                "be reported against the forced IVF-Flat control."
            ),
            "upstream_diagnostics_issue": (
                "https://github.com/sashakolpakov/dire-rapids/issues/13"
            ),
        }
    if args.method == "dire_ivf_flat_control":
        payload["configuration_origin"] = {
            "name": "fresh forced-IVF-Flat backend-policy control",
            "production_method": "dire_auto",
            "separation_policy": (
                "This dedicated profile does not overwrite or relabel the "
                "forced-IVF-Flat control benchmark records."
            ),
            "upstream_diagnostics_issue": (
                "https://github.com/sashakolpakov/dire-rapids/issues/13"
            ),
        }
    if args.method == "dire_spectral":
        payload["configuration_origin"] = {
            "name": "one-factor spectral-initialization sensitivity",
            "selection_independence": (
                "Only init changes from the default DiRe benchmark; all force "
                "parameters and the iteration budget are held fixed."
            ),
        }
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        X = load_worker_input(args.input_path, args.subset_path)
        if len(X) != args.n_samples:
            raise RuntimeError(
                f"worker expected {args.n_samples:,} rows but loaded {len(X):,}"
            )
        payload["n_features"] = int(X.shape[1])
        payload["host_input_bytes"] = int(X.nbytes)
        records = []
        canonical_embedding = None
        canonical_parameters = None
        last_diagnostics = None
        knn_audit = None
        for repeat in range(args.repeats):
            repeat_seed = args.seed + repeat
            cleanup_gpu()
            reducer, repeat_parameters = build_reducer(
                args.method,
                repeat_seed,
                args.n_neighbors,
                args.dire_iterations,
            )
            instrumentation = (
                instrument_dire_reducer(reducer)
                if args.method.startswith("dire")
                else None
            )
            monitor = NVMLMonitor(
                device_index=args.gpu_index,
                interval_sec=args.memory_poll_interval,
            )
            monitor.start()
            gpu_synchronize()
            started = time.perf_counter()
            try:
                embedding = to_numpy(reducer.fit_transform(X))
                gpu_synchronize()
                elapsed = time.perf_counter() - started
                memory = monitor.stop()
            except Exception:
                memory = monitor.stop()
                raise
            if len(embedding) != args.n_samples:
                raise RuntimeError(
                    f"embedding row count {len(embedding):,} != input {args.n_samples:,}"
                )
            if canonical_embedding is None:
                canonical_embedding = embedding
                canonical_parameters = repeat_parameters
            repeat_diagnostics = dire_runtime_diagnostics(
                reducer,
                instrumentation,
            )
            last_diagnostics = repeat_diagnostics
            records.append(
                {
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "method_parameters": repeat_parameters,
                    "reducer_diagnostics": repeat_diagnostics,
                    "fit_transform_wall_time_sec": elapsed,
                    "gpu_memory": memory,
                    "embedding_min": embedding.min(axis=0).tolist(),
                    "embedding_max": embedding.max(axis=0).tolist(),
                    "embedding_std": embedding.std(axis=0).tolist(),
                }
            )
            if (
                repeat == 0
                and args.method in ("dire_auto", "dire_ivf_flat_control")
                and hasattr(reducer, "_knn_indices")
            ):
                graph = np.asarray(reducer._knn_indices)
                if graph.ndim != 2 or len(graph) != args.n_samples:
                    raise RuntimeError(
                        "DiRe kNN audit graph has unexpected shape "
                        f"{graph.shape}"
                    )
                audit_size = min(args.knn_audit_sample_size, args.n_samples)
                audit_seed = args.seed + 900_000
                audit_rng = np.random.default_rng(audit_seed)
                query_indices = np.sort(
                    audit_rng.choice(
                        args.n_samples,
                        size=audit_size,
                        replace=False,
                    )
                ).astype(np.int32, copy=False)
                sampled_graph = np.ascontiguousarray(
                    graph[query_indices],
                    dtype=np.int32,
                )
                query_name = "knn_audit_query_indices.npy"
                neighbor_name = "knn_audit_neighbor_indices.npy"
                np.save(output_dir / query_name, query_indices)
                np.save(output_dir / neighbor_name, sampled_graph)
                knn_audit = {
                    "policy": (
                        "Fixed-seed uniform query rows shared by production "
                        "auto and forced IVF-Flat; neighbor identifiers refer "
                        "to the identical worker input ordering."
                    ),
                    "sample_size": int(audit_size),
                    "seed": int(audit_seed),
                    "k": int(sampled_graph.shape[1]),
                    "query_indices_file": query_name,
                    "neighbor_indices_file": neighbor_name,
                }
            if args.save_repeat_embeddings:
                repeat_embedding_name = f"embedding_repeat_{repeat:03d}.npy"
                np.save(output_dir / repeat_embedding_name, embedding)
                records[-1]["embedding_file"] = repeat_embedding_name
            del reducer, embedding
            cleanup_gpu()

        if canonical_embedding is None:
            raise RuntimeError("no embedding was produced")
        if args.save_repeat_embeddings:
            embedding_name = records[0]["embedding_file"]
        else:
            embedding_name = "embedding.npy"
            np.save(output_dir / embedding_name, canonical_embedding)
        times = np.asarray(
            [record["fit_transform_wall_time_sec"] for record in records],
            dtype=np.float64,
        )
        steady = times[1:] if len(times) > 1 else times
        peak_values = [
            record["gpu_memory"]["peak_used_bytes"]
            for record in records
            if record["gpu_memory"]["peak_used_bytes"] is not None
        ]
        incremental_values = [
            record["gpu_memory"]["peak_incremental_bytes"]
            for record in records
            if record["gpu_memory"]["peak_incremental_bytes"] is not None
        ]
        payload.update(
            {
                "status": "success",
                "method_parameters": canonical_parameters,
                "n_neighbors": (
                    canonical_parameters.get("n_neighbors", args.n_neighbors)
                ),
                "dire_iterations": (
                    canonical_parameters.get(
                        "max_iter_layout",
                        args.dire_iterations,
                    )
                ),
                "canonical_repeat": 0,
                "canonical_seed": args.seed,
                "reducer_diagnostics": last_diagnostics,
                "knn_graph_audit": knn_audit,
                "records": records,
                "timing": {
                    "cold_start_sec": float(times[0]),
                    "steady_mean_sec": float(steady.mean()),
                    "steady_std_sec": (
                        float(steady.std(ddof=1)) if len(steady) > 1 else 0.0
                    ),
                    "steady_n": int(len(steady)),
                    "all_mean_sec": float(times.mean()),
                    "all_std_sec": (
                        float(times.std(ddof=1)) if len(times) > 1 else 0.0
                    ),
                },
                "gpu_memory": {
                    "peak_used_bytes_max": max(peak_values) if peak_values else None,
                    "peak_incremental_bytes_max": (
                        max(incremental_values) if incremental_values else None
                    ),
                },
                "embedding_file": embedding_name,
                "finished_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
            }
        )
        write_json(result_path, payload)
        print(
            f"SUCCESS {args.dataset} {args.method} n={args.n_samples:,}: "
            f"cold={times[0]:.3f}s steady={steady.mean():.3f}s",
            flush=True,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "finished_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
            }
        )
        write_json(result_path, payload)
        traceback.print_exc()
        return 1


def dataset_paths(prepared_root: Path, dataset: str) -> tuple[Path, Path, dict]:
    config = DATASET_FILES[dataset]
    input_path = prepared_root / config["input"]
    labels_path = prepared_root / config["labels"]
    manifest_path = prepared_root / config["manifest"]
    missing = [
        path for path in (input_path, labels_path, manifest_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing prepared dataset files; run prepare_large_datasets.py first: "
            + ", ".join(str(path) for path in missing)
        )
    manifest = read_json(manifest_path)
    return input_path, labels_path, manifest


def build_nested_order(
    prepared_root: Path,
    dataset: str,
    n_samples: int,
    seed: int,
) -> Path:
    subset_root = prepared_root / "subsets"
    subset_root.mkdir(parents=True, exist_ok=True)
    order_path = subset_root / f"{dataset}_uniform_order_seed{seed}.npy"
    if order_path.exists():
        order = np.load(order_path, mmap_mode="r")
        if len(order) == n_samples:
            return order_path
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples).astype(
        np.int32 if n_samples <= np.iinfo(np.int32).max else np.int64
    )
    np.save(order_path, order)
    return order_path


def materialize_subset(
    prepared_root: Path,
    dataset: str,
    order_path: Path,
    subset_size: int,
    full_size: int,
    seed: int,
) -> Path | None:
    if subset_size == full_size:
        return None
    subset_root = prepared_root / "subsets"
    path = subset_root / f"{dataset}_n{subset_size:09d}_seed{seed}.npy"
    if path.exists():
        indices = np.load(path, mmap_mode="r")
        if len(indices) == subset_size:
            return path
    order = np.load(order_path, mmap_mode="r")
    # Sorting improves memmap read locality.  Membership, not order, determines
    # the nested random subset, and all methods receive the same sorted rows.
    indices = np.sort(np.asarray(order[:subset_size]))
    np.save(path, indices)
    return path


def successful_existing(
    result_path: Path,
    expected_n: int,
    require_repeat_embeddings: bool,
    require_knn_audit: bool,
) -> bool:
    if not result_path.exists():
        return False
    try:
        result = read_json(result_path)
    except (OSError, json.JSONDecodeError):
        return False
    embedding_path = result_path.parent / result.get("embedding_file", "")
    basic_success = (
        result.get("status") == "success"
        and result.get("n_samples") == expected_n
        and embedding_path.is_file()
    )
    if not basic_success:
        return False
    if require_knn_audit:
        audit = result.get("knn_graph_audit")
        if not isinstance(audit, dict):
            return False
        for key in ("query_indices_file", "neighbor_indices_file"):
            filename = audit.get(key)
            if (
                not isinstance(filename, str)
                or not (result_path.parent / filename).is_file()
            ):
                return False
        records = result.get("records", [])
        if not records or not all(
            isinstance(record.get("reducer_diagnostics"), dict)
            and isinstance(
                record["reducer_diagnostics"].get("stage_timings_sec"),
                dict,
            )
            for record in records
        ):
            return False
    if not require_repeat_embeddings:
        return True
    records = result.get("records", [])
    return bool(records) and all(
        isinstance(record.get("embedding_file"), str)
        and (result_path.parent / record["embedding_file"]).is_file()
        for record in records
    )


def validate_sizes(sizes: list[int], full_size: int, dataset: str) -> list[int]:
    clean = sorted(set(sizes))
    if not clean:
        raise ValueError(f"no sizes requested for {dataset}")
    invalid = [size for size in clean if size < 1 or size > full_size]
    if invalid:
        raise ValueError(
            f"invalid {dataset} sizes {invalid}; full dataset has {full_size:,} rows"
        )
    return clean


def run_parent(args) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    snapshot = environment_snapshot()
    resolved_commit = resolve_dire_commit(snapshot)
    if resolved_commit != EXPECTED_DIRE_COMMIT:
        raise RuntimeError(
            "DiRe-RAPIDS is not installed from the pinned benchmark commit: "
            f"expected {EXPECTED_DIRE_COMMIT}, resolved {resolved_commit!r}"
        )
    write_json(args.output_root / "environment.json", snapshot)

    requested_sizes = {
        "tenx": args.tenx_sizes,
        "arxiv": args.arxiv_sizes,
    }
    run_records = []
    any_failure = False
    for dataset in args.datasets:
        input_path, _labels_path, data_manifest = dataset_paths(
            args.prepared_root,
            dataset,
        )
        input_array = np.load(input_path, mmap_mode="r")
        full_size = len(input_array)
        if full_size != int(data_manifest["n_samples"]):
            raise RuntimeError(
                f"{dataset} manifest says {data_manifest['n_samples']:,} rows, "
                f"array has {full_size:,}"
            )
        sizes = validate_sizes(requested_sizes[dataset], full_size, dataset)
        order_path = build_nested_order(
            args.prepared_root,
            dataset,
            full_size,
            args.seed,
        )
        for size in sizes:
            subset_path = materialize_subset(
                args.prepared_root,
                dataset,
                order_path,
                size,
                full_size,
                args.seed,
            )
            for method in args.methods:
                output_dir = (
                    args.output_root
                    / dataset
                    / method
                    / f"n_{size:09d}"
                )
                result_path = output_dir / "result.json"
                log_path = output_dir / "worker.log"
                if args.resume and successful_existing(
                    result_path,
                    size,
                    args.save_repeat_embeddings,
                    (
                        method in ("dire_auto", "dire_ivf_flat_control")
                        and size == full_size
                    ),
                ):
                    print(
                        f"SKIP existing {dataset} {method} n={size:,}",
                        flush=True,
                    )
                    run_records.append(read_json(result_path))
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--dataset",
                    dataset,
                    "--method",
                    method,
                    "--n-samples",
                    str(size),
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    str(args.seed),
                    "--repeats",
                    str(args.repeats),
                    "--n-neighbors",
                    str(args.n_neighbors),
                    "--dire-iterations",
                    str(args.dire_iterations),
                    "--gpu-index",
                    str(args.gpu_index),
                    "--memory-poll-interval",
                    str(args.memory_poll_interval),
                    "--knn-audit-sample-size",
                    str(args.knn_audit_sample_size),
                ]
                if args.save_repeat_embeddings:
                    command.append("--save-repeat-embeddings")
                if subset_path is not None:
                    command.extend(["--subset-path", str(subset_path)])
                print(
                    f"RUN {dataset} {method} n={size:,} "
                    f"({args.repeats} fit(s))",
                    flush=True,
                )
                started = time.time()
                try:
                    with log_path.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=args.timeout_minutes * 60,
                            check=False,
                        )
                    return_code = completed.returncode
                except subprocess.TimeoutExpired:
                    return_code = 124
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(
                            f"\nTIMEOUT after {args.timeout_minutes} minutes\n"
                        )
                if result_path.exists():
                    result = read_json(result_path)
                else:
                    result = {
                        "schema_version": 1,
                        "status": "process_failure",
                        "dataset": dataset,
                        "method": method,
                        "display": DISPLAY[method],
                        "n_samples": size,
                        "return_code": return_code,
                        "worker_log": str(log_path),
                        "elapsed_before_failure_sec": time.time() - started,
                    }
                    write_json(result_path, result)
                result["worker_return_code"] = return_code
                result["worker_log"] = str(log_path.relative_to(args.output_root))
                write_json(result_path, result)
                run_records.append(result)
                if result.get("status") != "success":
                    any_failure = True
                    print(
                        f"FAILED {dataset} {method} n={size:,}; see {log_path}",
                        flush=True,
                    )

    manifest = {
        "schema_version": 1,
        "purpose": (
            "Same-H100 scaling benchmark requested in response to Scientific "
            "Reports Revision 3."
        ),
        "configuration": {
            "datasets": args.datasets,
            "methods": args.methods,
            "tenx_sizes": args.tenx_sizes,
            "arxiv_sizes": args.arxiv_sizes,
            "seed": args.seed,
            "repeats": args.repeats,
            "n_neighbors": args.n_neighbors,
            "dire_iterations": args.dire_iterations,
            "timeout_minutes": args.timeout_minutes,
            "subset_policy": (
                "Nested fixed-seed uniform random subsets, sorted only for "
                "read locality; identical rows and order for every method."
            ),
            "timing_policy": (
                "fit_transform wall clock with explicit GPU synchronization; "
                "first fit reported as cold start and later fits as steady state."
            ),
            "memory_policy": (
                "NVML total device memory sampled at fixed intervals, covering "
                "PyTorch, CuPy, cuVS, and cuML allocations."
            ),
            "save_repeat_embeddings": args.save_repeat_embeddings,
            "knn_audit_sample_size": args.knn_audit_sample_size,
        },
        "environment_file": "environment.json",
        "runs": run_records,
        "failures_present": any_failure,
    }
    write_json(args.output_root / "manifest.json", manifest)
    # A reducer failure is an experimental result (for example, an OOM at
    # scale), so finish the sweep and leave strict pass/fail to validation.
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_FILES),
        default=list(DATASET_FILES),
    )
    result.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
    )
    result.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/revision3/prepared"),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/revision3/large_results"),
    )
    result.add_argument(
        "--tenx-sizes",
        type=int,
        nargs="+",
        default=list(DATASET_FILES["tenx"]["default_sizes"]),
    )
    result.add_argument(
        "--arxiv-sizes",
        type=int,
        nargs="+",
        default=list(DATASET_FILES["arxiv"]["default_sizes"]),
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--n-neighbors", type=int, default=16)
    result.add_argument("--dire-iterations", type=int, default=128)
    result.add_argument("--gpu-index", type=int, default=0)
    result.add_argument("--memory-poll-interval", type=float, default=0.05)
    result.add_argument("--knn-audit-sample-size", type=int, default=20_000)
    result.add_argument("--timeout-minutes", type=int, default=120)
    result.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument(
        "--save-repeat-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "retain every repeated full layout, enabling separate layout-seed "
            "quality sensitivity analysis"
        ),
    )

    # Worker-only arguments.
    result.add_argument("--dataset", choices=tuple(DATASET_FILES))
    result.add_argument("--method", choices=METHODS)
    result.add_argument("--n-samples", type=int)
    result.add_argument("--input-path", type=Path)
    result.add_argument("--subset-path", type=Path)
    result.add_argument("--output-dir", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if args.dire_iterations < 1:
        raise ValueError("dire_iterations must be positive")
    if args.knn_audit_sample_size < 1:
        raise ValueError("knn_audit_sample_size must be positive")
    if not math.isfinite(args.memory_poll_interval) or args.memory_poll_interval <= 0:
        raise ValueError("memory_poll_interval must be positive and finite")
    if args.worker:
        required = {
            "dataset": args.dataset,
            "method": args.method,
            "n_samples": args.n_samples,
            "input_path": args.input_path,
            "output_dir": args.output_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"worker is missing required arguments: {missing}")
        raise SystemExit(worker_main(args))
    raise SystemExit(run_parent(args))


if __name__ == "__main__":
    main()
