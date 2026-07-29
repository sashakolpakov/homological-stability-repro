#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-data/fresh_benchmark_results}"
MAX_POINTS="${MAX_POINTS:-10000}"
FULL_DATASETS="${FULL_DATASETS:-0}"
CPU_MAX_POINTS="${CPU_MAX_POINTS:-${UMAP_MAX_POINTS:-10000}}"
METRIC_SUBSAMPLE="${METRIC_SUBSAMPLE:-0.1}"
TOPOLOGY_SAMPLE_SIZE="${TOPOLOGY_SAMPLE_SIZE:-1000}"
TOPOLOGY_STEPS="${TOPOLOGY_STEPS:-100}"
TOPOLOGY_SUBSET_SEED="${TOPOLOGY_SUBSET_SEED:-1042}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-10}"
CPU_REPEATS="${CPU_REPEATS:-3}"
CPU_JOBS="${CPU_JOBS:-1}"
TOPOLOGY_REPEATS="${TOPOLOGY_REPEATS:-1}"
DIRE_ITERATIONS="${DIRE_ITERATIONS:-128}"
METHODS="${METHODS:-dire cuml_tsne cuml_umap opentsne umap}"
JSON_LOG_ROOT="${JSON_LOG_ROOT:-data/fresh_json_logs}"
EMBEDDING_PNG_ROOT="${EMBEDDING_PNG_ROOT:-data/fresh_embedding_pngs}"
read -r -a METHOD_ARGS <<< "$METHODS"
if [[ "${TOPOLOGY:-1}" == "1" ]]; then
  TOPOLOGY_ARG="--topology"
else
  TOPOLOGY_ARG="--no-topology"
fi
if [[ "$FULL_DATASETS" == "1" ]]; then
  FULL_DATASETS_ARG="--full-datasets"
else
  FULL_DATASETS_ARG="--no-full-datasets"
fi

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --max-points "$MAX_POINTS"
  "$FULL_DATASETS_ARG"
  --cpu-max-points "$CPU_MAX_POINTS"
  --methods "${METHOD_ARGS[@]}"
  --dire-iterations "$DIRE_ITERATIONS"
  --cpu-jobs "$CPU_JOBS"
  --metric-subsample "$METRIC_SUBSAMPLE"
  --topology-sample-size "$TOPOLOGY_SAMPLE_SIZE"
  --topology-subset-seed "$TOPOLOGY_SUBSET_SEED"
  --topology-steps "$TOPOLOGY_STEPS"
  --seed "$SEED"
  --repeats "$REPEATS"
  --cpu-repeats "$CPU_REPEATS"
  --topology-repeats "$TOPOLOGY_REPEATS"
  "$TOPOLOGY_ARG"
)

"${PYTHON:-python3}" scripts/article_benchmarks.py \
  --output "$OUTPUT_ROOT/article_benchmark_results" \
  --datasets blobs disk moons \
  "${COMMON_ARGS[@]}"

"${PYTHON:-python3}" scripts/article_benchmarks.py \
  --output "$OUTPUT_ROOT/article_benchmark_results_mnist" \
  --datasets mnist \
  "${COMMON_ARGS[@]}"

"${PYTHON:-python3}" scripts/article_benchmarks.py \
  --output "$OUTPUT_ROOT/article_benchmark_results_levine" \
  --datasets levine13 levine32 \
  "${COMMON_ARGS[@]}"

"${PYTHON:-python3}" scripts/package_json_logs.py \
  --raw-root "$OUTPUT_ROOT" \
  --output "$JSON_LOG_ROOT"

"${PYTHON:-python3}" scripts/archive_embedding_pngs.py \
  --raw-root "$OUTPUT_ROOT" \
  --output "$EMBEDDING_PNG_ROOT"
