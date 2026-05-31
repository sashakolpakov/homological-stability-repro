#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-data/fresh_benchmark_results}"
MAX_POINTS="${MAX_POINTS:-10000}"
FULL_DATASETS="${FULL_DATASETS:-0}"
UMAP_MAX_POINTS="${UMAP_MAX_POINTS:-10000}"
METRIC_SUBSAMPLE="${METRIC_SUBSAMPLE:-0.1}"
TOPOLOGY_SAMPLE_FRACTION="${TOPOLOGY_SAMPLE_FRACTION:-0.05}"
TOPOLOGY_STEPS="${TOPOLOGY_STEPS:-100}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-10}"
TOPOLOGY_REPEATS="${TOPOLOGY_REPEATS:-1}"
JSON_LOG_ROOT="${JSON_LOG_ROOT:-data/fresh_json_logs}"
EMBEDDING_PNG_ROOT="${EMBEDDING_PNG_ROOT:-data/fresh_embedding_pngs}"
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

"${PYTHON:-python3}" scripts/article_benchmarks.py \
  --output "$OUTPUT_ROOT/article_benchmark_results" \
  --datasets blobs disk moons \
  --max-points "$MAX_POINTS" \
  "$FULL_DATASETS_ARG" \
  --umap-max-points "$UMAP_MAX_POINTS" \
  --metric-subsample "$METRIC_SUBSAMPLE" \
  --topology-sample-fraction "$TOPOLOGY_SAMPLE_FRACTION" \
  --topology-steps "$TOPOLOGY_STEPS" \
  --seed "$SEED" \
  --repeats "$REPEATS" \
  --topology-repeats "$TOPOLOGY_REPEATS" \
  "$TOPOLOGY_ARG"

"${PYTHON:-python3}" scripts/article_benchmarks.py \
  --output "$OUTPUT_ROOT/article_benchmark_results_mnist" \
  --datasets mnist \
  --max-points "$MAX_POINTS" \
  "$FULL_DATASETS_ARG" \
  --umap-max-points "$UMAP_MAX_POINTS" \
  --metric-subsample "$METRIC_SUBSAMPLE" \
  --topology-sample-fraction "$TOPOLOGY_SAMPLE_FRACTION" \
  --topology-steps "$TOPOLOGY_STEPS" \
  --seed "$SEED" \
  --repeats "$REPEATS" \
  --topology-repeats "$TOPOLOGY_REPEATS" \
  "$TOPOLOGY_ARG"

"${PYTHON:-python3}" scripts/article_benchmarks.py \
  --output "$OUTPUT_ROOT/article_benchmark_results_levine" \
  --datasets levine13 levine32 \
  --max-points "$MAX_POINTS" \
  "$FULL_DATASETS_ARG" \
  --umap-max-points "$UMAP_MAX_POINTS" \
  --metric-subsample "$METRIC_SUBSAMPLE" \
  --topology-sample-fraction "$TOPOLOGY_SAMPLE_FRACTION" \
  --topology-steps "$TOPOLOGY_STEPS" \
  --seed "$SEED" \
  --repeats "$REPEATS" \
  --topology-repeats "$TOPOLOGY_REPEATS" \
  "$TOPOLOGY_ARG"

"${PYTHON:-python3}" scripts/package_json_logs.py \
  --raw-root "$OUTPUT_ROOT" \
  --output "$JSON_LOG_ROOT"

"${PYTHON:-python3}" scripts/archive_embedding_pngs.py \
  --raw-root "$OUTPUT_ROOT" \
  --output "$EMBEDDING_PNG_ROOT"
