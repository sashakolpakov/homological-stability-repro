#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-data/revision3/ablation_results}"
PYTHON="${PYTHON:-python3}"
SEED="${SEED:-42}"
REPEATS="${ABLATION_REPEATS:-3}"
TOPOLOGY_REPEATS="${ABLATION_TOPOLOGY_REPEATS:-3}"
MAX_POINTS="${ABLATION_MAX_POINTS:-10000}"
METRIC_SUBSAMPLE="${ABLATION_METRIC_SUBSAMPLE:-0.05}"
TOPOLOGY_SAMPLE_SIZE="${ABLATION_TOPOLOGY_SAMPLE_SIZE:-1000}"
TOPOLOGY_STEPS="${ABLATION_TOPOLOGY_STEPS:-100}"
DIRE_ITERATIONS="${DIRE_ITERATIONS:-128}"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON" scripts/article_benchmarks.py \
  --output "$OUTPUT_ROOT" \
  --datasets blobs disk moons mnist levine13 levine32 \
  --methods dire_pca_init dire dire_spectral_init dire_spectral \
  --max-points "$MAX_POINTS" \
  --no-full-datasets \
  --dire-iterations "$DIRE_ITERATIONS" \
  --metric-subsample "$METRIC_SUBSAMPLE" \
  --topology-sample-size "$TOPOLOGY_SAMPLE_SIZE" \
  --topology-steps "$TOPOLOGY_STEPS" \
  --seed "$SEED" \
  --repeats "$REPEATS" \
  --topology-repeats "$TOPOLOGY_REPEATS" \
  --topology
