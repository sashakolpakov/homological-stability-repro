#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
AB_ROOT="${CUVS_KNN_AB_ROOT:-data/cuvs-knn-ab}"
PREPARED_ROOT="${CUVS_KNN_AB_PREPARED_ROOT:-data/revision3/prepared}"
PIPELINE_VERSION="${CUVS_KNN_AB_PIPELINE_VERSION:-v1}"
STAGE_ROOT="$AB_ROOT/stages/$PIPELINE_VERSION"
REPEATS="${CUVS_KNN_AB_REPEATS:-3}"
ITERATIONS="${CUVS_KNN_AB_DIRE_ITERATIONS:-128}"
TIMEOUT_MINUTES="${CUVS_KNN_AB_TIMEOUT_MINUTES:-180}"
TOPOLOGY_SUBSET_SIZE="${CUVS_KNN_AB_TOPOLOGY_SUBSET_SIZE:-4000}"
TOPOLOGY_SUBSET_COUNT="${CUVS_KNN_AB_TOPOLOGY_SUBSET_COUNT:-10}"
TOPOLOGY_WORKERS="${CUVS_KNN_AB_TOPOLOGY_WORKERS:-24}"
KNN_AUDIT_SAMPLE_SIZE="${CUVS_KNN_AB_GRAPH_AUDIT_SIZE:-20000}"

mkdir -p "$STAGE_ROOT"

run_stage() {
  local stage="$1"
  shift
  local marker="$STAGE_ROOT/$stage.done"
  if [[ "${CUVS_KNN_AB_FORCE_STAGES:-0}" != "1" && -s "$marker" ]]; then
    echo "[cuvs-knn-ab] reusing completed stage: $stage"
    return
  fi
  echo "[cuvs-knn-ab] starting stage: $stage"
  "$@"
  {
    date -u +"completed_utc=%Y-%m-%dT%H:%M:%SZ"
    printf 'pipeline_version=%s\n' "$PIPELINE_VERSION"
  } > "$marker"
  echo "[cuvs-knn-ab] completed stage: $stage"
}

run_stage prepare \
  "$PYTHON" scripts/prepare_large_datasets.py \
    --raw-root data/revision3/raw \
    --prepared-root "$PREPARED_ROOT"

run_stage pilot \
  "$PYTHON" scripts/large_scale_benchmarks.py \
    --prepared-root "$PREPARED_ROOT" \
    --output-root "$AB_ROOT/results" \
    --datasets tenx arxiv \
    --methods dire_index_search dire_all_neighbors \
    --tenx-sizes 100000 \
    --arxiv-sizes 100000 \
    --repeats "$REPEATS" \
    --dire-iterations "$ITERATIONS" \
    --knn-audit-sample-size "$KNN_AUDIT_SAMPLE_SIZE" \
    --timeout-minutes "$TIMEOUT_MINUTES"

run_stage validate_pilot \
  "$PYTHON" scripts/summarize_cuvs_knn_ab.py \
    --results-root "$AB_ROOT/results" \
    --evaluation-root "$AB_ROOT/evaluation" \
    --output-root "$AB_ROOT/pilot_summary" \
    --profiles tenx:100000 arxiv:100000

run_stage full_scale \
  "$PYTHON" scripts/large_scale_benchmarks.py \
    --prepared-root "$PREPARED_ROOT" \
    --output-root "$AB_ROOT/results" \
    --datasets tenx arxiv \
    --methods dire_index_search dire_all_neighbors \
    --tenx-sizes 1306127 \
    --arxiv-sizes 723457 \
    --repeats "$REPEATS" \
    --dire-iterations "$ITERATIONS" \
    --knn-audit-sample-size "$KNN_AUDIT_SAMPLE_SIZE" \
    --timeout-minutes "$TIMEOUT_MINUTES" \
    --save-repeat-embeddings

run_stage full_quality \
  "$PYTHON" scripts/evaluate_large_embeddings.py \
    --prepared-root "$PREPARED_ROOT" \
    --large-results-root "$AB_ROOT/not-run/large-results" \
    --backend-policy-results-root "$AB_ROOT/not-run/backend-policy-results" \
    --topology-sensitivity-results-root \
      "$AB_ROOT/not-run/topology-sensitivity-results" \
    --cuvs-knn-ab-results-root "$AB_ROOT/results" \
    --output-root "$AB_ROOT/evaluation" \
    --knn-sample-size "$KNN_AUDIT_SAMPLE_SIZE" \
    --classifier-sample-size 100000 \
    --topology-subset-size "$TOPOLOGY_SUBSET_SIZE" \
    --topology-subset-count "$TOPOLOGY_SUBSET_COUNT" \
    --topology-workers "$TOPOLOGY_WORKERS"

run_stage validate_full \
  "$PYTHON" scripts/summarize_cuvs_knn_ab.py \
    --results-root "$AB_ROOT/results" \
    --evaluation-root "$AB_ROOT/evaluation" \
    --output-root "$AB_ROOT/summary" \
    --profiles \
      tenx:100000 arxiv:100000 tenx:1306127 arxiv:723457 \
    --require-evaluation

echo "[cuvs-knn-ab] all experiment stages complete"
