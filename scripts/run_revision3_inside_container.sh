#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
REVISION3_ROOT="${REVISION3_ROOT:-data/revision3}"
PIPELINE_VERSION="${REVISION3_PIPELINE_VERSION:-v11}"
FORCE_STAGES="${FORCE_STAGES:-0}"
STAGE_ROOT="$REVISION3_ROOT/stages/$PIPELINE_VERSION"

LARGE_REPEATS="${LARGE_REPEATS:-3}"
DIRE_ITERATIONS="${DIRE_ITERATIONS:-128}"
LARGE_TIMEOUT_MINUTES="${LARGE_TIMEOUT_MINUTES:-180}"
EVALUATION_TOPOLOGY_SUBSET_SIZE="${EVALUATION_TOPOLOGY_SUBSET_SIZE:-4000}"
EVALUATION_TOPOLOGY_SUBSET_COUNT="${EVALUATION_TOPOLOGY_SUBSET_COUNT:-10}"
EVALUATION_TOPOLOGY_WORKERS="${EVALUATION_TOPOLOGY_WORKERS:-24}"
EVALUATION_KNN_SAMPLE_SIZE="${EVALUATION_KNN_SAMPLE_SIZE:-20000}"
EVALUATION_CLASSIFIER_SAMPLE_SIZE="${EVALUATION_CLASSIFIER_SAMPLE_SIZE:-100000}"
KNN_AUDIT_SAMPLE_SIZE="${KNN_AUDIT_SAMPLE_SIZE:-20000}"

mkdir -p "$STAGE_ROOT"

run_stage() {
  local stage="$1"
  shift
  local marker="$STAGE_ROOT/$stage.done"
  if [[ "$FORCE_STAGES" != "1" && -s "$marker" ]]; then
    echo "[revision3] reusing completed stage: $stage"
    return
  fi
  echo "[revision3] starting stage: $stage"
  "$@"
  {
    date -u +"completed_utc=%Y-%m-%dT%H:%M:%SZ"
    printf 'pipeline_version=%s\n' "$PIPELINE_VERSION"
  } > "$marker"
  echo "[revision3] completed stage: $stage"
}

run_stage prepare \
  "$PYTHON" scripts/prepare_large_datasets.py \
    --raw-root "$REVISION3_ROOT/raw" \
    --prepared-root "$REVISION3_ROOT/prepared"

run_stage large_scaling \
  "$PYTHON" scripts/large_scale_benchmarks.py \
    --prepared-root "$REVISION3_ROOT/prepared" \
    --output-root "$REVISION3_ROOT/large_results" \
    --repeats "$LARGE_REPEATS" \
    --dire-iterations "$DIRE_ITERATIONS" \
    --knn-audit-sample-size "$KNN_AUDIT_SAMPLE_SIZE" \
    --timeout-minutes "$LARGE_TIMEOUT_MINUTES"

run_stage backend_policy_profile \
  "$PYTHON" scripts/large_scale_benchmarks.py \
    --prepared-root "$REVISION3_ROOT/prepared" \
    --output-root "$REVISION3_ROOT/backend_policy_results" \
    --datasets tenx arxiv \
    --methods dire_auto dire_ivf_flat_control \
    --tenx-sizes 1306127 \
    --arxiv-sizes 723457 \
    --repeats "$LARGE_REPEATS" \
    --dire-iterations "$DIRE_ITERATIONS" \
    --knn-audit-sample-size "$KNN_AUDIT_SAMPLE_SIZE" \
    --timeout-minutes "$LARGE_TIMEOUT_MINUTES"

run_stage topology_sensitivity \
  "$PYTHON" scripts/large_scale_benchmarks.py \
    --prepared-root "$REVISION3_ROOT/prepared" \
    --output-root "$REVISION3_ROOT/topology_sensitivity_results" \
    --datasets tenx arxiv \
    --methods dire_auto dire dire_spectral cuml_umap cuml_tsne \
    --tenx-sizes 100000 1306127 \
    --arxiv-sizes 100000 723457 \
    --repeats "$LARGE_REPEATS" \
    --save-repeat-embeddings \
    --knn-audit-sample-size "$KNN_AUDIT_SAMPLE_SIZE" \
    --timeout-minutes "$LARGE_TIMEOUT_MINUTES"

run_stage large_evaluation \
  "$PYTHON" scripts/evaluate_large_embeddings.py \
    --prepared-root "$REVISION3_ROOT/prepared" \
    --large-results-root "$REVISION3_ROOT/large_results" \
    --backend-policy-results-root "$REVISION3_ROOT/backend_policy_results" \
    --topology-sensitivity-results-root \
      "$REVISION3_ROOT/topology_sensitivity_results" \
    --output-root "$REVISION3_ROOT/evaluation" \
    --knn-sample-size "$EVALUATION_KNN_SAMPLE_SIZE" \
    --classifier-sample-size "$EVALUATION_CLASSIFIER_SAMPLE_SIZE" \
    --topology-subset-size "$EVALUATION_TOPOLOGY_SUBSET_SIZE" \
    --topology-subset-count "$EVALUATION_TOPOLOGY_SUBSET_COUNT" \
    --topology-workers "$EVALUATION_TOPOLOGY_WORKERS"

run_stage small_reference_suite \
  env \
    PYTHON="$PYTHON" \
    OUTPUT_ROOT="$REVISION3_ROOT/small_results" \
    JSON_LOG_ROOT="$REVISION3_ROOT/small_json_logs" \
    EMBEDDING_PNG_ROOT="$REVISION3_ROOT/small_embedding_pngs" \
    METHODS="dire cuml_tsne cuml_umap opentsne umap" \
    REPEATS="${SMALL_GPU_REPEATS:-20}" \
    CPU_REPEATS="${SMALL_CPU_REPEATS:-10}" \
    CPU_JOBS="${SMALL_CPU_JOBS:-24}" \
    CPU_MAX_POINTS="${SMALL_CPU_MAX_POINTS:-10000}" \
    TOPOLOGY_REPEATS="${SMALL_TOPOLOGY_REPEATS:-20}" \
    TOPOLOGY_SAMPLE_SIZE="${SMALL_TOPOLOGY_SAMPLE_SIZE:-1000}" \
    DIRE_ITERATIONS="$DIRE_ITERATIONS" \
    scripts/run_benchmarks.sh "$REVISION3_ROOT/small_results"

run_stage initialization_ablation \
  env \
    PYTHON="$PYTHON" \
    DIRE_ITERATIONS="$DIRE_ITERATIONS" \
    ABLATION_REPEATS="${ABLATION_REPEATS:-3}" \
    ABLATION_TOPOLOGY_REPEATS="${ABLATION_TOPOLOGY_REPEATS:-3}" \
    scripts/run_initialization_ablation.sh \
      "$REVISION3_ROOT/ablation_results"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
"$PYTHON" scripts/package_revision3_results.py \
  --revision3-root "$REVISION3_ROOT" \
  --small-json-root "$REVISION3_ROOT/small_json_logs" \
  --small-embedding-root "$REVISION3_ROOT/small_embedding_pngs" \
  --ablation-root "$REVISION3_ROOT/ablation_results" \
  --bundles-root "$REVISION3_ROOT/bundles" \
  --run-id "$RUN_ID"

"$PYTHON" scripts/verify_revision3_bundle.py \
  --bundle-root \
    "$REVISION3_ROOT/bundle_staging/revision3-results-$RUN_ID" \
  --require-current-contract

echo "[revision3] all stages complete"
