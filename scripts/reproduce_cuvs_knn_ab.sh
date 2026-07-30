#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CUVS_KNN_AB_IMAGE:-homological-stability-repro:cuvs-knn-ab}"
AB_ROOT="${CUVS_KNN_AB_ROOT:-data/cuvs-knn-ab}"
MIN_GPU_MEMORY_MIB="${CUVS_KNN_AB_MIN_GPU_MEMORY_MIB:-70000}"
REMOTE_DOCKER="${REMOTE_DOCKER:-auto}"

if [[ "$AB_ROOT" = /* || "$AB_ROOT" == *".."* ]]; then
  echo "CUVS_KNN_AB_ROOT must be repository-relative without '..'." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required." >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required." >&2
  exit 1
fi

if [[ "$REMOTE_DOCKER" == "auto" ]]; then
  if docker version >/dev/null 2>&1; then
    REMOTE_DOCKER="docker"
  elif sudo -n docker version >/dev/null 2>&1; then
    REMOTE_DOCKER="sudo docker"
  else
    echo "Docker is unavailable to this user, including passwordless sudo." >&2
    exit 1
  fi
else
  $REMOTE_DOCKER version >/dev/null
fi

GPU_MEMORY_MIB="$(
  nvidia-smi \
    --query-gpu=memory.total \
    --format=csv,noheader,nounits |
    head -n 1 |
    tr -d ' '
)"
if [[ ! "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "Could not parse GPU memory: $GPU_MEMORY_MIB" >&2
  exit 1
fi
if (( GPU_MEMORY_MIB < MIN_GPU_MEMORY_MIB )); then
  echo "At least $MIN_GPU_MEMORY_MIB MiB is required; found $GPU_MEMORY_MIB." >&2
  exit 1
fi

mkdir -p \
  "$AB_ROOT/cache/home" \
  "$AB_ROOT/cache/huggingface" \
  "$AB_ROOT/cache/xdg" \
  "$AB_ROOT/cache/numba" \
  "$AB_ROOT/cache/matplotlib"

STATUS_FILE="$AB_ROOT/pipeline.status"
write_status() {
  local state="$1"
  local code="$2"
  local temporary="$STATUS_FILE.tmp"
  {
    printf 'state=%s\n' "$state"
    printf 'exit_code=%s\n' "$code"
    date -u +"updated_utc=%Y-%m-%dT%H:%M:%SZ"
  } > "$temporary"
  mv "$temporary" "$STATUS_FILE"
}
on_exit() {
  local code=$?
  if (( code == 0 )); then
    write_status success 0
  else
    write_status failed "$code"
  fi
}
trap on_exit EXIT
write_status running -1

{
  date -u +"timestamp_utc=%Y-%m-%dT%H:%M:%SZ"
  nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader
  git rev-parse HEAD
  git status --short
  $REMOTE_DOCKER version
} > "$AB_ROOT/host-preflight.txt"

echo "[cuvs-knn-ab] building pinned image $IMAGE"
$REMOTE_DOCKER build -f docker/Dockerfile -t "$IMAGE" .

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
$REMOTE_DOCKER run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --user "$HOST_UID:$HOST_GID" \
  -e HOME="/work/$AB_ROOT/cache/home" \
  -e HF_HOME="/work/$AB_ROOT/cache/huggingface" \
  -e XDG_CACHE_HOME="/work/$AB_ROOT/cache/xdg" \
  -e NUMBA_CACHE_DIR="/work/$AB_ROOT/cache/numba" \
  -e MPLCONFIGDIR="/work/$AB_ROOT/cache/matplotlib" \
  -e CUVS_KNN_AB_ROOT="$AB_ROOT" \
  -e CUVS_KNN_AB_REPEATS="${CUVS_KNN_AB_REPEATS:-3}" \
  -e CUVS_KNN_AB_DIRE_ITERATIONS="${CUVS_KNN_AB_DIRE_ITERATIONS:-128}" \
  -e CUVS_KNN_AB_TIMEOUT_MINUTES="${CUVS_KNN_AB_TIMEOUT_MINUTES:-180}" \
  -e CUVS_KNN_AB_TOPOLOGY_SUBSET_SIZE="${CUVS_KNN_AB_TOPOLOGY_SUBSET_SIZE:-4000}" \
  -e CUVS_KNN_AB_TOPOLOGY_SUBSET_COUNT="${CUVS_KNN_AB_TOPOLOGY_SUBSET_COUNT:-10}" \
  -e CUVS_KNN_AB_TOPOLOGY_WORKERS="${CUVS_KNN_AB_TOPOLOGY_WORKERS:-24}" \
  -e CUVS_KNN_AB_GRAPH_AUDIT_SIZE="${CUVS_KNN_AB_GRAPH_AUDIT_SIZE:-20000}" \
  -e CUVS_KNN_AB_FORCE_STAGES="${CUVS_KNN_AB_FORCE_STAGES:-0}" \
  -v "$PWD:/work" \
  -w /work \
  "$IMAGE" \
  scripts/run_cuvs_knn_ab_inside_container.sh

write_status success 0
RUN_ID="${CUVS_KNN_AB_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
ARCHIVE="$AB_ROOT/cuvs-knn-ab-$RUN_ID.tar.gz"
tar \
  --exclude="$AB_ROOT/cache" \
  --exclude="$AB_ROOT/*.tar.gz" \
  -czf "$ARCHIVE" \
  "$AB_ROOT/host-preflight.txt" \
  "$AB_ROOT/pipeline.status" \
  "$AB_ROOT/results" \
  "$AB_ROOT/evaluation" \
  "$AB_ROOT/pilot_summary" \
  "$AB_ROOT/summary" \
  "$AB_ROOT/stages"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
printf '%s\n' "${ARCHIVE##*/}" > "$AB_ROOT/LATEST"

echo "[cuvs-knn-ab] result archive: $ARCHIVE"
