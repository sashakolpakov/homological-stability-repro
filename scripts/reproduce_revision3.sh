#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-homological-stability-repro:revision3}"
REVISION3_ROOT="${REVISION3_ROOT:-data/revision3}"
REMOTE_DOCKER="${REMOTE_DOCKER:-auto}"
DOCKER_BUILD_FLAGS="${DOCKER_BUILD_FLAGS:-}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-70000}"
PIPELINE_VERSION="${REVISION3_PIPELINE_VERSION:-v11}"

if [[ "$REVISION3_ROOT" = /* || "$REVISION3_ROOT" == *".."* ]]; then
  echo "REVISION3_ROOT must be a repository-relative path without '..'." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the Revision 3 GPU rerun." >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for the Revision 3 GPU rerun." >&2
  exit 1
fi

if [[ "$REMOTE_DOCKER" == "auto" ]]; then
  if docker version >/dev/null 2>&1; then
    REMOTE_DOCKER="docker"
  elif sudo -n docker version >/dev/null 2>&1; then
    REMOTE_DOCKER="sudo docker"
  else
    echo "Docker is installed but unavailable to this user (including passwordless sudo)." >&2
    exit 1
  fi
else
  $REMOTE_DOCKER version >/dev/null
fi

GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "Could not determine GPU memory from nvidia-smi: $GPU_MEMORY_MIB" >&2
  exit 1
fi
if (( GPU_MEMORY_MIB < MIN_GPU_MEMORY_MIB )); then
  echo "This full profile requires at least $MIN_GPU_MEMORY_MIB MiB GPU memory; found $GPU_MEMORY_MIB MiB." >&2
  echo "Override MIN_GPU_MEMORY_MIB only if reduced-size/OOM records are acceptable." >&2
  exit 1
fi

mkdir -p \
  "$REVISION3_ROOT/cache/home" \
  "$REVISION3_ROOT/cache/huggingface" \
  "$REVISION3_ROOT/cache/xdg" \
  "$REVISION3_ROOT/cache/numba" \
  "$REVISION3_ROOT/cache/matplotlib"

STATUS_FILE="$REVISION3_ROOT/pipeline.status"
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
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  lscpu
  $REMOTE_DOCKER version
} > "$REVISION3_ROOT/host-preflight.txt"

echo "[revision3] building pinned container image: $IMAGE"
$REMOTE_DOCKER build $DOCKER_BUILD_FLAGS -f docker/Dockerfile -t "$IMAGE" .
{
  printf 'benchmark_image=%s\n' "$IMAGE"
  $REMOTE_DOCKER image inspect \
    --format 'image_id={{.Id}} repo_digests={{json .RepoDigests}}' \
    "$IMAGE"
} >> "$REVISION3_ROOT/host-preflight.txt"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
echo "[revision3] running full suite on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
$REMOTE_DOCKER run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --user "$HOST_UID:$HOST_GID" \
  -e HOME="/work/$REVISION3_ROOT/cache/home" \
  -e HF_HOME="/work/$REVISION3_ROOT/cache/huggingface" \
  -e XDG_CACHE_HOME="/work/$REVISION3_ROOT/cache/xdg" \
  -e NUMBA_CACHE_DIR="/work/$REVISION3_ROOT/cache/numba" \
  -e MPLCONFIGDIR="/work/$REVISION3_ROOT/cache/matplotlib" \
  -e REVISION3_ROOT="$REVISION3_ROOT" \
  -e REVISION3_PIPELINE_VERSION="$PIPELINE_VERSION" \
  -e FORCE_STAGES="${FORCE_STAGES:-0}" \
  -e LARGE_REPEATS="${LARGE_REPEATS:-3}" \
  -e DIRE_ITERATIONS="${DIRE_ITERATIONS:-128}" \
  -e LARGE_TIMEOUT_MINUTES="${LARGE_TIMEOUT_MINUTES:-180}" \
  -e EVALUATION_TOPOLOGY_SUBSET_SIZE="${EVALUATION_TOPOLOGY_SUBSET_SIZE:-4000}" \
  -e EVALUATION_TOPOLOGY_SUBSET_COUNT="${EVALUATION_TOPOLOGY_SUBSET_COUNT:-10}" \
  -e EVALUATION_TOPOLOGY_WORKERS="${EVALUATION_TOPOLOGY_WORKERS:-24}" \
  -e KNN_AUDIT_SAMPLE_SIZE="${KNN_AUDIT_SAMPLE_SIZE:-20000}" \
  -e SMALL_GPU_REPEATS="${SMALL_GPU_REPEATS:-20}" \
  -e SMALL_CPU_REPEATS="${SMALL_CPU_REPEATS:-10}" \
  -e SMALL_TOPOLOGY_REPEATS="${SMALL_TOPOLOGY_REPEATS:-20}" \
  -e SMALL_TOPOLOGY_SAMPLE_SIZE="${SMALL_TOPOLOGY_SAMPLE_SIZE:-1000}" \
  -e SMALL_CPU_JOBS="${SMALL_CPU_JOBS:-24}" \
  -e ABLATION_REPEATS="${ABLATION_REPEATS:-3}" \
  -e ABLATION_TOPOLOGY_REPEATS="${ABLATION_TOPOLOGY_REPEATS:-3}" \
  -v "$PWD:/work" \
  -w /work \
  "$IMAGE" \
  scripts/run_revision3_inside_container.sh

ARCHIVE_NAME="$(sed -n '1p' "$REVISION3_ROOT/bundles/LATEST")"
if [[ ! "$ARCHIVE_NAME" =~ ^revision3-results-[A-Za-z0-9._-]+\.tar\.gz$ ]]; then
  echo "Invalid result archive name after GPU pipeline: $ARCHIVE_NAME" >&2
  exit 1
fi
BUNDLE_NAME="${ARCHIVE_NAME%.tar.gz}"
AUDIT_DIRECTORY="$REVISION3_ROOT/cpu_audit"
AUDIT_NAME="$BUNDLE_NAME-topology-audit.json"
AUDIT_PATH="$AUDIT_DIRECTORY/$AUDIT_NAME"
mkdir -p "$AUDIT_DIRECTORY"

echo "[revision3] independently auditing bundled topology with no GPU exposed"
$REMOTE_DOCKER run --rm \
  --ipc=host \
  --user "$HOST_UID:$HOST_GID" \
  -e CUDA_VISIBLE_DEVICES="" \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e HOME="/work/$REVISION3_ROOT/cache/home" \
  -e XDG_CACHE_HOME="/work/$REVISION3_ROOT/cache/xdg" \
  -e NUMBA_CACHE_DIR="/work/$REVISION3_ROOT/cache/numba" \
  -e MPLCONFIGDIR="/work/$REVISION3_ROOT/cache/matplotlib" \
  -v "$PWD:/work" \
  -w /work \
  "$IMAGE" \
  python3 scripts/audit_revision3_topology.py \
    --bundle-root "$REVISION3_ROOT/bundle_staging/$BUNDLE_NAME" \
    --output "$AUDIT_PATH" \
    --workers "${CPU_AUDIT_WORKERS:-24}"

AUDIT_HASH="$(sha256sum "$AUDIT_PATH" | cut -d ' ' -f 1)"
AUDIT_SIDECAR_TEMP="$AUDIT_PATH.sha256.tmp"
printf '%s  %s\n' "$AUDIT_HASH" "$AUDIT_NAME" > "$AUDIT_SIDECAR_TEMP"
mv "$AUDIT_SIDECAR_TEMP" "$AUDIT_PATH.sha256"
AUDIT_POINTER_TEMP="$AUDIT_DIRECTORY/LATEST.tmp"
printf '%s\n' "$AUDIT_NAME" > "$AUDIT_POINTER_TEMP"
mv "$AUDIT_POINTER_TEMP" "$AUDIT_DIRECTORY/LATEST"

prune_children_except() {
  local directory="$1"
  shift
  if [[ ! -d "$directory" || -L "$directory" ]]; then
    echo "Refusing to prune missing or linked result directory: $directory" >&2
    return 1
  fi
  local candidate name keep expected
  shopt -s dotglob nullglob
  for candidate in "$directory"/*; do
    name="${candidate##*/}"
    keep=0
    for expected in "$@"; do
      if [[ "$name" == "$expected" ]]; then
        keep=1
        break
      fi
    done
    if (( keep == 1 )); then
      continue
    fi
    echo "[revision3] removing superseded result path: $candidate"
    if [[ -d "$candidate" && ! -L "$candidate" ]]; then
      rm -rf -- "$candidate"
    else
      rm -f -- "$candidate"
    fi
  done
  shopt -u dotglob nullglob
}

prune_children_except \
  "$REVISION3_ROOT/bundles" \
  "LATEST" "$ARCHIVE_NAME" "$ARCHIVE_NAME.sha256"
prune_children_except \
  "$REVISION3_ROOT/bundle_staging" \
  "$BUNDLE_NAME"
prune_children_except \
  "$REVISION3_ROOT/cpu_audit" \
  "LATEST" "$AUDIT_NAME" "$AUDIT_NAME.sha256"
prune_children_except \
  "$REVISION3_ROOT/stages" \
  "$PIPELINE_VERSION"

echo "[revision3] compact result bundle: $ARCHIVE_NAME"
echo "[revision3] CPU topology audit: $AUDIT_NAME"
