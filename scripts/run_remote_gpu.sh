#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_DIR="${REMOTE_DIR:-~/homological-stability-repro-run}"
IMAGE="${IMAGE:-homological-stability-repro:latest}"
REMOTE_DOCKER="${REMOTE_DOCKER:-auto}"
DOCKER_BUILD_FLAGS="${DOCKER_BUILD_FLAGS:-}"
GPU_TASK="${GPU_TASK:-benchmarks}"
REPEATS="${REPEATS:-10}"
TOPOLOGY_REPEATS="${TOPOLOGY_REPEATS:-1}"
MAX_POINTS="${MAX_POINTS:-10000}"
FULL_DATASETS="${FULL_DATASETS:-0}"
UMAP_MAX_POINTS="${UMAP_MAX_POINTS:-10000}"
METRIC_SUBSAMPLE="${METRIC_SUBSAMPLE:-0.1}"
TOPOLOGY_SAMPLE_FRACTION="${TOPOLOGY_SAMPLE_FRACTION:-0.05}"
TOPOLOGY_STEPS="${TOPOLOGY_STEPS:-100}"
SWEEP_FRACTIONS="${SWEEP_FRACTIONS:-0.05 0.1 0.2}"
SWEEP_REPEATS="${SWEEP_REPEATS:-1}"
WARN_TOPOLOGY_HOURS="${WARN_TOPOLOGY_HOURS:-1.0}"
LOCAL_JSON_ROOT="${LOCAL_JSON_ROOT:-data/remote_json_logs}"
LOCAL_EMBEDDING_ROOT="${LOCAL_EMBEDDING_ROOT:-data/remote_embedding_pngs}"
LOCAL_SWEEP_ROOT="${LOCAL_SWEEP_ROOT:-data/topology_sampling_sweep}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-data/fresh_benchmark_results}"
REMOTE_JSON_ROOT="${REMOTE_JSON_ROOT:-data/fresh_json_logs}"
REMOTE_EMBEDDING_ROOT="${REMOTE_EMBEDDING_ROOT:-data/fresh_embedding_pngs}"
REMOTE_SWEEP_ROOT="${REMOTE_SWEEP_ROOT:-data/topology_sampling_sweep}"

usage() {
  cat >&2 <<'EOF'
Usage:
  HOST=ubuntu@host SSH_KEY=/path/to/private_key scripts/run_remote_gpu.sh
  scripts/run_remote_gpu.sh --host ubuntu@host --ssh-key /path/to/private_key

Optional environment variables:
  GPU_TASK=benchmarks|topology-sweep        default: benchmarks
  REMOTE_DOCKER=auto|docker|"sudo docker"   default: auto
  DOCKER_BUILD_FLAGS="--no-cache"            force fresh dependency install
  REPEATS=10
  TOPOLOGY_REPEATS=1
  MAX_POINTS=10000
  FULL_DATASETS=0
  UMAP_MAX_POINTS=10000
  METRIC_SUBSAMPLE=0.1
  TOPOLOGY_SAMPLE_FRACTION=0.05
  TOPOLOGY_STEPS=100
  SWEEP_FRACTIONS="0.05 0.1 0.2"             topology-sweep only
  SWEEP_REPEATS=1                             topology-sweep only
  WARN_TOPOLOGY_HOURS=1.0                    topology-sweep only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --task)
      GPU_TASK="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$HOST" || -z "$SSH_KEY" ]]; then
  usage
  exit 2
fi

SSH=(ssh -i "$SSH_KEY" "$HOST")
SCP=(scp -i "$SSH_KEY")

if [[ ! -r "$SSH_KEY" ]]; then
  echo "SSH key is not readable: $SSH_KEY" >&2
  exit 2
fi

echo "[1/6] Checking remote SSH, Docker, and GPU prerequisites"
"${SSH[@]}" "true"
"${SSH[@]}" "command -v docker >/dev/null"
"${SSH[@]}" "nvidia-smi >/dev/null"
if [[ "$REMOTE_DOCKER" == "auto" ]]; then
  if "${SSH[@]}" "docker version >/dev/null 2>&1"; then
    REMOTE_DOCKER="docker"
  elif "${SSH[@]}" "sudo -n docker version >/dev/null 2>&1"; then
    REMOTE_DOCKER="sudo docker"
  else
    echo "Remote Docker is installed, but the SSH user cannot run docker or passwordless sudo docker." >&2
    echo "Set REMOTE_DOCKER explicitly after fixing Docker permissions on the host." >&2
    exit 1
  fi
else
  "${SSH[@]}" "$REMOTE_DOCKER version >/dev/null"
fi
echo "Using remote Docker command: $REMOTE_DOCKER"

echo "[2/6] Uploading reproducibility repository"
COPYFILE_DISABLE=1 tar \
  --exclude .git \
  --exclude .venv \
  --exclude .matplotlib-cache \
  --exclude __pycache__ \
  --exclude '*/__pycache__' \
  --exclude pics \
  --exclude docs/_build \
  --exclude docs/_static/paper \
  --exclude docs/figures.rst \
  --exclude docs/paper.rst \
  --exclude 'dire_short.aux' \
  --exclude 'dire_short.bbl' \
  --exclude 'dire_short.bcf' \
  --exclude 'dire_short.blg' \
  --exclude 'dire_short.fdb_latexmk' \
  --exclude 'dire_short.fls' \
  --exclude 'dire_short.log' \
  --exclude 'dire_short.out' \
  --exclude 'dire_short.pdf' \
  --exclude 'dire_short.run.xml' \
  --exclude data/fresh_benchmark_results \
  --exclude data/fresh_json_logs \
  --exclude data/fresh_embedding_pngs \
  --exclude data/remote_benchmark_results \
  --exclude data/remote_json_logs \
  --exclude data/remote_embedding_pngs \
  -czf - . | "${SSH[@]}" "mkdir -p $REMOTE_DIR && tar -xzf - -C $REMOTE_DIR"

echo "[3/6] Building benchmark Docker image on remote GPU host"
"${SSH[@]}" "cd $REMOTE_DIR && $REMOTE_DOCKER build $DOCKER_BUILD_FLAGS -f docker/Dockerfile -t $IMAGE ."

case "$GPU_TASK" in
  benchmarks)
    echo "[4/6] Running full benchmark suite in Docker"
    "${SSH[@]}" "cd $REMOTE_DIR && REMOTE_UID=\$(id -u) && REMOTE_GID=\$(id -g) && $REMOTE_DOCKER run --rm --gpus all --user \$REMOTE_UID:\$REMOTE_GID -e HOME=/tmp -v \$(pwd):/work -w /work $IMAGE make benchmarks OUTPUT_ROOT=$REMOTE_OUTPUT_ROOT JSON_LOG_ROOT=$REMOTE_JSON_ROOT EMBEDDING_PNG_ROOT=$REMOTE_EMBEDDING_ROOT REPEATS=$REPEATS TOPOLOGY_REPEATS=$TOPOLOGY_REPEATS MAX_POINTS=$MAX_POINTS FULL_DATASETS=$FULL_DATASETS UMAP_MAX_POINTS=$UMAP_MAX_POINTS METRIC_SUBSAMPLE=$METRIC_SUBSAMPLE TOPOLOGY_SAMPLE_FRACTION=$TOPOLOGY_SAMPLE_FRACTION TOPOLOGY_STEPS=$TOPOLOGY_STEPS"

    echo "[5/6] Downloading benchmark JSON logs and canonical embedding PNGs"
    rm -rf "$LOCAL_JSON_ROOT" "$LOCAL_EMBEDDING_ROOT"
    mkdir -p "$LOCAL_JSON_ROOT" "$LOCAL_EMBEDDING_ROOT"
    "${SCP[@]}" -r "$HOST:$REMOTE_DIR/$REMOTE_JSON_ROOT/." "$LOCAL_JSON_ROOT/"
    "${SCP[@]}" -r "$HOST:$REMOTE_DIR/$REMOTE_EMBEDDING_ROOT/." "$LOCAL_EMBEDDING_ROOT/"

    echo "[6/6] Verifying downloaded benchmark artifacts"
    JSON_COUNT=$(find "$LOCAL_JSON_ROOT" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
    PNG_COUNT=$(find "$LOCAL_EMBEDDING_ROOT" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
    echo "Downloaded $JSON_COUNT JSON files and $PNG_COUNT embedding PNG files."
    if [[ "$JSON_COUNT" -lt 6 || "$PNG_COUNT" -lt 24 ]]; then
      echo "Downloaded artifact counts look incomplete. Inspect the remote logs before using these results." >&2
      exit 1
    fi
    echo "Done. To build local figures/PDF/site from the remote output, run:"
    echo "  DATA_ROOT=$LOCAL_JSON_ROOT EMBEDDING_DATA_ROOT=$LOCAL_EMBEDDING_ROOT scripts/run_pipeline.sh all"
    ;;
  topology-sweep)
    echo "[4/6] Running blobs topology sampling sweep in Docker"
    "${SSH[@]}" "cd $REMOTE_DIR && REMOTE_UID=\$(id -u) && REMOTE_GID=\$(id -g) && $REMOTE_DOCKER run --rm --gpus all --user \$REMOTE_UID:\$REMOTE_GID -e HOME=/tmp -v \$(pwd):/work -w /work $IMAGE python3 scripts/sweep_topology_sampling.py --output $REMOTE_SWEEP_ROOT --fractions $SWEEP_FRACTIONS --repeats $SWEEP_REPEATS --topology-repeats $TOPOLOGY_REPEATS --metric-subsample $METRIC_SUBSAMPLE --max-points $MAX_POINTS --topology-steps $TOPOLOGY_STEPS --warn-topology-hours $WARN_TOPOLOGY_HOURS"

    echo "[5/6] Downloading topology sampling sweep artifacts"
    rm -rf "$LOCAL_SWEEP_ROOT"
    mkdir -p "$LOCAL_SWEEP_ROOT"
    "${SCP[@]}" -r "$HOST:$REMOTE_DIR/$REMOTE_SWEEP_ROOT/." "$LOCAL_SWEEP_ROOT/"

    echo "[6/6] Verifying downloaded topology sampling sweep"
    if [[ ! -s "$LOCAL_SWEEP_ROOT/summary.json" ]]; then
      echo "Missing topology sweep summary: $LOCAL_SWEEP_ROOT/summary.json" >&2
      exit 1
    fi
    echo "Done. Topology sampling summary:"
    echo "  $LOCAL_SWEEP_ROOT/summary.json"
    ;;
  *)
    echo "Unknown GPU_TASK: $GPU_TASK" >&2
    echo "Use GPU_TASK=benchmarks or GPU_TASK=topology-sweep." >&2
    exit 2
    ;;
esac
