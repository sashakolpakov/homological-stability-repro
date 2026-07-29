#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/homological-stability-repro-revision3}"
IMAGE="${IMAGE:-homological-stability-repro:revision3}"
WAIT=0
FORCE_LAUNCH=0
FRESH_REMOTE_DIR=0
PYTHON_BIN="${PYTHON:-python3}"
SOURCE_ARCHIVE=""

cleanup() {
  if [[ -n "$SOURCE_ARCHIVE" && -f "$SOURCE_ARCHIVE" ]]; then
    rm -f "$SOURCE_ARCHIVE"
  fi
}
trap cleanup EXIT

usage() {
  cat <<'EOF'
Usage:
  scripts/run_remote_revision3.sh \
    --host ubuntu@GPU_HOST --ssh-key /path/to/private_key [--wait]

This creates and verifies a small source-only archive, uploads it without any
committed benchmark data or derived publication artifacts, and starts the
complete Revision 3 pipeline under nohup. A dropped SSH connection does not
stop the run. Use scripts/fetch_revision3_results.sh after state=success.

Options:
  --host USER@HOST
  --ssh-key PATH
  --remote-dir ABSOLUTE_PATH
  --image IMAGE_NAME
  --wait                 poll until completion
  --force-launch         launch despite a live recorded pipeline PID
  --fresh-remote-dir     delete an existing idle remote worktree first;
                         use this for a from-zero data preparation and rerun
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --wait) WAIT=1; shift ;;
    --force-launch) FORCE_LAUNCH=1; shift ;;
    --fresh-remote-dir) FRESH_REMOTE_DIR=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOST" || -z "$SSH_KEY" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -r "$SSH_KEY" ]]; then
  echo "SSH key is not readable: $SSH_KEY" >&2
  exit 2
fi
if [[ "$REMOTE_DIR" == "/" || ! "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  echo "Remote directory must be an absolute path containing only letters, digits, '.', '_', '-', and '/'." >&2
  exit 2
fi
REMOTE_BASENAME="${REMOTE_DIR##*/}"
if [[ "$REMOTE_BASENAME" != homological-stability-repro-* ]]; then
  echo "Remote directory basename must start with homological-stability-repro-." >&2
  exit 2
fi
if [[ ! "$IMAGE" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
  echo "Unsafe Docker image name: $IMAGE" >&2
  exit 2
fi
if (( FRESH_REMOTE_DIR == 1 && FORCE_LAUNCH == 1 )); then
  echo "--fresh-remote-dir and --force-launch cannot be combined." >&2
  exit 2
fi

SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes "$HOST")

echo "[revision3-remote] checking SSH, Docker, and GPU"
"${SSH[@]}" "command -v docker >/dev/null && command -v nvidia-smi >/dev/null && nvidia-smi -L"

if (( FORCE_LAUNCH == 0 )) && "${SSH[@]}" \
  "test -s $REMOTE_DIR/data/revision3/pipeline.pid && kill -0 \$(cat $REMOTE_DIR/data/revision3/pipeline.pid) 2>/dev/null"; then
  echo "A recorded Revision 3 pipeline is already running on $HOST." >&2
  echo "Use --force-launch only after independently confirming that a second run is intended." >&2
  exit 1
fi

if (( FRESH_REMOTE_DIR == 1 )); then
  echo "[revision3-remote] removing any idle remote worktree for a from-zero run"
  "${SSH[@]}" \
    "if test -L $REMOTE_DIR; then echo 'Refusing linked remote worktree' >&2; exit 1; elif test -e $REMOTE_DIR; then rm -rf -- $REMOTE_DIR; fi"
fi

SOURCE_ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/revision3-source.XXXXXX")"
echo "[revision3-remote] creating and verifying source-only archive"
"$PYTHON_BIN" scripts/package_revision3_remote_source.py \
  --output "$SOURCE_ARCHIVE"

echo "[revision3-remote] uploading verified source-only archive"
"${SSH[@]}" \
  "mkdir -p $REMOTE_DIR && rm -rf -- $REMOTE_DIR/docker $REMOTE_DIR/requirements $REMOTE_DIR/scripts && rm -f -- $REMOTE_DIR/.dockerignore $REMOTE_DIR/Makefile $REMOTE_DIR/README.md && tar -xzf - -C $REMOTE_DIR" \
  < "$SOURCE_ARCHIVE"

echo "[revision3-remote] launching detached pipeline"
"${SSH[@]}" \
  "cd $REMOTE_DIR && mkdir -p data/revision3 && (nohup env IMAGE=$IMAGE scripts/reproduce_revision3.sh > data/revision3/pipeline.log 2>&1 < /dev/null & echo \$! > data/revision3/pipeline.pid)"

echo "Remote log: $REMOTE_DIR/data/revision3/pipeline.log"
echo "Remote status: $REMOTE_DIR/data/revision3/pipeline.status"
echo "Fetch after success:"
echo "  scripts/fetch_revision3_results.sh --host $HOST --ssh-key $SSH_KEY --remote-dir $REMOTE_DIR"

if (( WAIT == 1 )); then
  while true; do
    STATUS="$("${SSH[@]}" "if test -s $REMOTE_DIR/data/revision3/pipeline.status; then cat $REMOTE_DIR/data/revision3/pipeline.status; else echo state=starting; fi")"
    printf '%s\n' "$STATUS"
    if [[ "$STATUS" == *"state=success"* ]]; then
      exit 0
    fi
    if [[ "$STATUS" == *"state=failed"* ]]; then
      "${SSH[@]}" "tail -n 80 $REMOTE_DIR/data/revision3/pipeline.log" >&2
      exit 1
    fi
    sleep 30
  done
fi
