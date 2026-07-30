#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/homological-stability-repro-revision3}"
PYTHON="${PYTHON:-python3}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-data/revision3/downloads}"
FETCH_ROOT="${FETCH_ROOT:-data/revision3/fetched}"
OUTPUT_ROOT="${OUTPUT_ROOT:-generated/revision3}"
RENDER=1

usage() {
  cat <<'EOF'
Usage:
  scripts/fetch_revision3_results.sh \
    --host ubuntu@GPU_HOST --ssh-key /path/to/private_key

The script downloads the compact result archive and the independently produced
CPU-only topology audit. It verifies both outer SHA-256 sidecars and every
internal bundle file, extracts without following links or unsafe paths,
installs the matching successful audit, renders all figures/tables, updates the
local CURRENT pointer, and only then removes superseded Revision 3 downloads
and extractions.

Options:
  --host USER@HOST
  --ssh-key PATH
  --remote-dir ABSOLUTE_PATH
  --download-root PATH
  --fetch-root PATH
  --output-root PATH
  --no-render
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --download-root) DOWNLOAD_ROOT="$2"; shift 2 ;;
    --fetch-root) FETCH_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --no-render) RENDER=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOST" || -z "$SSH_KEY" || ! -r "$SSH_KEY" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  echo "Unsafe remote directory: $REMOTE_DIR" >&2
  exit 2
fi

SSH=(
  ssh
  -i "$SSH_KEY"
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  "$HOST"
)
printf -v RSYNC_RSH \
  'ssh -i %q -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4' \
  "$SSH_KEY"
RSYNC=(
  rsync
  -a
  --partial
  --append
  --timeout=60
  -e "$RSYNC_RSH"
)
STATUS="$("${SSH[@]}" "cat $REMOTE_DIR/data/revision3/pipeline.status 2>/dev/null || true")"
if [[ "$STATUS" != *"state=success"* ]]; then
  echo "Remote pipeline is not marked successful:" >&2
  printf '%s\n' "${STATUS:-state=unknown}" >&2
  exit 1
fi

ARCHIVE_NAME="$("${SSH[@]}" "sed -n '1p' $REMOTE_DIR/data/revision3/bundles/LATEST")"
if [[ ! "$ARCHIVE_NAME" =~ ^revision3-results-[A-Za-z0-9._-]+\.tar\.gz$ ]]; then
  echo "Remote LATEST contains an unsafe/invalid archive name: $ARCHIVE_NAME" >&2
  exit 1
fi
SIDECAR_NAME="$ARCHIVE_NAME.sha256"
mkdir -p "$DOWNLOAD_ROOT" "$FETCH_ROOT"
ARCHIVE_PATH="$DOWNLOAD_ROOT/$ARCHIVE_NAME"
SIDECAR_PATH="$DOWNLOAD_ROOT/$SIDECAR_NAME"

sha256_path() {
  "$PYTHON" -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
    "$1"
}

download_atomic() {
  local remote_subdirectory="$1"
  local remote_name="$2"
  local destination="$3"
  local temporary="$DOWNLOAD_ROOT/.revision3-download.$remote_name.partial"
  if [[ -L "$temporary" || ( -e "$temporary" && ! -f "$temporary" ) ]]; then
    echo "Refusing unsafe resumable download path: $temporary" >&2
    return 1
  fi
  if ! "${RSYNC[@]}" \
      "$HOST:$REMOTE_DIR/data/revision3/$remote_subdirectory/$remote_name" \
      "$temporary"; then
    echo "Resumable download retained for retry: $temporary" >&2
    return 1
  fi
  if ! mv "$temporary" "$destination"; then
    return 1
  fi
}

download_verified() {
  local remote_subdirectory="$1"
  local remote_name="$2"
  local destination="$3"
  local expected_hash="$4"
  local actual_hash=""
  if [[ -s "$destination" ]]; then
    actual_hash="$(sha256_path "$destination")"
    if [[ "$actual_hash" == "$expected_hash" ]]; then
      echo "Reusing verified local download: $destination"
      return
    fi
    echo "Replacing local download with wrong hash: $destination"
  fi
  download_atomic "$remote_subdirectory" "$remote_name" "$destination"
  actual_hash="$(sha256_path "$destination")"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    rm -f -- "$destination"
    echo "Downloaded SHA-256 mismatch: $destination" >&2
    return 1
  fi
}

prune_revision3_entries() {
  local directory="$1"
  local kind="$2"
  shift 2
  if [[ ! -d "$directory" || -L "$directory" ]]; then
    echo "Refusing to prune a missing or linked artifact root: $directory" >&2
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
    case "$kind:$name" in
      downloads:revision3-results-*|downloads:.revision3-download.*)
        ;;
      fetched:revision3-results-*|fetched:.revision3-results-*|fetched:.CURRENT.*)
        ;;
      *)
        echo "Unrecognized entry left for the cleanliness audit: $candidate" >&2
        continue
        ;;
    esac
    echo "Removing superseded artifact: $candidate"
    if [[ -d "$candidate" && ! -L "$candidate" ]]; then
      rm -rf -- "$candidate"
    else
      rm -f -- "$candidate"
    fi
  done
  shopt -u dotglob nullglob
}

download_atomic "bundles" "$SIDECAR_NAME" "$SIDECAR_PATH"

EXPECTED_HASH="$(sed -n '1s/[[:space:]].*$//p' "$SIDECAR_PATH")"
if [[ ! "$EXPECTED_HASH" =~ ^[0-9a-f]{64}$ ]]; then
  rm -f -- "$SIDECAR_PATH"
  echo "Invalid SHA-256 sidecar: $SIDECAR_PATH" >&2
  exit 1
fi
download_verified "bundles" "$ARCHIVE_NAME" "$ARCHIVE_PATH" "$EXPECTED_HASH"

"$PYTHON" scripts/verify_revision3_bundle.py \
  --archive "$ARCHIVE_PATH" \
  --destination-root "$FETCH_ROOT" \
  --expected-archive-sha256 "$EXPECTED_HASH" \
  --require-current-contract

BUNDLE_NAME="${ARCHIVE_NAME%.tar.gz}"
EXPECTED_AUDIT_NAME="$BUNDLE_NAME-topology-audit.json"
AUDIT_NAME="$("${SSH[@]}" \
  "sed -n '1p' $REMOTE_DIR/data/revision3/cpu_audit/LATEST 2>/dev/null || true")"
if [[ "$AUDIT_NAME" != "$EXPECTED_AUDIT_NAME" ]]; then
  echo "Remote CPU audit does not match $BUNDLE_NAME: $AUDIT_NAME" >&2
  exit 1
fi
AUDIT_SIDECAR_NAME="$AUDIT_NAME.sha256"
AUDIT_DOWNLOAD_PATH="$DOWNLOAD_ROOT/$AUDIT_NAME"
AUDIT_SIDECAR_PATH="$DOWNLOAD_ROOT/$AUDIT_SIDECAR_NAME"
download_atomic \
  "cpu_audit" "$AUDIT_SIDECAR_NAME" "$AUDIT_SIDECAR_PATH"
EXPECTED_AUDIT_HASH="$(
  sed -n '1s/[[:space:]].*$//p' "$AUDIT_SIDECAR_PATH"
)"
if [[ ! "$EXPECTED_AUDIT_HASH" =~ ^[0-9a-f]{64}$ ]]; then
  rm -f -- "$AUDIT_SIDECAR_PATH"
  echo "Invalid CPU-audit SHA-256 sidecar: $AUDIT_SIDECAR_PATH" >&2
  exit 1
fi
download_verified \
  "cpu_audit" "$AUDIT_NAME" "$AUDIT_DOWNLOAD_PATH" "$EXPECTED_AUDIT_HASH"
"$PYTHON" - "$AUDIT_DOWNLOAD_PATH" "$BUNDLE_NAME" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
bundle_name = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("source_bundle") != bundle_name:
    raise SystemExit(
        f"CPU audit belongs to {payload.get('source_bundle')!r}, "
        f"not {bundle_name!r}"
    )
if payload.get("status") != "success":
    raise SystemExit(f"CPU audit is not successful: {payload.get('status')!r}")
PY

mkdir -p "$OUTPUT_ROOT"
AUDIT_OUTPUT_TEMP="$(mktemp "$OUTPUT_ROOT/.topology-audit.XXXXXX")"
if ! cp "$AUDIT_DOWNLOAD_PATH" "$AUDIT_OUTPUT_TEMP"; then
  rm -f -- "$AUDIT_OUTPUT_TEMP"
  exit 1
fi
if ! mv "$AUDIT_OUTPUT_TEMP" "$OUTPUT_ROOT/topology-audit.json"; then
  rm -f -- "$AUDIT_OUTPUT_TEMP"
  exit 1
fi

if (( RENDER == 1 )); then
  PYTHON="$PYTHON" scripts/build_revision3_local.sh \
    --bundle-root "$FETCH_ROOT/$BUNDLE_NAME" \
    --output-root "$OUTPUT_ROOT"
fi

POINTER_TEMP="$(mktemp "$FETCH_ROOT/.CURRENT.XXXXXX")"
if ! printf '%s\n' "$BUNDLE_NAME" > "$POINTER_TEMP"; then
  rm -f -- "$POINTER_TEMP"
  exit 1
fi
if ! mv "$POINTER_TEMP" "$FETCH_ROOT/CURRENT"; then
  rm -f -- "$POINTER_TEMP"
  exit 1
fi

prune_revision3_entries \
  "$DOWNLOAD_ROOT" downloads \
  "$ARCHIVE_NAME" "$SIDECAR_NAME" "$AUDIT_NAME" "$AUDIT_SIDECAR_NAME"
prune_revision3_entries \
  "$FETCH_ROOT" fetched \
  "CURRENT" "$BUNDLE_NAME"

if (( RENDER == 1 )); then
  "$PYTHON" scripts/check_authorial_invariants.py \
    --require-clean-local-artifacts
else
  echo "Rendering was disabled; the final clean-artifact invariant is deferred."
fi

echo "Fetched and verified bundle: $FETCH_ROOT/$BUNDLE_NAME"
echo "Fetched and verified CPU audit: $OUTPUT_ROOT/topology-audit.json"
