#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
DATA_ROOT="${DATA_ROOT:-data/archived/json_logs}"
EMBEDDING_DATA_ROOT="${EMBEDDING_DATA_ROOT:-data/archived/embedding_pngs}"
LOG="${LOG:-reproduce_archived.log}"

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_archived.sh

Environment overrides:
  PYTHON=python3
  VENV=.venv
  DATA_ROOT=data/archived/json_logs
  EMBEDDING_DATA_ROOT=data/archived/embedding_pngs
  LOG=reproduce_archived.log

This is the GPU-free entry point. It creates or reuses a local Python virtual
environment, installs the lightweight rendering/docs dependencies, verifies the
archived benchmark artifacts, and rebuilds figures, the LaTeX PDF, and the
Sphinx HTML manuscript from the archived Lambda logs.
EOF
}

run_logged() {
  local label="$1"
  shift
  echo "$label"
  if ! "$@" >>"$LOG" 2>&1; then
    local status=$?
    echo "Command failed while running: $label" >&2
    echo "Full log: $LOG" >&2
    echo "Last 60 log lines:" >&2
    tail -n 60 "$LOG" >&2 || true
    exit "$status"
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

rm -f "$LOG"
touch "$LOG"
echo "Detailed log: $LOG"

echo "[1/5] Checking archived benchmark artifacts"
JSON_COUNT=$(find "$DATA_ROOT" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')
PNG_COUNT=$(find "$EMBEDDING_DATA_ROOT" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
if [[ "$JSON_COUNT" -lt 7 ]]; then
  echo "Expected at least 7 archived JSON files in $DATA_ROOT; found $JSON_COUNT." >&2
  exit 1
fi
if [[ "$PNG_COUNT" -lt 24 ]]; then
  echo "Expected 24 archived embedding PNG files in $EMBEDDING_DATA_ROOT; found $PNG_COUNT." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  run_logged "[2/5] Creating Python environment: $VENV" "$PYTHON_BIN" -m venv "$VENV"
else
  echo "[2/5] Reusing Python environment: $VENV"
fi
PY="$VENV/bin/python"
run_logged "      Upgrading pip" "$PY" -m pip install --upgrade pip
run_logged "      Installing rendering and documentation dependencies" "$PY" -m pip install -r requirements/render.txt -r requirements/docs.txt

echo "[3/5] Checking TeX tools"
for tool in latexmk pdflatex biber; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required for the local PDF build. Install TeX Live/MacTeX or use the Docker workflow in README.md." >&2
    exit 1
  fi
done

run_logged "[4/5] Running archived-data pipeline" env DATA_ROOT="$DATA_ROOT" EMBEDDING_DATA_ROOT="$EMBEDDING_DATA_ROOT" PYTHON="$PY" scripts/run_pipeline.sh all

echo "[5/5] Done"
echo "Figures: pics/"
echo "PDF: dire_short.pdf"
echo "Sphinx HTML: docs/_build/html/index.html"
echo "Detailed log: $LOG"
