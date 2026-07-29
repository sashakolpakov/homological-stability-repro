#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
DATA_ROOT="${DATA_ROOT:-data/archived/json_logs}"
EMBEDDING_DATA_ROOT="${EMBEDDING_DATA_ROOT:-data/archived/embedding_pngs}"
LOG="${LOG:-reproduce_archived.log}"
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_archived.sh [--check-only]

Options:
  --check-only   verify that the ordinary clone contains the complete committed
                 archived-data and Revision 3 publication payload, then stop

Environment overrides:
  PYTHON=python3
  VENV=.venv
  DATA_ROOT=data/archived/json_logs
  EMBEDDING_DATA_ROOT=data/archived/embedding_pngs
  LOG=reproduce_archived.log

This is the GPU-free entry point. It creates or reuses a local Python virtual
environment, installs the lightweight rendering/docs dependencies, verifies the
archived benchmark artifacts, and rebuilds figures, the LaTeX PDF, and the
Sphinx HTML manuscript from the archived Lambda logs and committed
publication-ready Revision 3 artifacts.
EOF
}

run_logged() {
  local label="$1"
  shift
  echo "$label"
  if "$@" >>"$LOG" 2>&1; then
    return 0
  else
    local status=$?
    echo "Command failed while running: $label" >&2
    echo "Full log: $LOG" >&2
    echo "Last 60 log lines:" >&2
    tail -n 60 "$LOG" >&2 || true
    exit "$status"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

echo "[1/6] Verifying ordinary-clone archived payload"
"$PYTHON_BIN" scripts/verify_archived_clone_payload.py \
  --repository-root "$ROOT"
if (( CHECK_ONLY == 1 )); then
  echo "Archived clone payload is complete; no environment was created."
  exit 0
fi

rm -f "$LOG"
touch "$LOG"
echo "Detailed log: $LOG"

if [[ ! -x "$VENV/bin/python" ]]; then
  run_logged "[2/6] Creating Python environment: $VENV" "$PYTHON_BIN" -m venv "$VENV"
else
  echo "[2/6] Reusing Python environment: $VENV"
fi
PY="$VENV/bin/python"
run_logged "      Upgrading pip" "$PY" -m pip install --upgrade pip
run_logged "      Installing rendering and documentation dependencies" "$PY" -m pip install -r requirements/render.txt -r requirements/docs.txt

echo "[3/6] Checking TeX tools"
for tool in latexmk pdflatex biber; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required for the local PDF build. Install TeX Live/MacTeX or use the Docker workflow in README.md." >&2
    exit 1
  fi
done

run_logged "[4/6] Running archived-data pipeline" env DATA_ROOT="$DATA_ROOT" EMBEDDING_DATA_ROOT="$EMBEDDING_DATA_ROOT" PYTHON="$PY" scripts/run_pipeline.sh all

echo "[5/6] Validating required build products"
for output in \
  dire_short.pdf \
  docs/_build/html/index.html \
  docs/_build/html/paper.html \
  docs/_build/html/revision3.html
do
  if [[ ! -s "$output" ]]; then
    echo "Required build product is missing or empty: $output" >&2
    exit 1
  fi
done

echo "[6/6] Done"
echo "Figures: pics/"
echo "PDF: dire_short.pdf"
echo "Sphinx HTML: docs/_build/html/index.html"
echo "Detailed log: $LOG"
