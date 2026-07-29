#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
DATA_ROOT="${DATA_ROOT:-data/archived/json_logs}"
EMBEDDING_DATA_ROOT="${EMBEDDING_DATA_ROOT:-data/archived/embedding_pngs}"
PYTHON_BIN="${PYTHON:-python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-.matplotlib-cache}"

case "$MODE" in
  figures)
    MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON_BIN" scripts/render_figures.py \
      --data-root "$DATA_ROOT" \
      --embedding-png-root "$EMBEDDING_DATA_ROOT" \
      --output .
    ;;
  manuscript)
    MPLCONFIGDIR="$MPLCONFIGDIR" PYTHON="$PYTHON_BIN" scripts/compile_manuscript.sh \
      --data-root "$DATA_ROOT" \
      --embedding-png-root "$EMBEDDING_DATA_ROOT"
    ;;
  docs)
    MPLCONFIGDIR="$MPLCONFIGDIR" PYTHON="$PYTHON_BIN" scripts/compile_manuscript.sh \
      --data-root "$DATA_ROOT" \
      --embedding-png-root "$EMBEDDING_DATA_ROOT"
    "$PYTHON_BIN" scripts/prepare_docs.py
    rm -rf docs/_build/html
    "$PYTHON_BIN" -m sphinx -b html docs docs/_build/html
    "$PYTHON_BIN" scripts/check_authorial_invariants.py --require-generated
    ;;
  all)
    "$0" figures
    "$0" manuscript
    "$0" docs
    ;;
  *)
    echo "Usage: $0 [figures|manuscript|docs|all]" >&2
    exit 2
    ;;
esac
