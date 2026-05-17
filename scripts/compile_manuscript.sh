#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/archived/json_logs"
EMBEDDING_PNG_ROOT="${EMBEDDING_DATA_ROOT:-data/archived/embedding_pngs}"
RENDER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --embedding-png-root)
      EMBEDDING_PNG_ROOT="$2"
      shift 2
      ;;
    --no-render)
      RENDER=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RENDER" == "1" ]]; then
  "${PYTHON:-python3}" scripts/render_figures.py \
    --data-root "$DATA_ROOT" \
    --embedding-png-root "$EMBEDDING_PNG_ROOT" \
    --output .
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is required to compile the manuscript." >&2
  exit 1
fi

if ! command -v biber >/dev/null 2>&1; then
  echo "biber is required because dire_short.tex uses biblatex with backend=biber." >&2
  exit 1
fi

latexmk -pdf -interaction=nonstopmode -halt-on-error dire_short.tex
