#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
FETCH_ROOT="${FETCH_ROOT:-data/revision3/fetched}"
OUTPUT_ROOT="${OUTPUT_ROOT:-generated/revision3}"
PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-.}"
MPLCONFIGDIR="${MPLCONFIGDIR:-.matplotlib-cache}"
BUNDLE_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  scripts/build_revision3_local.sh \
    [--bundle-root PATH] [--output-root PATH] [--paper-output-root PATH]

Without --bundle-root, the script reads data/revision3/fetched/CURRENT. It
verifies every bundled file and regenerates the Revision 3 figures, CSV tables,
TeX table fragments, topology-sensitivity table, result macros, and an outcome
summary without a GPU. It also regenerates the manuscript's canonical `pics/`
tree from the bundle's fresh twenty-run GPU and ten-run CPU small suite. The generated-artifact
directory and `pics/` tree are replaced so files from an older bundle cannot
survive; an existing successful CPU topology audit is retained only when it
names the same source bundle. Use
`make revision3-topology-audit` when the exact
topology environment is installed to recompute, rather than only render, all
bundled homology scores.
The default publication-ready destination is generated/revision3.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-root) BUNDLE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --paper-output-root) PAPER_OUTPUT_ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$BUNDLE_ROOT" ]]; then
  if [[ ! -s "$FETCH_ROOT/CURRENT" ]]; then
    echo "No fetched bundle pointer exists at $FETCH_ROOT/CURRENT." >&2
    exit 1
  fi
  CURRENT="$(sed -n '1p' "$FETCH_ROOT/CURRENT")"
  if [[ ! "$CURRENT" =~ ^revision3-results-[A-Za-z0-9._-]+$ ]]; then
    echo "Unsafe/invalid fetched bundle name: $CURRENT" >&2
    exit 1
  fi
  BUNDLE_ROOT="$FETCH_ROOT/$CURRENT"
fi

mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

"$PYTHON" scripts/verify_revision3_bundle.py \
  --bundle-root "$BUNDLE_ROOT" \
  --require-current-contract
"$PYTHON" scripts/render_revision3_artifacts.py \
  --bundle-root "$BUNDLE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --clean-output-root
"$PYTHON" scripts/render_figures.py \
  --data-root "$BUNDLE_ROOT/small_suite/json_logs" \
  --embedding-png-root "$BUNDLE_ROOT/small_suite/embedding_pngs" \
  --output "$PAPER_OUTPUT_ROOT"

echo "Revision 3 figures and tables: $OUTPUT_ROOT"
echo "Canonical manuscript figures: $PAPER_OUTPUT_ROOT/pics"
