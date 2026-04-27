#!/usr/bin/env bash
# End-to-end data pipeline. Usage:
#   ./run_pipeline.sh           full run
#   ./run_pipeline.sh --smoke   tiny sample for local sanity-check
#
# After this finishes, train pointwise BERT with:
#   python train.py --data_dir data/processed
# Then run the stacking pipeline (stacking/scripts/run_runpod.sh) for heads.
set -euo pipefail

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

if [ "$SMOKE" -eq 1 ]; then
  LIMIT=200
  CAP=30
  TOTAL=500
  CELL=20
else
  LIMIT=0
  CAP=620
  TOTAL=30000
  CELL=3000
fi

echo "[pipeline] fetch"
python pipeline/01_fetch_sources.py

echo "[pipeline] parse leetcode"
python pipeline/02_parse_leetcode.py --limit "$LIMIT"

echo "[pipeline] parse codecomplex"
python pipeline/03_parse_codecomplex.py --limit "$LIMIT"

echo "[pipeline] synthetic supplemental"
python pipeline/04_parse_supplemental.py

echo "[pipeline] normalize"
python pipeline/05_normalize_labels.py

echo "[pipeline] strip leakage (Solution wrappers + complexity comments)"
python pipeline/05b_strip_leakage.py

echo "[pipeline] dedupe + filter"
python pipeline/06_dedupe_filter.py

echo "[pipeline] balance + augment"
python pipeline/07_balance_augment.py --cap_per_class "$CAP"

echo "[pipeline] split"
python pipeline/08_split.py

echo "[pipeline] pointwise parquet"
python pipeline/09_make_pointwise.py

echo "[pipeline] pairwise parquet"
python pipeline/10_make_pairwise.py --target_total "$TOTAL" --per_cell_cap "$CELL"

echo "[pipeline] audit"
python pipeline/11_audit_report.py

echo "[pipeline] done. Artifacts in data/processed/"
