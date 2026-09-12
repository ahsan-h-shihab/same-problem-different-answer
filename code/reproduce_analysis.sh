#!/usr/bin/env bash
# Regenerate every reported number from the committed per-item predictions and raw LLM
# responses, then verify them. Minutes on a CPU; no API keys.
#
#   1. fetch/verify the pinned XNLI test data (gold labels, source text for cue tagging)
#   2. re-score the LLM study offline from the raw responses
#   3. recompute all metrics (results/analysis__test.json, results/analysis_subset.json)
#   4. regenerate the paper's number macros and tables (into paper/, not distributed)
#      and the figures (into figures/)
#   5. check the regenerated macros/tables against results/expected_generated_outputs.json
#   6. verify all SHA256 provenance hashes (read-only)
#
# Steps 3-4 rewrite their output files in place; the metrics JSON regenerates byte-identically,
# so steps 5-6 pass on an unmodified checkout. Figures regenerate with identical content but
# different embedded PDF timestamps.
set -euo pipefail
cd "$(dirname "$0")"
export USE_TF=0 PYTHONIOENCODING=utf-8

python fetch_xnli.py
python rescore_llm.py
python analyze.py --split test
python analyze_subset.py
mkdir -p ../paper/tables
python make_macros.py --splits test
python make_tables.py --split test
python make_figures.py --split test
python extract_examples.py
python make_examples.py
python make_llm.py
python check_generated.py
python verify_provenance.py --require-data
echo "REPRODUCE_ANALYSIS_OK"
