#!/usr/bin/env bash
# OPTIONAL: re-query the three frontier LLMs on the fixed 300-item subset (paid API calls).
#
# Hosted models are not bit-reproducible, so a re-query yields new samples rather than the
# committed responses; the reported results are re-derived offline by rescore_llm.py.
# Requires OPENROUTER_API_KEY in the environment (or in a local .env file, never committed).
# Outputs go to ../rerun_llm/ (or $RESULTS_DIR); committed files in ../results/ are untouched.
#
# Settings are those recorded in results/llm_models.json. The Claude Sonnet 5 record has no
# reasoning_effort field; llm_eval.py's default ("low") is used here.
set -euo pipefail
cd "$(dirname "$0")"
export USE_TF=0 PYTHONIOENCODING=utf-8

python fetch_xnli.py
export RESULTS_DIR="${RESULTS_DIR:-../rerun_llm}"
mkdir -p "$RESULTS_DIR"
cp ../results/llm_subset.json "$RESULTS_DIR/llm_subset.json"

python llm_eval.py --provider openrouter --model anthropic/claude-sonnet-5 --tag sonnet5 \
       --concurrency 6 --temperature 1.0
python llm_eval.py --provider openrouter --model openai/gpt-5.6 --tag gpt56 \
       --concurrency 3 --temperature 1.0 --reasoning_effort minimal
python llm_eval.py --provider openrouter --model google/gemini-3.1-pro-preview --tag gemini \
       --concurrency 4 --temperature 1.0 --reasoning_effort low
