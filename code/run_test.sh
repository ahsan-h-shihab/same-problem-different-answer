#!/usr/bin/env bash
# Re-run the two open-weight NLI models on the XNLI test split (5,010 items x 15 languages)
# and compare the predictions with the committed ones.
#
# CPU-only, full precision. Expect several hours per model on a laptop-class CPU; progress is
# checkpointed per language, so an interrupted run resumes. Outputs go to ../rerun/ (or
# $RESULTS_DIR); committed files in ../results/ are never modified.
set -euo pipefail
cd "$(dirname "$0")"
export USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8

python fetch_xnli.py
export RESULTS_DIR="${RESULTS_DIR:-../rerun}"
mkdir -p "$RESULTS_DIR"

python run_experiment.py --models mdeberta-base minilm-l6 --splits test --batch_size 32
python compare_predictions.py --candidate "$RESULTS_DIR" --tags mdeberta-base minilm-l6
