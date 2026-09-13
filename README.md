# Same Problem, Different Answer: Accuracy Hides Cross-Lingual Inconsistency in Multilingual Inference

Code, per-item predictions, raw LLM responses, analysis outputs, and SHA256 provenance for the
paper of the same title (LUHME 2026 workshop at EMNLP 2026).

The paper measures **item-level cross-lingual consistency** on the parallel XNLI test set
(5,010 premise–hypothesis items, each in 15 languages with one shared gold label): does a
model give the *same* label to the same item in every language? Two open-weight zero-shot
NLI models are the primary study; three hosted frontier LLMs on a fixed 300-item subset are a
robustness check.

| Model (XNLI test) | Mean per-language accuracy | Unanimous across 15 languages | Items whose prediction flips |
|---|---|---|---|
| mDeBERTa-v3-base | 80.8% | 38.9% | 61.1% |
| MiniLM-L6 | 71.3% | 21.9% | 78.1% |

## Quick start (minutes, CPU, no API keys)

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd code
python verify_provenance.py        # check every SHA256 in results/provenance.json (read-only)
bash reproduce_analysis.sh         # regenerate all reported numbers and verify them
```

`reproduce_analysis.sh` fetches the pinned XNLI test file, re-scores the LLM study offline
from the raw responses, recomputes all metrics, regenerates the LaTeX macro and table files
that hold every number reported in the paper (written to `paper/`, which is not distributed)
and the figures, then checks the macros and tables against
`results/expected_generated_outputs.json` and re-verifies all provenance hashes. On an
unmodified checkout every check passes and the metrics JSON regenerates byte-identically.
The figure PDFs regenerate with identical content but different embedded timestamps, so they
will show as modified.

## Full re-runs (optional)

| What | Command | Notes |
|---|---|---|
| Open-weight inference | `bash code/run_test.sh` | Both models on the full test set. CPU-only; several hours per model; checkpointed per language. Writes to `rerun/` and compares with the committed predictions (`compare_predictions.py`). |
| Hosted-LLM re-query | `bash code/run_llm.sh` | Paid API calls via OpenRouter (`OPENROUTER_API_KEY`); needs `requirements-llm.txt`. Hosted models are not bit-reproducible, so this draws new samples; the reported numbers come from the committed responses via `rescore_llm.py`. Writes to `rerun_llm/`. |
| Truncation check | `python code/check_truncation.py` | Tokenizes all 75,150 test pairs for both models: no pair exceeds the 256-token limit (longest: 216 tokens, mDeBERTa). |

## Pinned inputs

All pins live in `code/revisions.py`.

| Input | Source | Revision / checksum |
|---|---|---|
| XNLI test split | Hugging Face `facebook/xnli`, file `all_languages/test-00000-of-00001.parquet` | commit `b8dd5d7af51114dbda02c0e3f6133f332186418e`; SHA256 `599e3a0191403f19cbe802afdf69841152000b41eaed725e4f463d432c0ffb49` |
| mDeBERTa-v3-base | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | commit `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c` |
| MiniLM-L6 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | commit `0a71e92a985b6e1ad1828cf67ce9c459639c1dca` |
| Frontier LLMs | OpenRouter `openai/gpt-5.6`, `anthropic/claude-sonnet-5`, `google/gemini-3.1-pro-preview` | settings in `results/llm_models.json` |

The `all_languages` configuration is used because it keeps the 15 language versions of each
item on one row; the per-language XNLI configurations are not row-aligned. The XNLI
validation split is not used: both checkpoints were fine-tuned on it.

## Repository layout

```
code/      pipeline, analysis, and verification tooling
results/   per-item predictions, raw LLM responses, metrics, provenance.json,
           expected hashes of the regenerated paper macros/tables
logs/      provenance.log (human-readable SHA256 record)
figures/   figure PDFs
```

The manuscript itself is not part of this repository; the published paper will be available
from the conference proceedings.

### Results files

| File | Contents |
|---|---|
| `preds__{mdeberta-base,minilm-l6}__test.npz` | `probs` [5010, 15, 3] class probabilities (entailment, neutral, contradiction), `preds` [5010, 15] labels, `gold` [5010] |
| `meta__*__test.json` | model id, label mapping, language order, per-language accuracy |
| `llm_subset.json` | the fixed subset: seed 42, 300 test-item indices, gold-label distribution |
| `llm_raw__{gpt56,sonnet5,gemini}.jsonl` | one line per API response: `idx`, `lang`, `output`, `error` |
| `preds__{gpt56,sonnet5,gemini}__test300.npz` | `preds` [300, 15] parsed labels (−1 = unparseable), `gold`, `indices`, `langs` |
| `llm_models.json` | provider, model id, decoding settings, invalid counts |
| `analysis__test.json`, `analysis_subset.json` | all reported metrics |
| `flip_examples.json` | candidate items for the paper's qualitative-example table |
| `provenance.json` | SHA256 of the subset, raw and parsed predictions, and final metrics |
| `expected_generated_outputs.json` | SHA256 of the regenerated paper macro and table files |

### How the LLM responses are scored

Each (item, language) instance is scored from the **first successful response in file
order**; if no call for that instance succeeded, from its first recorded attempt. Outputs are
lowercased and matched to the three labels (exact, then prefix, then alphabetic substring);
anything else is invalid.

For 1,658 of the 4,500 GPT-5.6 instances the raw file contains two successful responses, and
68 of these pairs parse to different labels; scoring deterministically uses the first (paper,
Appendix B). `python rescore_llm.py --duplicate-sensitivity` recomputes the GPT-5.6 subset
metrics with the later response instead: mean accuracy 80.1% (unchanged), flip rate 57.9%
(reported: 58.5%), unanimity 42.1% (41.5%), pairwise agreement 80.9% (81.0%), Fleiss' kappa
0.698 (0.699).

API failures that carried no model output were removed before release, which changes no
scored record; see `PROVENANCE_CHANGES.md`.

## Provenance

`results/provenance.json` and `logs/provenance.log` record SHA256 hashes for the sampled
subset (index file and item content), raw model outputs, parsed predictions, and final
metrics. `code/verify_provenance.py` re-checks all of them without writing anything.
`code/hash_artifacts.py` regenerates the record; it is not part of the normal workflow,
because it overwrites the committed record. `PROVENANCE_CHANGES.md` documents the cleaning
applied for release and lists the original hashes of the six changed files.

## Third-party material

| Material | Upstream source | Terms | In this repository |
|---|---|---|---|
| XNLI (Conneau et al., 2018) | <https://github.com/facebookresearch/XNLI>; Hugging Face `facebook/xnli` | CC BY-NC 4.0 (upstream `LICENSE`) | Not redistributed; `code/fetch_xnli.py` downloads the pinned file. Exceptions: 40 English premise–hypothesis pairs in `results/flip_examples.json` and the gold labels stored in `results/preds__*.npz`, which remain under CC BY-NC 4.0. |
| mDeBERTa-v3-base-mnli-xnli, multilingual-MiniLMv2-L6-mnli-xnli | Hugging Face `MoritzLaurer/...` | MIT (model cards) | Not redistributed; downloaded at the pinned revisions. |
| Hosted-LLM responses | OpenRouter (`openai/gpt-5.6`, `anthropic/claude-sonnet-5`, `google/gemini-3.1-pro-preview`) | Provider terms of service | `results/llm_raw__*.jsonl`: the models' raw text outputs under the prompt in `code/llm_eval.py`. |

## Citation

Please cite the paper; `CITATION.cff` holds the citation metadata. Bibliographic details
(proceedings pages, DOI) will be added once the proceedings are published.

## Licence

Code is released under the MIT License (`LICENSE`). Experimental outputs and documentation are
released under CC BY 4.0, with the third-party exceptions above; see `LICENSE-OUTPUTS.md`.
