# Provenance changes made for the public release

The experiments' original SHA256 provenance record was generated on 2026-07-16 (UTC). For
the public release, some files were cleaned or re-encoded and the record was regenerated on
2026-09-12 (UTC) with the same software environment (Python 3.12.1, numpy 2.4.4,
torch 2.10.0+cpu, transformers 4.57.1, datasets 5.0.0). **No prediction, parsed label, metric,
or reported number changed.**

## What changed

Six of the 20 recorded file hashes differ from the original record.

| Artifact | Change | Original SHA256 (bytes) | Released SHA256 (bytes) |
|---|---|---|---|
| `results/llm_models.json` | Removed the entry `gemini_keytest`: a 4-call API connectivity test (2 items x 2 languages, direct Gemini API, all 4 outputs invalid), not used by any analysis, table, figure, or reported number. Line endings normalised to LF. | `ae2686637995f8c169027bb234b8735a20cd94f815c15009a8b02c5c6bd098d1` (951) | `da11aa167c37f1cc872080b39ad8b430aa0d392e548e6ca6a2b9a8b96e927bc7` (686) |
| `results/llm_raw__gpt56.jsonl` | Removed 1,589 API credit/quota-error records, 8 transport-error records (5 connection errors, 3 HTTP 520) for instances that later received a successful response, and 3 truncated partial lines from interrupted writes. Kept all 6,193 model responses: 6,157 successful and 36 empty completions. | `0e2a8da4f92ab36500bd41be7baca95b81d81f5ac13dd2740590b4bc7d7e43e3` (772,933) | `4c725d7ffc5c7ea39d0b49db466e99a03944508ffb7834dcbc9365ebf18bd736` (415,884) |
| `results/llm_raw__gemini.jsonl` | Removed 6 API credit-error records. Kept all 4,506 model responses: 4,499 successful and 7 empty completions. | `2821de95c060e49ba0f2abb6defd9d5d8fb4d4f5e16b3f3e8ce704e709d84639` (305,516) | `5ba3106a4c14fdf2e0b56ada0aa21db41b78f83e524e7c43c09c81441c4c9ef5` (304,286) |
| `results/llm_subset.json` | Line endings normalised from CRLF to LF; parsed JSON identical. | `63946e5c2b16ffa9e21b9924fac5b4c51161a85fbce00c034d0937c8ff4d54ec` (3,379) | `c32aa34d38b9e3d1daa584e84b3a2854262214847b98103822beec146ed57d97` (3,068) |
| `results/analysis__test.json` | Line endings normalised from CRLF to LF; parsed JSON identical. | `1bea871c8819ac39ec701569c8ce7c8bd3dc2713e435334955016f5186267987` (7,912) | `b35fe32e37441602101280e7ec62fa5e76f30a395e93d1e04117b89ea1a63990` (7,669) |
| `results/analysis_subset.json` | Line endings normalised from CRLF to LF; parsed JSON identical. | `1e9cb57ba0027aaabb822f949b0d87edb83619ac01335b259f4ac2042976af37` (12,338) | `42e41ef8cece39ccdaccbd64d9abbf739b696153ee287c68ad4ee3d91eb4855e` (11,952) |

Removed records were API failures (no model output). Kept raw records are the original bytes
of the original lines, in the original order, including their original CRLF line endings.

The line-ending normalisation makes regeneration byte-identical on every platform: all
scripts now write UTF-8 text with LF line endings, whereas the originals were written on
Windows with CRLF. `results/flip_examples.json` and the two `results/meta__*.json` files were
normalised the same way (they are not in the provenance record).

## What did not change

The other 14 of the 20 recorded hashes are identical to the original record:

- the content hash of the 300 x 15 sampled items;
- all five parsed-prediction arrays (`preds__*.npz:preds`) and both open-weight probability
  arrays (`preds__*.npz:probs`);
- all five prediction files (`preds__*.npz`);
- `llm_raw__sonnet5.jsonl` (it contained no error records).

## How the change was checked

- For each of the 13,500 LLM instances (3 models x 300 items x 15 languages), the record
  selected for scoring (`llm_eval.load_cache`: first successful response, else first attempt)
  is identical before and after cleaning.
- Re-scoring the cleaned raw files (`code/rescore_llm.py`) reproduces all three committed
  prediction matrices exactly and the invalid counts in `llm_models.json`
  (GPT-5.6 1/4,500; Claude Sonnet 5 0/4,500; Gemini 3.1 Pro 4/4,500).
- Every line-ending-normalised JSON file parses to the same object as the original, and the
  pipeline regenerates the normalised bytes exactly.
- `code/verify_provenance.py --require-data` reports 20 match, 0 mismatch against the
  regenerated record.
