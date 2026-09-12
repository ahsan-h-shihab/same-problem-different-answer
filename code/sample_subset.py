"""Deterministically sample ~300 XNLI test items (fixed seed) for the LLM robustness study.
Preserves the parallel structure: we store item INDICES into the all_languages test split,
so every sampled item keeps all 15 language versions and its shared gold label.
"""
import os, json
import numpy as np
import xnli_load

SEED = 42
N = 300
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    _, gold = xnli_load.load_all("test") if False else (None, None)
    # only need the count + gold; load en source (cheap) for count & gold
    _, _, gold = xnli_load.load_en_source("test")
    total = len(gold)
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(total, size=N, replace=False).tolist())
    # sanity: label balance of the sample
    dist = {int(k): int((gold[idx] == k).sum()) for k in [0, 1, 2]}
    out = {"seed": SEED, "n": N, "total_test": int(total), "indices": idx, "gold_dist": dist}
    path = os.path.join(RESULTS, "llm_subset.json")
    json.dump(out, open(path, "w", encoding="utf-8", newline="\n"), indent=2)
    print(f"sampled {N}/{total} test items (seed={SEED}); gold_dist={dist}; wrote {path}")


if __name__ == "__main__":
    main()
