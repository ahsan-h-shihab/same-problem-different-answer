"""Re-score the frontier-LLM study offline from the committed raw responses.

No API key and no network are needed. For each (item, language) the scored record is the
first successful response in file order; if no call for that instance succeeded, it is the
first recorded attempt (llm_eval.load_cache). Labels are parsed with llm_eval.parse_label.

    python rescore_llm.py                   # compare with results/preds__*__test300.npz
    python rescore_llm.py --write ../rescored   # also write re-scored .npz files there
    python rescore_llm.py --duplicate-sensitivity   # GPT-5.6: first vs later duplicate response

The comparison covers the parsed prediction matrix, item indices, language order, and the
invalid counts recorded in results/llm_models.json. Exit status 0 when everything matches,
1 on any mismatch. Committed files under results/ are never modified.
"""
import os, sys, json, argparse, collections
import numpy as np
from llm_eval import load_cache, parse_label, LANGS, _ok

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TAGS = ["gpt56", "sonnet5", "gemini"]


def rescore(tag, indices):
    cache = load_cache(os.path.join(RESULTS, f"llm_raw__{tag}.jsonl"))
    preds = np.full((len(indices), len(LANGS)), -1, dtype=np.int8)
    for ii, i in enumerate(indices):
        for lj, lg in enumerate(LANGS):
            rec = cache.get((i, lg))
            preds[ii, lj] = parse_label(rec["output"]) if rec else -1
    return preds


def duplicate_sensitivity(indices):
    """For GPT-5.6 instances with two successful responses, score the later one instead of the
    first and recompute the subset metrics with analyze_subset.metrics_from_preds (the code
    that produced results/analysis_subset.json). Needs the XNLI test data for cue tagging."""
    import xnli_load, phenomena
    from analyze_subset import metrics_from_preds
    succ = collections.defaultdict(list)
    for line in open(os.path.join(RESULTS, "llm_raw__gpt56.jsonl"), encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if _ok(r):
            succ[(r["idx"], r["lang"])].append(r["output"])
    first = rescore("gpt56", indices)
    later = first.copy()
    for ii, i in enumerate(indices):
        for lj, lg in enumerate(LANGS):
            v = succ.get((i, lg))
            if v and len(v) > 1:
                later[ii, lj] = parse_label(v[-1])
    prem, hyp, gold_full = xnli_load.load_en_source("test")
    idx = np.array(indices)
    cues = {k: v[idx] for k, v in phenomena.tag_all(prem, hyp).items()}
    gold = gold_full[idx].astype(int)
    n_dup = sum(len(v) > 1 for v in succ.values())
    print(f"GPT-5.6 instances with two successful responses: {n_dup}; "
          f"labels differing between first and later: {int((first != later).sum())}")
    for name, preds in [("first (reported)", first), ("later", later)]:
        m = metrics_from_preds(preds.astype(int), gold, cues)
        print(f"  {name:17s} acc={100 * m['mean_per_lang_acc']:.1f} flip={100 * m['flip_rate(non_unanimous)']:.1f} "
              f"unanimity={100 * m['unanimity']:.1f} agree={100 * m['mean_pairwise_agreement']:.1f} "
              f"kappa={m['fleiss_kappa']:.3f} n_used={m['n_used']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="DIR", help="write re-scored npz files to DIR")
    ap.add_argument("--duplicate-sensitivity", action="store_true",
                    help="report GPT-5.6 metrics when the later duplicate response is scored")
    args = ap.parse_args()

    sub = json.load(open(os.path.join(RESULTS, "llm_subset.json")))
    indices = sub["indices"]
    models = json.load(open(os.path.join(RESULTS, "llm_models.json")))
    failures = 0
    gold = None
    if args.write:
        import xnli_load  # gold labels come from the XNLI data, not from committed outputs
        _, all_gold = xnli_load.load_all("test")
        gold = np.array([all_gold[i] for i in indices], dtype=np.int8)
        os.makedirs(args.write, exist_ok=True)

    for tag in TAGS:
        preds = rescore(tag, indices)
        z = np.load(os.path.join(RESULTS, f"preds__{tag}__test300.npz"))
        checks = {
            "preds": np.array_equal(preds, z["preds"]),
            "indices": list(map(int, z["indices"])) == list(indices),
            "langs": [str(x) for x in z["langs"]] == LANGS,
            "invalid_count": int((preds < 0).sum()) == int(models[tag]["invalid"]),
        }
        ok = all(checks.values())
        failures += not ok
        print(f"{'OK ' if ok else 'FAIL'} {tag:8s} invalid={int((preds < 0).sum())}/{preds.size} "
              + " ".join(f"{k}={'match' if v else 'MISMATCH'}" for k, v in checks.items()))
        if args.write:
            np.savez_compressed(os.path.join(args.write, f"preds__{tag}__test300.npz"),
                                preds=preds, gold=gold, indices=np.array(indices), langs=np.array(LANGS))
    print(f"{len(TAGS) - failures}/{len(TAGS)} models re-scored identically")
    if args.duplicate_sensitivity:
        duplicate_sensitivity(indices)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
