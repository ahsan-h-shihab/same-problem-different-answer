"""Compute the SAME consistency metrics on the fixed 300-item subset, for both the
frontier LLMs (preds__{tag}__test300.npz) and the open-weight models restricted to the
identical subset indices (from preds__{tag}__test.npz). Enables an apples-to-apples
robustness comparison. Output: results/analysis_subset.json.
"""
import os, json, glob, argparse
import numpy as np
import xnli_load
import phenomena
from analyze import (unanimity, item_agreement, fleiss_kappa, majority, chi2_2x2,
                     bootstrap_ci, LANGS)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def metrics_from_preds(preds, gold, cues):
    """preds [N,L] int (>=0 valid; -1 invalid). Complete-case over items with all-valid rows."""
    N, L = preds.shape
    valid_row = (preds >= 0).all(axis=1)
    n_excluded = int((~valid_row).sum())
    P = preds[valid_row]
    G = gold[valid_row]
    cuesv = {k: v[valid_row] for k, v in cues.items()}
    per_lang_acc = {lg: float((P[:, i] == G).mean()) for i, lg in enumerate(LANGS)}
    non_unan = np.array([len(set(row)) > 1 for row in P])
    flip = float(non_unan.mean())
    _, agree = item_agreement(P)
    kappa = fleiss_kappa(P)
    maj = majority(P)
    maj_correct_on_split = float((maj[non_unan] == G[non_unan]).mean()) if non_unan.any() else float("nan")
    flip_involve = {lg: float(np.mean(P[:, i] != maj)) for i, lg in enumerate(LANGS)}
    phen = {}
    for key in ["negation", "modality", "quantifier", "any"]:
        m = cuesv[key]
        fc = float(non_unan[m].mean()) if m.any() else float("nan")
        fu = float(non_unan[~m].mean()) if (~m).any() else float("nan")
        a = int(non_unan[m].sum()); b = int((~non_unan[m]).sum())
        c = int(non_unan[~m].sum()); d = int((~non_unan[~m]).sum())
        phen[key] = {"n_cued": int(m.sum()), "flip_cued": fc, "flip_uncued": fu,
                     "risk_ratio": float(fc / fu) if fu else float("nan"),
                     "chi2_yates": chi2_2x2(a, b, c, d)}
    return {
        "n_used": int(valid_row.sum()), "n_excluded_invalid": n_excluded,
        "mean_per_lang_acc": float(np.mean(list(per_lang_acc.values()))),
        "per_lang_acc": per_lang_acc,
        "flip_rate(non_unanimous)": flip, "unanimity": 1 - flip,
        "mean_pairwise_agreement": agree, "fleiss_kappa": kappa,
        "maj_correct_on_split": maj_correct_on_split,
        "per_lang_flip_involvement": flip_involve, "phenomena": phen,
    }


def main():
    sub = json.load(open(os.path.join(RESULTS, "llm_subset.json")))
    idx = np.array(sub["indices"])
    prem_en, hyp_en, gold_full = xnli_load.load_en_source("test")
    cues_full = phenomena.tag_all(prem_en, hyp_en)
    cues = {k: v[idx] for k, v in cues_full.items()}
    gold_sub = gold_full[idx]

    out = {"n": int(len(idx)), "seed": sub["seed"], "models": {},
           "cue_counts": {k: int(v.sum()) for k, v in cues.items()}}

    # open-weight models restricted to subset indices
    for f in sorted(glob.glob(os.path.join(RESULTS, "preds__*__test.npz"))):
        tag = os.path.basename(f).split("__")[1]
        d = np.load(f)
        preds = d["preds"].astype(int)[idx]  # [300,15]
        gold = d["gold"].astype(int)[idx]
        assert (gold == gold_sub).all(), f"gold mismatch for {tag}"
        out["models"][tag + " (subset)"] = {"kind": "open-weight",
                                             **metrics_from_preds(preds, gold_sub, cues)}

    # LLM models (already on subset)
    for f in sorted(glob.glob(os.path.join(RESULTS, "preds__*__test300.npz"))):
        tag = os.path.basename(f).split("__")[1]
        d = np.load(f)
        preds = d["preds"].astype(int)
        gold = d["gold"].astype(int)
        assert (gold == gold_sub).all(), f"gold mismatch for {tag}"
        out["models"][tag] = {"kind": "llm", **metrics_from_preds(preds, gold_sub, cues)}

    path = os.path.join(RESULTS, "analysis_subset.json")
    json.dump(out, open(path, "w", encoding="utf-8", newline="\n"), indent=2)
    print("wrote", path)
    for tag, r in out["models"].items():
        print(f"[{tag:22s}] acc={r['mean_per_lang_acc']:.3f} flip={r['flip_rate(non_unanimous)']:.3f} "
              f"kappa={r['fleiss_kappa']:.3f} negRR={r['phenomena']['negation']['risk_ratio']:.2f} "
              f"(n={r['n_used']}, excl={r['n_excluded_invalid']})")


if __name__ == "__main__":
    main()
