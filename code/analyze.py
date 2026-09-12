"""Analyze saved XNLI predictions: cross-lingual item-level consistency vs accuracy,
per-phenomenon flip concentration, per-language flip involvement, robustness.

Reads results/preds__{tag}__{split}.npz and produces results/analysis__{split}.json
plus tidy CSVs used directly by the figure/table scripts and the paper. No numbers are
computed anywhere else; the paper cites these files.
"""
import os, json, glob, argparse
import numpy as np
import phenomena
import xnli_load

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
# Rough pretraining-resource tier from CommonCrawl sizes reported in the XLM-R paper
# (Conneau et al., 2020). Higher = more resourced. Used only as an ordinal proxy.
RESOURCE_RANK = {  # 1 = lowest resource among these 15, 15 = highest
    "en": 15, "ru": 14, "de": 13, "fr": 12, "es": 11, "zh": 10, "vi": 9, "th": 8,
    "tr": 7, "ar": 6, "bg": 5, "el": 4, "hi": 3, "ur": 2, "sw": 1,
}
RNG = np.random.default_rng(0)


def load_preds(tag, split):
    f = os.path.join(RESULTS, f"preds__{tag}__{split}.npz")
    d = np.load(f)
    return d["probs"], d["preds"].astype(int), d["gold"].astype(int)


def unanimity(preds):
    """Fraction of items where all languages predict the same label."""
    return float(np.mean([len(set(row)) == 1 for row in preds]))


def item_agreement(preds):
    """Per-item average pairwise agreement across languages, and its mean.
    For an item with counts n_c per class over L langs: agree = (sum n_c^2 - L)/(L*(L-1))."""
    L = preds.shape[1]
    agrees = np.zeros(preds.shape[0])
    for i, row in enumerate(preds):
        counts = np.bincount(row, minlength=3)
        agrees[i] = (np.sum(counts**2) - L) / (L * (L - 1))
    return agrees, float(agrees.mean())


def fleiss_kappa(preds, k=3):
    """Fleiss' kappa treating L languages as raters, items as subjects."""
    N, L = preds.shape
    P_i = np.zeros(N)
    col_totals = np.zeros(k)
    for i, row in enumerate(preds):
        counts = np.bincount(row, minlength=k)
        col_totals += counts
        P_i[i] = (np.sum(counts**2) - L) / (L * (L - 1))
    Pbar = P_i.mean()
    p_j = col_totals / (N * L)
    Pe = np.sum(p_j**2)
    if abs(1 - Pe) < 1e-12:
        return float("nan")
    return float((Pbar - Pe) / (1 - Pe))


def majority(preds):
    """Per-item majority label (ties -> lowest label id)."""
    return np.array([np.bincount(row, minlength=3).argmax() for row in preds])


def bootstrap_ci(values_fn, n_items, B=1000):
    """Bootstrap 95% CI over items. values_fn takes an index array, returns scalar."""
    stats = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n_items, n_items)
        stats[b] = values_fn(idx)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def chi2_2x2(a, b, c, d):
    """2x2 chi-square with Yates continuity correction. Table [[a,b],[c,d]]."""
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if min(row1, row2, col1, col2) == 0:
        return 0.0
    exp = [row1 * col1 / n, row1 * col2 / n, row2 * col1 / n, row2 * col2 / n]
    obs = [a, b, c, d]
    chi2 = sum((abs(o - e) - 0.5) ** 2 / e for o, e in zip(obs, exp))
    return float(chi2)


def analyze_model(tag, split, cues):
    probs, preds, gold = load_preds(tag, split)
    N, L, _ = probs.shape
    conf = probs.max(-1)  # per (item,lang) confidence

    per_lang_acc = {lg: float((preds[:, i] == gold).mean()) for i, lg in enumerate(LANGS)}
    mean_acc = float(np.mean(list(per_lang_acc.values())))

    non_unan = np.array([len(set(row)) > 1 for row in preds])
    flip_rate = float(non_unan.mean())
    unan = 1.0 - flip_rate
    agrees, mean_agree = item_agreement(preds)
    kappa = fleiss_kappa(preds)
    maj = majority(preds)

    # Decomposition: unanimous-correct / unanimous-wrong / split
    unan_correct = float(np.mean([(len(set(row)) == 1 and row[0] == gold[i]) for i, row in enumerate(preds)]))
    unan_wrong = float(np.mean([(len(set(row)) == 1 and row[0] != gold[i]) for i, row in enumerate(preds)]))
    split_items = flip_rate  # remaining

    # Among split items, is the majority label correct? (instability on otherwise-answerable items)
    split_mask = non_unan
    maj_correct_on_split = float((maj[split_mask] == gold[split_mask]).mean()) if split_mask.any() else float("nan")

    # Confident flips: on split items, confidence of majority vs minority predictions
    min_conf, maj_conf, conf_flip_frac = [], [], []
    for i in np.where(split_mask)[0]:
        row = preds[i]
        m = maj[i]
        is_min = row != m
        maj_conf.extend(conf[i, ~is_min].tolist())
        min_conf.extend(conf[i, is_min].tolist())
        # a "confident flip" = a minority prediction made with conf>0.9
        conf_flip_frac.append(float(np.mean(conf[i, is_min] > 0.9)) if is_min.any() else 0.0)
    conf_summary = {
        "mean_conf_majority_on_split": float(np.mean(maj_conf)) if maj_conf else float("nan"),
        "mean_conf_minority_on_split": float(np.mean(min_conf)) if min_conf else float("nan"),
        "frac_minority_preds_high_conf(>0.9)": float(np.mean(np.array(min_conf) > 0.9)) if min_conf else float("nan"),
    }

    # Per-language flip involvement: fraction of items where lang disagrees with majority
    flip_involve = {lg: float(np.mean(preds[:, i] != maj)) for i, lg in enumerate(LANGS)}
    # correlation with resource rank
    langs_sorted = LANGS
    fi = np.array([flip_involve[lg] for lg in langs_sorted])
    rr = np.array([RESOURCE_RANK[lg] for lg in langs_sorted])
    acc_arr = np.array([per_lang_acc[lg] for lg in langs_sorted])
    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])
    corr_flip_resource = pearson(fi, rr)
    corr_flip_acc = pearson(fi, acc_arr)

    # Per-phenomenon flip concentration
    phen = {}
    for key in ["negation", "modality", "quantifier", "any"]:
        m = cues[key]
        fr_cued = float(non_unan[m].mean()) if m.any() else float("nan")
        fr_uncued = float(non_unan[~m].mean()) if (~m).any() else float("nan")
        a = int(non_unan[m].sum()); b = int((~non_unan[m]).sum())
        c = int(non_unan[~m].sum()); d = int((~non_unan[~m]).sum())
        chi2 = chi2_2x2(a, b, c, d)
        lo, hi = bootstrap_ci(lambda idx, mm=m, nu=non_unan: (nu[idx][mm[idx]].mean() - nu[idx][~mm[idx]].mean())
                              if mm[idx].any() and (~mm[idx]).any() else 0.0, N)
        phen[key] = {
            "n_cued": int(m.sum()), "flip_cued": fr_cued, "flip_uncued": fr_uncued,
            "risk_ratio": float(fr_cued / fr_uncued) if fr_uncued else float("nan"),
            "abs_diff": fr_cued - fr_uncued, "abs_diff_ci95": [lo, hi],
            "chi2_yates": chi2,  # df=1; crit 3.84 (p<.05), 6.63 (p<.01), 10.83 (p<.001)
        }

    # Bootstrap CI on headline flip rate and accuracy
    flip_ci = bootstrap_ci(lambda idx: non_unan[idx].mean(), N)
    acc_ci = bootstrap_ci(lambda idx: float((preds[idx] == gold[idx, None]).mean()), N)

    return {
        "tag": tag, "split": split, "n_items": N, "n_langs": L,
        "mean_per_lang_acc": mean_acc, "mean_acc_ci95": acc_ci,
        "per_lang_acc": per_lang_acc,
        "flip_rate(non_unanimous)": flip_rate, "flip_rate_ci95": flip_ci,
        "unanimity": unan, "mean_pairwise_agreement": mean_agree, "fleiss_kappa": kappa,
        "decomp": {"unanimous_correct": unan_correct, "unanimous_wrong": unan_wrong, "split": split_items},
        "maj_correct_on_split": maj_correct_on_split,
        "confidence": conf_summary,
        "per_lang_flip_involvement": flip_involve,
        "corr_flipinvolve_resource": corr_flip_resource,
        "corr_flipinvolve_accuracy": corr_flip_acc,
        "phenomena": phen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    split = args.split

    # English source for cue tagging (index-aligned across langs)
    prem_en, hyp_en, _ = xnli_load.load_en_source(split)
    cues = phenomena.tag_all(prem_en, hyp_en)
    cue_counts = {k: int(v.sum()) for k, v in cues.items()}
    n_en = len(prem_en)

    tags = sorted({os.path.basename(p).split("__")[1]
                   for p in glob.glob(os.path.join(RESULTS, f"preds__*__{split}.npz"))})
    print(f"split={split} models={tags} cue_counts={cue_counts} n={n_en}")

    out = {"split": split, "cue_counts": cue_counts, "n_items": n_en, "models": {}}
    for tag in tags:
        print(f"analyzing {tag} ...")
        out["models"][tag] = analyze_model(tag, split, cues)

    path = os.path.join(RESULTS, f"analysis__{split}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)
    # quick console summary
    for tag, r in out["models"].items():
        print(f"\n[{tag}] mean_acc={r['mean_per_lang_acc']:.3f}  "
              f"flip_rate={r['flip_rate(non_unanimous)']:.3f}  kappa={r['fleiss_kappa']:.3f}")
        for key in ["negation", "modality", "quantifier"]:
            p = r["phenomena"][key]
            print(f"    {key:10s} flip cued={p['flip_cued']:.3f} uncued={p['flip_uncued']:.3f} "
                  f"RR={p['risk_ratio']:.2f} chi2={p['chi2_yates']:.1f}")


if __name__ == "__main__":
    main()
