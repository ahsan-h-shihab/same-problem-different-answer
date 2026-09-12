"""Generate publication figures from results/analysis__{split}.json (+ preds for confidence).
Every value is read from committed result files; nothing is recomputed ad hoc here.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
# Colorblind-safe (Okabe-Ito subset)
C = {"acc": "#0072B2", "unan": "#D55E00", "agree": "#009E73",
     "cued": "#D55E00", "uncued": "#56B4E9", "pt": "#0072B2"}

MODEL_ORDER = ["minilm-l6", "mdeberta-base", "xlmr-large"]
MODEL_LABEL = {"minilm-l6": "MiniLM-L6\n(small)", "mdeberta-base": "mDeBERTa-base",
               "xlmr-large": "XLM-R-large"}
LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
RESOURCE_RANK = {"en":15,"ru":14,"de":13,"fr":12,"es":11,"zh":10,"vi":9,"th":8,
                 "tr":7,"ar":6,"bg":5,"el":4,"hi":3,"ur":2,"sw":1}


def load(split):
    with open(os.path.join(RESULTS, f"analysis__{split}.json")) as f:
        return json.load(f)


def present_models(A):
    return [m for m in MODEL_ORDER if m in A["models"]]


def fig1_acc_vs_consistency(A, split):
    models = present_models(A)
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    x = np.arange(len(models)); w = 0.26
    acc = [A["models"][m]["mean_per_lang_acc"] for m in models]
    unan = [A["models"][m]["unanimity"] for m in models]
    agree = [A["models"][m]["mean_pairwise_agreement"] for m in models]
    ax.bar(x - w, acc, w, label="Mean per-language accuracy", color=C["acc"])
    ax.bar(x, agree, w, label="Mean cross-lingual agreement", color=C["agree"])
    ax.bar(x + w, unan, w, label="Unanimity (all 15 agree)", color=C["unan"])
    for xi, a, g, u in zip(x, acc, agree, unan):
        for dx, v in [(-w, a), (0, g), (w, u)]:
            ax.text(xi + dx, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)
    fig.savefig(os.path.join(FIGDIR, f"fig1_acc_vs_consistency__{split}.pdf"))
    plt.close(fig)


def fig2_phenomena(A, split, model="mdeberta-base"):
    if model not in A["models"]:
        model = present_models(A)[-1]
    ph = A["models"][model]["phenomena"]
    keys = ["negation", "modality", "quantifier"]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    x = np.arange(len(keys)); w = 0.34
    cued = [ph[k]["flip_cued"] for k in keys]
    unc = [ph[k]["flip_uncued"] for k in keys]
    # bootstrap CI on the difference is stored; show cued/uncued bars, annotate RR & chi2
    ax.bar(x - w/2, cued, w, label="cue present", color=C["cued"])
    ax.bar(x + w/2, unc, w, label="cue absent", color=C["uncued"])
    # value labels inside the bars: unambiguous and needs no extra vertical space
    for xi, v in list(zip(x - w/2, cued)) + list(zip(x + w/2, unc)):
        ax.text(xi, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=6.5, color="white")
    for xi, k in zip(x, keys):
        rr = ph[k]["risk_ratio"]; chi2 = ph[k]["chi2_yates"]
        top = max(ph[k]["flip_cued"], ph[k]["flip_uncued"])
        star = "***" if chi2 > 10.83 else "**" if chi2 > 6.63 else "*" if chi2 > 3.84 else "n.s."
        ax.text(xi, top + 0.012, f"RR={rr:.2f} ({star})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([k.capitalize() for k in keys])
    ax.set_ylabel("Cross-lingual flip rate")
    # tight headroom: room for the RR annotations, no dead whitespace
    ax.set_ylim(0, max(max(cued), max(unc)) * 1.16)
    # legend ABOVE the axes -> cannot collide with the per-group annotations.
    # (model identity is stated in the caption, so the in-axes title is redundant)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              handlelength=1.4, columnspacing=1.8)
    fig.savefig(os.path.join(FIGDIR, f"fig2_phenomena__{split}.pdf"))
    plt.close(fig)


def fig3_resource(A, split, model="mdeberta-base"):
    if model not in A["models"]:
        model = present_models(A)[-1]
    fi = A["models"][model]["per_lang_flip_involvement"]
    acc = A["models"][model]["per_lang_acc"]
    r = A["models"][model]["corr_flipinvolve_resource"]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    xs = [RESOURCE_RANK[lg] for lg in LANGS]
    ys = [fi[lg] for lg in LANGS]
    ax.scatter(xs, ys, c=C["pt"], s=18)
    for lg in LANGS:
        ax.annotate(lg, (RESOURCE_RANK[lg], fi[lg]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    # trend line
    b, a = np.polyfit(xs, ys, 1)
    xx = np.array([min(xs), max(xs)])
    ax.plot(xx, a + b * xx, color=C["unan"], lw=1, ls="--")
    ax.set_xlabel("Pretraining-resource rank (1=low, 15=high)")
    ax.set_ylabel("Flip involvement\n(disagree w/ majority)")
    ax.set_title(f"r = {r:.2f}", fontsize=8)
    fig.savefig(os.path.join(FIGDIR, f"fig3_resource__{split}.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    A = load(args.split)
    fig1_acc_vs_consistency(A, args.split)
    fig2_phenomena(A, args.split)
    fig3_resource(A, args.split)
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
