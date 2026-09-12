"""Emit LaTeX tables (booktabs) into paper/tables/ from results/analysis__{split}.json."""
import os, json, argparse

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
TABDIR = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(TABDIR, exist_ok=True)

LANGS = ["ar","bg","de","el","en","es","fr","hi","ru","sw","th","tr","ur","vi","zh"]
MODEL_ORDER = ["minilm-l6", "mdeberta-base", "xlmr-large"]
MODEL_LABEL = {"minilm-l6": "MiniLM-L6", "mdeberta-base": "mDeBERTa-base", "xlmr-large": "XLM-R-large"}


def present(A):
    return [m for m in MODEL_ORDER if m in A["models"]]


def p(x, d=1):
    return f"{100*x:.{d}f}"


def table_consistency(A, split):
    ms = present(A)
    rows = []
    rows.append(r"\begin{tabular}{@{}lrrrrr@{}}")
    rows.append(r"\toprule")
    rows.append(r"Model & Acc. & Agree & Unan. & $\kappa$ & Flip \\")
    rows.append(r"\midrule")
    for m in ms:
        r = A["models"][m]
        rows.append(f"{MODEL_LABEL[m]} & {p(r['mean_per_lang_acc'])} & "
                    f"{p(r['mean_pairwise_agreement'])} & {p(r['unanimity'])} & "
                    f"{r['fleiss_kappa']:.2f} & {p(r['flip_rate(non_unanimous)'])} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    open(os.path.join(TABDIR, f"tab_consistency__{split}.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")


def table_phenomena(A, split):
    ms = present(A)
    rows = [r"\begin{tabular}{@{}ll" + "r" * (len(ms)) + "@{}}", r"\toprule"]
    rows.append("Phenomenon & & " + " & ".join(MODEL_LABEL[m] for m in ms) + r" \\")
    rows.append(r"\midrule")
    for key, disp in [("negation", "Negation"), ("modality", "Modality"), ("quantifier", "Quantifier")]:
        cued = " & ".join(p(A["models"][m]["phenomena"][key]["flip_cued"]) for m in ms)
        unc = " & ".join(p(A["models"][m]["phenomena"][key]["flip_uncued"]) for m in ms)
        rr = " & ".join(f"{A['models'][m]['phenomena'][key]['risk_ratio']:.2f}" for m in ms)
        rows.append(f"{disp} & cue present & {cued} \\\\")
        rows.append(f" & cue absent & {unc} \\\\")
        rows.append(f" & risk ratio & {rr} \\\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR, f"tab_phenomena__{split}.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")


def table_perlang(A, split):
    ms = present(A)
    rows = [r"\begin{tabular}{l" + "r" * (len(ms) + 1) + "}", r"\toprule"]
    rows.append("Lang & " + " & ".join(MODEL_LABEL[m] for m in ms) + r" & Flip inv. \\")
    rows.append(r"\midrule")
    main = ms[-1] if ms else None
    for lg in LANGS:
        accs = " & ".join(p(A["models"][m]["per_lang_acc"][lg]) for m in ms)
        fi = p(A["models"][main]["per_lang_flip_involvement"][lg]) if main else ""
        rows.append(f"{lg} & {accs} & {fi} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TABDIR, f"tab_perlang__{split}.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    A = json.load(open(os.path.join(RESULTS, f"analysis__{args.split}.json")))
    table_consistency(A, args.split)
    table_phenomena(A, args.split)
    table_perlang(A, args.split)
    print("tables written to", TABDIR)


if __name__ == "__main__":
    main()
