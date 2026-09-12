"""Generate a compact qualitative flip-examples table (paper/tables/tab_examples.tex)
from results/flip_examples.json. Illustrates confident single-language cross-lingual flips
(mDeBERTa): correct in 14 languages, confidently wrong in one."""
import os, json

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
TABDIR = os.path.join(HERE, "..", "paper", "tables")
SHOW_IDX = [331, 4280, 2260]  # negation examples with distinct dissenting languages (ru, ur, el)
ABBR = {"entail": "ent", "neutral": "neu", "contra": "con"}


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def clip(s, n):
    s = " ".join(s.split())
    return esc(s[:n] + ("\\ldots" if len(s) > n else ""))


def main():
    ex = {e["idx"]: e for e in json.load(open(os.path.join(RESULTS, "flip_examples.json")))}
    # full-width (table*) layout: ~16cm available, so text columns can breathe
    rows = [r"\begin{tabular}{p{6.6cm}p{4.6cm}ccc}", r"\toprule",
            r"Premise (English) & Hypothesis (English) & Gold & Cue & Dissent \\", r"\midrule"]
    for i in SHOW_IDX:
        if i not in ex:
            continue
        e = ex[i]
        diss = [(lg, e["preds"][lg]) for lg in e["preds"] if e["preds"][lg] != e["gold"]]
        diss_s = ", ".join(f"{lg}:\\textbf{{{ABBR.get(pr, pr)}}}" for lg, pr in diss)
        cue = "/".join(c[:3] for c in e["cue"]) if e["cue"] else "--"
        rows.append(f"{clip(e['premise_en'],86)} & {clip(e['hypothesis_en'],58)} & "
                    f"{ABBR.get(e['gold'], e['gold'])} & {cue} & {diss_s} \\\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABDIR, "tab_examples.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote paper/tables/tab_examples.tex with", len(SHOW_IDX), "examples")


if __name__ == "__main__":
    main()
