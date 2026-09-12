"""Surface concrete cross-lingual flip examples for qualitative analysis.
Finds items where the SAME reasoning problem gets different answers across languages,
prioritising cue-bearing, high-confidence flips where the majority is correct.
Outputs results/flip_examples.json (human-inspected; a few are chosen for the paper).
"""
import os, json
import numpy as np
import phenomena
import xnli_load

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
LANGS = ["ar","bg","de","el","en","es","fr","hi","ru","sw","th","tr","ur","vi","zh"]
LABEL = {0: "entail", 1: "neutral", 2: "contra"}
SHOW = ["en", "de", "fr", "ru", "zh", "ar", "hi", "sw", "ur", "th"]  # subset to display
MODEL = "mdeberta-base"


def main():
    d = np.load(os.path.join(RESULTS, f"preds__{MODEL}__test.npz"))
    probs, preds, gold = d["probs"], d["preds"].astype(int), d["gold"].astype(int)
    conf = probs.max(-1)
    N = len(gold)

    prem_en, hyp_en, _ = xnli_load.load_en_source("test")
    cues = phenomena.tag_all(prem_en, hyp_en)

    maj = np.array([np.bincount(row, minlength=3).argmax() for row in preds])
    non_unan = np.array([len(set(row)) > 1 for row in preds])

    cand = []
    for i in np.where(non_unan)[0]:
        row = preds[i]
        if maj[i] != gold[i]:
            continue  # want majority correct -> flip is the error
        minority = row != maj[i]
        n_flip = int(minority.sum())
        if n_flip < 1 or n_flip > 6:  # a few languages flip, not chaos
            continue
        # prioritise cue-bearing, confident flips
        cue = [k for k in ["negation", "modality", "quantifier"] if cues[k][i]]
        min_conf = float(conf[i, minority].mean())
        score = (2 if cue else 0) + min_conf + 0.1 * (6 - n_flip)
        cand.append((score, i, n_flip, cue, min_conf))

    cand.sort(reverse=True)
    out = []
    for score, i, n_flip, cue, min_conf in cand[:40]:
        row = preds[i]
        out.append({
            "idx": int(i), "gold": LABEL[int(gold[i])], "cue": cue,
            "n_flip": n_flip, "min_conf": round(min_conf, 3),
            "premise_en": prem_en[i], "hypothesis_en": hyp_en[i],
            "preds": {lg: LABEL[int(preds[i, j])] for j, lg in enumerate(LANGS)},
            "conf": {lg: round(float(conf[i, j]), 2) for j, lg in enumerate(LANGS)},
        })
    json.dump(out, open(os.path.join(RESULTS, "flip_examples.json"), "w", encoding="utf-8", newline="\n"), indent=2, ensure_ascii=False)
    print(f"wrote {len(out)} candidate flip examples")
    for e in out[:6]:
        flipped = [lg for lg in LANGS if e["preds"][lg] != LABEL[int(maj[e["idx"]])]]
        print(f"\nidx={e['idx']} gold={e['gold']} cue={e['cue']} flipped={flipped}")
        print("  P:", e["premise_en"][:90])
        print("  H:", e["hypothesis_en"][:90])
        print("  preds:", {lg: e["preds"][lg] for lg in SHOW})


if __name__ == "__main__":
    main()
