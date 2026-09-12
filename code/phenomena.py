"""Pragmatic-cue tagging on the English XNLI source (the original, pre-translation text).

XNLI premises come from English MultiNLI and hypotheses were authored in English, then
professionally translated into 14 languages. So the 'en' config is the source; tagging cues
there defines the phenomenon for the index-aligned item in ALL languages. Cues are explicit
surface lexical markers -> transparent & reproducible, at the cost of recall (documented as a
limitation). We tag on premise+hypothesis combined.
"""
import re

NEGATION = {
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor",
    "cannot", "without", "hardly", "barely", "scarcely",
}
# contraction marker handled separately: any token containing "n't" (isn't, don't, won't...)
MODALITY = {
    "must", "should", "shall", "may", "might", "can", "could", "would", "ought",
}
MODALITY_MW = [  # multiword deontic/necessity markers
    "have to", "has to", "had to", "need to", "needs to", "needed to",
    "supposed to", "allowed to", "required to",
]
QUANTIFIER = {
    "all", "every", "each", "some", "any", "most", "many", "much", "few", "several",
    "both", "either", "none", "no", "everyone", "everything", "everybody",
    "someone", "something", "anybody", "anything", "nobody", "nothing",
}

_tok = re.compile(r"[a-z']+")


def _tokens(text):
    return _tok.findall(text.lower())


def tag(premise, hypothesis):
    """Return dict of bool flags for negation/modality/quantifier + 'any'."""
    text = (premise or "") + " " + (hypothesis or "")
    toks = _tokens(text)
    tokset = set(toks)
    low = text.lower()

    neg = bool(tokset & NEGATION) or any("n't" in t for t in toks)
    mod = bool(tokset & MODALITY) or any(mw in low for mw in MODALITY_MW)
    quant = bool(tokset & QUANTIFIER)
    return {
        "negation": neg,
        "modality": mod,
        "quantifier": quant,
        "any": neg or mod or quant,
    }


def tag_all(premises, hypotheses):
    import numpy as np
    keys = ["negation", "modality", "quantifier", "any"]
    out = {k: np.zeros(len(premises), dtype=bool) for k in keys}
    for i, (p, h) in enumerate(zip(premises, hypotheses)):
        t = tag(p, h)
        for k in keys:
            out[k][i] = t[k]
    return out


if __name__ == "__main__":
    tests = [
        ("The man is not happy.", "He is sad."),
        ("You must not take this with alcohol.", "Alcohol is allowed."),
        ("Some patients recovered.", "All patients recovered."),
        ("The cat sat on the mat.", "A cat is resting."),
    ]
    for p, h in tests:
        print(tag(p, h), "|", p, "/", h)
