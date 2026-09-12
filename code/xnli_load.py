"""XNLI loader using the `all_languages` config, which stores the TRUE parallel structure:
each row is ONE item carrying all 15 language versions (premise as a Translation dict,
hypothesis as TranslationVariableLanguages) plus ONE shared gold label. This guarantees
index alignment across languages -- the per-language configs are NOT row-aligned.

Label order (ClassLabel): 0=entailment, 1=neutral, 2=contradiction (== model/XNLI order).
Only the requested split's parquet is read (train is never downloaded).
"""
import os
import numpy as np
from datasets import load_dataset
from revisions import XNLI_REPO, XNLI_REVISION

LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
_DATA = os.environ.get("XNLI_DATA_DIR") or os.path.join(os.path.dirname(__file__), "..", "data", "xnli")


def _load(split):
    """Prefer the local parquet copy (see fetch_xnli.py); fall back to the pinned Hub revision."""
    fname = "validation" if split in ("validation", "dev") else split
    local = os.path.join(_DATA, f"{fname}.parquet")
    src = local if os.path.exists(local) else \
        f"hf://datasets/{XNLI_REPO}@{XNLI_REVISION}/all_languages/{fname}-00000-of-00001.parquet"
    return load_dataset("parquet", data_files={fname: src}, split=fname)


def _hyp_map(hy):
    """hypothesis is {'language': [...], 'translation': [...]} -> {lang: text}."""
    return dict(zip(hy["language"], hy["translation"]))


def load_all(split):
    """Return (data, gold): data[lang] = (premises, hypotheses) index-aligned; gold shared."""
    ds = _load(split)
    gold = np.array(ds["label"], dtype=np.int8)
    prem_col = list(ds["premise"])       # list of {lang: text}
    hyp_col = list(ds["hypothesis"])     # list of {'language':[], 'translation':[]}
    premises = {lg: [] for lg in LANGS}
    hyps = {lg: [] for lg in LANGS}
    for pr, hy in zip(prem_col, hyp_col):
        hm = _hyp_map(hy)
        for lg in LANGS:
            premises[lg].append(pr[lg])
            hyps[lg].append(hm[lg])
    data = {lg: (premises[lg], hyps[lg]) for lg in LANGS}
    return data, gold


def load_en_source(split):
    """English source (premise, hypothesis) + gold, in the SAME row order as load_all."""
    ds = _load(split)
    gold = np.array(ds["label"], dtype=np.int8)
    prem = [p["en"] for p in list(ds["premise"])]
    hyp = [_hyp_map(h)["en"] for h in list(ds["hypothesis"])]
    return prem, hyp, gold
