"""Measure tokenized premise+hypothesis lengths on the FULL XNLI test set, per language, for
both open-weight models, to confirm that max_length=256 truncates no pair in any language.

    python check_truncation.py

Prints per-language max length and the number of pairs longer than 256 tokens, then a
summary. Exit status 0 when no pair exceeds 256 tokens for either model, 1 otherwise.
"""
import os, sys
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np
from transformers import AutoTokenizer
import xnli_load
from xnli_load import LANGS
from revisions import MODEL_REVISIONS

MAX_LENGTH = 256


def main():
    data, gold = xnli_load.load_all("test")
    any_over = False
    for name, rev in MODEL_REVISIONS.items():
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        print(f"\n{name}@{rev[:8]}")
        print(f"{'lang':4s} {'p50':>5s} {'p99':>5s} {'max':>5s} {'>256':>5s}")
        overall_max, overall_where, n_over = 0, None, 0
        for lg in LANGS:
            enc = tok(data[lg][0], data[lg][1], truncation=False)["input_ids"]
            lens = np.array([len(x) for x in enc])
            over = int((lens > MAX_LENGTH).sum())
            n_over += over
            if lens.max() > overall_max:
                overall_max, overall_where = int(lens.max()), (lg, int(lens.argmax()))
            print(f"{lg:4s} {int(np.percentile(lens, 50)):5d} {int(np.percentile(lens, 99)):5d} "
                  f"{int(lens.max()):5d} {over:5d}")
        print(f"pairs={len(gold) * len(LANGS)}  longest={overall_max} tokens "
              f"(lang={overall_where[0]}, item={overall_where[1]})  pairs>{MAX_LENGTH}={n_over}")
        any_over = any_over or n_over > 0
    return 1 if any_over else 0


if __name__ == "__main__":
    sys.exit(main())
