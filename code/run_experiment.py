"""Run zero-shot multilingual NLI models over parallel XNLI, save per-item predictions.

Design principles:
- Reproducible: fixed model list, fixed lang order, deterministic inference, seed set.
- Traceable: saves full softmax probs + predicted XNLI label per (model, split, lang).
- Robust label mapping: reads each model's config.id2label and maps to XNLI order
  (0=entailment, 1=neutral, 2=contradiction).

Outputs (per model, per split) in results/:
  preds__{model_tag}__{split}.npz  with arrays:
     probs  : float32 [n_items, n_langs, 3]  (XNLI-ordered class probs)
     preds  : int8    [n_items, n_langs]     (argmax XNLI label)
     gold   : int8    [n_items]              (shared gold label)
     langs  : list[str] (json in meta)
  meta__{model_tag}__{split}.json
"""
import os, sys, json, time, argparse
# Prevent TensorFlow import (transformers auto-imports it if present -> minutes of overhead)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import xnli_load
from xnli_load import LANGS
from revisions import MODEL_REVISIONS

torch.set_num_threads(int(os.environ.get("NLI_THREADS", "6")))
torch.manual_seed(0)

RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

XNLI_ORDER = ["entailment", "neutral", "contradiction"]  # -> ids 0,1,2

MODELS = {
    "mdeberta-base":  "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "minilm-l6":      "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
}


def build_label_map(model):
    id2label = {int(k): v.lower().strip() for k, v in model.config.id2label.items()}
    lab2id = {v: k for k, v in id2label.items()}
    for name in XNLI_ORDER:
        if name not in lab2id:
            raise ValueError(f"Model label {name!r} missing; id2label={id2label}")
    # column permutation so that output[:, j] == prob of XNLI class j
    perm = [lab2id[name] for name in XNLI_ORDER]
    return perm


@torch.no_grad()
def infer(model, tok, premises, hyps, perm, batch_size=32, max_length=256):
    """Length-bucketed inference: sort by (approx) length to minimise padding waste
    on CPU, then scatter results back to original order (predictions are unchanged).
    max_length=256 covers 100% of XNLI pairs in all 15 languages (max observed 216 tokens on the full test set),
    so no language is disproportionately truncated (see check_truncation.py)."""
    n = len(premises)
    order = sorted(range(n), key=lambda i: len(premises[i]) + len(hyps[i]))
    out = np.zeros((n, 3), dtype=np.float32)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        p = [premises[i] for i in idx]
        h = [hyps[i] for i in idx]
        enc = tok(p, h, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # model order
        for k, i in enumerate(idx):
            out[i] = probs[k, perm]                            # -> XNLI order, original index
    return out


def run_model(tag, split, data, gold, batch_size):
    """Per-language checkpointing: after each language we persist a checkpoint so a crash /
    power loss resumes from the last completed language (loses <= one language of compute)."""
    name = MODELS[tag]
    n = len(gold)
    final = os.path.join(RESULTS, f"preds__{tag}__{split}.npz")
    if os.path.exists(final):
        print(f"[{tag}] {split}: final predictions already exist, skipping.", flush=True)
        return
    ckpt = os.path.join(RESULTS, f"ckpt__{tag}__{split}.npz")

    probs = np.zeros((n, len(LANGS), 3), dtype=np.float32)
    done = set()
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        if z["probs"].shape == probs.shape:
            probs = z["probs"]
            done = {int(x) for x in z["done"]}
            print(f"[{tag}] {split}: resuming, {len(done)}/{len(LANGS)} languages done.", flush=True)

    rev = MODEL_REVISIONS[name]
    id2label = {int(k): v for k, v in AutoConfig.from_pretrained(name, revision=rev).id2label.items()}
    if len(done) < len(LANGS):
        print(f"[{tag}] loading {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev)
        model.eval()
        perm = build_label_map(model)
        print(f"[{tag}] id2label={model.config.id2label} perm={perm}", flush=True)
        for li, lg in enumerate(LANGS):
            if li in done:
                continue
            t0 = time.time()
            pr, hy = data[lg]
            probs[:, li, :] = infer(model, tok, pr, hy, perm, batch_size=batch_size)
            done.add(li)
            np.savez_compressed(ckpt, probs=probs, done=np.array(sorted(done)))  # checkpoint
            dt = time.time() - t0
            acc = float((probs[:, li, :].argmax(-1) == gold).mean())
            print(f"[{tag}] {split} {lg}: acc={acc:.3f}  {n} items in {dt:.0f}s ({n/dt:.1f} it/s)", flush=True)
        del model, tok

    preds = probs.argmax(-1).astype(np.int8)
    np.savez_compressed(final, probs=probs, preds=preds, gold=gold)
    meta = {
        "model": name, "revision": rev, "tag": tag, "split": split, "langs": LANGS,
        "n_items": int(n), "xnli_order": XNLI_ORDER, "id2label": id2label,
        "per_lang_acc": {lg: float((preds[:, i] == gold).mean()) for i, lg in enumerate(LANGS)},
    }
    with open(os.path.join(RESULTS, f"meta__{tag}__{split}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    if os.path.exists(ckpt):
        os.remove(ckpt)  # checkpoint no longer needed once final is written
    print(f"[{tag}] saved. mean acc={np.mean(list(meta['per_lang_acc'].values())):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    for split in args.splits:
        print(f"=== loading XNLI split={split} ===", flush=True)
        t0 = time.time()
        data, gold = xnli_load.load_all(split)
        print(f"loaded {len(gold)} parallel items x {len(LANGS)} langs in {time.time()-t0:.0f}s", flush=True)
        for tag in args.models:
            run_model(tag, split, data, gold, args.batch_size)


if __name__ == "__main__":
    main()
