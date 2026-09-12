"""Compute SHA256 provenance hashes for every immutable experimental artifact, so any
reported number can be traced to a fixed input. Categories:
  1. sampled XNLI subset      2. raw model predictions
  3. parsed predictions       4. final metrics
Writes logs/provenance.log (human-readable) and results/provenance.json (machine-readable).
Run AFTER metrics are computed but BEFORE any paper tables/figures are generated.
"""
import os, sys, json, glob, hashlib, platform
from datetime import datetime, timezone
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
LOGS = os.path.join(HERE, "..", "logs")
os.makedirs(LOGS, exist_ok=True)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(path)


def sha_array(arr):
    a = np.ascontiguousarray(arr)
    meta = f"{a.dtype.str}|{a.shape}".encode()
    return sha_bytes(meta + a.tobytes()), a.nbytes


def sha_json_canonical(obj):
    b = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return sha_bytes(b), len(b)


def record(entries, category, name, digest, nbytes, extra=None):
    e = {"category": category, "artifact": name, "sha256": digest, "bytes": int(nbytes)}
    if extra:
        e.update(extra)
    entries.append(e)


def main():
    entries = []

    # 1. sampled XNLI subset: the index file AND the immutable content it selects
    sub_path = os.path.join(RESULTS, "llm_subset.json")
    if os.path.exists(sub_path):
        d, n = sha_file(sub_path)
        record(entries, "sampled_subset", "llm_subset.json", d, n)
        # content hash of the actual sampled parallel items (ties hash to data, not just indices)
        try:
            import xnli_load
            sub = json.load(open(sub_path))
            data, gold = xnli_load.load_all("test")
            idx = sub["indices"]
            content = []
            for i in idx:
                content.append({"i": int(i), "gold": int(gold[i]),
                                "p": {lg: data[lg][0][i] for lg in xnli_load.LANGS},
                                "h": {lg: data[lg][1][i] for lg in xnli_load.LANGS}})
            dd, nn = sha_json_canonical({"seed": sub["seed"], "items": content})
            record(entries, "sampled_subset", "subset_content(300x15 parallel items)", dd, nn,
                   {"n_items": len(idx)})
        except Exception as e:
            record(entries, "sampled_subset", "subset_content", f"ERROR:{type(e).__name__}", 0)

    # 2/3. open-weight raw (probs) + parsed (preds) predictions, per npz
    for f in sorted(glob.glob(os.path.join(RESULTS, "preds__*.npz"))):
        name = os.path.basename(f)
        fd, fn = sha_file(f)
        record(entries, "prediction_file", name, fd, fn)
        z = np.load(f)
        if "probs" in z:  # open-weight: raw class probabilities
            d, n = sha_array(z["probs"])
            record(entries, "raw_predictions", f"{name}:probs", d, n, {"shape": list(z["probs"].shape)})
        if "preds" in z:  # parsed argmax labels (both open-weight & LLM)
            d, n = sha_array(z["preds"])
            record(entries, "parsed_predictions", f"{name}:preds", d, n, {"shape": list(z["preds"].shape)})

    # 2. LLM raw API responses (jsonl)
    for f in sorted(glob.glob(os.path.join(RESULTS, "llm_raw__*.jsonl"))):
        d, n = sha_file(f)
        record(entries, "raw_predictions", os.path.basename(f), d, n)

    # 4. final metrics
    for pat in ["analysis__*.json", "analysis_subset.json", "llm_models.json"]:
        for f in sorted(glob.glob(os.path.join(RESULTS, pat))):
            d, n = sha_file(f)
            record(entries, "final_metrics", os.path.basename(f), d, n)

    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for mod in ["torch", "transformers", "datasets"]:
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "n/a"

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "environment": env, "n_artifacts": len(entries), "artifacts": entries}
    json.dump(out, open(os.path.join(RESULTS, "provenance.json"), "w", encoding="utf-8", newline="\n"), indent=2)

    # human-readable log
    lines = [f"# Provenance (SHA256) — generated {out['generated_utc']}",
             f"# env: python={env['python']} torch={env.get('torch')} "
             f"transformers={env.get('transformers')} datasets={env.get('datasets')} numpy={env['numpy']}",
             f"# {len(entries)} artifacts", ""]
    by_cat = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    for cat in ["sampled_subset", "raw_predictions", "parsed_predictions", "final_metrics", "prediction_file"]:
        if cat not in by_cat:
            continue
        lines.append(f"## {cat}")
        for e in by_cat[cat]:
            lines.append(f"{e['sha256']}  {e['bytes']:>12d}  {e['artifact']}")
        lines.append("")
    with open(os.path.join(LOGS, "provenance.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"hashed {len(entries)} artifacts -> logs/provenance.log, results/provenance.json")
    for cat in ["sampled_subset", "raw_predictions", "parsed_predictions", "final_metrics"]:
        print(f"  {cat}: {len(by_cat.get(cat, []))} artifact(s)")


if __name__ == "__main__":
    main()
