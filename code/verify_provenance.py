"""Read-only verification of every SHA256 in results/provenance.json.

Recomputes each recorded hash from the files in results/ and checks that logs/provenance.log
lists the same hashes. Nothing is written. Hash definitions match hash_artifacts.py:
  - whole files:           SHA256 of the file bytes
  - "<file>.npz:<array>":  SHA256 of dtype|shape header + contiguous array bytes
  - subset content:        SHA256 of the canonical JSON of the 300 sampled items
                           (all 15 languages + gold), which requires the XNLI test data

    python verify_provenance.py                 # subset-content check SKIPPED if data absent
    python verify_provenance.py --require-data  # treat a missing XNLI copy as a failure

Exit status 0 when every checked hash matches (and, with --require-data, nothing was
skipped); 1 otherwise.
"""
import os, sys, re, json, hashlib, argparse
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
LOG = os.path.join(HERE, "..", "logs", "provenance.log")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(path)


def sha_array(arr):
    a = np.ascontiguousarray(arr)
    meta = f"{a.dtype.str}|{a.shape}".encode()
    return hashlib.sha256(meta + a.tobytes()).hexdigest(), a.nbytes


def sha_subset_content():
    import xnli_load
    sub = json.load(open(os.path.join(RESULTS, "llm_subset.json")))
    data, gold = xnli_load.load_all("test")
    items = [{"i": int(i), "gold": int(gold[i]),
              "p": {lg: data[lg][0][i] for lg in xnli_load.LANGS},
              "h": {lg: data[lg][1][i] for lg in xnli_load.LANGS}} for i in sub["indices"]]
    b = json.dumps({"seed": sub["seed"], "items": items}, sort_keys=True, ensure_ascii=True,
                   separators=(",", ":")).encode()
    return hashlib.sha256(b).hexdigest(), len(b)


def data_available():
    from xnli_load import _DATA
    return os.path.exists(os.path.join(_DATA, "test.parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-data", action="store_true")
    args = ap.parse_args()

    prov = json.load(open(os.path.join(RESULTS, "provenance.json")))
    match = mismatch = missing = skipped = 0
    for e in prov["artifacts"]:
        name = e["artifact"]
        try:
            if name.startswith("subset_content"):
                if not data_available():
                    skipped += 1
                    print(f"SKIP     {name}: XNLI test data not found (python fetch_xnli.py)")
                    continue
                digest, nbytes = sha_subset_content()
            elif ":" in name:
                fname, key = name.split(":", 1)
                digest, nbytes = sha_array(np.load(os.path.join(RESULTS, fname))[key])
            else:
                digest, nbytes = sha_file(os.path.join(RESULTS, name))
        except FileNotFoundError:
            missing += 1
            print(f"MISSING  {name}")
            continue
        if digest == e["sha256"] and nbytes == e["bytes"]:
            match += 1
        else:
            mismatch += 1
            print(f"MISMATCH {name}\n  recorded {e['sha256']} ({e['bytes']} bytes)\n"
                  f"  found    {digest} ({nbytes} bytes)")

    # the human-readable log must list exactly the same (hash, artifact) pairs
    log_pairs = set(re.findall(r"^([0-9a-f]{64})\s+\d+\s+(.+?)\s*$", open(LOG, encoding="utf-8").read(), re.M))
    json_pairs = {(e["sha256"], e["artifact"]) for e in prov["artifacts"]}
    log_ok = log_pairs == json_pairs
    if not log_ok:
        print(f"LOG MISMATCH logs/provenance.log vs results/provenance.json "
              f"({len(log_pairs ^ json_pairs)} differing entries)")

    total = len(prov["artifacts"])
    print(f"{match} match, {mismatch} mismatch, {missing} missing, {skipped} skipped "
          f"(of {total}); provenance.log {'consistent' if log_ok else 'INCONSISTENT'}")
    failed = mismatch or missing or not log_ok or (args.require_data and skipped)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
