"""Check that regenerated paper macros and tables match the versions used in the paper.

code/reproduce_analysis.sh writes LaTeX macro and table files (every number reported in the
paper) into paper/, which is not distributed. This script compares their SHA256 with
results/expected_generated_outputs.json. Read-only.

    python check_generated.py

Exit status 0 when every listed file exists and matches, 1 otherwise.
"""
import os, sys, json, hashlib

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    manifest = json.load(open(os.path.join(ROOT, "results", "expected_generated_outputs.json")))
    match = mismatch = missing = 0
    for rel, expected in manifest["files"].items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing += 1
            print(f"MISSING  {rel} (run: bash reproduce_analysis.sh)")
            continue
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest == expected:
            match += 1
        else:
            mismatch += 1
            print(f"MISMATCH {rel}\n  expected {expected}\n  found    {digest}")
    print(f"generated outputs: {match} match, {mismatch} mismatch, {missing} missing "
          f"(of {len(manifest['files'])})")
    return 1 if (mismatch or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
