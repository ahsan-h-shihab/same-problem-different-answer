"""Compare re-run open-weight predictions against the committed ones.

    python compare_predictions.py --candidate ../rerun

For each model, reports whether the argmax prediction matrix [5010 x 15] is identical and
the maximum absolute difference in class probabilities. Exit status 0 when every compared
prediction matrix is identical, 1 otherwise (probability differences at float precision are
reported but do not fail the comparison).
"""
import os, sys, argparse
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="directory with re-run preds__*__test.npz")
    ap.add_argument("--tags", nargs="+", default=["mdeberta-base", "minilm-l6"])
    args = ap.parse_args()

    failures = 0
    for tag in args.tags:
        ref = np.load(os.path.join(RESULTS, f"preds__{tag}__test.npz"))
        path = os.path.join(args.candidate, f"preds__{tag}__test.npz")
        if not os.path.exists(path):
            print(f"MISSING {path}")
            failures += 1
            continue
        cand = np.load(path)
        same = np.array_equal(ref["preds"], cand["preds"]) and np.array_equal(ref["gold"], cand["gold"])
        n_diff = int((ref["preds"] != cand["preds"]).sum())
        dprob = float(np.abs(ref["probs"] - cand["probs"]).max())
        failures += not same
        print(f"{'OK ' if same else 'FAIL'} {tag}: differing predictions={n_diff}, max |dprob|={dprob:.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
