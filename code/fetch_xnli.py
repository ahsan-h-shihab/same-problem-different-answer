"""Fetch the pinned XNLI test parquet into data/xnli/ and verify its SHA256.

XNLI is not redistributed in this repository. This script downloads the exact file used for
every reported result from the Hugging Face Hub (facebook/xnli at a pinned commit).

    python fetch_xnli.py            # download (if missing) and verify
    python fetch_xnli.py --check    # verify an existing copy only; never downloads

Exit status 0 when data/xnli/test.parquet exists and matches the pinned SHA256, 1 otherwise.
Set XNLI_DATA_DIR to use a different directory.
"""
import os, sys, shutil, hashlib, argparse
from revisions import XNLI_REPO, XNLI_REVISION, XNLI_TEST_FILE, XNLI_TEST_SHA256

DATA = os.environ.get("XNLI_DATA_DIR") or os.path.join(os.path.dirname(__file__), "..", "data", "xnli")
TARGET = os.path.join(DATA, "test.parquet")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not download")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        if args.check:
            print(f"MISSING {TARGET} (run: python fetch_xnli.py)")
            return 1
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(repo_id=XNLI_REPO, filename=XNLI_TEST_FILE,
                              repo_type="dataset", revision=XNLI_REVISION)
        os.makedirs(DATA, exist_ok=True)
        shutil.copyfile(src, TARGET)
        print(f"downloaded {XNLI_REPO}@{XNLI_REVISION}:{XNLI_TEST_FILE}")

    digest = sha256(TARGET)
    if digest != XNLI_TEST_SHA256:
        print(f"SHA256 MISMATCH {TARGET}\n  expected {XNLI_TEST_SHA256}\n  found    {digest}")
        return 1
    print(f"OK {TARGET} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
