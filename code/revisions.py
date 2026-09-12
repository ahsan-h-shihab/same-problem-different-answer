"""Pinned upstream revisions for every reported result.

The model and dataset revisions are the Hugging Face commit hashes of the snapshots used for
the experiments. XNLI_TEST_SHA256 is the SHA256 of the exact test parquet file that was read.
"""

XNLI_REPO = "facebook/xnli"
XNLI_REVISION = "b8dd5d7af51114dbda02c0e3f6133f332186418e"
XNLI_TEST_FILE = "all_languages/test-00000-of-00001.parquet"
XNLI_TEST_SHA256 = "599e3a0191403f19cbe802afdf69841152000b41eaed725e4f463d432c0ffb49"
XNLI_TEST_ROWS = 5010

MODEL_REVISIONS = {
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli": "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c",
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli": "0a71e92a985b6e1ad1828cf67ce9c459639c1dca",
}
