"""Settings for the ML jobs, overridable through environment variables."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))        # the VM's ~/dblp, mounted read-only
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))     # the api's serving databases (read-only)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))  # writable: model artifacts + metrics

DUCKDB_MEMORY = os.environ.get("DUCKDB_MEMORY", "6GB")
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "4"))

# Dataset caps. They bound the within-block pair explosion: a block like "Wei Wang" has 522
# labelled people, so all pairs of all their papers would be hundreds of millions.
MAX_PEOPLE_PER_BLOCK = int(os.environ.get("MAX_PEOPLE_PER_BLOCK", "25"))
MAX_PAPERS_PER_PERSON = int(os.environ.get("MAX_PAPERS_PER_PERSON", "6"))
MIN_PEOPLE_PER_BLOCK = int(os.environ.get("MIN_PEOPLE_PER_BLOCK", "3"))
MAX_BLOCKS = int(os.environ.get("MAX_BLOCKS", "4000"))
POS_PAIRS_PER_BLOCK = int(os.environ.get("POS_PAIRS_PER_BLOCK", "120"))
NEG_PAIRS_PER_BLOCK = int(os.environ.get("NEG_PAIRS_PER_BLOCK", "360"))

# Blocks are split by hash so the same block never appears in two splits.
TRAIN_BUCKETS = set(range(0, 7))   # 70%
VAL_BUCKETS = {7}                  # 10% - threshold tuning only
TEST_BUCKETS = {8, 9}              # 20% - reported metrics

SEED = 7
