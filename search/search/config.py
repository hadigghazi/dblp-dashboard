"""Settings for the search service, overridable through environment variables."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
TMP_DIR = Path(os.environ.get("SEARCH_TMP_DIR", str(MODELS_DIR / "tmp")))
DUCKDB_MEMORY = os.environ.get("DUCKDB_MEMORY", "4GB")
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "4"))

# Same population journal/conference recommendation scores: recent enough to matter, real venue.
# Older papers and other record kinds (preprints, theses, books, ...) fall back to exact-word search.
FIRST_YEAR = int(os.environ.get("SEARCH_FIRST_YEAR", "2010"))

MODEL_NAME = os.environ.get("SEARCH_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("SEARCH_EMBED_DIM", "384"))
# bge models are trained to expect this instruction on the query side only; the title side gets none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ENCODE_BATCH = int(os.environ.get("SEARCH_ENCODE_BATCH", "256"))
ENCODE_THREADS = int(os.environ.get("SEARCH_ENCODE_THREADS", "8"))
CHECKPOINT_EVERY = int(os.environ.get("SEARCH_CHECKPOINT_EVERY", "40"))   # batches between flushes

# Vectors are memory-mapped, not loaded whole into RAM: the process footprint stays small on a VM
# shared with other services, at the cost of a page-fault on first touch of each region.
VECTOR_DTYPE = "float16"

TOP_SPARSE = int(os.environ.get("SEARCH_TOP_SPARSE", "300"))
TOP_DENSE = int(os.environ.get("SEARCH_TOP_DENSE", "2000"))   # over-fetched, then filtered by kind/year
MAX_QUERY_TOKENS = int(os.environ.get("SEARCH_MAX_QUERY_TOKENS", "16"))
RRF_K = 60   # standard reciprocal-rank-fusion constant; results are not sensitive to small changes

MIN_DF = int(os.environ.get("SEARCH_MIN_DF", "2"))
BM25_K1 = 1.5
BM25_B = 0.75

EVAL_PAPERS = int(os.environ.get("SEARCH_EVAL_PAPERS", "1500"))   # each does a full dense pass; ~15 min at this size

SEED = 7
