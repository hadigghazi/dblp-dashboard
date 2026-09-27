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

# A local sentence-transformers model, or "openai:<model>" for the API - the two implement the
# same two methods, so nothing downstream knows which one produced a vector.
MODEL_NAME = os.environ.get("SEARCH_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("SEARCH_EMBED_DIM", "384"))
# bge models are trained to expect this instruction on the query side only; the title side gets none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ENCODE_BATCH = int(os.environ.get("SEARCH_ENCODE_BATCH", "256"))
ENCODE_THREADS = int(os.environ.get("SEARCH_ENCODE_THREADS", "8"))

# The API path. It is latency-bound rather than CPU-bound, so a batch handed down by the build
# loop is split into requests that fly together.
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_REQUEST_SIZE = int(os.environ.get("SEARCH_API_REQUEST_SIZE", "256"))
API_CONCURRENCY = int(os.environ.get("SEARCH_API_CONCURRENCY", "6"))
# The account's own tokens-per-minute limit, with headroom. Requests wait for room in this
# budget before they are sent: a rate limit is not something retries can get around.
API_TOKENS_PER_MINUTE = int(os.environ.get("SEARCH_API_TPM", "900000"))
API_TIMEOUT = float(os.environ.get("SEARCH_API_TIMEOUT", "120"))
API_RETRIES = int(os.environ.get("SEARCH_API_RETRIES", "8"))
CHECKPOINT_EVERY = int(os.environ.get("SEARCH_CHECKPOINT_EVERY", "40"))   # batches between flushes

# Vectors are memory-mapped, not loaded whole into RAM: the process footprint stays small on a VM
# shared with other services, at the cost of a page-fault on first touch of each region.
VECTOR_DTYPE = "float16"

TOP_SPARSE = int(os.environ.get("SEARCH_TOP_SPARSE", "300"))
TOP_DENSE = int(os.environ.get("SEARCH_TOP_DENSE", "2000"))   # over-fetched, then filtered by kind/year
MAX_QUERY_TOKENS = int(os.environ.get("SEARCH_MAX_QUERY_TOKENS", "16"))
# A title query's rarest tokens are rare, so BM25 touches few postings. A description ("papers about
# making transformers cheaper to run") can consist entirely of common words, where even the rarest
# sixteen are each held by hundreds of thousands of papers and the join becomes millions of rows.
# Bounding the postings, rather than the token count, keeps a long query as cheap as a short one.
MAX_POSTINGS = int(os.environ.get("SEARCH_MAX_POSTINGS", "3000000"))
MIN_QUERY_TOKENS = 3

# Reciprocal rank fusion gives both rankings equal say. That is right when both have something to
# say, and wrong when one does not: on a paraphrase query BM25 returns noise, and blending it in
# cost every top-1 hit the embeddings had found.
# Measured on the match, not on the query: how much of what you typed the best word match
# explains. A title with one word swapped leaves a hit covering most of the query; a
# description of the same paper leaves a hit covering two tokens out of a dozen.
COVERAGE_FLOOR = float(os.environ.get("SEARCH_COVERAGE_FLOOR", "0.5"))
COVERAGE_FULL = float(os.environ.get("SEARCH_COVERAGE_FULL", "0.7"))
# Zero, not a small number. Measured on paraphrase queries: the word ranking found the right paper
# 0 times in 56, at any rank - so it is not weak evidence there, it is no evidence. At a tenth of a
# vote a word hit at rank 1 still outranks a semantic hit at rank 100, which is where the right
# paper sits when somebody describes it. The exact-word top-up still guarantees recall.
SPARSE_FLOOR = float(os.environ.get("SEARCH_SPARSE_FLOOR", "0.0"))
RRF_K = 60   # standard reciprocal-rank-fusion constant; results are not sensitive to small changes


def ranker_version():
    """Everything that decides an answer's ORDER. The disk cache keys on this, so changing a
    threshold invalidates the answers it produced - without it, the next measurement silently scores
    the previous ranker."""
    return f"{RRF_K}:{COVERAGE_FLOOR}:{COVERAGE_FULL}:{SPARSE_FLOOR}:{MAX_QUERY_TOKENS}:{MAX_POSTINGS}"

MIN_DF = int(os.environ.get("SEARCH_MIN_DF", "2"))
BM25_K1 = 1.5
BM25_B = 0.75

EVAL_PAPERS = int(os.environ.get("SEARCH_EVAL_PAPERS", "1500"))   # each does a full dense pass; ~15 min at this size

SEED = 7
