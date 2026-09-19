"""Settings for the venue-recommendation jobs, overridable through environment variables."""
import os

FIRST_YEAR = int(os.environ.get("VENUES_FIRST_YEAR", "2010"))   # older papers do not enter the store

# Years. None = derived from the dump: the test papers are the last complete year, scored with
# statistics from everything before it; the ranker learns on papers two years earlier, with
# statistics from before *that* year, so no paper ever contributes to the statistics it is scored by.
T_TEST = int(os.environ["VENUES_T_TEST"]) if os.environ.get("VENUES_T_TEST") else None
RANK_LAG = 2

# The class set at a snapshot: series with a body of work and recent activity.
MIN_SERIES_PAPERS = int(os.environ.get("VENUES_MIN_SERIES_PAPERS", "100"))
ACTIVE_YEARS = 3

# Text. Tokens are title words and word bigrams; a query keeps its rarest tokens, which carry the
# information, and bounds the scoring join.
MIN_DF = int(os.environ.get("VENUES_MIN_DF", "3"))
MAX_QUERY_TOKENS = int(os.environ.get("VENUES_MAX_QUERY_TOKENS", "12"))
KNN_MAX_DF = int(os.environ.get("VENUES_KNN_MAX_DF", "200000"))   # related-paper search skips commoner tokens
NB_ALPHA = 0.05

# Candidates per paper: the top of each content scorer, plus every series an author published in.
TOP_CONTENT = int(os.environ.get("VENUES_TOP_CONTENT", "20"))
MAX_HISTORY = int(os.environ.get("VENUES_MAX_HISTORY", "30"))

# Papers sampled for the ranker and for the test.
RANK_PAPERS = int(os.environ.get("VENUES_RANK_PAPERS", "150000"))
TEST_PAPERS = int(os.environ.get("VENUES_TEST_PAPERS", "50000"))

SEED = 7
