"""Settings for the chat service, overridable through environment variables."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))      # the api's serving databases (read-only)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))   # writable: leaderboard store, budget ledger
TMP_DIR = Path(os.environ.get("CHAT_TMP_DIR", str(MODELS_DIR / "tmp")))

DUCKDB_MEMORY = os.environ.get("DUCKDB_MEMORY", "2GB")
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "2"))

# ---- the language model -----------------------------------------------------
# Any OpenAI-compatible /chat/completions endpoint works (OpenAI, Azure, a local vLLM, ...), so the
# provider is a base URL rather than a code path. Tool calling is the only capability required.
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# The router/answerer. The deep model is used only for the final answer of a multi-step question,
# where the cheap model's synthesis is the weak link.
MODEL_FAST = os.environ.get("CHAT_MODEL_FAST", "gpt-4.1-mini")
MODEL_DEEP = os.environ.get("CHAT_MODEL_DEEP", "gpt-4.1")
ESCALATE_AFTER_ROUNDS = int(os.environ.get("CHAT_ESCALATE_AFTER_ROUNDS", "2"))
TEMPERATURE = float(os.environ.get("CHAT_TEMPERATURE", "0"))

# Per 1M tokens, only for reporting a request's cost in the UI; set to your contract's numbers.
PRICE_IN = {MODEL_FAST: float(os.environ.get("CHAT_PRICE_FAST_IN", "0.40")),
            MODEL_DEEP: float(os.environ.get("CHAT_PRICE_DEEP_IN", "2.00"))}
PRICE_OUT = {MODEL_FAST: float(os.environ.get("CHAT_PRICE_FAST_OUT", "1.60")),
             MODEL_DEEP: float(os.environ.get("CHAT_PRICE_DEEP_OUT", "8.00"))}

# ---- limits -----------------------------------------------------------------
MAX_QUESTION_CHARS = int(os.environ.get("CHAT_MAX_QUESTION_CHARS", "400"))
MAX_HISTORY_TURNS = int(os.environ.get("CHAT_MAX_HISTORY_TURNS", "6"))
MAX_ROUNDS = int(os.environ.get("CHAT_MAX_ROUNDS", "3"))          # tool rounds before answering anyway
MAX_TOOL_CALLS = int(os.environ.get("CHAT_MAX_TOOL_CALLS", "8"))   # per question, across rounds
TIME_BUDGET_SECONDS = float(os.environ.get("CHAT_TIME_BUDGET_SECONDS", "25"))
TOOL_TIMEOUT = float(os.environ.get("CHAT_TOOL_TIMEOUT", "10"))
SQL_TIMEOUT = float(os.environ.get("CHAT_SQL_TIMEOUT", "5"))
SQL_ROW_CAP = int(os.environ.get("CHAT_SQL_ROW_CAP", "200"))
ROWS_TO_MODEL = int(os.environ.get("CHAT_ROWS_TO_MODEL", "25"))    # a tool shows the model this many rows
LLM_TIMEOUT = float(os.environ.get("CHAT_LLM_TIMEOUT", "40"))
# a rate limit means "wait", not "no": retried, honouring the provider's own reset hint
RATE_RETRIES = int(os.environ.get("CHAT_RATE_RETRIES", "4"))
RATE_WAIT_MAX = float(os.environ.get("CHAT_RATE_WAIT_MAX", "15"))

# ---- access -----------------------------------------------------------------
# A shared token, because every question spends money. Unset (the default in dev) means open.
TOKEN = os.environ.get("CHAT_TOKEN", "")
RATE_PER_MINUTE = int(os.environ.get("CHAT_RATE_PER_MINUTE", "6"))
RATE_PER_DAY = int(os.environ.get("CHAT_RATE_PER_DAY", "200"))
# the terminal has its own allowance: an evaluation run is 57 questions, and exhausting the page's
# allowance from a maintenance task is a self-inflicted outage
RATE_PER_DAY_CLI = int(os.environ.get("CHAT_RATE_PER_DAY_CLI", "1000"))
BUDGET_USD_PER_DAY = float(os.environ.get("CHAT_BUDGET_USD_PER_DAY", "5.00"))

# ---- the other services -----------------------------------------------------
SEARCH_URL = os.environ.get("CHAT_SEARCH_URL", "http://searchapi:8003").rstrip("/")
# the dashboard's own api: used only by the QA harness, as the reference for any number
# the assistant also computes
DASHBOARD_URL = os.environ.get("CHAT_DASHBOARD_URL", "http://api:8000").rstrip("/")
ML_URL = os.environ.get("CHAT_ML_URL", "http://mlapi:8001").rstrip("/")
UPSTREAM_TIMEOUT = float(os.environ.get("CHAT_UPSTREAM_TIMEOUT", "20"))
# a local Ollama container on the compose network, for running open models (experiments only)
OLLAMA_URL = os.environ.get("CHAT_OLLAMA_URL", "http://ollama:11434/v1").rstrip("/")

# the first year the search index covers, so the paraphrase test samples the same population
SEARCH_FIRST_YEAR = int(os.environ.get("SEARCH_FIRST_YEAR", "2010"))

# ---- what papers say (content.py) ---------------------------------------------
# dblp has titles, not abstracts: questions about a paper's content are answered from abstracts
# OpenAlex returns, found the way the DBLP-QA study found best. CHAT_CONTENT_TOOL=0 turns the tool
# off and puts back the old rule that such questions are declined.
CONTENT_TOOL = os.environ.get("CHAT_CONTENT_TOOL", "1") != "0"
OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
# the free key allows $1 a day: a search costs $0.001, a lookup by DOI $0.0001
OPENALEX_CALLS_PER_DAY = int(os.environ.get("CHAT_OPENALEX_CALLS_PER_DAY", "1400"))
# OpenAlex's semantic search sometimes takes ten seconds: the abstract search gets a longer limit than
# other tools, and still answers in about five when the services are quick
CONTENT_DEADLINE = float(os.environ.get("CHAT_CONTENT_DEADLINE", "14"))
CONTENT_TOOL_TIMEOUT = float(os.environ.get("CHAT_CONTENT_TOOL_TIMEOUT", "15"))
CONTENT_POOL = 50                 # results asked of each search, as in the study
CONTENT_TOP = 5                   # abstracts given to the model, as in the study
CONTENT_ABSTRACT_CHARS = 2000
CONTENT_MAX_CHARS = 16000         # what the model reads of the tool's result (other tools: 6,000)
CONTENT_CALLS_PER_QUESTION = 2
CONTENT_CACHE_DAYS = 30
# Version 2 of the content path, from the failure analysis of version 1 on DBLP-QA and Fresh: the rule
# that never declines a research question before searching and searches a second time with the
# question's distinctive terms (2; 1 is the first rule); abstracts from Semantic Scholar and Crossref for
# the best candidates OpenAlex has none for; and a stronger model writing answers built on abstracts
# ("" leaves them to the router). The evaluation switches these per variant to measure each version.
CONTENT_RULE_VERSION = int(os.environ.get("CHAT_CONTENT_RULE_VERSION", "2"))
CONTENT_FALLBACK = os.environ.get("CHAT_CONTENT_FALLBACK", "1") != "0"
CONTENT_WRITER = os.environ.get("CHAT_CONTENT_WRITER", MODEL_DEEP)
# Version 3: Dewey's own abstract index (abstractindex.py) - the abstracts of dblp's papers from
# OpenAlex's snapshot, searched with BM25 beside the live searches and the first place an abstract is
# looked for by key. CHAT_CONTENT_SEARCH=local leaves out every outside service (the index and dblp's
# own title search only): the closed world the evaluation sets beside RAGScholar's.
CONTENT_LOCAL_INDEX = os.environ.get("CHAT_CONTENT_LOCAL_INDEX", "1") != "0"
CONTENT_SEARCH = os.environ.get("CHAT_CONTENT_SEARCH", "live")
# how the pool is ordered: "pool" = BM25 with the pool's own statistics, as the study did; "rrf" = that
# fused with the index's whole-corpus order (reciprocal rank fusion), when the index took part
CONTENT_FUSION = os.environ.get("CHAT_CONTENT_FUSION", "pool")


def configured() -> bool:
    return bool(API_KEY)
