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
# the first year the search index covers, so the paraphrase test samples the same population
SEARCH_FIRST_YEAR = int(os.environ.get("SEARCH_FIRST_YEAR", "2010"))


def configured() -> bool:
    return bool(API_KEY)
