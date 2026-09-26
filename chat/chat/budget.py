"""
Spend and rate limits.

The page is gated by a token, but a token can leak and a loop can be written by accident, so the
service also refuses to spend beyond a daily ceiling. The ledger is a small JSON file next to the
models, written after every request, so a restart does not reset the day's spend.

Both limits fail closed: if the ledger cannot be read, the request is refused rather than served for
free.
"""
import json
import logging
import threading
import time
from collections import deque
from datetime import date

from . import config

log = logging.getLogger("dblp.chat.budget")


def _price(model, tokens_in, tokens_out):
    pin = config.PRICE_IN.get(model, config.PRICE_IN.get(config.MODEL_FAST, 0.0))
    pout = config.PRICE_OUT.get(model, config.PRICE_OUT.get(config.MODEL_FAST, 0.0))
    return (tokens_in / 1e6) * pin + (tokens_out / 1e6) * pout


class Ledger:
    def __init__(self, path=None):
        self.path = path or (config.MODELS_DIR / "chat-budget.json")
        self._lock = threading.Lock()
        self._recent = deque()          # request timestamps, for the per-minute limit
        self._state = None

    # ---- persistence ----
    def _load(self):
        today = date.today().isoformat()
        if self._state is not None and self._state.get("date") == today:
            return self._state
        state = {"date": today, "requests": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0,
                 "by_model": {}}
        try:
            if self.path.exists():
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                if stored.get("date") == today:
                    state.update(stored)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("budget ledger unreadable (%s); starting the day at zero", e)
        self._state = state
        return state

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:
            log.warning("could not write the budget ledger: %s", e)

    # ---- checks ----
    def check(self):
        """(ok, reason). Called before a question is sent to the provider."""
        with self._lock:
            state = self._load()
            now = time.time()
            while self._recent and now - self._recent[0] > 60:
                self._recent.popleft()
            if len(self._recent) >= config.RATE_PER_MINUTE:
                return False, (f"too many questions in the last minute "
                               f"(limit {config.RATE_PER_MINUTE}); try again shortly")
            if state["requests"] >= config.RATE_PER_DAY:
                return False, f"the daily question limit ({config.RATE_PER_DAY}) is used up; it resets at UTC midnight"
            if state["usd"] >= config.BUDGET_USD_PER_DAY:
                return False, (f"today's model budget (${config.BUDGET_USD_PER_DAY:.2f}) is used up; "
                               f"it resets at midnight")
            self._recent.append(now)
            return True, ""

    def record(self, model, usage):
        with self._lock:
            state = self._load()
            tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
            cost = _price(model, tin, tout)
            state["input_tokens"] += tin
            state["output_tokens"] += tout
            state["usd"] = round(state["usd"] + cost, 6)
            per = state["by_model"].setdefault(model, {"requests": 0, "input_tokens": 0,
                                                       "output_tokens": 0, "usd": 0.0})
            per["input_tokens"] += tin
            per["output_tokens"] += tout
            per["usd"] = round(per["usd"] + cost, 6)
            self._save()
            return cost

    def finish_request(self, model):
        with self._lock:
            state = self._load()
            state["requests"] += 1
            state["by_model"].setdefault(model, {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                                                "usd": 0.0})["requests"] += 1
            self._save()

    def snapshot(self):
        with self._lock:
            state = dict(self._load())
        state["usd_limit"] = config.BUDGET_USD_PER_DAY
        state["usd_left"] = round(max(0.0, config.BUDGET_USD_PER_DAY - state["usd"]), 4)
        state["requests_limit"] = config.RATE_PER_DAY
        state["requests_left"] = max(0, config.RATE_PER_DAY - state["requests"])
        return state


ledger = Ledger()
