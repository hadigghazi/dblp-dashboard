"""
HTTP front for the chat service.

POST /chat/ask streams server-sent events: the tools it calls (with their tables and timings), then
the answer token by token, then one `done` event with usage and cost. The stream is the audit trail -
a reader can see which query produced the number before the sentence finishes.

Access is gated by a shared token when CHAT_TOKEN is set, because every question spends money, and
by a per-minute rate limit plus a daily budget that fails closed.
"""
import hashlib
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import agent, budget, config, data, docs, store, tools as T, usage
from .llm import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dblp.chat.server")

# Suggested questions for the assistant panel. Every one is a question the gold set verifies - word
# for word, and for a follow-up the whole exchange - so the panel never advertises something that
# fails; tests/test_examples.py holds it to that. "hard" ones carry a line, in plain words, saying what
# makes them hard, because the point of showing them is to show what the assistant can do.
EXAMPLES = [
    # the easy way in
    {"q": "Which author has the most papers in dblp?", "level": "basic"},
    {"q": "Tell me about CVPR", "level": "basic"},
    {"q": "How many papers were published in 2024?", "level": "basic", "then": "And in 2014?"},
    {"q": "Find papers about learning robot manipulation from a few demonstrations", "level": "basic"},

    # the ones that show what it can do
    {"q": "How many different people are called Wei Wang?", "level": "hard",
     "why": "One name, hundreds of different people - it tells them apart"},
    {"q": "What do Yang Liu's papers look like?", "level": "hard",
     "why": "A name many people share - it says so instead of mixing them up"},
    {"q": "Which authors bridge the most research communities?", "level": "hard",
     "why": "The bridges between fields are not the people who publish most"},
    {"q": "How many degrees of separation are there between computer scientists?", "level": "hard",
     "why": "Measured across about four million authors"},
    {"q": "How central is Yoshua Bengio in the co-authorship network?", "level": "hard",
     "why": "His rank among about four million authors, on four different measures"},
    {"q": "Have Yann LeCun and Yoshua Bengio written a paper together?", "level": "hard",
     "why": "Checks every paper the two of them are on"},
    {"q": "Who are Geoffrey Hinton's co-authors that also publish at ICML?", "level": "hard",
     "why": "A person, their collaborators and a venue, all in one question"},
    {"q": "Has NeurIPS grown faster than ICML since 2015?", "level": "hard",
     "why": "Two venues compared year by year"},
    {"q": "How many journal papers did Jürgen Schmidhuber publish since 2020?", "level": "hard",
     "why": "One author, one kind of paper, one period - all at once"},
    {"q": "Where should a paper called 'Contrastive pretraining for medical image segmentation' be "
          "submitted?", "level": "hard",
     "why": "Asks the venue-recommendation model"},
    {"q": "What is Geoffrey Hinton's h-index?", "level": "hard",
     "why": "dblp has no citations - watch it say so instead of guessing"},
    {"q": "Who is the most central author by betweenness?", "level": "hard",
     "then": "And how many papers do they have?",
     "why": "Then ask the follow-up - it remembers who you meant"},
]


class State:
    def __init__(self):
        self.client = Client()
        self.http = httpx.Client(timeout=config.UPSTREAM_TIMEOUT)
        self.store_meta = {}
        self.error = None
        self.lock = threading.Lock()

    def load(self):
        if not data.pool.load():
            self.error = data.pool.error
            return False
        try:
            # ATTACH on the real connection, so every cursor of this instance sees the store
            self.store_meta = store.attach(data.pool.connection(), data.pool.meta)
            self.error = None
        except Exception as e:
            self.error = f"leaderboard store unavailable: {e}"
            log.exception("could not attach the chat store")
            return False
        log.info("chat ready: dump %s, store built %s", data.pool.fingerprint(),
                 self.store_meta.get("built_at"))
        return True

    def watch(self, every=300):
        def loop():
            while True:
                time.sleep(every)
                try:
                    current = data.find_serving_db()
                    if str(current) != str(data.pool.serving_path):
                        log.info("new serving database (%s); reloading", current.name)
                        self.load()
                except Exception:
                    log.exception("serving-database watch failed")
        threading.Thread(target=loop, name="chat-watch", daemon=True).start()

    def ctx(self):
        return agent.Ctx(data.pool, self.http, self.store_meta)

    def ready(self):
        return data.pool.ready() and self.error is None


state = State()


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=state.load, name="chat-load", daemon=True).start()   # never block startup
    state.watch()
    yield
    state.http.close()
    state.client.close()


app = FastAPI(title="dblp chat", lifespan=lifespan, docs_url="/chat/docs",
              openapi_url="/chat/openapi.json")


def require_token(x_chat_token: Optional[str] = Header(None), token: Optional[str] = Query(None)):
    if not config.TOKEN:
        return True            # unset: open, which is the local-development case
    if (x_chat_token or token) != config.TOKEN:
        raise HTTPException(401, detail="This assistant needs an access token.")
    return True


class Turn(BaseModel):
    role: str = Field("user")
    content: str = ""


class Ask(BaseModel):
    question: str = Field(..., min_length=3)
    history: List[Turn] = Field(default_factory=list)
    refresh: bool = False


@app.get("/chat/health")
def health():
    return {"ok": True, "ready": state.ready(), "configured": state.client.configured()}


def _evaluation():
    """The gold-set report for this dump, if `chat.cli evaluate` has been run against it. The rest of
    the site never shows a model's output without its measured accuracy; neither does this."""
    path = config.MODELS_DIR / "chat-eval" / f"{data.pool.fingerprint()}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("summary")
    except (OSError, json.JSONDecodeError) as e:
        log.warning("gold-set report unreadable: %s", e)
        return None


@app.get("/chat/status")
def status(_ok=Depends(require_token)):
    return {
        "evaluation": _evaluation(),
        "configured": state.client.configured(),
        "ready": state.ready(),
        "error": state.error,
        "models": {"router": config.MODEL_FAST, "answers": config.MODEL_DEEP,
                   "provider": config.BASE_URL},
        "dump": {"fingerprint": data.pool.fingerprint(),
                 "latest_mdate": data.pool.meta.get("latest_mdate"),
                 "last_full_year": data.pool.last_full_year() if data.pool.ready() else None,
                 "records": data.pool.meta.get("records")},
        "store": state.store_meta,
        "budget": budget.ledger.snapshot(),
        "limits": {"question_chars": config.MAX_QUESTION_CHARS, "rounds": config.MAX_ROUNDS,
                   "tool_calls": config.MAX_TOOL_CALLS, "seconds": config.TIME_BUDGET_SECONDS,
                   "per_minute": config.RATE_PER_MINUTE},
        "tools": [{"name": s["name"], "description": s["description"]} for s in T.SPECS],
        "documentation_topics": docs.titles(),
        "cannot_answer": docs.LIMITS,
        "examples": EXAMPLES,
    }


def _cache_path(question, fingerprint):
    tag = hashlib.sha1(f"{fingerprint}|{question.strip().lower()}".encode()).hexdigest()[:16]
    d = config.MODELS_DIR / "chat-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.json"


def _sse(event):
    return f"data: {json.dumps(event, default=str)}\n\n"


def _run(question, history):
    """Yield events as they happen: the agent runs in a worker thread, this drains its queue."""
    events = queue.Queue()

    def work():
        try:
            agent.answer(state.ctx(), state.client, question, history=history,
                         emit=events.put, ledger=budget.ledger)
        except Exception as e:                      # never leave the browser hanging on a stack trace
            log.exception("question failed")
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=work, name="chat-answer", daemon=True).start()
    while True:
        event = events.get()
        if event is None:
            return
        yield event


def ask_stream(question, history, refresh=False):
    fingerprint = data.pool.fingerprint()
    path = _cache_path(question, fingerprint) if not history else None
    if path and refresh:
        path.unlink(missing_ok=True)
    if path and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            for event in cached:
                if event.get("type") == "done":
                    event = dict(event, cached=True, cost_usd=0.0)
                yield _sse(event)
            return
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)

    ok, why = budget.ledger.check()
    if not ok:
        yield _sse({"type": "error", "message": why})
        return
    collected, finished = [], False
    for event in _run(question, history):
        collected.append(event)
        if event.get("type") == "done":
            finished = True
        yield _sse(event)
    # what was asked, and what it took - the only record of which tools the catalogue is missing
    usage.record(usage.from_events(question, collected, data.pool.fingerprint()))
    if path and finished:
        try:
            path.write_text(json.dumps(collected, default=str), encoding="utf-8")
        except OSError as e:
            log.warning("could not cache the answer: %s", e)


def _guard(question):
    if not state.client.configured():
        raise HTTPException(503, detail="No model provider configured: set OPENAI_API_KEY on the service.")
    if not state.ready():
        raise HTTPException(503, detail=state.error or "The dblp snapshot is still being attached; "
                                                      "try again in a moment.")
    if len(question) > config.MAX_QUESTION_CHARS:
        raise HTTPException(422, detail=f"Question too long (limit {config.MAX_QUESTION_CHARS} characters).")


@app.post("/chat/ask")
def ask(body: Ask, request: Request, _ok=Depends(require_token)):
    _guard(body.question)
    log.info("ask from %s: %r", request.client.host if request.client else "?", body.question[:120])
    history = [t.model_dump() for t in body.history]
    return StreamingResponse(ask_stream(body.question, history, body.refresh),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/chat/ask")
def ask_get(question: str = Query(..., min_length=3), refresh: bool = False, _ok=Depends(require_token)):
    """The same thing for a terminal: curl -N '.../chat/ask?question=...&token=...'"""
    _guard(question)
    return StreamingResponse(ask_stream(question, [], refresh), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
