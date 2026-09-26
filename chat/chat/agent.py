"""
The agent loop.

One question becomes: ask the model which tools to call, call them (in parallel), hand the results
back, stream the answer. Two provider calls in the common case, three when a question needs a second
round of tools.

The loop is deliberately short and bounded - rounds, tool calls, wall-clock - because an agent that
can spend an unbounded amount of somebody's money and time on a bibliography question is a bug, not
a feature. When a budget runs out the model is told so and answers with what it has.

Everything is synchronous: DuckDB is synchronous, and the endpoint streams from a worker thread.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from . import config, docs, tools as T
from .llm import LLMError

log = logging.getLogger("dblp.chat.agent")

MAX_RESULT_CHARS = 6000


SYSTEM = """You answer questions about the dblp computer-science bibliography by calling tools over a
live snapshot of the dump. You are part of a dashboard; every figure you give must come from a tool.

DATA
- Snapshot of dblp: {records} records, {publications} publications, {author_pages} author pages.
  Latest record edit {latest_mdate}. The last COMPLETE year is {last_full_year}; the snapshot's own
  year is partial, so never present it as a finished year.
- {limits}

HOW TO WORK
1. Call tools. Never state a number, name, year, count or ranking that did not come from a tool in
   this conversation. If you cannot get it from a tool, say you cannot.
2. A name is not an identity: call resolve_author or resolve_venue first and use the returned key.
   If several pages share the name, either ask which one, or answer for the most likely and say
   which page you used.
3. Prefer a typed tool over run_sql. Use run_sql only when no tool fits, keep it aggregated, and say
   in the answer that it was an ad-hoc query.
4. Carry each tool's `note` into your answer when it changes the meaning of the number: whether
   preprints are in, whether disambiguation bins are excluded, whether papers are counted per author
   slot. State it in a clause, not a lecture.
5. If a tool returns `refused` or an empty table, say so plainly. An empty table never means zero.
6. When a question needs something dblp does not have (citations, abstracts, affiliations, impact,
   awards, demographics), say it is not in the data, in one sentence, then offer the nearest thing
   that IS answerable and answer that if it is obvious.
7. For accuracy claims about the ML models or search, call model_cards and quote the measured
   numbers with their baseline. Never estimate them.

STYLE
- Two to four sentences. Lead with the answer.
- The interface already renders each tool's table under your answer: do not repeat more than the two
  or three rows your sentence needs, and never reformat a whole table as markdown.
- Give exact numbers with thousands separators, and name the year or window they cover.
- No greetings, no "great question", no restating the question.
- If you had to choose an interpretation (a person, a venue, a window), say which in a short clause.
"""


class Ctx:
    """What a tool handler gets: a cursor factory, the dump's metadata, and an HTTP client."""

    def __init__(self, pool, http, store_meta=None):
        self.pool = pool
        self.http = http
        self.store_meta = store_meta or {}

    def cursor(self):
        return self.pool.cursor()

    @property
    def meta(self):
        return self.pool.meta

    @property
    def serving_path(self):
        return self.pool.serving_path

    def last_full_year(self):
        return self.pool.last_full_year()


def system_prompt(ctx):
    facts = {}
    try:
        facts = T.store.facts(ctx.cursor())
    except Exception as e:                       # the store may not be attached yet
        log.warning("dataset facts unavailable for the prompt: %s", e)
    fmt = lambda k: f"{facts[k]:,}" if k in facts else "an unknown number of"
    return SYSTEM.format(records=fmt("records"), publications=fmt("publications"),
                         author_pages=fmt("author_pages"),
                         latest_mdate=ctx.meta.get("latest_mdate", "unknown"),
                         last_full_year=ctx.last_full_year(), limits=docs.LIMITS)


def _trim(payload):
    text = json.dumps(payload, default=str)
    if len(text) <= MAX_RESULT_CHARS:
        return text
    rows = payload.get("rows")
    if isinstance(rows, list) and len(rows) > 5:
        payload = dict(payload, rows=rows[:5], rows_omitted=len(rows) - 5)
        text = json.dumps(payload, default=str)
    return text[:MAX_RESULT_CHARS] + '..." (truncated)'


def run_tools(ctx, calls, emit, budget_left):
    """Execute tool calls in parallel; returns the tool messages for the next model call."""
    messages = []
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(calls)))) as pool:
        started = {}
        for call in calls:
            emit({"type": "tool_start", "name": call["name"], "arguments": call["arguments"]})
            started[pool.submit(T.call, ctx, call["name"], call["arguments"])] = (call, time.time())
        for future, (call, t0) in list(started.items()):
            timeout = max(0.5, min(config.TOOL_TIMEOUT, budget_left()))
            try:
                out = future.result(timeout=timeout)
            except FutureTimeout:
                out = {"summary": f"{call['name']} took longer than {timeout:.0f}s and was abandoned; "
                                  f"answer without it or ask for a narrower window.", "refused": True}
            except Exception as e:                      # a handler that raises must not kill the turn
                log.exception("tool %s failed", call["name"])
                out = {"summary": f"{call['name']} failed: {type(e).__name__}: {e}", "refused": True}
            ms = int(1000 * (time.time() - t0))
            event = {"type": "tool", "name": call["name"], "arguments": call["arguments"], "ms": ms,
                     "summary": out.get("summary"), "note": out.get("note"), "link": out.get("link"),
                     "refused": bool(out.get("refused")), "sql": out.get("sql")}
            if out.get("rows") is not None:
                event["columns"] = out.get("columns")
                event["rows"] = out.get("rows")
            emit(event)
            log.info("tool %s %s -> %sms", call["name"], call["arguments"], ms)
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                             "content": _trim(out)})
    return messages


def answer(ctx, client, question, history=None, emit=None, ledger=None):
    """Run one question. `emit` receives events; returns a summary of the run."""
    emit = emit or (lambda _e: None)
    started = time.time()
    budget_left = lambda: config.TIME_BUDGET_SECONDS - (time.time() - started)
    messages = [{"role": "system", "content": system_prompt(ctx)}]
    for turn in (history or [])[-config.MAX_HISTORY_TURNS:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        content = (turn.get("content") or "")[:2000]
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    usage_total = {"input_tokens": 0, "output_tokens": 0}
    cost_total = 0.0
    used_tools, rounds, calls_made = [], 0, 0
    model = config.MODEL_FAST

    def account(model_name, usage):
        nonlocal cost_total
        usage_total["input_tokens"] += usage.get("input_tokens", 0)
        usage_total["output_tokens"] += usage.get("output_tokens", 0)
        if ledger is not None:
            cost_total += ledger.record(model_name, usage)

    direct = None
    for rounds in range(1, config.MAX_ROUNDS + 1):
        emit({"type": "status", "text": "choosing tools" if rounds == 1 else "following up"})
        try:
            step = client.complete(messages, model=config.MODEL_FAST, tools=T.schemas())
        except LLMError as e:
            emit({"type": "error", "message": str(e)})
            return {"error": str(e), "rounds": rounds}
        account(step.get("model", config.MODEL_FAST), step.get("usage", {}))
        calls = step.get("tool_calls") or []
        if not calls:
            direct = step.get("content") or ""
            break
        room = max(0, config.MAX_TOOL_CALLS - calls_made)
        if len(calls) > room:
            calls = calls[:room]
        calls_made += len(calls)
        messages.append({"role": "assistant", "content": step.get("content") or None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": json.dumps(c["arguments"])}}
                                        for c in calls]})
        used_tools += [c["name"] for c in calls]
        messages += run_tools(ctx, calls, emit, budget_left)
        if calls_made >= config.MAX_TOOL_CALLS or budget_left() < 5:
            messages.append({"role": "user", "content":
                             "Answer now with the tool results above; there is no time for more tools. "
                             "Say plainly if something is missing."})
            break

    if direct is not None:
        # the model answered without tools: usually a definition or a clarifying question
        for piece in direct.split(" "):
            emit({"type": "token", "text": piece + " "})
        text = direct
    else:
        model = config.MODEL_DEEP if rounds >= config.ESCALATE_AFTER_ROUNDS else config.MODEL_FAST
        emit({"type": "status", "text": "writing"})
        pieces = []
        try:
            for kind, value in client.stream(messages, model=model):
                if kind == "token":
                    pieces.append(value)
                    emit({"type": "token", "text": value})
                elif kind == "usage":
                    account(model, value)
        except LLMError as e:
            emit({"type": "error", "message": str(e)})
            return {"error": str(e), "rounds": rounds, "tools": used_tools}
        text = "".join(pieces)

    if ledger is not None:
        ledger.finish_request(model)
    done = {"type": "done", "answer": text, "rounds": rounds, "tools": used_tools,
            "usage": usage_total, "cost_usd": round(cost_total, 5),
            "seconds": round(time.time() - started, 2), "model": model,
            "dump": {"fingerprint": ctx.meta.get("fingerprint"),
                     "latest_mdate": ctx.meta.get("latest_mdate")}}
    emit(done)
    return done
