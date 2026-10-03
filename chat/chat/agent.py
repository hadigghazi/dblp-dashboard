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
   If several pages share the name and the question already says which one - a number such as 0002,
   an affiliation, a country, a field, a period - pick that page and answer the question in this same
   turn: call the tool it needs now ("tell me about X" is author_profile). Do not offer to. Otherwise
   either ask which one, or answer for the most likely and say which you used. dblp writes names in
   Latin script: transliterate a name written in Arabic, Chinese or another script yourself before
   calling resolve_author, try the usual spellings if the first finds nothing, and never ask the user
   to transliterate it.
   Never build a key or a venue id yourself: they are opaque (homepages/165/0820-2) and a guessed one
   either fails or, worse, matches nothing. Use one a tool returned, or pass the exact name the user
   gave (a page name such as "Wei Wang 0003", a venue name such as "IEEE Access") - both are accepted.
3. Prefer a typed tool over run_sql. Use run_sql only when no tool fits, keep it aggregated, and say
   in the answer that it was an ad-hoc query.
4. Carry each tool's `note` into your answer when it changes the meaning of the number: whether
   preprints are in, whether disambiguation bins are excluded, whether papers are counted per author
   slot. State it in a clause, not a lecture.
5. If a tool returns `refused` or an empty table, say so plainly. An empty table never means zero.
   Never explain a failed lookup with a guess about the person or the data ("he probably has no
   co-authors"): a wrong key is a wrong key - resolve the name again. An empty result is only evidence
   of "never" or "none" when every filter behind it came from a tool, not from a guess.
6. When a question needs something dblp does not have (citations, abstracts, affiliations, impact,
   awards, demographics), say it is not in the data, in one sentence, then offer the nearest thing
   that IS answerable and answer that if it is obvious.
7. For accuracy claims about the ML models or search, call model_cards and quote the measured
   numbers with their baseline. Never estimate them.
8. A follow-up ("and his co-authors?", "what about 2014?", "the second one") is about the subject of
   the previous turn. Each earlier answer ends with a bracketed list of the pages its tools found,
   with their keys. "The first one", "the second one", "the last one" count in the order YOUR ANSWER
   named things - the co-author you wrote first is "the first one" - never in the order of that list,
   which is only where you look up the key of the thing you meant. If the key you need is not in the
   list, call resolve_author or resolve_venue again with the name. Never guess a key, and never
   ask the user for one - nobody knows dblp keys. Name the subject in your answer so it cannot be
   misread.

STYLE
- Two to four sentences. Lead with the answer.
- The interface already renders each tool's table under your answer: do not repeat more than the two
  or three rows your sentence needs, and never reformat a whole table as markdown.
- Give exact numbers with thousands separators, and name the year or window they cover.
- No greetings, no "great question", no restating the question.
- If you had to choose an interpretation (a person, a venue, a window), say which in a short clause.
- Never write a dblp key (homepages/..., journals/..., conf/...) or a bracketed list of pages in the
  answer, unless the user typed a key: the interface links every page, and the list used for
  follow-ups is attached automatically. Name a person by name, adding the number (Wei Wang 0003) or
  the affiliation when two share a name.
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


def run_tools(ctx, calls, emit, budget_left, collect=None):
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
            if collect is not None:
                # the whole payload, including the `meta` the UI event drops: anything checking the
                # answer against its sources has to see exactly what the model was given
                collect.append({"name": call["name"], "arguments": call["arguments"], "result": out})
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                             "content": _trim(out)})
    return messages


MEMORY_ITEMS = 12
MEMORY_CHARS = 1200


def remembered(payloads):
    """The pages and records an answer's tools returned, with their keys, in the order they came back.

    The conversation goes back to the model as text, and the tool results do not go with it - so
    without this, a follow-up like "the second one" has the person's name but not their key, and the
    model guesses one. This is the part of the tool results a follow-up can need, small enough to send
    back every turn."""
    seen, lines = set(), []
    for p in payloads:
        out = p.get("result") or {}
        if out.get("refused"):
            continue
        for row in out.get("rows") or []:
            if not isinstance(row, dict):
                continue
            key = row.get("key") or row.get("sid")
            label = row.get("name") or row.get("title")
            if not key or not label or key in seen:
                continue
            seen.add(key)
            extra = [str(row[k]) for k in ("papers", "records", "affiliation", "year", "kind")
                     if row.get(k) not in (None, "")]
            lines.append(f"{len(lines) + 1}. {label} - {key}" + (f" ({', '.join(extra)})" if extra else ""))
            if len(lines) >= MEMORY_ITEMS:
                break
        # the subject of a profile is in the call, not in its rows
        args = p.get("arguments") or {}
        for arg in ("key", "author_key", "sid"):
            key = args.get(arg)
            if key and key not in seen:
                seen.add(key)
                lines.append(f"{len(lines) + 1}. (looked up) {key}")
        if len(lines) >= MEMORY_ITEMS:
            break
    text = "\n".join(lines)
    return text[:MEMORY_CHARS]


def with_memory(answer_text, payloads):
    """What a turn is remembered as: the answer, then the pages behind it. Built only here - the panel
    and the evaluation both send it back exactly as it was given."""
    memory = remembered(payloads)
    if not memory:
        return answer_text
    return (f"{answer_text}\n\n[Pages and records this answer used, in the order the tools returned "
            f"them; use these keys for follow-ups:\n{memory}]")


def answer(ctx, client, question, history=None, emit=None, ledger=None, collect=None,
           channel="web"):
    """Run one question. `emit` receives events; returns a summary of the run."""
    emit = emit or (lambda _e: None)
    # every tool result is kept for this answer's memory, whether or not the caller collects them
    collect = collect if collect is not None else []
    started = time.time()
    budget_left = lambda: config.TIME_BUDGET_SECONDS - (time.time() - started)
    messages = [{"role": "system", "content": system_prompt(ctx)}]
    for turn in (history or [])[-config.MAX_HISTORY_TURNS:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        content = (turn.get("content") or "")[:3000]   # room for the answer and its memory
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
        emit({"type": "status", "text": "looking it up" if rounds == 1 else "checking one more thing"})
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
        messages += run_tools(ctx, calls, emit, budget_left, collect)
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
        emit({"type": "status", "text": "writing the answer"})
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
        ledger.finish_request(model, channel)
    done = {"type": "done", "answer": text, "rounds": rounds, "tools": used_tools,
            "usage": usage_total, "cost_usd": round(cost_total, 5),
            "seconds": round(time.time() - started, 2), "model": model,
            "memory": with_memory(text, collect),
            "dump": {"fingerprint": ctx.meta.get("fingerprint"),
                     "latest_mdate": ctx.meta.get("latest_mdate")}}
    emit(done)
    return done
