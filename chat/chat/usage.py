"""
What people actually asked.

The tool catalogue was written from my guesses about the questions a user would type. Every question
that falls through to ad-hoc SQL is the catalogue saying a tool is missing, every refused tool call
is an argument the model could not get right, and every slow question is a query worth precomputing -
but none of that is visible unless it is written down.

One JSON line per question, appended to a monthly file next to the models. Nothing about the person
is recorded: no address, no token, no session - only the question, what it took to answer it, and
what it cost. `chat.cli questions` reads them back.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger("dblp.chat.usage")


def log_dir() -> Path:
    d = config.MODELS_DIR / "chat-log"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_for(when=None) -> Path:
    when = when or datetime.now(timezone.utc)
    return log_dir() / f"{when:%Y-%m}.jsonl"


def from_events(question, events, fingerprint=None):
    """Build an entry from what the stream emitted, so logging can never disagree with the answer."""
    tools = [e for e in events if e.get("type") == "tool"]
    done = next((e for e in events if e.get("type") == "done"), {})
    error = next((e["message"] for e in events if e.get("type") == "error"), None)
    return {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "tools": [t["name"] for t in tools],
        "refused_tools": [t["name"] for t in tools if t.get("refused")],
        "used_sql": any(t["name"] == "run_sql" for t in tools),
        "no_tool": not tools,
        "rounds": done.get("rounds"),
        "seconds": done.get("seconds"),
        "cost_usd": done.get("cost_usd"),
        "cached": bool(done.get("cached")),
        "model": done.get("model"),
        "answered": bool(done.get("answer")),
        "error": error,
        "dump": fingerprint,
    }


def record(entry, path=None):
    """Appending must never break an answer that already worked."""
    try:
        target = Path(path) if path else path_for()
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return target
    except OSError as e:
        log.warning("could not record the question: %s", e)
        return None


def read(days=30, directory=None):
    d = Path(directory) if directory else log_dir()
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    out = []
    for file in sorted(d.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                if datetime.fromisoformat(entry["at"]).timestamp() >= cutoff:
                    out.append(entry)
            except (KeyError, ValueError):
                out.append(entry)
    return out


def summarize(days=30, directory=None):
    """The three things worth acting on: what is missing from the catalogue, what the model gets
    wrong about a tool's arguments, and what is slow."""
    entries = read(days, directory)
    n = len(entries) or 1
    counts, refused = {}, {}
    for entry in entries:
        for name in entry.get("tools") or []:
            counts[name] = counts.get(name, 0) + 1
        for name in entry.get("refused_tools") or []:
            refused[name] = refused.get(name, 0) + 1
    times = sorted(e["seconds"] for e in entries if e.get("seconds") is not None)
    return {
        "days": days,
        "questions": len(entries),
        "cost_usd": round(sum(e.get("cost_usd") or 0 for e in entries), 4),
        "cached_share": round(sum(bool(e.get("cached")) for e in entries) / n, 3),
        "median_seconds": times[len(times) // 2] if times else None,
        "slowest": sorted([e for e in entries if e.get("seconds") is not None],
                          key=lambda e: -e["seconds"])[:5],
        "tools": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "refused_tool_calls": dict(sorted(refused.items(), key=lambda kv: -kv[1])),
        # the catalogue's to-do list: questions no typed tool could answer
        "fell_back_to_sql": [e["question"] for e in entries if e.get("used_sql")],
        "answered_without_a_tool": [e["question"] for e in entries if e.get("no_tool")],
        "errors": [{"question": e["question"], "error": e["error"]} for e in entries if e.get("error")],
        "recent": [{"question": e["question"], "tools": e.get("tools"), "seconds": e.get("seconds")}
                   for e in entries[-20:]],
    }
