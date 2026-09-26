"""
Command line, for the VM.

  python -m chat.cli store                      build the leaderboard store for this dump
  python -m chat.cli tools                      list the tool catalogue
  python -m chat.cli ask "who has most papers"  one question, printed as it happens
  python -m chat.cli evaluate [--limit N]       run the gold set, write the report
  python -m chat.cli qa [--only X]              check the tools against the dashboard (no model calls)

`ask` is the same code path the web endpoint uses, so a question that works here works there.
"""
import argparse
import json
import logging
import sys

import httpx

from . import agent, budget, config, data, evaluate as E, qa as Q, store, tools as T
from .llm import Client


def _ready():
    """The same state the server builds: serving database attached, leaderboard store attached."""
    if not data.pool.load():
        sys.exit(data.pool.error)
    store_meta = store.attach(data.pool.connection(), data.pool.meta)
    return agent.Ctx(data.pool, httpx.Client(timeout=config.UPSTREAM_TIMEOUT), store_meta)


def cmd_store(_args):
    con, meta = data.connect()
    try:
        path = store.build(con, meta)
        print(f"built {path}")
    finally:
        con.close()


def cmd_tools(_args):
    for spec in T.SPECS:
        print(f"{spec['name']:<20} {spec['description'][:110]}")
    print(f"\n{len(T.SPECS)} tools")


def cmd_ask(args):
    ctx = _ready()
    client = Client()
    if not client.configured():
        sys.exit("No OPENAI_API_KEY set: the tools work, but nothing can route a question.")

    def emit(event):
        kind = event.get("type")
        if kind == "tool":
            print(f"\n  · {event['name']}({json.dumps(event['arguments'], default=str)[:90]}) "
                  f"{event['ms']}ms -> {event.get('summary')}", file=sys.stderr)
        elif kind == "token":
            sys.stdout.write(event["text"])
            sys.stdout.flush()
        elif kind == "error":
            print(f"\nerror: {event['message']}", file=sys.stderr)
        elif kind == "done":
            print(f"\n\n[{event['seconds']}s, {event['rounds']} round(s), "
                  f"{event['usage']['input_tokens']}+{event['usage']['output_tokens']} tokens, "
                  f"${event['cost_usd']}, tools: {', '.join(event['tools']) or 'none'}]", file=sys.stderr)

    agent.answer(ctx, client, args.question, emit=emit, ledger=budget.ledger)


def cmd_evaluate(args):
    ctx = _ready()
    client = Client()
    if not client.configured():
        sys.exit("No OPENAI_API_KEY set.")
    payload = E.run(ctx, client, limit=args.limit)
    path = E.save(payload, ctx.meta.get("fingerprint", "unknown"))
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nwritten to {path}")


def cmd_qa(args):
    ctx = _ready()
    report = Q.run(ctx, only=args.only)
    width = max(len(r["name"]) for r in report["results"]) + 2
    for r in report["results"]:
        mark = {"pass": "ok  ", "FAIL": "FAIL", "skip": "skip"}[r["status"]]
        print(f"{mark} {r['name']:<{width}} {r['seconds']:>5.2f}s  {r['detail']}")
    s = report["summary"]
    print(f"\n{s['passed']}/{s['checks']} passed, {s['failed']} failed, "
          f"{s['skipped']} skipped in {s['seconds']}s")
    path = config.MODELS_DIR / "chat-eval" / f"qa-{ctx.meta.get('fingerprint', 'unknown')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"written to {path}")
    sys.exit(1 if s["failed"] else 0)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="chat.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("store", help="build the leaderboard store").set_defaults(fn=cmd_store)
    sub.add_parser("tools", help="list the tool catalogue").set_defaults(fn=cmd_tools)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.set_defaults(fn=cmd_ask)
    ev = sub.add_parser("evaluate", help="run the gold set")
    ev.add_argument("--limit", type=int, default=None)
    ev.set_defaults(fn=cmd_evaluate)
    q = sub.add_parser("qa", help="check the tools against the dashboard, without a model")
    q.add_argument("--only", default=None, help="a check name fragment, or a group: "
                                                "dashboard, invariant, behaviour")
    q.set_defaults(fn=cmd_qa)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
