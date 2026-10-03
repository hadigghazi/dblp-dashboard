"""
Command line, for the VM.

  python -m chat.cli store                      build the leaderboard store for this dump
  python -m chat.cli tools                      list the tool catalogue
  python -m chat.cli ask "who has most papers"  one question, printed as it happens
  python -m chat.cli evaluate [--limit N]       run the gold set, write the report
  python -m chat.cli qa [--only X]              check the tools against the dashboard (no model calls)
  python -m chat.cli questions [--days N]      what people asked, and what the catalogue is missing
  python -m chat.cli search-eval [--papers N]  can search find a paper from a description of it?
  python -m chat.cli scenarios [--suite S]     one person's questions about themselves, answers in full
  python -m chat.cli dblpqa closed-book         DBLP-QA with no retrieval, judged automatically
  python -m chat.cli dblpqa retrieval           DBLP-QA: does retrieval find each question's paper?
  python -m chat.cli dblpqa rag --ranker hybrid DBLP-QA answered from the top-5 retrieved abstracts

`ask` is the same code path the web endpoint uses, so a question that works here works there.
"""
import argparse
import json
import logging
import sys

import httpx

from . import agent, budget, config, data, evaluate as E, qa as Q, searcheval as SE, store, tools as T, usage
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
    events = []
    client = Client()
    if not client.configured():
        sys.exit("No OPENAI_API_KEY set: the tools work, but nothing can route a question.")

    def emit(event):
        events.append(event)
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

    agent.answer(ctx, client, args.question, emit=emit, ledger=budget.ledger, channel="cli")
    usage.record(usage.from_events(args.question, events, ctx.meta.get("fingerprint")))


def cmd_evaluate(args):
    ctx = _ready()
    client = Client()
    if not client.configured():
        sys.exit("No OPENAI_API_KEY set.")
    payload = E.run(ctx, client, limit=args.limit)
    if payload["summary"].get("stopped"):
        # not saved: a run cut short by the provider would replace the last real measurement
        print(json.dumps(payload["summary"], indent=2))
        sys.exit("\nThe evaluation stopped early because the model provider refused - see 'stopped' "
                 "above. Nothing was saved; the last complete report still stands.")
    path = E.save(payload, ctx.meta.get("fingerprint", "unknown"))
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nwritten to {path}")


def cmd_dblpqa(args):
    from . import dblpqa as DQ
    client = Client()
    if not client.configured():
        sys.exit("No OPENAI_API_KEY set.")
    if args.condition == "report":
        DQ.report(args.run)
        return
    if args.condition == "rescore":
        if not args.run:
            sys.exit("rescore needs --run <run directory name>")
        DQ.rescore(client, args.run, args.judge)
        return
    rows, sha = DQ.load_dataset()
    if args.limit:
        rows = rows[:args.limit]
    condition, contexts = args.condition, None
    if condition in ("retrieval", "rag"):
        from . import dblpqa_rag as RAG
        pools, rankings, report = RAG.prepare(rows)
        RAG.print_retrieval(report)
        print(f"embeddings ${report['embedding_cost_usd']}")
        if condition == "retrieval":
            return
        condition = f"{DQ.RAG_PREFIX}{args.ranker}"
        contexts = RAG.rag_contexts(rows, pools, rankings, args.ranker, k=args.k)
    elif condition == "oracle":
        contexts = DQ.fetch_abstracts(rows)
    print(f"DBLP-QA: {len(rows)} questions (sha256 {sha[:12]}), {condition}, judge {args.judge}, "
          f"sampling {args.sampling}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    DQ.run_condition(client, models, args.judge, rows, sha, condition, contexts=contexts,
                     force=args.force, sampling=args.sampling,
                     reuse_controls=not args.recheck_judge and not args.limit)


def cmd_scenarios(args):
    from . import scenarios as SC
    ctx = _ready()
    client = Client()
    if not client.configured():
        sys.exit("No OPENAI_API_KEY set.")
    payload = SC.run(ctx, client, suite=args.suite, limit=args.limit)
    path = SC.save(payload, ctx.meta.get("fingerprint", "unknown"))
    print(f"written to {path}")


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


def cmd_search_eval(args):
    ctx = _ready()
    client = Client()
    if not client.configured():
        sys.exit("No OPENAI_API_KEY set: this test needs a model to write the descriptions.")
    payload = SE.run(ctx, client, n_papers=args.papers, seed=args.seed, ledger=budget.ledger,
                     regenerate=args.regenerate)
    path = SE.save(payload, ctx.meta.get("fingerprint", "unknown"))
    print(json.dumps({k: v for k, v in payload.items() if k not in ("examples", "hybrid_lost_to_embeddings_alone")},
                     indent=2))
    print("\nexample generated queries:")
    for e in payload["examples"]:
        print(f"  {e['query']}\n      -> {e['title']}")
    print(f"\nwritten to {path}")


def cmd_questions(args):
    report = usage.summarize(args.days)
    if not report["questions"]:
        print(f"No questions logged in the last {args.days} days.")
        return
    print(f"{report['questions']} questions in {args.days} days, median {report['median_seconds']}s, "
          f"${report['cost_usd']}, {report['cached_share']:.0%} served from cache\n")
    print("tools used:")
    for name, count in list(report["tools"].items())[:12]:
        print(f"  {count:>4}  {name}")
    if report["refused_tool_calls"]:
        print("\nrefused tool calls (the model gets these arguments wrong):")
        for name, count in report["refused_tool_calls"].items():
            print(f"  {count:>4}  {name}")
    if report["fell_back_to_sql"]:
        print("\nanswered with ad-hoc SQL - each of these is a tool the catalogue is missing:")
        for q in report["fell_back_to_sql"][-10:]:
            print(f"  - {q}")
    if report["answered_without_a_tool"]:
        print("\nanswered with no tool at all:")
        for q in report["answered_without_a_tool"][-10:]:
            print(f"  - {q}")
    if report["errors"]:
        print("\nerrors:")
        for e in report["errors"][-5:]:
            print(f"  - {e['question']}: {e['error']}")
    print("\nslowest:")
    for e in report["slowest"]:
        print(f"  {e['seconds']:>5.1f}s  {e['question']}")


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
    se = sub.add_parser("search-eval", help="can search find a paper from a description of it?")
    se.add_argument("--papers", type=int, default=60)
    se.add_argument("--seed", type=int, default=7)
    se.add_argument("--regenerate", action="store_true",
                    help="write new descriptions instead of reusing the saved ones")
    se.set_defaults(fn=cmd_search_eval)
    ql = sub.add_parser("questions", help="what people asked, and what the catalogue is missing")
    ql.add_argument("--days", type=int, default=30)
    ql.set_defaults(fn=cmd_questions)
    dq = sub.add_parser("dblpqa", help="experiments on the DBLP-QA benchmark")
    dq.add_argument("condition", choices=["closed-book", "oracle", "retrieval", "rag", "report", "rescore"],
                    help="oracle = each question with the abstract it was written from; retrieval = where "
                         "each ranker puts that paper (no answers); rag = answers from the top --k papers "
                         "of --ranker; rescore = score the unscored answers of --run")
    dq.add_argument("--ranker", default="hybrid",
                    choices=["bm25", "dense", "hybrid", "dblp-search", "s2-search"],
                    help="for rag: bm25 is RAGScholar's method, hybrid = bm25 + embeddings fused")
    dq.add_argument("--k", type=int, default=5, help="for rag: papers in the context (the paper's best: 5)")
    dq.add_argument("--run", default=None, help="for report: a run directory (default: the latest)")
    dq.add_argument("--sampling", choices=["ours", "paper"], default="ours",
                    help="'paper' = the paper's Table 1 settings, for reproducing its models")
    dq.add_argument("--recheck-judge", action="store_true",
                    help="run the judge's controls again even if they passed on this dataset before")
    dq.add_argument("--models", default="gpt-4.1-mini,gpt-4.1",
                    help="comma-separated; 'ollama:<tag>' runs an open model on the local Ollama container")
    dq.add_argument("--judge", default="gpt-4.1",
                    help="'none' generates and saves answers without scoring them (rescore later)")
    dq.add_argument("--limit", type=int, default=None, help="first N questions only, for a dry run")
    dq.add_argument("--force", action="store_true", help="run even if the judge fails its controls")
    dq.set_defaults(fn=cmd_dblpqa)

    sc = sub.add_parser("scenarios", help="one person's questions about themselves, answers in full")
    sc.add_argument("--suite", default="instructor")
    sc.add_argument("--limit", type=int, default=None)
    sc.set_defaults(fn=cmd_scenarios)

    q = sub.add_parser("qa", help="check the tools against the dashboard, without a model")
    q.add_argument("--only", default=None, help="a check name fragment, or a group: "
                                                "dashboard, invariant, behaviour")
    q.set_defaults(fn=cmd_qa)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
