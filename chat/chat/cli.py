"""
Command line, for the VM.

  python -m chat.cli store                      build the leaderboard store for this dump
  python -m chat.cli paper-ids                  build the DOI/arXiv index the abstract search maps papers with
  python -m chat.cli tools                      list the tool catalogue
  python -m chat.cli ask "who has most papers"  one question, printed as it happens
  python -m chat.cli evaluate [--limit N]       run the gold set, write the report
  python -m chat.cli qa [--only X]              check the tools against the dashboard (no model calls)
  python -m chat.cli questions [--days N]      what people asked, and what the catalogue is missing
  python -m chat.cli search-eval [--papers N]  can search find a paper from a description of it?
  python -m chat.cli scenarios [--suite S]     one person's questions about themselves, answers in full
  python -m chat.cli dblpqa closed-book         DBLP-QA with no retrieval, judged automatically
  python -m chat.cli dblpqa retrieval           DBLP-QA: does retrieval find each question's paper?
  python -m chat.cli dblpqa rag --ranker bm25   DBLP-QA answered from the top-5 retrieved abstracts
  python -m chat.cli dblpqa rag --mode gated    ... keeping only the abstracts a relevance check passes
  python -m chat.cli dblpqa audit               which questions one gold answer can grade, and why points were lost
  python -m chat.cli dblpqa regrade             the missed questions graded fairly: any correct answer counts
  python -m chat.cli dblpqa crossjudge          the main answer sets re-judged by an open model of another family
  python -m chat.cli dblpqa fresh-build         DBLP-QA-Fresh: questions from papers newer than the models
  python -m chat.cli dblpqa grid --models M     the original paper's ten context variants for M, judged
  python -m chat.cli dblpqa bearing             the original paper's RQ1: does a top abstract answer the question?
  python -m chat.cli dblpqa replication         the grid beside the original paper's Table 3, finding by finding
  python -m chat.cli dblpqa dewey --variant V   Dewey itself answers every question (live, frozen pool, or v0)
  python -m chat.cli dblpqa dewey-retrieval     Dewey's live abstract search alone: where it puts the source paper
  python -m chat.cli dblpqa structured-build    questions about dblp's records, answers by SQL over the parquet
  python -m chat.cli dblpqa structured --arm A  Dewey, RAGScholar's pipeline or no retrieval on them (no judge)
  python -m chat.cli dblpqa structured-compare  the arms side by side, Dewey against each
  python -m chat.cli dblpqa --dataset fresh ... any of the above on DBLP-QA-Fresh instead

`ask` is the same code path the web endpoint uses, so a question that works here works there.
"""
import argparse
import json
import logging
import sys

import httpx

from . import agent, budget, config, data, evaluate as E, paperids, qa as Q, searcheval as SE, store, tools as T, usage
from .llm import Client


def _ready():
    """The same state the server builds: serving database attached, leaderboard store attached."""
    if not data.pool.load():
        sys.exit(data.pool.error)
    store_meta = store.attach(data.pool.connection(), data.pool.meta)
    paperids.attach(data.pool.connection(), data.pool.meta)
    return agent.Ctx(data.pool, httpx.Client(timeout=config.UPSTREAM_TIMEOUT), store_meta)


def cmd_store(_args):
    con, meta = data.connect()
    try:
        path = store.build(con, meta)
        print(f"built {path}")
    finally:
        con.close()


def cmd_paper_ids(_args):
    """The DOI/arXiv index the abstract search maps OpenAlex papers to dblp with: one scan of the
    parquet per dump. The service picks it up within five minutes."""
    con, meta = data.connect()
    try:
        path = paperids.build(con, meta)
        con.execute(f"ATTACH '{path}' AS x (READ_ONLY)")
        print(f"built {path}: {dict(con.execute('SELECT k, v FROM x._meta').fetchall())}")
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
    if args.no_content_tool:
        config.CONTENT_TOOL = False          # Dewey as it was before the abstract search, for comparison
    payload = E.run(ctx, client, limit=args.limit, only=args.only)
    if args.only:
        # a diagnostic subset: printed, never saved over the full report
        print(json.dumps(payload["summary"], indent=2))
        for case in payload.get("results") or []:
            print(f"\n== {case['question']} | passed {case['passed']} {case['reason']}\n   tools {case['tools']}\n"
                  f"   {case['answer']}")
        return
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
    DQ.use_dataset("fresh" if args.condition == "fresh-build" else args.dataset)
    if args.condition == "fresh-build":
        from . import dblpqa_fresh as FR
        FR.build(client, target=args.target, candidates=args.candidates)
        return
    if args.condition == "report":
        DQ.report(args.run)
        return
    if args.condition in ("structured-build", "structured", "structured-compare"):
        # dblp's records rather than DBLP-QA's questions: its own question set and folder
        from . import dblpqa_structured as ST
        if args.condition == "structured-build":
            ST.build(seed=args.seed)
        elif args.condition == "structured-compare":
            ST.compare()
        else:
            ctx = _ready()
            arms_models = ["dewey"] if args.arm == "dewey" else [m.strip() for m in args.models.split(",") if m.strip()]
            for model in arms_models:
                ST.run(ctx, client, args.arm, model=model, limit=args.limit)
        return
    if args.condition == "rescore":
        if not args.run:
            sys.exit("rescore needs --run <run directory name>")
        DQ.rescore(client, args.run, args.judge)
        return
    rows, sha = DQ.load_dataset()
    if args.limit:
        rows = rows[:args.limit]
    if args.condition == "crossjudge":
        from . import dblpqa_crossjudge as CJ
        CJ.run(client, rows, judge_model=args.second_judge, pool=args.pool, modes=args.modes or "plain",
               force=args.force, include=[c.strip() for c in (args.include or "").split(",") if c.strip()])
        return
    if args.condition in ("audit", "regrade"):
        from . import dblpqa_audit as AU
        if args.condition == "audit":
            AU.run(client, rows, pool=args.pool)
        elif args.judge == "none":
            sys.exit("regrade needs a judge")
        else:
            AU.regrade(client, rows, pool=args.pool, judge_model=args.judge, force=args.force,
                       version=args.regrade_version, modes=args.modes)
        return
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.condition in ("dewey", "dewey-retrieval"):
        # Dewey itself: the same serving database, store and paper ids as the web endpoint
        from . import dblpqa_dewey as DW
        ctx = _ready()
        if args.condition == "dewey-retrieval":
            DW.retrieval_check(ctx, rows, allow_incomplete=args.allow_incomplete_pool)
            return
        variant = {"live": "dewey", "frozen": "dewey-frozen", "v0": "dewey-v0"}[args.variant]
        DW.run(ctx, client, rows, sha, variant=variant, judge_model=args.judge,
               allow_incomplete=args.allow_incomplete_pool)
        return
    if args.condition in ("grid", "bearing", "replication"):
        from . import dblpqa_replicate as RP
        if args.condition == "bearing":
            rankers = [r.strip() for r in (args.rankers or ",".join(RP.BEARING_RANKERS)).split(",") if r.strip()]
            RP.run_bearing(client, rows, rankers=rankers, judge_model=args.judge,
                           allow_incomplete=args.allow_incomplete_pool, force=args.force)
            return
        if args.condition == "grid":
            labels = [v.strip() for v in args.variants.split(",")] if args.variants else []
            unknown = [v for v in labels if v not in RP.BY_LABEL]
            if unknown:
                sys.exit(f"unknown variants {unknown}; the paper's are {', '.join(RP.BY_LABEL)}")
            variants = [RP.BY_LABEL[v] for v in labels] or list(RP.VARIANTS)
            RP.run_grid(client, rows, sha, models, sampling=args.sampling, judge_model=args.judge,
                        ranker=args.ranker, variants=variants, allow_incomplete=args.allow_incomplete_pool)
        RP.report(rows, models, sampling=args.sampling, ranker=args.ranker,
                  allow_incomplete=args.allow_incomplete_pool)
        return
    condition, contexts = args.condition, None
    notes = None
    if condition in ("retrieval", "rag"):
        from . import dblpqa_rag as RAG
        # retrieval builds (and finishes) the pool; rag only reads it, so every rag run sees the same one
        pools, rankings, report = RAG.prepare(rows, frozen=condition == "rag",
                                              allow_incomplete=args.allow_incomplete_pool)
        RAG.print_retrieval(report)
        if condition == "retrieval":
            return
        if args.strategy != "cd":
            # one of the paper's other strategies: contexts depend on the model (Concatenated Answers
            # are built from that model's own single-abstract answers), so one run per model
            from . import dblpqa_replicate as RP
            if args.mode != "plain":
                sys.exit("--mode applies to concatenated documents (--strategy cd) only")
            for model in models:
                variant = (args.strategy, args.k)
                contexts, notes = RP.contexts_for(variant, rows, pools, rankings, args.ranker, model,
                                                  args.sampling, report["pool"]["sha256"],
                                                  DQ.study_dir() / "runs")
                DQ.run_condition(client, [model], args.judge, rows, sha, RP.condition_for(variant, args.ranker),
                                 contexts=contexts, force=args.force, sampling=args.sampling,
                                 reuse_controls=not args.recheck_judge and not args.limit, notes=notes)
            return
        contexts = RAG.rag_contexts(rows, pools, rankings, args.ranker, k=args.k)
        notes = {"pool_sha256": report["pool"]["sha256"],
                 "retrieval": {"ranker": args.ranker, "k": args.k, "mode": args.mode}}
        if args.mode == "gated":
            contexts, notes["gate"] = RAG.gate_contexts(client, rows, pools, contexts, args.gate_model)
        condition = RAG.condition_name(args.ranker, args.mode, "cd", args.k)
    elif condition == "oracle":
        contexts = DQ.fetch_abstracts(rows)
    print(f"DBLP-QA: {len(rows)} questions (sha256 {sha[:12]}), {condition}, judge {args.judge}, "
          f"sampling {args.sampling}")
    DQ.run_condition(client, models, args.judge, rows, sha, condition, contexts=contexts,
                     force=args.force, sampling=args.sampling,
                     reuse_controls=not args.recheck_judge and not args.limit, notes=notes)


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
    sub.add_parser("paper-ids", help="build the DOI/arXiv index for the abstract search").set_defaults(fn=cmd_paper_ids)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.set_defaults(fn=cmd_ask)
    ev = sub.add_parser("evaluate", help="run the gold set")
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--only", default=None,
                    help="comma-separated fragments: run only the cases whose question contains one (not saved)")
    ev.add_argument("--no-content-tool", action="store_true",
                    help="switch the abstract search off, as Dewey was before it")
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
    dq.add_argument("condition", choices=["closed-book", "oracle", "retrieval", "rag", "audit", "regrade", "crossjudge", "report",
                             "rescore", "fresh-build", "grid", "bearing", "replication", "dewey",
                             "dewey-retrieval", "structured-build", "structured", "structured-compare"],
                    help="oracle = each question with the abstract it was written from; retrieval = where "
                         "each ranker puts that paper (no answers); rag = answers from the top --k papers "
                         "of --ranker; rescore = score the unscored answers of --run; grid = the original "
                         "paper's ten context variants for --models; bearing = the paper's RQ1 measure "
                         "(does a top abstract answer the question?); replication = the grid beside the "
                         "paper's Table 3")
    dq.add_argument("--arm", choices=["dewey", "rag", "closed"], default="dewey",
                    help="for structured: Dewey, RAGScholar's pipeline over the records (--models), or the "
                         "question alone (--models)")
    dq.add_argument("--seed", type=int, default=7, help="for structured-build: which questions are drawn")
    dq.add_argument("--variant", choices=["live", "frozen", "v0"], default="live",
                    help="for dewey: live = as deployed; frozen = its abstract search ranks the study's frozen "
                         "pool; v0 = as it was before the abstract search (declines content questions)")
    dq.add_argument("--strategy", choices=["cd", "single", "ca"], default="cd",
                    help="for rag: the paper's context strategies - cd = top --k abstracts concatenated; "
                         "single = only the --k-th ranked abstract (A1-A5); ca = an answer per top --k "
                         "abstract, combined into one (needs the single runs first)")
    dq.add_argument("--variants", default=None,
                    help="for grid: comma-separated, e.g. A1,Top-3-CD (default: all ten)")
    dq.add_argument("--include", default=None,
                    help="for crossjudge: only these conditions (e.g. dewey,dewey-frozen) and the baselines "
                         "they are compared with")
    dq.add_argument("--rankers", default=None,
                    help="for bearing: comma-separated rankers (default: bm25, dense, hybrid and three first stages)")
    dq.add_argument("--ranker", default="bm25",
                    choices=["bm25", "dense", "hybrid", "dblp-search", "s2-search", "openalex-search",
                             "openalex-semantic"],
                    help="for rag: bm25 is RAGScholar's method, hybrid = bm25 + embeddings fused")
    dq.add_argument("--mode", default="plain", choices=["plain", "permissive", "gated"],
                    help="for rag: permissive = told the abstracts may be off-topic; gated = a relevance "
                         "check keeps only abstracts that address the question (none kept = closed-book)")
    dq.add_argument("--gate-model", default="gpt-4.1-mini", help="for --mode gated")
    dq.add_argument("--dataset", choices=["dblpqa", "fresh"], default="dblpqa",
                    help="the original 50 questions, or DBLP-QA-Fresh (built by fresh-build)")
    dq.add_argument("--target", type=int, default=100, help="for fresh-build: questions to keep")
    dq.add_argument("--candidates", type=int, default=None,
                    help="for fresh-build: papers to sample (default 4x --target)")
    dq.add_argument("--second-judge", default="ollama:qwen2.5:14b",
                    help="for crossjudge: the independent judge, normally an open model on the local Ollama")
    dq.add_argument("--modes", choices=["plain", "all"], default=None,
                    help="for crossjudge and regrade: plain RAG only, or the permissive and gated runs too "
                         "(default: plain for crossjudge, all for regrade)")
    dq.add_argument("--regrade-version", type=int, choices=[1, 2], default=1,
                    help="for regrade: 1 = answer before the abstracts (the method); 2 = after them (faster "
                         "on a local judge; a variant, gpt-4.1 failed its controls with it)")
    dq.add_argument("--pool", default=None,
                    help="for audit/regrade: the retrieval pool's fingerprint (default: the latest rag run's)")
    dq.add_argument("--allow-incomplete-pool", action="store_true",
                    help="for rag: run even if some first-stage search never succeeded")
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
