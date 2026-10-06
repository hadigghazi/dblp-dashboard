"""
Dewey end to end on DBLP-QA and DBLP-QA-Fresh.

The study's conditions send a fixed prompt straight to a model. Here each question goes to Dewey as
a user would type it - its router, its tool catalogue, its rules - and the answer it writes is judged
exactly like every other run: the same judge, the same controls, paired question by question with the
study's runs. Three variants:

  dewey         as deployed: the abstract search runs live (dblp's title search and OpenAlex);
  dewey-frozen  the abstract search ranks the study's frozen pool for each question instead of
                searching, so Dewey is given the candidates the fixed pipeline was given; what still
                differs is the agent - whether it searches at all, the query it sends, its prompt, and
                which of its models writes the answer;
  dewey-v0      Dewey before it could read abstracts: the tool switched off and the old rule back,
                under which such questions are declined.

Every record keeps what the agent did: the tools it called, the query it sent, the papers it was
given, whether the source paper was among them, the [n] it cited, which model wrote the answer, and
rounds, seconds and cost. Dewey's own spending goes to a ledger of its own, so an evaluation never
uses up the website's daily budget.

`retrieval_check` runs the live abstract search alone, without a model, on every question as it is
written - the tool's Recall@k against the frozen pipeline's - and fills the search cache, so the live
variant's searches with the question as written cost nothing again.
"""
import json
import logging
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from . import agent, budget, config, content, dblpqa as DQ, dblpqa_rag as RAG, dblpqa_replicate as RP
from .llm import LLMError

log = logging.getLogger("dblp.chat.dblpqa_dewey")

VARIANTS = ("dewey", "dewey-frozen", "dewey-v0")
MODEL = "dewey"
CITE = re.compile(r"\[(\d+)\]")
# what each Dewey run is set beside: (label, condition, model, sampling, on the same pool)
BASELINES = [
    ("closed-book, gpt-4.1-mini (Dewey's answering model)", "closed-book", "gpt-4.1-mini", "ours", False),
    ("closed-book, gpt-4.1", "closed-book", "gpt-4.1", "ours", False),
    ("plain RAG, gpt-4.1-mini", "rag-bm25", "gpt-4.1-mini", "ours", True),
    ("plain RAG, gpt-4.1", "rag-bm25", "gpt-4.1", "ours", True),
    ("permissive RAG, gpt-4.1", "rag-bm25-permissive", "gpt-4.1", "ours", True),
    ("RAGScholar's configuration (Mistral-7B, BM25 top-5 concatenated)", "rag-bm25", "ollama:mistral:v0.1",
     "paper", True),
    ("oracle, gpt-4.1-mini", "oracle", "gpt-4.1-mini", "ours", False),
    ("Dewey as deployed", "dewey", MODEL, "ours", False),
    ("Dewey on the frozen pool", "dewey-frozen", MODEL, "ours", True),
    ("Dewey before the abstract search", "dewey-v0", MODEL, "ours", False),
]


def _norm(text):
    return " ".join((text or "").lower().split())


def source_hits(payloads, aliases, gold_title):
    """What the abstract search gave the model, and whether the source paper was in it: by the study's
    rule, a returned record is the source if its key is one of the source's aliases or its normalised
    title is the source's. The source counts as given only with its abstract."""
    retrieved, first_rank, listed, given, calls = [], None, False, False, 0
    for p in payloads:
        out = p.get("result") or {}
        if p.get("name") != "search_abstracts" or out.get("refused"):
            continue
        calls += 1
        for row in out.get("rows") or []:
            key = row.get("key")
            retrieved.append(key)
            is_source = key in aliases or (gold_title and RAG.norm_title(row.get("title")) == gold_title)
            if not is_source:
                continue
            listed = True
            if row.get("abstract") and row["abstract"] != "(no abstract available)":
                given = True
            if first_rank is None and calls == 1:
                first_rank = row.get("n")
    return {"retrieved": retrieved, "source_listed": listed, "source_in_context": given,
            "source_rank": first_rank, "content_calls": calls}


def citations(answer, payloads):
    """The [n] an answer cites, and any n that no abstract search returned."""
    numbers = sorted({int(n) for n in CITE.findall(answer or "")})
    shown = max((len((p.get("result") or {}).get("rows") or []) for p in payloads
                 if p.get("name") == "search_abstracts"), default=0)
    return {"cited": numbers, "invalid": [n for n in numbers if n < 1 or n > shown]}


def _paired(mine, theirs):
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        return None
    diffs = [mine[q] - theirs[q] for q in shared]
    return {"questions": len(shared), "delta": DQ.bootstrap_ci(diffs),
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}


def summarize(records, condition, runs_dir, pool):
    judged = [r for r in records if r.get("score") is not None]
    scores = {r["id"]: r["score"] for r in judged}
    seconds = sorted(r["seconds"] for r in records if r.get("seconds") is not None)
    writers = {}
    for r in records:
        writers[r.get("writer") or "none"] = writers.get(r.get("writer") or "none", 0) + 1
    entry = {
        "answered": len(records), "judged": len(judged),
        "judge_score": DQ.bootstrap_ci(list(scores.values())) if scores else None,
        "distribution": {s: list(scores.values()).count(s) for s in (2, 1, 0)},
        "rouge_l": round(sum(r["rouge_l"] for r in records) / max(1, len(records)), 4),
        "answer_words": round(statistics.mean(r["answer_words"] for r in records), 1) if records else None,
        "called_the_abstract_search": sum(1 for r in records if r["content_calls"]),
        "query_as_written": sum(1 for r in records if r["content_calls"] and r["query_verbatim"]),
        "source_in_context": sum(1 for r in records if r["source_in_context"]),
        "source_in_first_five": sum(1 for r in records if r["source_rank"]),
        "answers_citing": sum(1 for r in records if r["cited"]),
        "invalid_citations": sum(len(r["invalid_citations"]) for r in records),
        "writers": writers, "errors": sum(1 for r in records if r.get("error")),
        "degraded_searches": sum(1 for r in records if r.get("search_degraded")),
        "median_seconds": seconds[len(seconds) // 2] if seconds else None,
        "p90_seconds": seconds[int(0.9 * (len(seconds) - 1))] if seconds else None,
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in records), 4),
        "paired": {}, "by_retrieval": {},
    }
    for label, cond, model, sampling, same_pool in BASELINES:
        if cond == condition:
            continue
        name, recs = RP.latest_answers(runs_dir, cond, model, sampling, pool if same_pool else None)
        theirs = {q: r["score"] for q, r in recs.items() if r.get("score") is not None}
        got = _paired(scores, theirs) if name else None
        if got:
            entry["paired"][label] = dict(got, against_run=name)
    base_name, base = RP.latest_answers(runs_dir, "closed-book", "gpt-4.1-mini", "ours")
    base_scores = {q: r["score"] for q, r in base.items() if r.get("score") is not None}
    for label, hit in (("source given", True), ("source not given", False)):
        part = {r["id"]: r["score"] for r in judged if bool(r["source_in_context"]) == hit}
        if part:
            entry["by_retrieval"][label] = {"questions": len(part), "judge_score": DQ.bootstrap_ci(list(part.values())),
                                            "vs_closed_book_mini": _paired(part, base_scores)}
    return entry


def print_summary(entry, condition, out=print):
    ci = entry["judge_score"]
    if not ci:
        out(f"{condition}: {entry['answered']} answered, none scored")
        return
    d = entry["distribution"]
    out(f"{condition}: {ci['mean']:.2f} / 2 (95% CI {ci['low']:.2f}-{ci['high']:.2f})  2s: {d[2]}  1s: {d[1]}  "
        f"0s: {d[0]}  ROUGE-L {entry['rouge_l']:.3f}  {entry['answer_words']} words per answer")
    n = entry["answered"]
    out(f"  abstract search called on {entry['called_the_abstract_search']}/{n} questions "
        f"({entry['query_as_written']} with the question as written); source paper given on "
        f"{entry['source_in_context']}/{n}, in the first call's top five on {entry['source_in_first_five']}")
    out(f"  answers citing [n]: {entry['answers_citing']}/{n}, citations to nothing shown: {entry['invalid_citations']}; "
        f"written by {entry['writers']}; errors {entry['errors']}")
    out(f"  median {entry['median_seconds']}s, p90 {entry['p90_seconds']}s, ${entry['cost_usd']} for Dewey's own calls")
    for label, p in entry["paired"].items():
        dd = p["delta"]
        out(f"  vs {label}: {dd['mean']:+.2f} (95% CI {dd['low']:+.2f} to {dd['high']:+.2f}) on {p['questions']} "
            f"questions; better on {p['better']}, worse on {p['worse']}")
    for label, part in entry["by_retrieval"].items():
        line = f"  {label}: {part['questions']} questions, {part['judge_score']['mean']:.2f}"
        if part["vs_closed_book_mini"]:
            line += f", vs closed-book gpt-4.1-mini {part['vs_closed_book_mini']['delta']['mean']:+.2f}"
        out(line)


def _answer(ctx, client, question, ledger, payloads):
    """One question through Dewey, retried once if the provider was only busy."""
    for attempt in range(2):
        payloads.clear()
        done = agent.answer(ctx, client, question, emit=lambda _e: None, ledger=ledger, collect=payloads,
                            channel="cli")
        if not done.get("error") or done.get("error_kind") in ("credit", "key") or attempt:
            return done
        time.sleep(20)
    return done


def run(ctx, client, rows, sha, variant="dewey", judge_model="gpt-4.1", allow_incomplete=False, out=print):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    pools, _rankings, report = RAG.prepare(rows, frozen=True, allow_incomplete=allow_incomplete,
                                           out=lambda *_: None)
    pool = report["pool"]["sha256"]
    oracle = DQ.fetch_abstracts(rows, out=lambda *_: None)
    runs_dir = DQ.study_dir() / "runs"
    meter = DQ.Meter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = runs_dir / f"{stamp}-{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    controls = DQ.reusable_controls(judge_model, sha, runs_dir, len(rows))
    if controls:
        out(f"judge controls: passed on this dataset in run {controls['reused_from']} - reused")
    else:
        controls = DQ.judge_controls(client, meter, judge_model, rows, out)
        if not controls["passed"]:
            out("stopped: the judge failed its controls")
            return None
    ledger = budget.Ledger(path=DQ.study_dir() / "dewey-ledger.json")
    was = config.CONTENT_TOOL
    config.CONTENT_TOOL = variant != "dewey-v0"
    started, records, stopped = time.time(), [], None
    try:
        with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
            for i, row in enumerate(rows, 1):
                qid = row["id"]
                ctx.frozen_pool = pools.get(qid) if variant == "dewey-frozen" else None
                payloads = []
                done = _answer(ctx, client, row["question"], ledger, payloads)
                if done.get("error_kind") in ("credit", "key"):
                    stopped = f"stopped at {qid}: {done.get('error')}"
                    out(f"!! {stopped}")
                    break
                text = done.get("answer") or ""
                verdict = DQ.judge(client, meter, judge_model, row["question"], row["answer"], text) if text else \
                    {"score": 0, "reason": "no answer: " + str(done.get("error"))}
                aliases = set((pools.get(qid) or {}).get("aliases") or [row["dblp_key"]])
                hits = source_hits(payloads, aliases, RAG.norm_title((oracle.get(qid) or {}).get("title")))
                cites = citations(text, payloads)
                queries = [(p.get("arguments") or {}).get("question") for p in payloads
                           if p.get("name") == "search_abstracts"]
                metas = [((p.get("result") or {}).get("meta") or {}) for p in payloads
                         if p.get("name") == "search_abstracts"]
                rec = {"condition": variant, "model": MODEL, "id": qid, "question": row["question"],
                       "gold": row["answer"], "answer": text, "score": verdict["score"], "reason": verdict["reason"],
                       "rouge_l": round(DQ.rouge_l(text, row["answer"]), 4), "answer_words": len(text.split()),
                       "writer": done.get("model"), "rounds": done.get("rounds"), "tools": done.get("tools") or [],
                       "queries": queries,
                       "query_verbatim": bool(queries) and all(_norm(q) == _norm(row["question"]) for q in queries if q),
                       "cited": cites["cited"], "invalid_citations": cites["invalid"],
                       "seconds": done.get("seconds"), "cost_usd": done.get("cost_usd"), "error": done.get("error"),
                       "search_degraded": any(m.get("degraded") or any(v != "ok" for v in (m.get("sources") or {}).values())
                                              for m in metas),
                       "search_cached": [m.get("cached") for m in metas],
                       **hits}
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(rows):
                    out(f"  {variant}: {i}/{len(rows)} answered, {time.time() - started:.0f}s")
    finally:
        config.CONTENT_TOOL = was
        ctx.frozen_pool = None
    entry = summarize(records, variant, runs_dir, pool)
    print_summary(entry, variant, out)
    payload = {
        "condition": variant, "run_at": stamp, "dataset_sha256": sha, "questions": len(records),
        "questions_in_dataset": len(rows), "models": [MODEL], "judge": judge_model,
        "sampling": {"ours": DQ.SAMPLING["ours"]}, "judge_controls": controls, "results": {MODEL: entry},
        "pool_sha256": pool, "stopped": stopped,
        "agent": {"router": config.MODEL_FAST, "deep": config.MODEL_DEEP,
                  "escalate_after_rounds": config.ESCALATE_AFTER_ROUNDS, "max_rounds": config.MAX_ROUNDS,
                  "max_tool_calls": config.MAX_TOOL_CALLS, "time_budget": config.TIME_BUDGET_SECONDS,
                  "content_tool": variant != "dewey-v0", "content_top": config.CONTENT_TOP,
                  "content_pool": config.CONTENT_POOL},
        "judge_cost_usd": meter.cost(), "seconds": round(time.time() - started, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out(f"\njudge ${meter.cost()}, Dewey ${entry['cost_usd']} · written to {out_dir}")
    return payload


def retrieval_check(ctx, rows, allow_incomplete=False, out=print):
    """The live abstract search on every question as written, without a model: where it puts the
    source paper (the study's rule), how many candidates and abstracts it has, how long it takes."""
    pools, _rankings, report = RAG.prepare(rows, frozen=True, allow_incomplete=allow_incomplete,
                                           out=lambda *_: None)
    oracle = DQ.fetch_abstracts(rows, out=lambda *_: None)
    ranks, per_q, times, statuses = [], {}, [], {}
    for i, row in enumerate(rows, 1):
        t0 = time.time()
        ctx.frozen_pool = None
        cands, status, degraded, cached = content.pool_for(ctx, row["question"], t0 + config.CONTENT_DEADLINE)
        ranking = content.rank(row["question"], cands)
        took = time.time() - t0
        aliases = set((pools.get(row["id"]) or {}).get("aliases") or [row["dblp_key"]])
        gold_title = RAG.norm_title((oracle.get(row["id"]) or {}).get("title"))
        rank = next((n for n, k in enumerate(ranking, 1)
                     if k in aliases or (gold_title and RAG.norm_title(cands[k].get("title")) == gold_title)), None)
        ranks.append(rank)
        if not cached:
            times.append(took)
        for source, state in status.items():
            statuses.setdefault(source, {}).setdefault(state, 0)
            statuses[source][state] += 1
        per_q[row["id"]] = {"rank": rank, "candidates": len(cands),
                            "with_abstract": sum(1 for c in cands.values() if c.get("abstract")),
                            "seconds": round(took, 2), "cached": cached, "degraded": degraded}
        if i % 10 == 0:
            out(f"  {i}/{len(rows)} searched")
    times.sort()
    result = {"metrics": RAG.retrieval_metrics(ranks), "questions": per_q, "sources": statuses,
              "pool_sha256_for_aliases": report["pool"]["sha256"],
              "frozen_pipeline": {k: report["rankers"]["bm25"][k] for k in ("recall@1", "recall@5", "ranked_at_all")},
              "mean_candidates": round(statistics.mean(q["candidates"] for q in per_q.values()), 1),
              "share_with_abstract": round(sum(q["with_abstract"] for q in per_q.values())
                                           / max(1, sum(q["candidates"] for q in per_q.values())), 3),
              "live_seconds": {"median": times[len(times) // 2] if times else None,
                               "p90": times[int(0.9 * (len(times) - 1))] if times else None, "searched": len(times)}}
    (DQ.study_dir() / "dewey-retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    m, f = result["metrics"], result["frozen_pipeline"]
    out(f"Dewey's live abstract search, questions as written: R@1 {m['recall@1']:.2f}, R@5 {m['recall@5']:.2f}, "
        f"found {m['ranked_at_all']:.2f} (frozen pipeline: {f['recall@1']:.2f} / {f['recall@5']:.2f} / "
        f"{f['ranked_at_all']:.2f}); {result['mean_candidates']} candidates, "
        f"{result['share_with_abstract']:.0%} with an abstract; sources {statuses}; "
        f"live search median {result['live_seconds']['median']}s, p90 {result['live_seconds']['p90']}s")
    return result
