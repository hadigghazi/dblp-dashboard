"""
A second, independent judge for DBLP-QA.

Every score so far comes from gpt-4.1, which also judged answers written by gpt-4.1 and gpt-4.1-mini,
and LLM judges are known to favour their own family's answers. This re-judges the main answer sets
with an open model from another family - Qwen2.5-14B by default, on the VM's own Ollama, so it costs
nothing but time - using the same rubric, after the same controls the first judge passed, and reports:

  * how far the two judges agree: exact agreement, Cohen's kappa, quadratic-weighted kappa (the scale
    is ordinal, so 0-vs-2 is a worse disagreement than 1-vs-2), and the confusion matrix;
  * each answer set's mean under both judges;
  * the paired contrasts the study rests on - plain RAG and the oracle against closed-book, and plain
    RAG on the questions whose source was retrieved or missed - under both judges;
  * a self-preference check: how much more gpt-4.1 gives than the second judge, per answering model.
    If gpt-4.1 favoured its own family, that gap would be larger for the GPT answers than for
    Mistral's.

Verdicts are cached and saved as they come, so an interrupted run resumes where it stopped.
"""
import hashlib
import json
import time
from pathlib import Path

from . import config, dblpqa as DQ, dblpqa_audit as AU

VERSION = 1
DEFAULT_JUDGE = "ollama:qwen2.5:14b"
FIRST_JUDGE = "gpt-4.1"


def weighted_kappa(a, b, cats=(0, 1, 2)):
    """Quadratic-weighted kappa between two judges' scores on an ordinal scale."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n, k = len(pairs), len(cats)
    if not n:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    seen = [[0] * k for _ in range(k)]
    for x, y in pairs:
        seen[idx[x]][idx[y]] += 1
    rows = [sum(seen[i]) for i in range(k)]
    cols = [sum(seen[i][j] for i in range(k)) for j in range(k)]
    weight = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    observed = sum(weight(i, j) * seen[i][j] for i in range(k) for j in range(k))
    expected = sum(weight(i, j) * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    return 1.0 if expected == 0 else round(1 - observed / expected, 3)


def answer_sets(runs, pool, modes="plain"):
    """[(condition, model, sampling, run name, {id: record})]: per answering model, its latest
    closed-book, oracle and plain-RAG runs (and the selective modes too with modes="all")."""
    chosen = [c for c in AU.rag_runs(runs, pool) if modes == "all" or DQ.rag_parts(c[0])[1] == "plain"]
    sets = []
    for model, sampling in dict.fromkeys((c[1], c[2]) for c in chosen):
        for condition in ("closed-book", "oracle"):
            name, recs = AU.latest(runs, condition, model, sampling)
            if recs:
                sets.append((condition, model, sampling, name, recs))
    order = {"closed-book": 0, "oracle": 2}
    return sorted(sets + chosen, key=lambda s: (s[2] != "ours", s[1], order.get(s[0], 1), s[0]))


def _paired(first, second, ids):
    """The difference second-minus-first over shared questions, with its interval."""
    shared = [q for q in ids if first.get(q) is not None and second.get(q) is not None]
    diffs = [second[q] - first[q] for q in shared]
    return {"questions": len(shared), "delta": DQ.bootstrap_ci(diffs) if diffs else None,
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}


def run(client, rows, judge_model=DEFAULT_JUDGE, out=print, pool=None, cache_dir=None, modes="plain",
        force=False):
    cache_dir = Path(cache_dir or config.MODELS_DIR / "dblpqa")
    runs = AU.load_runs(cache_dir / "runs")
    if pool is None:
        pool = next((s.get("pool_sha256") for _, s, _ in reversed(runs) if s.get("pool_sha256")), None)
    sets = answer_sets(runs, pool, modes)
    if not sets:
        raise SystemExit(f"no answer sets on pool {pool}")
    by_id = {r["id"]: r for r in rows}
    cache_path = cache_dir / "crossjudge-labels.json"
    cache = AU._load(cache_path)
    meter = DQ.Meter()
    started, fresh = time.time(), [0]
    todo = len(rows) * 2 + sum(len(s[4]) for s in sets)

    def grade(qid, answer):
        key = hashlib.sha1(f"{VERSION}|{judge_model}|{qid}|{answer}".encode("utf-8")).hexdigest()
        if key not in cache:
            try:
                cache[key] = DQ.judge(client, meter, judge_model, by_id[qid]["question"], by_id[qid]["answer"], answer)
            except ValueError as e:      # a reply with no score in it: left out, and counted
                cache[key] = {"score": None, "reason": f"unreadable: {e}"[:300]}
            fresh[0] += 1
            if fresh[0] % 20 == 0:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                rate = (time.time() - started) / fresh[0]
                out(f"  {fresh[0]} verdicts in {time.time() - started:.0f}s ({rate:.0f}s each; at most "
                    f"{todo} in all, cached ones are free)")
        return cache[key]["score"]

    out(f"second judge {judge_model} on pool {pool}: {len(sets)} answer sets")
    swap = DQ.derangement(len(rows))
    gold_ok = sum(grade(r["id"], r["answer"]) == 2 for r in rows)
    swap_ok = sum(grade(r["id"], rows[swap[i]]["answer"]) == 0 for i, r in enumerate(rows))
    n = len(rows)
    controls = {"gold_scored_2": gold_ok, "swapped_scored_0": swap_ok, "questions": n,
                "passed": gold_ok / n >= 0.95 and swap_ok / n >= 0.95}
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    out(f"controls: gold answers scored 2 in {gold_ok}/{n}, another question's answer scored 0 in "
        f"{swap_ok}/{n} -> {'PASS' if controls['passed'] else 'FAIL'}")
    report = {"judge": judge_model, "first_judge": FIRST_JUDGE, "pool_sha256": pool, "version": VERSION,
              "controls": controls}
    if not controls["passed"] and not force:
        report["stopped"] = "the second judge failed its controls"
        (cache_dir / "crossjudge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        out("stopped: a judge that fails its controls is not used (--force to judge anyway)")
        return report

    scores = {}
    for condition, model, sampling, name, recs in sets:
        scores[(condition, model, sampling)] = {
            q: (rec["score"], grade(q, rec["answer"])) for q, rec in recs.items() if q in by_id}
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    report.update(summarize(sets, scores))
    report["cost_usd"] = meter.cost()
    (cache_dir / "crossjudge.json").write_text(json.dumps(dict(report, scores={
        f"{c}|{m}|{s}": v for (c, m, s), v in scores.items()}), indent=2), encoding="utf-8")
    print_report(report, out)
    return report


def summarize(sets, scores):
    pairs = [(f, s, model) for (c, model, _), per_q in scores.items() for f, s in per_q.values()]
    usable = [(f, s, m) for f, s, m in pairs if f is not None and s is not None]
    agreement = {"answers": len(pairs), "unreadable": sum(1 for _, s, _ in pairs if s is None),
                 "exact": sum(f == s for f, s, _ in usable),
                 "kappa": AU.cohen_kappa([f for f, _, _ in usable], [s for _, s, _ in usable]),
                 "weighted_kappa": weighted_kappa([f for f, _, _ in usable], [s for _, s, _ in usable]),
                 "confusion": {f"{a}->{b}": sum(1 for f, s, _ in usable if (f, s) == (a, b))
                               for a in (0, 1, 2) for b in (0, 1, 2)}}

    per_set = []
    for condition, model, sampling, name, _ in sets:
        per_q = scores[(condition, model, sampling)]
        per_set.append({"condition": condition, "model": model, "sampling": sampling, "run": name,
                        "questions": len(per_q),
                        "first": DQ.bootstrap_ci([f for f, _ in per_q.values() if f is not None]),
                        "second": DQ.bootstrap_ci([s for _, s in per_q.values() if s is not None])})

    contrasts = []
    for model, sampling in dict.fromkeys((m, s) for _, m, s, _, _ in sets):
        closed = scores.get(("closed-book", model, sampling)) or {}
        for (condition, m, s), per_q in scores.items():
            if (m, s) != (model, sampling) or condition == "closed-book":
                continue
            recs = next(r for c, mm, ss, _, r in sets if (c, mm, ss) == (condition, m, s))
            groups = {"all": list(per_q)}
            if condition.startswith(DQ.RAG_PREFIX):
                groups["source retrieved"] = [q for q in per_q if recs[q].get("source_in_context")]
                groups["source missed"] = [q for q in per_q if not recs[q].get("source_in_context")]
            for group, ids in groups.items():
                contrasts.append({"model": model, "sampling": sampling, "contrast": f"{condition} - closed-book",
                                  "questions": group,
                                  "first": _paired({q: v[0] for q, v in closed.items()}, {q: v[0] for q, v in per_q.items()}, ids),
                                  "second": _paired({q: v[1] for q, v in closed.items()}, {q: v[1] for q, v in per_q.items()}, ids)})

    # self-preference: what gpt-4.1 gives beyond the second judge, per answering model
    preference = {}
    for model in dict.fromkeys(m for _, _, m in usable):
        preference[model] = DQ.bootstrap_ci([f - s for f, s, m in usable if m == model])
    return {"agreement": agreement, "sets": per_set, "contrasts": contrasts, "first_minus_second": preference}


def print_report(report, out=print):
    a = report["agreement"]
    out(f"\nagreement with {report['first_judge']} over {a['answers']} answers: exact {a['exact']}, "
        f"kappa {a['kappa']}, quadratic-weighted kappa {a['weighted_kappa']}"
        + (f"; {a['unreadable']} unreadable verdicts left out" if a["unreadable"] else ""))
    out("rows: first judge, columns: second judge  " + "  ".join(f"{k}: {v}" for k, v in a["confusion"].items()))
    out(f"\n{'model':20s} {'condition':14s} {report['first_judge']:>18s} {'second judge':>18s}")
    ci = lambda c: f"{c['mean']:6.2f} ({c['low']:.2f}-{c['high']:.2f})" if c and c["mean"] is not None else "-"
    for e in report["sets"]:
        out(f"{e['model'][-20:]:20s} {e['condition']:14s} {ci(e['first']):>18s} {ci(e['second']):>18s}")
    out(f"\n{'model':20s} {'contrast':26s} {'questions':17s} {report['first_judge'] + ' delta':>26s} {'second judge delta':>26s}")
    for c in report["contrasts"]:
        cells = []
        for side in ("first", "second"):
            d = c[side]["delta"]
            cells.append(f"{d['mean']:+.2f} ({d['low']:+.2f} to {d['high']:+.2f})" if d else "-")
        out(f"{c['model'][-20:]:20s} {c['contrast']:26s} {c['questions'] + ' (' + str(c['first']['questions']) + ')':17s} "
            f"{cells[0]:>26s} {cells[1]:>26s}")
    out("\nself-preference check - how much more the first judge gives than the second, per answering model:")
    for model, ci in report["first_minus_second"].items():
        out(f"  {model:22s} {ci['mean']:+.2f} (95% CI {ci['low']:+.2f} to {ci['high']:+.2f})")
    out(f"(if {report['first_judge']} favoured its own family, the GPT rows would sit above Mistral's)  "
        f"${report.get('cost_usd', 0)}")
