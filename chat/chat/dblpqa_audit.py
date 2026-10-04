"""
An automatic audit of DBLP-QA: which questions a single gold answer can fairly grade, and why answers
lost points when retrieval missed the source paper.

With realistic retrieval the strong models scored below their own closed-book answers on questions
whose source paper was not retrieved, and a relevance gate did not help: it judged the wrong papers'
abstracts relevant. The likely reason is the benchmark, not the retrieval. Each question was written
from one abstract and graded against that abstract's answer, yet many are worded generically ("Why is
a new design for SLO auditing needed?"), so an answer drawn from another, equally relevant paper is
graded as wrong. This module measures how much of the loss that explains, without anyone labelling by
hand:

  * every question is labelled general (textbook knowledge), identifiable (specific enough to find
    the paper) or underspecified (many papers could answer it, differently);
  * every answer that lost points on a question whose source was not retrieved is labelled
    valid_other_paper (a correct answer to the question as worded, supported by a retrieved abstract
    from another paper), misled (it used the abstracts but does not answer the question as worded) or
    own_knowledge (not based on the abstracts).

Two labellers, gpt-4.1 and gpt-4.1-mini, label everything independently; their agreement (Cohen's
kappa) is reported and gpt-4.1's labels are used. Labels are cached, so a re-run pays for nothing
already labelled.

The audit credits only RAG answers, and its labellers have no controls, so it cannot by itself say
that retrieval helps on the missed questions. `regrade` can: on the questions where retrieval missed
the source, every answer set - closed-book included - is graded again by a judge that accepts any
correct answer to the question as worded, given the same evidence for all (the reference answer and
the plain run's five retrieved abstracts). The judge must first pass three controls on those very
questions: the reference answer scores 2, another question's reference scores 0, and another
question's RAG answer scores 0 - the last one catches a judge that rewards any fluent, on-topic-looking
text.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from . import config, dblpqa as DQ

VERSION = 1
LABELERS = ("gpt-4.1", "gpt-4.1-mini")
QUESTION_LABELS = ("general", "identifiable", "underspecified")
ANSWER_LABELS = ("valid_other_paper", "misled", "own_knowledge")

QUESTION_SYSTEM = (
    "You audit a question-answering benchmark about computer-science research. Each question was written "
    "from one paper's abstract, and its ground-truth answer was taken from that abstract. Put the question "
    "in exactly one category:\n"
    "- general: the ground truth is standard knowledge in the field; an expert would give essentially this "
    "answer without knowing the paper.\n"
    "- identifiable: the ground truth is specific to this paper, and the question has enough specific "
    "detail (a named system or method, an unusual combination of topics) to tell which paper it asks about.\n"
    "- underspecified: the ground truth is specific to this paper, but the question as worded does not "
    "identify the paper - many papers could answer it, with different correct answers.\n"
    'Reply with JSON only: {"label": "general" | "identifiable" | "underspecified", "reason": "<one sentence>"}')

ANSWER_SYSTEM = (
    "You audit answers from a question-answering benchmark about computer-science research. Each question "
    "was written from one paper's abstract, and its ground truth comes from that paper. For this question "
    "the search did NOT find that paper: the system answered from other papers' abstracts, and its answer "
    "was graded below full marks against the ground truth. Put the answer in exactly one category:\n"
    "- valid_other_paper: the answer is a correct, reasonable answer to the question as it is worded, and "
    "it is supported by one of the given abstracts; it describes a different paper's work than the ground "
    "truth, so it lost marks only because the question does not say which paper it means.\n"
    "- misled: the answer relies on the given abstracts but does not correctly answer the question as "
    "worded (it answers a different question, misreads an abstract, or applies a claim where it does not "
    "fit).\n"
    "- own_knowledge: the answer is not based on the given abstracts, and it falls short of the ground truth.\n"
    'Reply with JSON only: {"label": "valid_other_paper" | "misled" | "own_knowledge", "reason": "<one sentence>"}')


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_label(text, allowed):
    """{"label", "reason"} from a labeller's reply; ValueError if there is no allowed label in it."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        got = json.loads(match.group(0)) if match else {}
    except ValueError:
        got = {}
    label = str(got.get("label") or "").strip().lower()
    if label not in allowed:
        found = [a for a in allowed if a in (text or "").lower()]
        if len(found) != 1:
            raise ValueError(f"no single label in {(text or '')[:200]!r}")
        label = found[0]
    return {"label": label, "reason": str(got.get("reason", ""))[:300]}


def cohen_kappa(a, b):
    """Agreement between two labellers beyond chance; 1 is perfect, 0 is chance."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if not n:
        return None
    observed = sum(x == y for x, y in pairs) / n
    cats = {x for p in pairs for x in p}
    expected = sum(sum(x == c for x, _ in pairs) / n * sum(y == c for _, y in pairs) / n for c in cats)
    return 1.0 if expected == 1 else round((observed - expected) / (1 - expected), 3)


def _label(client, meter, labeler, system, prompt, allowed):
    for attempt in range(2):
        step = client.complete([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                               model=labeler, temperature=0)
        meter.add(labeler, step.get("usage", {}))
        try:
            return parse_label(step.get("content"), allowed)
        except ValueError as e:
            error = str(e)
    return {"label": None, "reason": f"unreadable: {error}"}


# --------------------------------------------------------------------------- the runs

def load_runs(runs_dir):
    """[(name, summary, records)] for every run with answers, oldest first."""
    found = []
    for path in sorted(Path(runs_dir).glob("*/summary.json")):
        answers = path.parent / "answers.jsonl"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            records = [json.loads(x) for x in answers.read_text(encoding="utf-8").splitlines() if x.strip()]
        except (OSError, ValueError):
            continue
        found.append((path.parent.name, summary, records))
    return found


def sampling_of(summary):
    return next(iter(summary.get("sampling") or {"ours": None}))


def latest(runs, condition, model, sampling, pool=None):
    """(run name, {question id: scored record}) of the latest `condition` run with this model."""
    for name, summary, records in reversed(runs):
        if summary.get("condition") != condition or model not in (summary.get("models") or []):
            continue
        if sampling_of(summary) != sampling or (pool is not None and summary.get("pool_sha256") != pool):
            continue
        mine = {r["id"]: r for r in records if r["model"] == model and r.get("score") is not None}
        if mine:
            return name, mine
    return None, {}


def rag_runs(runs, pool):
    """The latest run per (rag condition, model, sampling) on this pool: [(condition, model, sampling,
    run name, {id: record})]."""
    seen, chosen = set(), []
    for name, summary, records in reversed(runs):
        condition = summary.get("condition") or ""
        if not condition.startswith(DQ.RAG_PREFIX) or summary.get("pool_sha256") != pool:
            continue
        for model in summary.get("models") or []:
            key = (condition, model, sampling_of(summary))
            if key in seen:
                continue
            mine = {r["id"]: r for r in records if r["model"] == model and r.get("score") is not None}
            if mine:
                seen.add(key)
                chosen.append((condition, model, sampling_of(summary), name, mine))
    return sorted(chosen, key=lambda c: (c[2] != "ours", c[1], c[0]))


def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 2) if values else None


def _blocks(cands, keys):
    return "\n\n".join(f"[{i}] {cands.get(k, {}).get('title') or ''}\n{cands.get(k, {}).get('abstract') or '(no abstract)'}"
                       for i, k in enumerate(keys, 1))


# --------------------------------------------------------------------------- the audit

def run(client, rows, out=print, pool=None, cache_dir=None, labelers=LABELERS):
    cache_dir = Path(cache_dir or DQ.study_dir())
    runs = load_runs(cache_dir / "runs")
    if pool is None:
        pool = next((s.get("pool_sha256") for _, s, _ in reversed(runs) if s.get("pool_sha256")), None)
    chosen = rag_runs(runs, pool)
    if not chosen:
        raise SystemExit(f"no rag runs on pool {pool}")
    pools = _load(cache_dir / "pools.json")
    oracle = DQ.fetch_abstracts(rows, cache_dir=cache_dir, out=lambda *_: None)
    cache_path = cache_dir / "audit-labels.json"
    cache = _load(cache_path)
    meter = DQ.Meter()

    # every question, by both labellers
    questions = {}
    for row in rows:
        src = oracle.get(row["id"]) or {}
        prompt = (f"Question: {row['question']}\nGround truth: {row['answer']}\n"
                  f"Source paper: {src.get('title') or '(title unknown)'}\n"
                  f"Abstract: {src.get('abstract') or '(abstract unavailable)'}")
        questions[row["id"]] = {}
        for labeler in labelers:
            key = f"q|{VERSION}|{labeler}|{row['id']}"
            if key not in cache:
                cache[key] = _label(client, meter, labeler, QUESTION_SYSTEM, prompt, QUESTION_LABELS)
            questions[row["id"]][labeler] = cache[key]
    out(f"questions labelled (${meter.cost()} so far)")

    # every answer that lost points where retrieval missed the source, by both labellers
    by_id = {r["id"]: r for r in rows}
    answers = []
    for condition, model, sampling, name, recs in chosen:
        for qid, rec in recs.items():
            if rec.get("source_in_context") or rec["score"] >= 2:
                continue
            keys = rec["kept"] if "kept" in rec else rec.get("retrieved") or []
            labels = {}
            for labeler in labelers:
                if not keys:
                    labels[labeler] = {"label": "own_knowledge", "reason": "no abstract was given (gate kept none)"}
                    continue
                key = "a|" + hashlib.sha1(f"{VERSION}|{labeler}|{name}|{model}|{qid}".encode()).hexdigest()
                if key not in cache:
                    prompt = (f"Question: {by_id[qid]['question']}\nGround truth: {by_id[qid]['answer']}\n"
                              f"Answer: {rec['answer']}\n\nAbstracts the system was given:\n"
                              f"{_blocks((pools.get(qid) or {}).get('candidates') or {}, keys)}")
                    cache[key] = _label(client, meter, labeler, ANSWER_SYSTEM, prompt, ANSWER_LABELS)
                labels[labeler] = cache[key]
            answers.append({"condition": condition, "model": model, "sampling": sampling, "run": name,
                            "id": qid, "score": rec["score"], "labels": labels})
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    report = summarize(rows, runs, chosen, questions, answers, labelers)
    report.update(pool_sha256=pool, labelers=list(labelers), cost_usd=meter.cost(), version=VERSION)
    (cache_dir / "audit.json").write_text(json.dumps(dict(report, questions=questions, answers=answers),
                                                      indent=2), encoding="utf-8")
    print_report(report, out)
    return report


def summarize(rows, runs, chosen, questions, answers, labelers):
    main, second = labelers[0], labelers[-1]
    qlabel = {qid: q[main]["label"] for qid, q in questions.items()}
    report = {"question_agreement": {
        "kappa": cohen_kappa([q[main]["label"] for q in questions.values()],
                             [q[second]["label"] for q in questions.values()]),
        "same": sum(q[main]["label"] == q[second]["label"] for q in questions.values()),
        "of": len(questions)},
        "answer_agreement": {
            "kappa": cohen_kappa([a["labels"][main]["label"] for a in answers],
                                 [a["labels"][second]["label"] for a in answers]),
            "same": sum(a["labels"][main]["label"] == a["labels"][second]["label"] for a in answers),
            "of": len(answers)}}

    # per question label: how findable, how well known, how much plain RAG helps
    plain = [c for c in chosen if DQ.rag_parts(c[0])[1] == "plain"]
    found = {}
    for _, _, _, _, recs in plain:
        for qid, rec in recs.items():
            found.setdefault(qid, bool(rec.get("source_in_context")))
    by_label = {}
    for label in QUESTION_LABELS:
        ids = [r["id"] for r in rows if qlabel.get(r["id"]) == label]
        entry = {"questions": len(ids), "source_retrieved": sum(found.get(q, False) for q in ids), "models": {}}
        for _, model, sampling, _, recs in plain:
            _, closed = latest(runs, "closed-book", model, sampling)
            shared = [q for q in ids if q in recs and q in closed]
            entry["models"][model] = {"closed_book": _mean(closed[q]["score"] for q in shared),
                                      "plain_rag": _mean(recs[q]["score"] for q in shared)}
        by_label[label] = entry
    report["by_question_label"] = by_label

    # per rag run: the missed questions as graded, and again crediting answers valid for the question
    per_run = []
    for condition, model, sampling, name, recs in chosen:
        _, closed = latest(runs, "closed-book", model, sampling)
        missed = [q for q, r in recs.items() if not r.get("source_in_context") and q in closed]
        mine = {a["id"]: a for a in answers if a["run"] == name and a["model"] == model}
        counts = {label: sum(1 for a in mine.values() if a["labels"][main]["label"] == label) for label in ANSWER_LABELS}
        credited = {q: 2 if q in mine and mine[q]["labels"][main]["label"] == "valid_other_paper" else recs[q]["score"]
                    for q in missed}
        per_run.append({"condition": condition, "model": model, "sampling": sampling, "run": name,
                        "missed": len(missed), "lost_points": len([q for q in missed if q in mine]),
                        "labels": counts, "graded": _mean(recs[q]["score"] for q in missed),
                        "crediting_valid": _mean(credited.values()),
                        "closed_book": _mean(closed[q]["score"] for q in missed),
                        "by_question_label": {label: sum(1 for q in mine if qlabel.get(q) == label)
                                              for label in QUESTION_LABELS}})
    report["missed_questions"] = per_run
    return report


def print_report(report, out=print):
    qa, aa = report["question_agreement"], report["answer_agreement"]
    out(f"\nquestion labels by {report['labelers'][0]}; agreement with {report['labelers'][-1]}: "
        f"{qa['same']}/{qa['of']}, kappa {qa['kappa']}")
    models = list(next(iter(report["by_question_label"].values()))["models"])
    out(f"{'label':16s} {'n':>3s} {'found':>6s}  " + "  ".join(f"{m[-12:]:>21s}" for m in models))
    out(f"{'':16s} {'':>3s} {'':>6s}  " + "  ".join(f"{'closed -> plain RAG':>21s}" for _ in models))
    for label, e in report["by_question_label"].items():
        cells = [f"{str(e['models'][m]['closed_book']):>9s} -> {str(e['models'][m]['plain_rag']):>8s}" for m in models]
        out(f"{label:16s} {e['questions']:3d} {e['source_retrieved']:6d}  " + "  ".join(f"{c:>21s}" for c in cells))
    out(f"\nanswers that lost points where retrieval missed the source; agreement {aa['same']}/{aa['of']}, "
        f"kappa {aa['kappa']}")
    out(f"{'run':40s} {'missed':>6s} {'lost':>5s} {'valid':>6s} {'misled':>7s} {'own':>4s}   "
        f"{'closed':>6s} {'graded':>6s} {'credited':>8s}")
    for r in report["missed_questions"]:
        name = f"{r['model'][-19:]} {r['condition']}"
        lab = r["labels"]
        out(f"{name:40s} {r['missed']:6d} {r['lost_points']:5d} {lab['valid_other_paper']:6d} {lab['misled']:7d} "
            f"{lab['own_knowledge']:4d}   {str(r['closed_book']):>6s} {str(r['graded']):>6s} {str(r['crediting_valid']):>8s}")
    out("(closed = closed-book on the same questions; credited = graded, with valid answers about another "
        f"paper given full marks)  ${report['cost_usd']}")


# --------------------------------------------------------------------------- the fair re-grade

# The prompt's order. 1 (the method): the answer before the abstracts. 2: the answer after them, so that
# every verdict on one question shares a long prefix a local judge can reuse - five times faster on a CPU,
# but with it gpt-4.1 failed the mismatched-reference control (19/21), so it is a reported variant only.
REGRADE_VERSIONS = {1: "answer before the abstracts", 2: "answer after the abstracts"}
REGRADE_VERSION = 1
MULTI_SYSTEM = (
    "You grade answers to questions about computer-science research. Each question was written from one "
    "paper's abstract and comes with that paper's answer (the reference). The question may have other "
    "correct answers, so you are also given abstracts of other papers on the topic. Grade the answer to "
    "the question as it is worded:\n"
    "2 - correct and complete: it matches the reference, or it is an equally valid answer to the question "
    "as worded that is supported by one of the abstracts or is established knowledge in the field;\n"
    "1 - partially correct or incomplete;\n"
    "0 - incorrect, unsupported, or it does not answer the question.\n"
    "Do not reward an answer for length or fluency.\n"
    'Reply with JSON only: {"score": 0, 1 or 2, "reason": "<one sentence>"}')


def _condition_order(condition):
    return ({"closed-book": 0}.get(condition, 1), condition)


def regrade(client, rows, out=print, pool=None, cache_dir=None, judge_model="gpt-4.1", force=False,
            version=REGRADE_VERSION, modes="all"):
    """Every answer set on the questions where retrieval missed the source, graded again against the
    reference and the same retrieved abstracts, after controls; paired with the same model's
    closed-book answers graded the same way. modes="plain" keeps closed-book and plain RAG only."""
    cache_dir = Path(cache_dir or DQ.study_dir())
    runs = load_runs(cache_dir / "runs")
    if pool is None:
        pool = next((s.get("pool_sha256") for _, s, _ in reversed(runs) if s.get("pool_sha256")), None)
    chosen = rag_runs(runs, pool)
    plain = next((c for c in chosen if DQ.rag_parts(c[0])[1] == "plain"), None)
    if not plain:
        raise SystemExit(f"no plain rag run on pool {pool}")
    pools = _load(cache_dir / "pools.json")
    by_id = {r["id"]: r for r in rows}
    # retrieval is the same for every model and mode on one pool: the plain run says which were missed
    missed = [r["id"] for r in rows if r["id"] in plain[4] and not plain[4][r["id"]].get("source_in_context")]
    evidence = {q: _blocks((pools.get(q) or {}).get("candidates") or {}, plain[4][q].get("retrieved") or [])
                or "(none)" for q in missed}

    sets = []
    for model, sampling in dict.fromkeys((c[1], c[2]) for c in chosen):
        name, closed = latest(runs, "closed-book", model, sampling)
        if closed:
            sets.append(("closed-book", model, sampling, name, closed))
    graded = [c for c in chosen if modes == "all" or DQ.rag_parts(c[0])[1] == "plain"]
    sets = sorted(sets + graded, key=lambda s: (s[2] != "ours", s[1], _condition_order(s[0])))
    slug = re.sub(r"[^a-z0-9.]+", "-", judge_model.lower())
    report_path = cache_dir / f"regrade-{slug}-v{version}{'-plain' if modes == 'plain' else ''}.json"

    cache_path = cache_dir / "regrade-labels.json"
    cache = _load(cache_path)
    meter = DQ.Meter()
    started, fresh = time.time(), [0]
    most = 3 * len(missed) + sum(1 for s in sets for q in missed if q in s[4])

    def grade(qid, answer):
        key = hashlib.sha1(f"{version}|{judge_model}|{qid}|{answer}".encode("utf-8")).hexdigest()
        if key not in cache:
            head = (f"Question: {by_id[qid]['question']}\n"
                    f"Reference answer (from the paper the question was written from): {by_id[qid]['answer']}\n")
            abstracts = f"Abstracts of other papers on the topic:\n{evidence[qid]}"
            prompt = (f"{head}Answer to grade: {answer}\n\n{abstracts}" if version == 1
                      else f"{head}\n{abstracts}\n\nAnswer to grade: {answer}")
            step = DQ.judge_call(client, meter, judge_model, [{"role": "system", "content": MULTI_SYSTEM},
                                                              {"role": "user", "content": prompt}])
            try:
                cache[key] = DQ.parse_judgement(step.get("content"))
            except ValueError as e:      # a local judge's reply with no score in it: counted, left out
                cache[key] = {"score": None, "reason": f"unreadable: {e}"[:300]}
            fresh[0] += 1
            # a local judge on a CPU takes most of a minute per verdict: say where it is, and keep what it did
            if fresh[0] % 10 == 0:
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                out(f"  {fresh[0]} verdicts in {time.time() - started:.0f}s "
                    f"({(time.time() - started) / fresh[0]:.0f}s each; at most {most} in all, cached ones are free)")
        return cache[key]

    order = DQ.derangement(len(missed))
    if version == 2:
        # every verdict on one question in a row: a local judge reads the question's five abstracts once
        # and reuses them, so only the answer is new in each verdict. Nothing is reported unless the
        # controls below pass.
        for i, q in enumerate(missed):
            other = missed[order[i]]
            for answer in ([by_id[q]["answer"], by_id[other]["answer"], plain[4][other]["answer"]]
                           + [s[4][q]["answer"] for s in sets if q in s[4]]):
                grade(q, answer)

    # the controls, on these very questions and their evidence
    n = len(missed)
    checks = {"reference scored 2": [grade(q, by_id[q]["answer"])["score"] == 2 for q in missed],
              "another question's reference scored 0":
                  [grade(q, by_id[missed[order[i]]]["answer"])["score"] == 0 for i, q in enumerate(missed)],
              "another question's RAG answer scored 0":
                  [grade(q, plain[4][missed[order[i]]]["answer"])["score"] == 0 for i, q in enumerate(missed)]}
    controls = {name: sum(ok) for name, ok in checks.items()}
    controls["questions"] = n
    controls["passed"] = all(sum(ok) / n >= 0.95 for ok in checks.values()) if n else False
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    out(f"re-grade of the {n} questions where retrieval missed the source paper (judge {judge_model}, pool {pool}, "
        f"prompt v{version}: {REGRADE_VERSIONS[version]})")
    out("controls: " + "; ".join(f"{name} in {controls[name]}/{n}" for name in checks)
        + f" -> {'PASS' if controls['passed'] else 'FAIL'}")
    report = {"pool_sha256": pool, "judge": judge_model, "version": version, "modes": modes, "missed": missed,
              "controls": controls, "sets": []}
    if not controls["passed"] and not force:
        report.update(stopped="the judge failed its controls", cost_usd=meter.cost())
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        out("stopped: a judge that fails its controls is not used (--force to grade anyway)")
        return report

    regraded = {}
    for condition, model, sampling, name, recs in sets:
        scored = {q: grade(q, recs[q]["answer"])["score"] for q in missed if q in recs}
        regraded[(condition, model, sampling)] = {q: v for q, v in scored.items() if v is not None}
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    for condition, model, sampling, name, recs in sets:
        mine = regraded[(condition, model, sampling)]
        before = {q: recs[q]["score"] for q in mine}
        entry = {"condition": condition, "model": model, "sampling": sampling, "run": name, "questions": len(mine),
                 "gold_graded": _mean(before.values()), "regraded": DQ.bootstrap_ci(list(mine.values())),
                 "raised": sum(mine[q] > before[q] for q in mine), "lowered": sum(mine[q] < before[q] for q in mine)}
        closed = regraded.get(("closed-book", model, sampling))
        if condition != "closed-book" and closed:
            shared = [q for q in mine if q in closed]
            diffs = [mine[q] - closed[q] for q in shared]
            entry["vs_closed_book"] = {"questions": len(shared), "delta": DQ.bootstrap_ci(diffs),
                                       "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}
        report["sets"].append(entry)
    report["cost_usd"] = meter.cost()
    report_path.write_text(json.dumps(dict(report, scores={
        f"{c}|{m}|{s}": v for (c, m, s), v in regraded.items()}), indent=2), encoding="utf-8")
    print_regrade(report, out)
    return report


def print_regrade(report, out=print):
    out(f"\n{'model':20s} {'condition':22s} {'gold-graded':>11s} {'re-graded':>18s} {'raised':>7s} {'lowered':>8s}"
        f"   vs closed-book (re-graded)")
    for e in report["sets"]:
        ci = e["regraded"]
        line = (f"{e['model'][-20:]:20s} {e['condition']:22s} {str(e['gold_graded']):>11s} "
                f"{ci['mean']:6.2f} ({ci['low']:.2f}-{ci['high']:.2f}) {e['raised']:7d} {e['lowered']:8d}")
        paired = e.get("vs_closed_book")
        if paired:
            d = paired["delta"]
            line += (f"   {d['mean']:+.2f} (95% CI {d['low']:+.2f} to {d['high']:+.2f}); better on "
                     f"{paired['better']}, worse on {paired['worse']}")
        out(line)
    out(f"(gold-graded = the original score against the one reference; re-graded = any answer correct for the "
        f"question as worded counts)  ${report['cost_usd']}")
