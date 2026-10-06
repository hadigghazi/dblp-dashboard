"""
Every experiment of the original paper, re-run automatically on the same questions.

RAGScholar & DBLP-QA (Neekhra, Nilles & Schenkel, SCOLIA '26) asks four questions: does BM25 retrieve a
paper that answers the question (RQ1); are single or concatenated abstracts the better context (RQ2);
are answers written per abstract and then concatenated better than concatenated abstracts (RQ3); and
how much the model matters (RQ4). It answers them with one person's 0-2 ratings, plus ROUGE-L and
BERTScore, over a Lucene index that can no longer be rebuilt. Its scores cannot be set beside ours -
the same models score differently under our judge (no context: Mistral-7B 1.04 against its 0.80,
TinyLlama 0.58 against its 1.10) - but its experiments can be re-run and its conclusions tested:

  * RQ1, automatically: a judge, held to controls, decides whether each retrieved abstract states the
    answer. That gives the paper's own measure - the rank of the first abstract that answers the
    question, as Recall@k and MRR@3 - for any ranking of the frozen pool, beside the rank of the paper
    the question was written from.
  * RQ2-RQ4: the paper's ten context variants - no context, each of the top five abstracts alone
    (A1-A5), the top 3 or 5 abstracts concatenated (Top-3/5-CD), and an answer per abstract for the top
    3 or 5 combined into one (Top-3/5-CA) - for its models and ours, judged like every other run.

`report` sets the grid beside the paper's Table 3: per model, the rank correlation of its manual scores
with our judge's across the ten variants, and whether each of its findings holds here.
"""
import hashlib
import json
import logging
import math
import re
from pathlib import Path

from . import dblpqa as DQ, dblpqa_rag as RAG

log = logging.getLogger("dblp.chat.dblpqa_replicate")

# the paper's ten variants, in the order of its Table 3
VARIANTS = (("single", 1), ("single", 2), ("single", 3), ("single", 4), ("single", 5),
            ("cd", 3), ("cd", 5), ("ca", 3), ("ca", 5), ("none", 0))
LABELS = {("single", j): f"A{j}" for j in range(1, 6)}
LABELS.update({("cd", 3): "Top-3-CD", ("cd", 5): "Top-5-CD", ("ca", 3): "Top-3-CA", ("ca", 5): "Top-5-CA",
               ("none", 0): "no context"})
BY_LABEL = {v: k for k, v in LABELS.items()}

# Table 3 of the paper: manual 0-2 score, BERTScore F1 and ROUGE-L F1, in VARIANTS order
PAPER_TABLE3 = {
    "manual": {
        "Mistral-7B": [1.72, 1.06, 1.00, 0.84, 0.86, 1.63, 1.74, 1.60, 1.49, 0.80],
        "Phi-4": [1.56, 0.78, 0.68, 0.58, 0.59, 1.66, 1.66, 1.63, 1.51, 0.40],
        "TinyLlama-1.1B": [1.63, 1.10, 0.94, 0.82, 0.70, 1.58, 1.52, 1.44, 1.34, 1.10],
        "FLAN-T5-Large": [1.34, 0.70, 0.70, 0.54, 0.56, 1.32, 1.34, 1.36, 1.34, 0.30],
        "FLAN-T5-XXL": [0.98, 0.24, 0.16, 0.06, 0.014, 1.04, 0.92, 0.96, 0.90, 0.60],
    },
    "bertscore": {
        "Mistral-7B": [0.58, 0.51, 0.47, 0.44, 0.39, 0.51, 0.54, 0.51, 0.47, 0.35],
        "Phi-4": [0.37, 0.38, 0.34, 0.30, 0.27, 0.31, 0.36, 0.36, 0.31, 0.28],
        "TinyLlama-1.1B": [0.45, 0.44, 0.41, 0.38, 0.36, 0.42, 0.41, 0.42, 0.39, 0.41],
        "FLAN-T5-Large": [0.47, 0.46, 0.38, 0.38, 0.38, 0.45, 0.47, 0.48, 0.45, 0.14],
        "FLAN-T5-XXL": [0.40, 0.38, 0.32, 0.28, 0.26, 0.42, 0.38, 0.38, 0.35, 0.21],
    },
    "rouge_l": {
        "Mistral-7B": [0.32, 0.30, 0.25, 0.24, 0.24, 0.34, 0.34, 0.32, 0.29, 0.21],
        "Phi-4": [0.28, 0.27, 0.21, 0.20, 0.15, 0.37, 0.33, 0.22, 0.23, 0.10],
        "TinyLlama-1.1B": [0.35, 0.24, 0.23, 0.22, 0.24, 0.28, 0.23, 0.28, 0.30, 0.26],
        "FLAN-T5-Large": [0.27, 0.20, 0.18, 0.19, 0.15, 0.27, 0.26, 0.24, 0.24, 0.08],
        "FLAN-T5-XXL": [0.20, 0.14, 0.12, 0.13, 0.10, 0.21, 0.20, 0.20, 0.13, 0.14],
    },
}
# the paper's RQ1, Section 5.1: the first abstract containing an answer was at rank 1 for 44 questions,
# 2 for 4 and 3 for 2; the paper a question was written from at rank 1 for 32, 2 for 7, 3 for 2, 4 for 1
# and outside the top five for 5 (three questions are not accounted for)
PAPER_RQ1 = {"answer_bearing": {"recall@1": 0.88, "recall@3": 1.0, "mrr@3": 0.93},
             "source": {"recall@1": 0.64, "recall@3": 0.82, "recall@5": 0.84}}
PAPER_MODEL = {"ollama:mistral:v0.1": "Mistral-7B", "ollama:phi4": "Phi-4",
               "ollama:tinyllama:1.1b-chat": "TinyLlama-1.1B"}
# a model trained on a short window: the prompt must fit, or the server cuts its beginning - the
# instructions and the top-ranked abstracts. Each abstract is shortened evenly instead (and said so).
CONTEXT_TOKENS = {"ollama:tinyllama:1.1b-chat": 2048}
ANSWER_RESERVE = 512 + 120      # the paper's 512-token answers, plus the instructions and the question
WORDS_PER_TOKEN = 1 / 1.4       # English text in a Llama tokenizer, conservatively


def condition_for(variant, ranker="bm25"):
    strategy, k = variant
    if strategy == "none":
        return "closed-book"
    return RAG.condition_name(ranker, "plain", strategy, k)


def fit_words(text, max_words):
    words = (text or "").split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " [...]"


def _fit_block(block, max_words):
    """A "[n] title" line and its abstract: the abstract is cut, the title kept."""
    head, _, body = block.partition("\n")
    return f"{head}\n{fit_words(body, max_words)}" if body else block


def fitted(model, contexts, k):
    """For a short-window model, each abstract cut to an even share of the window."""
    window = CONTEXT_TOKENS.get(model)
    if not window or not contexts:
        return contexts, None
    per = max(40, int((window - ANSWER_RESERVE) / max(1, k) * WORDS_PER_TOKEN))
    out = {}
    for qid, ctx in contexts.items():
        blocks = (ctx.get("abstract") or "").split("\n\n")
        out[qid] = dict(ctx, abstract="\n\n".join(_fit_block(b, per) for b in blocks))
    return out, {"window_tokens": window, "words_per_abstract": per}


# --------------------------------------------------------------------------- finding earlier runs

def _summaries(runs_dir):
    found = []
    for path in sorted(Path(runs_dir).glob("*/summary.json")):
        try:
            found.append((path.parent, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return found


def _sampling(summary):
    return next(iter(summary.get("sampling") or {"ours": None}))


def latest_answers(runs_dir, condition, model, sampling, pool=None, need=None):
    """(run name, {id: record}) of the newest run of `condition` with this model and sampling - on this
    pool for a rag condition - that answered (and scored, when `need` is given) every question."""
    for run, summary in reversed(_summaries(runs_dir)):
        if summary.get("condition") != condition or model not in (summary.get("models") or []):
            continue
        if _sampling(summary) != sampling:
            continue
        if condition.startswith(DQ.RAG_PREFIX) and pool and summary.get("pool_sha256") != pool:
            continue
        records = {}
        for line in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line) if line.strip() else None
            if rec and rec["model"] == model:
                records[rec["id"]] = rec
        if need is not None and (len(records) < need or any(r.get("score") is None for r in records.values())):
            continue
        return run.name, records
    return None, {}


def ca_contexts(rows, pools, rankings, ranker, k, model, sampling, pool, runs_dir):
    """Concatenated Answers: this model's own answers from A1 ... Ak, numbered, as the context for one
    final answer. The single-abstract runs must exist - the grid runs them first."""
    per_rank, used = [], []
    for j in range(1, k + 1):
        name, recs = latest_answers(runs_dir, condition_for(("single", j), ranker), model, sampling, pool)
        if not name:
            raise SystemExit(f"Top-{k}-CA for {model} needs its A{j} answers first "
                             f"(dblpqa grid, or dblpqa rag --strategy single --k {j})")
        per_rank.append(recs)
        used.append(name)
    contexts = {}
    for row in rows:
        ranking = rankings[ranker].get(row["id"]) or []
        rank = RAG.source_rank(ranking, set(RAG.pool_of(pools, row["id"], ranker)["aliases"]))
        answers = [recs[row["id"]]["answer"] for recs in per_rank if row["id"] in recs]
        contexts[row["id"]] = {"abstract": "\n\n".join(f"[{i}] {a}" for i, a in enumerate(answers, 1))
                               or "(no answers)", "source": f"{ranker}@ca{k}", "retrieved": ranking[:k],
                               "source_rank": rank, "source_in_context": rank is not None and rank <= k}
    return contexts, used


def contexts_for(variant, rows, pools, rankings, ranker, model, sampling, pool, runs_dir):
    """(contexts, notes) for one variant and one model."""
    strategy, k = variant
    if strategy == "none":
        return None, None
    notes = {"pool_sha256": pool, "retrieval": {"ranker": ranker, "k": k, "mode": "plain", "strategy": strategy}}
    if strategy == "single":
        contexts = RAG.single_contexts(rows, pools, rankings, ranker, k)
    elif strategy == "cd":
        contexts = RAG.rag_contexts(rows, pools, rankings, ranker, k=k)
    else:
        contexts, used = ca_contexts(rows, pools, rankings, ranker, k, model, sampling, pool, runs_dir)
        notes["concatenated_answers_from"] = used
    if strategy != "ca":                        # answers are short; only abstracts need fitting
        contexts, fit = fitted(model, contexts, 1 if strategy == "single" else k)
        if fit:
            notes["abstracts_shortened"] = fit
    return contexts, notes


# --------------------------------------------------------------------------- RQ2-RQ4: the grid

def run_grid(client, rows, sha, models, sampling="paper", judge_model="gpt-4.1", ranker="bm25",
             variants=VARIANTS, allow_incomplete=False, out=print):
    """Every variant for every model, skipping any that already has a complete, scored run."""
    pools, rankings, report = RAG.prepare(rows, frozen=True, allow_incomplete=allow_incomplete, out=out)
    pool = report["pool"]["sha256"]
    runs_dir = DQ.study_dir() / "runs"
    # Concatenated Answers are built from the single-abstract answers, so those always go first
    order = sorted(variants, key=lambda v: {"none": 0, "single": 1, "cd": 2, "ca": 3}[v[0]])
    for model in models:
        for variant in order:
            condition = condition_for(variant, ranker)
            have, _ = latest_answers(runs_dir, condition, model, sampling, pool, need=len(rows))
            if have:
                out(f"{model} {LABELS[variant]}: have {have}")
                continue
            out(f"\n== {model} {LABELS[variant]} ({condition})")
            contexts, notes = contexts_for(variant, rows, pools, rankings, ranker, model, sampling, pool, runs_dir)
            DQ.run_condition(client, [model], judge_model, rows, sha, condition, contexts=contexts, out=out,
                             sampling=sampling, notes=notes)
    return report


# --------------------------------------------------------------------------- RQ1: abstracts that answer

BEARING_VERSION = 1
BEARING_SYSTEM = """You check whether one paper abstract answers a question about computer-science
research. You are given the question, its ground-truth answer, and the abstract. Say true if the
abstract itself states what the ground-truth answer says, in any wording, so that someone who read
only this abstract could answer the question correctly. Say false if it does not; an abstract on the
same topic that does not state it is false.

Reply with JSON only: {"answers": true} or {"answers": false}"""
BEARING_RANKERS = ("bm25", "dense", "hybrid", "dblp-search", "openalex-search", "openalex-semantic")


def parse_bearing(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            got = json.loads(match.group(0)).get("answers")
            if isinstance(got, bool):
                return got
        except (ValueError, AttributeError):
            pass
    found = re.search(r'"?answers"?\s*[:=]\s*(true|false)', text or "", re.I)
    if found:
        return found.group(1).lower() == "true"
    raise ValueError(f"no verdict in {(text or '')[:200]!r}")


class Bearing:
    """Cached verdicts: does this abstract answer this question? Keyed by the texts themselves, so an
    abstract that two rankers return - or two dblp records share - is judged once."""

    def __init__(self, client, judge_model, path):
        self.client, self.judge_model, self.path = client, judge_model, Path(path)
        self.cache = RAG._load(self.path)
        self.meter = DQ.Meter()
        self.unreadable = 0

    def __call__(self, question, gold, abstract):
        ck = hashlib.sha1(f"{BEARING_VERSION}|{self.judge_model}|{question}|{gold}|{abstract}".encode()).hexdigest()
        if ck not in self.cache:
            step = DQ.judge_call(self.client, self.meter, self.judge_model, [
                {"role": "system", "content": BEARING_SYSTEM},
                {"role": "user", "content": f"Question: {question}\nGround-truth answer: {gold}\n\nAbstract:\n{abstract}"}])
            try:
                self.cache[ck] = parse_bearing(step.get("content"))
            except ValueError:
                self.unreadable += 1
                return None                         # not cached: asked again next time
            if len(self.cache) % 25 == 0:
                self.save()
        return self.cache[ck]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache), encoding="utf-8")


def bearing_controls(judge, rows, oracle, out=print):
    """The source abstract must count as answering its own question, and another question's source
    abstract must not - each for at least 95% of the questions that have one."""
    have = [r for r in rows if (oracle.get(r["id"]) or {}).get("abstract")]
    swap = DQ.derangement(len(have))
    own_ok = other_ok = 0
    misses = []
    for i, row in enumerate(have):
        own = judge(row["question"], row["answer"], oracle[row["id"]]["abstract"])
        other = judge(row["question"], row["answer"], oracle[have[swap[i]]["id"]]["abstract"])
        own_ok += own is True
        other_ok += other is False
        if own is not True:
            misses.append({"id": row["id"], "control": "its own abstract should answer it", "got": own})
        if other is not False:
            misses.append({"id": row["id"], "control": "another question's abstract should not", "got": other})
    n = len(have) or 1
    result = {"questions": len(have), "own_abstract_answers": round(own_ok / n, 3),
              "other_abstract_does_not": round(other_ok / n, 3), "misses": misses,
              "passed": own_ok / n >= 0.95 and other_ok / n >= 0.95}
    out(f"bearing controls: own abstract answers {own_ok}/{len(have)}, another's does not {other_ok}/{len(have)}"
        f" -> {'PASS' if result['passed'] else 'FAIL'}")
    return result


def first_bearing(judge, row, cands, keys):
    """1-based rank of the first of `keys` whose abstract answers the question, or None. A candidate
    with no abstract cannot answer it (the paper's index held an abstract for every paper; ours does
    not, and is ranked on its title alone)."""
    for i, key in enumerate(keys, 1):
        abstract = cands[key].get("abstract")
        if abstract and judge(row["question"], row["answer"], abstract):
            return i
    return None


def bearing_metrics(firsts, ks=(1, 3, 5)):
    n = len(firsts) or 1
    got = {f"recall@{k}": round(sum(1 for f in firsts if f is not None and f <= k) / n, 3) for k in ks}
    got["mrr@3"] = round(sum(1 / f for f in firsts if f is not None and f <= 3) / n, 3)
    got["mrr@5"] = round(sum(1 / f for f in firsts if f is not None and f <= 5) / n, 3)
    return got


def run_bearing(client, rows, rankers=BEARING_RANKERS, judge_model="gpt-4.1", k=RAG.TOP_K,
                allow_incomplete=False, force=False, out=print):
    """The paper's RQ1 measure for each ranker of the frozen pool, beside the source paper's ranks."""
    pools, rankings, report = RAG.prepare(rows, frozen=True, allow_incomplete=allow_incomplete, out=out)
    oracle = DQ.fetch_abstracts(rows, out=out)
    judge = Bearing(client, judge_model, DQ.study_dir() / "bearing.json")
    try:
        controls = bearing_controls(judge, rows, oracle, out)
        if not controls["passed"] and not force:
            out("stopped: the judge failed its controls; --force to run anyway")
            return None
        result = {"judge": judge_model, "version": BEARING_VERSION, "k": k, "pool_sha256": report["pool"]["sha256"],
                  "controls": controls, "rankers": {}, "paper": PAPER_RQ1}
        for name in rankers:
            firsts, per_q = [], {}
            if name not in rankings:
                out(f"{name}: no ranking (Dewey's index not built?) - left out")
                continue
            for row in rows:
                cands = RAG.pool_of(pools, row["id"], name)["candidates"]
                keys = (rankings[name].get(row["id"]) or [])[:k]
                first = first_bearing(judge, row, cands, keys)
                firsts.append(first)
                per_q[row["id"]] = first
            source = report["rankers"][name]
            result["rankers"][name] = {"answer_bearing": bearing_metrics(firsts), "first_answering_rank": per_q,
                                       "source": {k_: source[k_] for k_ in ("recall@1", "recall@3", "recall@5")}}
    finally:
        judge.save()
    result["cost_usd"] = judge.meter.cost()
    result["unreadable"] = judge.unreadable
    (DQ.study_dir() / "bearing-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_bearing(result, out)
    return result


def print_bearing(result, out=print):
    out(f"\nRQ1 as the paper measures it - the first top-{result['k']} abstract that answers the question "
        f"(judge {result['judge']}), beside the paper the question was written from")
    out(f"{'ranker':18s} {'ans R@1':>8s} {'ans R@3':>8s} {'ans R@5':>8s} {'MRR@3':>7s}   "
        f"{'src R@1':>8s} {'src R@3':>8s} {'src R@5':>8s}")
    rows = list(result["rankers"].items())
    rows.append(("paper (closed)", {"answer_bearing": result["paper"]["answer_bearing"],
                                    "source": result["paper"]["source"]}))
    for name, m in rows:
        a, s = m["answer_bearing"], m["source"]
        out(f"{name:18s} {a.get('recall@1', float('nan')):8.2f} {a.get('recall@3', float('nan')):8.2f} "
            f"{a.get('recall@5', float('nan')):8.2f} {a.get('mrr@3', float('nan')):7.3f}   "
            f"{s.get('recall@1', float('nan')):8.2f} {s.get('recall@3', float('nan')):8.2f} "
            f"{s.get('recall@5', float('nan')):8.2f}")
    out(f"(${result.get('cost_usd', 0)}; the paper's figures are over its own closed index and are not comparable "
        f"in level, only in kind)")


# --------------------------------------------------------------------------- the report

def spearman(xs, ys):
    """Rank correlation with average ranks for ties; None when either side is constant."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out, i = [0.0] * len(vals), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for m in range(i, j + 1):
                out[order[m]] = (i + j) / 2 + 1
            i = j + 1
        return out
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return round(cov / (vx * vy), 3) if vx and vy else None


def _paired(a, b):
    """Mean difference a - b over the questions both answered, with its interval and the counts."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = [a[q] - b[q] for q in shared]
    return {"questions": len(shared), "delta": DQ.bootstrap_ci(diffs),
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}


def grid_table(rows, models, sampling, pool, ranker="bm25"):
    """{model: {label: {"run", "mean", "rouge_l", "scores": {id: score}, "rouges": {id: rouge}}}} from
    the latest complete, scored run of each variant."""
    runs_dir = DQ.study_dir() / "runs"
    table = {}
    for model in models:
        table[model] = {}
        for variant in VARIANTS:
            name, recs = latest_answers(runs_dir, condition_for(variant, ranker), model, sampling, pool,
                                        need=len(rows))
            if not name:
                continue
            scores = {q: r["score"] for q, r in recs.items()}
            rouges = {q: r["rouge_l"] for q, r in recs.items()}
            table[model][LABELS[variant]] = {
                "run": name, "mean": round(sum(scores.values()) / len(scores), 3),
                "rouge_l": round(sum(rouges.values()) / len(rouges), 3), "scores": scores, "rouges": rouges}
    return table


def findings(cells):
    """The paper's conclusions (Sections 5.2-5.4 and 7), each tested on one model's grid. A paired
    claim holds when its interval is above zero; "compare" claims only report the difference."""
    def s(label):
        return (cells.get(label) or {}).get("scores")
    checks = []
    for label in ("Top-5-CD", "Top-3-CD", "A1"):
        if s(label) and s("no context"):
            checks.append({"claim": f"{label} beats no context", "paired": _paired(s(label), s("no context"))})
    singles = {f"A{j}": cells[f"A{j}"]["mean"] for j in range(1, 6) if f"A{j}" in cells}
    if len(singles) == 5:
        best = max(singles, key=singles.get)
        checks.append({"claim": "A1 is the best single-document variant", "holds": best == "A1", "means": singles})
    for k in (3, 5):
        if s(f"Top-{k}-CD") and s(f"Top-{k}-CA"):
            checks.append({"claim": f"Top-{k}-CD beats Top-{k}-CA (documents before answers)",
                           "paired": _paired(s(f"Top-{k}-CD"), s(f"Top-{k}-CA"))})
    if s("Top-5-CD") and s("Top-3-CD"):
        checks.append({"claim": "Top-5-CD against Top-3-CD (the paper: depends on the model)",
                       "paired": _paired(s("Top-5-CD"), s("Top-3-CD")), "compare": True})
    for check in checks:
        p = check.get("paired")
        if p and not check.get("compare"):
            check["holds"] = p["delta"]["low"] is not None and p["delta"]["low"] > 0
    return checks


def report(rows, models, sampling="paper", ranker="bm25", allow_incomplete=False, out=print):
    _pools, _rankings, ret = RAG.prepare(rows, frozen=True, allow_incomplete=allow_incomplete,
                                         out=lambda *_: None)
    pool = ret["pool"]["sha256"]
    table = grid_table(rows, models, sampling, pool, ranker)
    labels = [LABELS[v] for v in VARIANTS]
    result = {"pool_sha256": pool, "sampling": sampling, "ranker": ranker, "models": {}}
    out(f"Our judge's mean score per variant (pool {pool}, {sampling} sampling); the paper's manual score below")
    out(f"{'model':28s} " + " ".join(f"{l[:9]:>9s}" for l in labels))
    for model, cells in table.items():
        mine = [cells.get(l, {}).get("mean") for l in labels]
        out(f"{model[-28:]:28s} " + " ".join(f"{m:9.2f}" if m is not None else f"{'-':>9s}" for m in mine))
        entry = {"means": dict(zip(labels, mine)),
                 "rouge_l": {l: cells[l]["rouge_l"] for l in labels if l in cells},
                 "runs": {l: cells[l]["run"] for l in labels if l in cells}, "findings": findings(cells)}
        paper = PAPER_MODEL.get(model)
        if paper:
            theirs = PAPER_TABLE3["manual"][paper]
            out(f"{'  paper (manual)':28s} " + " ".join(f"{t:9.2f}" for t in theirs))
            both = [(m, t) for m, t in zip(mine, theirs) if m is not None]
            ours_rouge = [cells.get(l, {}).get("rouge_l") for l in labels]
            both_r = [(m, t) for m, t in zip(ours_rouge, PAPER_TABLE3["rouge_l"][paper]) if m is not None]
            entry.update(paper=paper,
                         spearman_with_paper_manual=spearman([m for m, _ in both], [t for _, t in both]),
                         spearman_rouge_with_paper_rouge=spearman([m for m, _ in both_r], [t for _, t in both_r]))
            out(f"{'':28s} rank correlation with the paper's manual scores across {len(both)} variants: "
                f"{entry['spearman_with_paper_manual']}; our ROUGE-L with its ROUGE-L: "
                f"{entry['spearman_rouge_with_paper_rouge']}")
        # answer by answer, does our judge agree with ROUGE-L? (the paper saw "a slight correlation"
        # between its manual scores and the automatic metrics)
        scores = [c["scores"][q] for c in cells.values() for q in c["scores"]]
        rouges = [c["rouges"][q] for c in cells.values() for q in c["scores"]]
        entry["answer_level_spearman_judge_rouge"] = spearman(scores, rouges)
        out(f"{'':28s} judge vs ROUGE-L over {len(scores)} answers: {entry['answer_level_spearman_judge_rouge']}")
        for check in entry["findings"]:
            p = check.get("paired")
            verdict = "compare" if check.get("compare") else ("holds" if check.get("holds") else "does not hold")
            tail = (f" {p['delta']['mean']:+.2f} ({p['delta']['low']:+.2f} to {p['delta']['high']:+.2f}), "
                    f"{p['better']} better / {p['worse']} worse" if p else f" {check.get('means')}")
            out(f"    {verdict}: {check['claim']}:{tail}")
        result["models"][model] = {k: v for k, v in entry.items()}
    (DQ.study_dir() / "replication.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
