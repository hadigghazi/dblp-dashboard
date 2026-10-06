"""
Experiments on DBLP-QA (Neekhra, Nilles & Schenkel, "RAGScholar & DBLP-QA", SCOLIA '26).

DBLP-QA is 50 questions, each written from one paper's abstract, with a 1-3 sentence answer taken from
that abstract. The paper scores answers by hand on a 0-2 scale - 2 correct and complete, 1 correct but
incomplete, 0 incorrect or irrelevant - plus ROUGE-L and BERTScore.

This module runs the benchmark under named conditions and scores every answer automatically, so the
results can be compared, re-run and written up without anyone rating answers by hand. The first
condition is the one the whole study turns on:

  closed-book - the question alone, no retrieval. Many DBLP-QA questions are textbook definitions
                ("What is Compressive Sensing?"), and even the paper's 1.1B model scored 1.10/2 with
                no context. If modern models score high here, much of the benchmark measures what a
                model remembers rather than what retrieval adds.

The judge applies the paper's own rubric, with the paper's own worked example as its anchor, and is
checked on every run by two controls it must get right: each question's gold answer must score 2, and
another question's gold answer must score 0. A judge that fails those is not used.
"""
import csv
import hashlib
import io
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import config
from .llm import LLMError

log = logging.getLogger("dblp.chat.dblpqa")

DATASET_URL = "https://seafile.rlp.net/f/6581519cdd1d4782bccc/?dl=1"
# Which question set the harness works on: the original 50, or DBLP-QA-Fresh (dblpqa_fresh), questions
# from papers too recent for the models to have seen. Each has its own folder, so runs, caches and the
# pairings between runs never mix.
# fresh2 is a held-out set built the same way from other papers: Dewey's second version was designed
# from its first version's failures on dblpqa and fresh, so it is measured on questions never looked at
DATASETS = {"dblpqa": {"dir": "dblpqa", "file": "dblp-qa.csv", "url": DATASET_URL},
            "fresh": {"dir": "dblpqa-fresh", "file": "fresh.csv", "url": None},
            "fresh2": {"dir": "dblpqa-fresh2", "file": "fresh.csv", "url": None}}
DATASET = "dblpqa"


def use_dataset(name):
    global DATASET
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    DATASET = name


def study_dir():
    return config.MODELS_DIR / DATASETS[DATASET]["dir"]
COLUMNS = ["id", "question", "answer", "dblp_key", "semantic_scholar_id"]
SEED = 7

ANSWER_SYSTEM = "You answer questions about computer-science research. Answer in one to three sentences."
CONTEXT_SYSTEM = ("You answer questions about computer-science research using the abstract you are given. "
                  "Answer in one to three sentences.")
# the retrieved papers are numbered and titled, as any search result list is; nothing tells the model
# that some of them may be off-topic - finding that out is part of what is being measured
RAG_SYSTEM = ("You answer questions about computer-science research using the paper abstracts you are "
              "given. Answer in one to three sentences.")
RAG_PREFIX = "rag-"
# selective retrieval (dblpqa_rag): "permissive" says the abstracts may be off-topic; "gated" filters
# them first, and a question left with none is asked closed-book
PERMISSIVE_SYSTEM = ("You answer questions about computer-science research. You are given abstracts of "
                     "papers a search returned for the question; they may or may not be relevant. Use them "
                     "where they address the question; if none does, ignore them and answer from your own "
                     "knowledge. Answer in one to three sentences.")
RAG_MODES = ("permissive", "gated")
# the paper's other context strategies (dblpqa_replicate): one retrieved abstract on its own (A1 ... A5),
# and Concatenated Answers - an answer written from each of the top k abstracts alone, then one answer
# written from those. The paper publishes neither prompt; these mirror RAG_SYSTEM.
SINGLE_SYSTEM = ("You answer questions about computer-science research using the paper abstract you are "
                 "given. Answer in one to three sentences.")
CA_SYSTEM = ("You answer questions about computer-science research. You are given answers to the question, "
             "each written from the abstract of one paper a search returned. Combine them into one answer, "
             "in one to three sentences.")
CONDITIONS = ("closed-book", "oracle")      # plus rag-<ranker>, see dblpqa_rag

# Generation settings. "paper" is Table 1 of the paper (Mistral-7B and TinyLlama rows): what a
# reproduction of their models has to use. "ours" is deterministic, for the hosted models.
SAMPLING = {
    "ours": {"temperature": 0, "extra": {}},
    "paper": {"temperature": 0.7, "extra": {"top_p": 0.9, "max_tokens": 512}},
}
OLLAMA_PREFIX = "ollama:"
LOCAL_RETRIES = 3


def client_for(model, hosted):
    """The hosted client, or one pointed at the local Ollama container for "ollama:<tag>" models. Both
    speak the same API, so the experiment code does not know the difference."""
    if model.startswith(OLLAMA_PREFIX):
        from .llm import Client
        return Client(api_key="ollama", base_url=config.OLLAMA_URL, timeout=600), model[len(OLLAMA_PREFIX):]
    return hosted, model

# The paper's rubric, with its own example (Table 2) as the anchor, so the judge reproduces the scale
# rather than inventing one.
JUDGE_SYSTEM = """You grade answers to questions about computer-science papers against a ground-truth
answer, on the scale used by the DBLP-QA benchmark:

  2 = correct and complete: says what the ground-truth answer says (wording may differ)
  1 = correct but incomplete: related to and consistent with the ground truth, but misses its key point
      or is only partly right
  0 = incorrect or irrelevant: contradicts the ground truth, describes something else, or does not answer

Grade only against the ground truth. Extra correct detail does not lower a score; a wrong claim that
contradicts the ground truth does. Do not reward fluency.

Worked example from the benchmark's authors.
Question: What is Compressive Sensing?
Ground truth: Compressive Sensing (CS) is an advanced signal processing technique that enables the
reconstruction of a signal using far fewer measurements than required by the traditional
Nyquist-Shannon sampling theorem.
- "Compressive sensing is a signal processing technique that allows for the reconstruction of signals or
  images from a small number of measurements, significantly fewer than what is typically required by the
  Nyquist-Shannon sampling theorem." -> 2 (answered in the same way as the ground truth)
- "Compressive Sensing, also known as Compressed Sensing (CS), is a technique used in signal processing
  and imaging that allows for the detection and recovery of signals that are inherently sparse or
  low-rank, such as images or signals with noise." -> 1 (related to the ground truth)
- "Compressive Sensing is a method used to improve on what any nonadaptive method can achieve in the
  context of recursive bisection method. It is a technique that establishes a non-asymptotic lower bound
  that applies to all methods, regardless of their computational complexity." -> 0 (unrelated)

Reply with JSON only: {"score": 0, 1 or 2, "reason": "<one short sentence>"}"""


# --------------------------------------------------------------------------- the dataset

def load_dataset(cache_dir=None, url=None):
    """The current dataset's questions - DBLP-QA is downloaded once and kept, Fresh must have been
    built. Returns (rows, sha256 of the file)."""
    spec = DATASETS[DATASET]
    cache = Path(cache_dir or study_dir())
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / spec["file"]
    if not path.exists():
        url = url or spec["url"]
        if not url:
            raise SystemExit(f"no {path.name} in {cache}: build it first with `dblpqa --dataset fresh fresh-build`")
        r = httpx.get(url, follow_redirects=True, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = path.read_bytes()
    rows = parse(raw.decode("utf-8", "replace"))
    return rows, hashlib.sha256(raw).hexdigest()


def parse(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or list(rows[0].keys()) != COLUMNS:
        raise ValueError(f"unexpected DBLP-QA columns: {list(rows[0].keys()) if rows else 'empty file'}")
    return [{k: (v or "").strip() for k, v in row.items()} for row in rows]


# --------------------------------------------------------------------------- abstracts

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "title,abstract,externalIds"


def _get(http, url, **kw):
    """GET or POST with patience: these are free public APIs with shared, unannounced rate limits."""
    method = kw.pop("method", "GET")
    for attempt in range(6):
        r = http.request(method, url, timeout=60, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(30, 2 ** attempt))
            continue
        return r
    return r


def _arxiv(http, arxiv_id):
    r = _get(http, "https://export.arxiv.org/api/query", params={"id_list": arxiv_id})
    found = re.search(r"<entry>.*?<summary>(.*?)</summary>", r.text, re.S) if r.status_code == 200 else None
    return " ".join(found.group(1).split()) if found else None


def _openalex(http, doi):
    r = _get(http, f"https://api.openalex.org/works/https://doi.org/{doi}")
    if r.status_code != 200:
        return None
    index = (r.json() or {}).get("abstract_inverted_index") or {}
    if not index:
        return None
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return " ".join(word for _, word in words)


def _crossref(http, doi):
    r = _get(http, f"https://api.crossref.org/works/{doi}")
    if r.status_code != 200:
        return None
    raw = ((r.json() or {}).get("message") or {}).get("abstract") or ""
    text = " ".join(re.sub(r"<[^>]+>", " ", raw).split())
    return re.sub(r"^Abstract\s+", "", text) or None


def fetch_abstracts(rows, cache_dir=None, http=None, out=print):
    """{question id: {"abstract", "source", "title"}} for every question, cached. Semantic Scholar first,
    then arXiv, OpenAlex and Crossref for any it withholds."""
    cache = Path(cache_dir or study_dir()) / "abstracts.json"
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [r for r in rows if r["id"] not in have]
    if todo:
        http = http or httpx.Client(follow_redirects=True, headers={"User-Agent": "dblp-explorer-research"})
        r = _get(http, S2_BATCH, method="POST", params={"fields": S2_FIELDS},
                 json={"ids": [f"CorpusId:{row['semantic_scholar_id']}" for row in todo]})
        papers = r.json() if r.status_code == 200 else [None] * len(todo)
        for row, paper in zip(todo, papers):
            paper = paper or {}
            ids = paper.get("externalIds") or {}
            entry = {"title": paper.get("title"), "abstract": paper.get("abstract"),
                     "source": "semantic-scholar" if paper.get("abstract") else None,
                     "doi": ids.get("DOI"), "arxiv": ids.get("ArXiv")}
            for source, fetch, key in (("arxiv", _arxiv, entry["arxiv"]),
                                       ("openalex", _openalex, entry["doi"]),
                                       ("crossref", _crossref, entry["doi"])):
                if entry["abstract"] or not key:
                    continue
                try:
                    text = fetch(http, key)
                except (httpx.HTTPError, ValueError):
                    text = None
                if text and len(text.split()) >= 20:
                    entry.update(abstract=text, source=source)
            have[row["id"]] = entry
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have, indent=2, ensure_ascii=False), encoding="utf-8")
    sources = {}
    for row in rows:
        src = have.get(row["id"], {}).get("source") or "none"
        sources[src] = sources.get(src, 0) + 1
    out("abstracts: " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    return have


# --------------------------------------------------------------------------- metrics

def _tokens(text):
    """The tokenisation of the reference rouge-score package without stemming: lower case, anything that
    is not a letter or digit becomes a space."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()


def rouge_l(prediction, reference):
    """ROUGE-L F1, longest common subsequence over tokens - equal to rouge-score's rougeL with
    use_stemmer=False. The paper does not say whether it stemmed, so this is stated wherever reported."""
    p, r = _tokens(prediction), _tokens(reference)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def bootstrap_ci(values, resamples=10000, seed=SEED, alpha=0.05):
    """Mean and a percentile bootstrap interval. With 50 questions an interval is the difference between
    a finding and a coincidence, so no mean is reported without one."""
    values = list(values)
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return {"mean": round(sum(values) / n, 3), "low": round(means[int(alpha / 2 * resamples)], 3),
            "high": round(means[int((1 - alpha / 2) * resamples) - 1], 3), "n": n}


def parse_judgement(text):
    """{"score": int, "reason": str} from the judge's reply, tolerating prose around the JSON."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            got = json.loads(match.group(0))
            score = int(got.get("score"))
            if score in (0, 1, 2):
                return {"score": score, "reason": str(got.get("reason", ""))[:300]}
        except (ValueError, TypeError):
            pass
    found = re.search(r'"?score"?\s*[:=]\s*([012])', text or "")
    if found:
        return {"score": int(found.group(1)), "reason": "(parsed from a non-JSON reply)"}
    raise ValueError(f"the judge did not return a score: {text[:200]!r}")


# --------------------------------------------------------------------------- the model calls

class Meter:
    """Tokens and dollars per model, from the provider's own usage counts."""

    def __init__(self):
        self.usage = {}

    def add(self, model, usage):
        u = self.usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        u["input_tokens"] += usage.get("input_tokens", 0)
        u["output_tokens"] += usage.get("output_tokens", 0)
        u["calls"] += 1

    def cost(self):
        total = 0.0
        for model, u in self.usage.items():
            pin = config.PRICE_IN.get(model, config.PRICE_IN.get(config.MODEL_DEEP, 2.0))
            pout = config.PRICE_OUT.get(model, config.PRICE_OUT.get(config.MODEL_DEEP, 8.0))
            total += u["input_tokens"] / 1e6 * pin + u["output_tokens"] / 1e6 * pout
        return round(total, 4)


def messages_for(condition, question, abstract=None):
    if condition == "oracle":
        return [{"role": "system", "content": CONTEXT_SYSTEM},
                {"role": "user", "content": f"Abstract:\n{abstract}\n\nQuestion: {question}"}]
    if condition.startswith(RAG_PREFIX) and abstract is not None:
        label = {"single": "Abstract", "ca": "Answers"}.get(rag_strategy(condition)[0], "Abstracts")
        return [{"role": "system", "content": system_prompt(condition)},
                {"role": "user", "content": f"{label}:\n{abstract}\n\nQuestion: {question}"}]
    # closed-book - and a gated question whose gate kept no abstract, asked exactly as closed-book
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": question}]


def rag_parts(condition):
    """("rag-bm25", "gated") from "rag-bm25-gated"; the mode is "plain" when there is no suffix."""
    for mode in RAG_MODES:
        if condition.endswith(f"-{mode}"):
            return condition[:-len(mode) - 1], mode
    return condition, "plain"


def rag_strategy(condition):
    """The paper's context strategy behind a rag condition, as (strategy, k): rag-bm25 is Top-5
    Concatenated Documents, the study's main condition and the name every earlier run has; -cd3 is
    Top-3, -a2 the second-ranked abstract alone, -ca5 Top-5 Concatenated Answers."""
    base = rag_parts(condition)[0]
    found = re.search(r"-(cd|a|ca)(\d+)$", base)
    if not found:
        return "cd", 5
    return {"cd": "cd", "a": "single", "ca": "ca"}[found.group(1)], int(found.group(2))


def system_prompt(condition):
    if condition == "oracle":
        return CONTEXT_SYSTEM
    if condition.startswith(RAG_PREFIX):
        strategy = rag_strategy(condition)[0]
        if strategy == "single":
            return SINGLE_SYSTEM
        if strategy == "ca":
            return CA_SYSTEM
        return PERMISSIVE_SYSTEM if rag_parts(condition)[1] == "permissive" else RAG_SYSTEM
    return ANSWER_SYSTEM


def answer_closed_book(client, meter, model, question, sampling="ours"):
    return answer(client, meter, model, messages_for("closed-book", question), sampling)


def answer(client, meter, model, messages, sampling="ours"):
    settings = SAMPLING[sampling]
    target, name = client_for(model, client)
    # a local server on a busy CPU fails the odd request mid-generation (a 500 after ~15 good answers
    # stopped the first calibration run); one failure must not throw away half an hour of answers
    for attempt in range(LOCAL_RETRIES + 1):
        try:
            step = target.complete(messages, model=name,
                                   temperature=settings["temperature"], extra=settings["extra"])
            break
        except Exception as e:
            raw = getattr(e, "raw", "") or str(e)
            if attempt == LOCAL_RETRIES or not model.startswith(OLLAMA_PREFIX):
                # the client's message is written for a website visitor; an experiment needs the server's
                raise RuntimeError(f"{model} failed: {raw}") from e
            log.warning("%s failed (%s), retrying", model, raw[:200])
            time.sleep(3 * (attempt + 1))
    if not model.startswith(OLLAMA_PREFIX):          # a local model costs nothing
        meter.add(model, step.get("usage", {}))
    return (step.get("content") or "").strip()


LOCAL_JUDGE_TOKENS = 200      # a local judge that rambles costs minutes per verdict on a CPU


def judge_call(client, meter, model, messages):
    """One deterministic judging call: hosted, or on the local server for an "ollama:<tag>" judge -
    retried there, since a busy CPU fails the odd request - and billed only when hosted."""
    target, name = client_for(model, client)
    local = model.startswith(OLLAMA_PREFIX)
    for attempt in range(LOCAL_RETRIES + 1):
        try:
            step = target.complete(messages, model=name, temperature=0,
                                   extra={"max_tokens": LOCAL_JUDGE_TOKENS} if local else None)
            break
        except Exception as e:
            if not local or attempt == LOCAL_RETRIES:
                raise
            log.warning("%s failed (%s), retrying", model, (getattr(e, "raw", "") or str(e))[:200])
            time.sleep(3 * (attempt + 1))
    if not local:
        meter.add(model, step.get("usage", {}))
    return step


def judge(client, meter, model, question, gold, candidate):
    prompt = f"Question: {question}\nGround truth: {gold}\nAnswer to grade: {candidate}"
    step = judge_call(client, meter, model, [{"role": "system", "content": JUDGE_SYSTEM},
                                             {"role": "user", "content": prompt}])
    return parse_judgement(step.get("content"))


# --------------------------------------------------------------------------- one run

def derangement(n, seed=SEED):
    """A shuffle in which nothing stays in place: question i is paired with another question's gold."""
    rng = random.Random(seed)
    order = list(range(n))
    while any(i == j for i, j in enumerate(order)):
        rng.shuffle(order)
    return order


def judge_controls(client, meter, judge_model, rows, out=print):
    """The judge has to give each gold answer 2 and someone else's gold answer 0. If it cannot, its
    scores mean nothing, and the run says so rather than reporting them."""
    swap = derangement(len(rows))
    gold_ok, swap_ok, misses = 0, 0, []
    for i, row in enumerate(rows):
        g = judge(client, meter, judge_model, row["question"], row["answer"], row["answer"])
        s = judge(client, meter, judge_model, row["question"], row["answer"], rows[swap[i]]["answer"])
        gold_ok += g["score"] == 2
        swap_ok += s["score"] == 0
        if g["score"] != 2:
            misses.append({"id": row["id"], "control": "gold should score 2", "got": g})
        if s["score"] != 0:
            misses.append({"id": row["id"], "control": "another question's answer should score 0", "got": s})
    n = len(rows)
    result = {"gold_scored_2": round(gold_ok / n, 3), "swapped_scored_0": round(swap_ok / n, 3),
              "misses": misses, "passed": gold_ok / n >= 0.95 and swap_ok / n >= 0.95}
    out(f"judge controls: gold answers scored 2 in {gold_ok}/{n}, another question's answer scored 0 "
        f"in {swap_ok}/{n} -> {'PASS' if result['passed'] else 'FAIL'}")
    return result


def reusable_controls(judge_model, sha, runs_dir=None, questions=None):
    """A judge that passed its controls on this exact dataset - all of it, not a --limit slice - already
    has: no need to pay again."""
    runs = Path(runs_dir or study_dir() / "runs")
    for summary in sorted(runs.glob("*/summary.json"), reverse=True):
        try:
            got = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        controls = got.get("judge_controls") or {}
        if (got.get("judge") == judge_model and got.get("dataset_sha256") == sha
                and controls.get("passed")
                and (questions is None or got.get("questions_in_dataset", got.get("questions")) == questions)):
            return dict(controls, reused_from=summary.parent.name)
    return None


def paired_delta(model, sampling, records, runs_dir, against="closed-book", pool=None):
    """This run's scores against the same model's scores in the latest `against` run with the same
    sampling, question by question: the mean difference with a paired bootstrap interval, and how many
    questions got better, worse or stayed the same. With `pool`, only a run that was given the same
    retrieval pool counts."""
    mine = {r["id"]: r["score"] for r in records if r["model"] == model}
    for summary in sorted(Path(runs_dir).glob(f"*-{against}/summary.json"), reverse=True):
        try:
            got = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if model not in (got.get("models") or []) or sampling not in (got.get("sampling") or {"ours": None}):
            continue
        if pool is not None and got.get("pool_sha256") != pool:
            continue
        theirs = {}
        for line in (summary.parent / "answers.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line) if line.strip() else None
            if rec and rec["model"] == model and rec.get("score") is not None:
                theirs[rec["id"]] = rec["score"]
        shared = sorted(set(mine) & set(theirs))
        if not shared:
            return None
        diffs = [mine[q] - theirs[q] for q in shared]
        return {"against_run": summary.parent.name, "questions": len(shared),
                "delta": bootstrap_ci(diffs),
                "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs),
                "same": sum(d == 0 for d in diffs)}
    return None


def run_closed_book(client, models, judge_model, rows, sha, out_dir=None, out=print, force=False,
                    sampling="ours", reuse_controls=True):
    return run_condition(client, models, judge_model, rows, sha, "closed-book", out_dir=out_dir, out=out,
                         force=force, sampling=sampling, reuse_controls=reuse_controls)


def run_condition(client, models, judge_model, rows, sha, condition, contexts=None, out_dir=None,
                  out=print, force=False, sampling="ours", reuse_controls=True, notes=None):
    """One condition for every model, judged. `notes` (the retrieval pool's fingerprint, the ranker,
    the gate) is saved with the run."""
    started = time.time()
    meter = Meter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_rows = rows
    if condition == "oracle":
        # a question with no abstract from any source cannot be given one; counted, not hidden
        rows = [r for r in rows if (contexts or {}).get(r["id"], {}).get("abstract")]
        out(f"oracle: {len(rows)} of {len(all_rows)} questions have an abstract")
    out_dir = Path(out_dir or study_dir() / "runs" / f"{stamp}-{condition}")
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = reusable_controls(judge_model, sha, out_dir.parent, len(all_rows)) if reuse_controls else None
    if judge_model == "none":
        controls = {"passed": True, "skipped": "answers generated without scoring; see dblpqa rescore"}
        out("no judge: answers are generated and saved; score them later with dblpqa rescore")
    elif controls:
        out(f"judge controls: passed on this dataset in run {controls['reused_from']} - reused")
    else:
        controls = judge_controls(client, meter, judge_model, all_rows, out)
    if not controls["passed"] and not force:
        # no point paying for answers that a judge which fails its own controls would score
        (out_dir / "summary.json").write_text(json.dumps(
            {"condition": condition, "run_at": stamp, "judge": judge_model, "stopped":
             "the judge failed its controls", "judge_controls": controls, "usage": meter.usage,
             "cost_usd": meter.cost()}, indent=2), encoding="utf-8")
        out(f"stopped: the judge failed its controls (see {out_dir / 'summary.json'}); --force to run anyway")
        return None
    records, judging, stopped = [], judge_model != "none", None
    # written as produced: a crash, or the credit running out, loses nothing already generated
    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
        for model in models:
            model_started = time.time()
            for i, row in enumerate(rows, 1):
                abstract = (contexts or {}).get(row["id"], {}).get("abstract")
                got = answer(client, meter, model, messages_for(condition, row["question"], abstract),
                             sampling)
                if i % 10 == 0 or i == len(rows):
                    # a CPU-bound model takes half an hour for fifty answers; say where it is
                    out(f"  {model}: {i}/{len(rows)} answered, {time.time() - model_started:.0f}s")
                verdict = None
                if judging:
                    try:
                        verdict = judge(client, meter, judge_model, row["question"], row["answer"], got)
                    except LLMError as e:
                        if not e.terminal:
                            raise
                        # out of credit: keep generating, score later
                        judging, stopped = False, f"judging stopped at {model} {row['id']}: {e}"
                        out(f"!! {stopped}\n   answers are still generated and saved; score them later "
                            f"with: dblpqa rescore --run {out_dir.name}")
                ctx = (contexts or {}).get(row["id"], {})
                rec = {"condition": condition, "model": model, "id": row["id"],
                       "context_source": ctx.get("source") if condition != "closed-book" else None,
                       "question": row["question"], "gold": row["answer"], "answer": got,
                       "score": verdict["score"] if verdict else None,
                       "reason": verdict["reason"] if verdict else None,
                       "rouge_l": round(rouge_l(got, row["answer"]), 4)}
                if condition.startswith(RAG_PREFIX):
                    rec.update(retrieved=ctx.get("retrieved"), source_rank=ctx.get("source_rank"),
                               source_in_context=ctx.get("source_in_context"))
                    if "kept" in ctx:
                        rec.update(kept=ctx["kept"], gate_kept_source=ctx["gate_kept_source"])
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    summary = summarize(records, condition, sampling, out_dir.parent, pool=(notes or {}).get("pool_sha256"))
    print_summary(summary, condition, out)
    payload = {
        "condition": condition, "run_at": stamp, "dataset_sha256": sha, "questions": len(rows),
        "questions_in_dataset": len(all_rows),
        "models": models, "judge": judge_model, "sampling": {sampling: SAMPLING[sampling]},
        "answer_prompt": system_prompt(condition),
        "rouge_l": "LCS F1 over lower-cased alphanumeric tokens, no stemming",
        "judge_controls": controls, "results": summary,
        "paper_reference": {"note": "the paper's no-context baseline, manual 0-2 score",
                            "Mistral-7B": 0.80, "Phi-4": 0.40, "TinyLlama-1.1B": 1.10,
                            "FLAN-T5-Large": 0.30, "FLAN-T5-XXL": 0.60,
                            "best_with_retrieval": {"Mistral-7B Top-5-CD": 1.74}},
        "usage": meter.usage, "cost_usd": meter.cost(), "seconds": round(time.time() - started, 1),
        "stopped": stopped,
    }
    payload.update(notes or {})
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out(f"\n${payload['cost_usd']} · {payload['seconds']}s · written to {out_dir}")
    return payload


def summarize(records, condition, sampling, runs_dir, pool=None):
    """Per model: answered and judged counts, the mean with its interval over the judged answers, and -
    for any condition but closed-book - the paired difference from the same model's closed-book run.
    A rag run is also paired with the oracle and split by whether retrieval found the source paper; a
    selective one is paired with plain RAG on the same pool."""
    summary = {}
    for model in dict.fromkeys(r["model"] for r in records):
        mine = [r for r in records if r["model"] == model]
        judged = [r for r in mine if r.get("score") is not None]
        scores = [r["score"] for r in judged]
        entry = {"answered": len(mine), "judged": len(judged),
                 "judge_score": bootstrap_ci(scores) if scores else None,
                 "distribution": {s: scores.count(s) for s in (2, 1, 0)},
                 "rouge_l": round(sum(r["rouge_l"] for r in mine) / len(mine), 4)}
        if condition != "closed-book" and judged:
            entry["vs_closed_book"] = paired_delta(model, sampling, judged, runs_dir)
        if condition.startswith(RAG_PREFIX) and judged:
            base, mode = rag_parts(condition)
            entry["vs_oracle"] = paired_delta(model, sampling, judged, runs_dir, against="oracle")
            if mode != "plain":
                entry["vs_plain"] = paired_delta(model, sampling, judged, runs_dir, against=base, pool=pool)
            # the question the oracle could not ask: when retrieval brings the wrong papers, does a
            # model do worse than if it had been given nothing at all?
            entry["by_retrieval"] = {}
            for label, hit in (("source retrieved", True), ("source missed", False)):
                part = [r for r in judged if bool(r.get("source_in_context")) == hit]
                if not part:
                    continue
                split = {"questions": len(part), "judge_score": bootstrap_ci([r["score"] for r in part]),
                         "vs_closed_book": paired_delta(model, sampling, part, runs_dir)}
                if mode != "plain":
                    split["vs_plain"] = paired_delta(model, sampling, part, runs_dir, against=base, pool=pool)
                if any("kept" in r for r in part):
                    split["gate_kept_source"] = sum(bool(r.get("gate_kept_source")) for r in part)
                    split["gate_passed_nothing"] = sum(not r.get("kept") for r in part)
                entry["by_retrieval"][label] = split
        summary[model] = entry
    return summary


def print_summary(summary, condition, out=print):
    for model, entry in summary.items():
        ci = entry["judge_score"]
        if not ci:
            out(f"{model:14s} {condition}: {entry['answered']} answered, none scored yet")
            continue
        partial = f" ({entry['judged']} of {entry['answered']} scored)" if entry["judged"] < entry["answered"] else ""
        dist = entry["distribution"]
        out(f"{model:14s} {condition}: {ci['mean']:.2f} / 2  (95% CI {ci['low']:.2f}-{ci['high']:.2f})  "
            f"2s: {dist[2]}  1s: {dist[1]}  0s: {dist[0]}  ROUGE-L {entry['rouge_l']:.3f}{partial}")
        paired = entry.get("vs_closed_book")
        if paired:
            d = paired["delta"]
            out(f"{'':14s} vs closed-book on the same {paired['questions']} questions: "
                f"{d['mean']:+.2f} (95% CI {d['low']:+.2f} to {d['high']:+.2f}); better on "
                f"{paired['better']}, worse on {paired['worse']}, same on {paired['same']}")
        elif condition != "closed-book":
            out(f"{'':14s} (no closed-book run with this model and sampling to compare with)")
        oracle = entry.get("vs_oracle")
        if oracle:
            d = oracle["delta"]
            out(f"{'':14s} vs oracle on the same {oracle['questions']} questions: {d['mean']:+.2f} "
                f"(95% CI {d['low']:+.2f} to {d['high']:+.2f})")
        plain = entry.get("vs_plain")
        if plain:
            d = plain["delta"]
            out(f"{'':14s} vs plain RAG on the same pool and {plain['questions']} questions: {d['mean']:+.2f} "
                f"(95% CI {d['low']:+.2f} to {d['high']:+.2f}); better on {plain['better']}, worse on "
                f"{plain['worse']}")
        elif condition.startswith(RAG_PREFIX) and rag_parts(condition)[1] != "plain":
            out(f"{'':14s} (no plain RAG run on this pool with this model and sampling to compare with)")
        for label, part in (entry.get("by_retrieval") or {}).items():
            ci, paired = part["judge_score"], part.get("vs_closed_book")
            line = f"{'':14s} {label}: {part['questions']} questions, {ci['mean']:.2f}"
            if paired:
                line += (f", vs closed-book {paired['delta']['mean']:+.2f}; better on {paired['better']}, "
                         f"worse on {paired['worse']}")
            if part.get("vs_plain"):
                line += f"; vs plain RAG {part['vs_plain']['delta']['mean']:+.2f}"
            if "gate_kept_source" in part:
                line += (f"; gate kept the source on {part['gate_kept_source']}" if label == "source retrieved"
                         else f"; gate passed nothing on {part['gate_passed_nothing']}")
            out(line)


def rescore(client, run_dir, judge_model, out=print):
    """Score whatever a run left unscored - generated with --judge none, or cut short when the credit
    ran out - and recompute its summary. Already-scored answers are not paid for again."""
    runs = study_dir() / "runs"
    run_dir = Path(run_dir) if run_dir and Path(run_dir).is_absolute() else runs / (run_dir or "")
    if not (run_dir / "answers.jsonl").exists():
        raise SystemExit(f"no answers in {run_dir}")
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meter = Meter()
    todo = [r for r in records if r.get("score") is None]
    if todo:
        rows, sha = load_dataset()
        controls = reusable_controls(judge_model, sha, runs, len(rows))
        if not controls:
            controls = judge_controls(client, meter, judge_model, rows, out)
            if not controls["passed"]:
                raise SystemExit("the judge failed its controls; nothing was scored")
        payload["judge_controls"] = controls
    for i, rec in enumerate(todo, 1):
        verdict = judge(client, meter, judge_model, rec["question"], rec["gold"], rec["answer"])
        rec.update(score=verdict["score"], reason=verdict["reason"])
        if i % 10 == 0 or i == len(todo):
            out(f"  scored {i}/{len(todo)}")
    with open(run_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sampling = next(iter(payload.get("sampling") or {"ours": None}))
    payload["results"] = summarize(records, payload["condition"], sampling, run_dir.parent,
                                   pool=payload.get("pool_sha256"))
    payload["judge"] = judge_model
    payload["stopped"] = None
    payload["cost_usd"] = round((payload.get("cost_usd") or 0) + meter.cost(), 4)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out(f"{run_dir.name}: {len(todo)} answers scored, ${meter.cost()}")
    print_summary(payload["results"], payload["condition"], out)
    return payload


def report(run_dir=None, out=print):
    """One line per question with every model's score side by side, then the judge's reasons for
    anything below 2 - the readable form of a run."""
    runs = study_dir() / "runs"
    run_dir = Path(run_dir) if run_dir else max((d for d in runs.iterdir() if d.is_dir()),
                                                key=lambda d: d.name)
    records = [json.loads(line) for line in
               (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    models = list(dict.fromkeys(r["model"] for r in records))
    by_id = {}
    for r in records:
        by_id.setdefault(r["id"], {"question": r["question"]})[r["model"]] = r
    out(f"{run_dir.name}\n")
    out("id    " + "  ".join(f"{m[-12:]:>12s}" for m in models) + "  question")
    for qid, row in by_id.items():
        out(f"{qid:5s} " + "  ".join(f"{row[m]['score'] if m in row else '-':>12}" for m in models)
            + f"  {row['question'][:70]}")
    out("\nbelow 2, with the judge's reason:")
    for qid, row in by_id.items():
        for m in models:
            if m in row and row[m]["score"] < 2:
                out(f"  {qid} {m} = {row[m]['score']}: {row[m]['reason']}")
    return by_id
