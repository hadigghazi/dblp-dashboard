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

log = logging.getLogger("dblp.chat.dblpqa")

DATASET_URL = "https://seafile.rlp.net/f/6581519cdd1d4782bccc/?dl=1"
COLUMNS = ["id", "question", "answer", "dblp_key", "semantic_scholar_id"]
SEED = 7

ANSWER_SYSTEM = "You answer questions about computer-science research. Answer in one to three sentences."

# Generation settings. "paper" is Table 1 of the paper (Mistral-7B and TinyLlama rows): what a
# reproduction of their models has to use. "ours" is deterministic, for the hosted models.
SAMPLING = {
    "ours": {"temperature": 0, "extra": {}},
    "paper": {"temperature": 0.7, "extra": {"top_p": 0.9, "max_tokens": 512}},
}
OLLAMA_PREFIX = "ollama:"


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

def load_dataset(cache_dir=None, url=DATASET_URL):
    """The 50 questions, downloaded once and kept. Returns (rows, sha256 of the file)."""
    cache = Path(cache_dir or config.MODELS_DIR / "dblpqa")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "dblp-qa.csv"
    if not path.exists():
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


def answer_closed_book(client, meter, model, question, sampling="ours"):
    settings = SAMPLING[sampling]
    target, name = client_for(model, client)
    try:
        step = target.complete([{"role": "system", "content": ANSWER_SYSTEM},
                                {"role": "user", "content": question}], model=name,
                               temperature=settings["temperature"], extra=settings["extra"])
    except Exception as e:
        # the client's message is written for a website visitor; an experiment needs the server's own
        raise RuntimeError(f"{model} failed: {getattr(e, 'raw', '') or e}") from e
    if not model.startswith(OLLAMA_PREFIX):          # a local model costs nothing
        meter.add(model, step.get("usage", {}))
    return (step.get("content") or "").strip()


def judge(client, meter, model, question, gold, candidate):
    prompt = f"Question: {question}\nGround truth: {gold}\nAnswer to grade: {candidate}"
    step = client.complete([{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt}], model=model, temperature=0)
    meter.add(model, step.get("usage", {}))
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


def reusable_controls(judge_model, sha, runs_dir=None):
    """A judge that passed its controls on this exact dataset already has: no need to pay again."""
    runs = Path(runs_dir or config.MODELS_DIR / "dblpqa" / "runs")
    for summary in sorted(runs.glob("*/summary.json"), reverse=True):
        try:
            got = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        controls = got.get("judge_controls") or {}
        if (got.get("judge") == judge_model and got.get("dataset_sha256") == sha
                and controls.get("passed") and got.get("questions") == 50):
            return dict(controls, reused_from=summary.parent.name)
    return None


def run_closed_book(client, models, judge_model, rows, sha, out_dir=None, out=print, force=False,
                    sampling="ours", reuse_controls=True):
    started = time.time()
    meter = Meter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_dir or config.MODELS_DIR / "dblpqa" / "runs" / f"{stamp}-closed-book")
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = reusable_controls(judge_model, sha, out_dir.parent) if reuse_controls else None
    if controls:
        out(f"judge controls: passed on this dataset in run {controls['reused_from']} - reused")
    else:
        controls = judge_controls(client, meter, judge_model, rows, out)
    if not controls["passed"] and not force:
        # no point paying for answers that a judge which fails its own controls would score
        (out_dir / "summary.json").write_text(json.dumps(
            {"condition": "closed-book", "run_at": stamp, "judge": judge_model, "stopped":
             "the judge failed its controls", "judge_controls": controls, "usage": meter.usage,
             "cost_usd": meter.cost()}, indent=2), encoding="utf-8")
        out(f"stopped: the judge failed its controls (see {out_dir / 'summary.json'}); --force to run anyway")
        return None
    records, summary = [], {}
    for model in models:
        scores, rouges = [], []
        for row in rows:
            answer = answer_closed_book(client, meter, model, row["question"], sampling)
            verdict = judge(client, meter, judge_model, row["question"], row["answer"], answer)
            rl = rouge_l(answer, row["answer"])
            scores.append(verdict["score"])
            rouges.append(rl)
            records.append({"condition": "closed-book", "model": model, "id": row["id"],
                            "question": row["question"], "gold": row["answer"], "answer": answer,
                            "score": verdict["score"], "reason": verdict["reason"], "rouge_l": round(rl, 4)})
        summary[model] = {"judge_score": bootstrap_ci(scores),
                          "distribution": {s: scores.count(s) for s in (2, 1, 0)},
                          "rouge_l": round(sum(rouges) / len(rouges), 4)}
        ci = summary[model]["judge_score"]
        out(f"{model:14s} closed-book: {ci['mean']:.2f} / 2  (95% CI {ci['low']:.2f}-{ci['high']:.2f})  "
            f"2s: {scores.count(2)}  1s: {scores.count(1)}  0s: {scores.count(0)}  "
            f"ROUGE-L {summary[model]['rouge_l']:.3f}")

    payload = {
        "condition": "closed-book", "run_at": stamp, "dataset_sha256": sha, "questions": len(rows),
        "models": models, "judge": judge_model, "sampling": {sampling: SAMPLING[sampling]},
        "answer_prompt": ANSWER_SYSTEM,
        "rouge_l": "LCS F1 over lower-cased alphanumeric tokens, no stemming",
        "judge_controls": controls, "results": summary,
        "paper_reference": {"note": "the paper's no-context baseline, manual 0-2 score",
                            "Mistral-7B": 0.80, "Phi-4": 0.40, "TinyLlama-1.1B": 1.10,
                            "FLAN-T5-Large": 0.30, "FLAN-T5-XXL": 0.60,
                            "best_with_retrieval": {"Mistral-7B Top-5-CD": 1.74}},
        "usage": meter.usage, "cost_usd": meter.cost(), "seconds": round(time.time() - started, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out(f"\n${payload['cost_usd']} · {payload['seconds']}s · written to {out_dir}")
    return payload


def report(run_dir=None, out=print):
    """One line per question with every model's score side by side, then the judge's reasons for
    anything below 2 - the readable form of a run."""
    runs = config.MODELS_DIR / "dblpqa" / "runs"
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
