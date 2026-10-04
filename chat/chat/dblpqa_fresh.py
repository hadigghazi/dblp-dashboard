"""
DBLP-QA-Fresh: DBLP-QA's kind of question, about papers too recent for the models to have seen.

On DBLP-QA, strong models answer most questions without retrieval, and an audit labels 58% of them
general knowledge. That is consistent with two explanations: the questions ask what any expert knows,
or the models have read these (mostly older) papers. A companion set built the same way from papers
published after the models' training data separates the two: if memory of the papers matters, closed-
book scores fall on the fresh set within the same question type, and retrieval gains more.

The set is built automatically, the way DBLP-QA's authors built theirs by hand:

  1. papers - dblp journal and conference papers (no CoRR preprints) from FIRST_YEAR on, after
     gpt-4.1's June 2024 training cutoff and Mistral-7B v0.1's 2023 one, sampled with a fixed seed;
  2. abstracts - by DOI from Semantic Scholar, then OpenAlex; papers with a short or no abstract are
     passed over;
  3. questions - gpt-4.1 reads one abstract and writes one question it answers and a one-to-three
     sentence answer taken from it, with four DBLP-QA pairs as style examples;
  4. a check - gpt-4.1-mini, given the abstract, must answer the question fully (the study's judge
     gives 2); a question its own abstract does not answer is dropped.

Nothing is filtered on closed-book answers: a set kept only where the models fail without the abstract
would show they fail by construction. Instead the questions go through the same audit as DBLP-QA, and
the two sets are compared within each audit label. The build is cached per paper, so it resumes.
"""
import csv
import json
import re
import time
from pathlib import Path

import httpx

from . import dblpqa as DQ, dblpqa_rag as RAG

FIRST_YEAR = 2025
SEED = "dblpqa-fresh-1"
GENERATOR = "gpt-4.1"
CHECKER = "gpt-4.1-mini"
MIN_WORDS = 80

# DBLP-QA's own pairs: the example in its paper, and three from the dataset
STYLE = [
    ("What is Magnetic resonance tagging?",
     "Magnetic Resonance (MR) Tagging is a technique used to measure heart deformations by creating a "
     "stripe grid pattern on cardiac images."),
    ("What is Compressive Sensing?",
     "Compressive Sensing (CS) is an advanced signal processing technique that enables the reconstruction "
     "of a signal using far fewer measurements than required by the traditional Nyquist-Shannon sampling "
     "theorem."),
    ("What is CtRL-Sim?",
     "CtRL-Sim is a method that leverages return-conditioned offline reinforcement learning to efficiently "
     "generate reactive and controllable traffic agents."),
    ("Why is a new design for SLO auditing needed?",
     "A new design is necessary to minimize the effort required for adapting to frequent changes in "
     "service landscapes and QoS parameters."),
]

GEN_SYSTEM = (
    "You write questions for a benchmark about computer-science research, the way its authors did. You are "
    "given one paper's abstract. Write one question that the abstract answers, about the paper's main idea, "
    "method, finding or motivation, and its answer: one to three sentences taken from the abstract. The "
    "question must make sense on its own - never 'this paper', 'the authors' or 'the proposed method'. "
    "Examples of questions and answers from the benchmark:\n"
    + "\n".join(f"Q: {q}\nA: {a}" for q, a in STYLE)
    + '\nReply with JSON only: {"question": "...", "answer": "..."}')


def sample_papers(n, parquet=None, first_year=FIRST_YEAR, seed=SEED):
    """[(key, title, year, ee)] - n recent journal/conference papers with a DOI, in a fixed random order."""
    return RAG._parquet_rows("""
        SELECT key, title, TRY_CAST(year AS INTEGER) AS y, ee FROM {src}
        WHERE type IN ('article', 'inproceedings') AND TRY_CAST(year AS INTEGER) >= ?
          AND coalesce(journal, '') <> 'CoRR' AND coalesce(publtype, '') NOT LIKE '%informal%'
          AND array_to_string(ee, ' ') LIKE '%doi.org/10.%'
        ORDER BY hash(key || ?) LIMIT ?""", [first_year, seed, n], parquet)


def parse_pair(text):
    """(question, answer) from the generator's reply, or ValueError."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        got = json.loads(match.group(0)) if match else {}
    except ValueError:
        got = {}
    question, answer = str(got.get("question", "")).strip(), str(got.get("answer", "")).strip()
    if not question.endswith("?") or not 10 <= len(question) <= 250 or not 10 <= len(answer) <= 700:
        raise ValueError(f"not a question and answer: {(text or '')[:200]!r}")
    if re.search(r"\b(this|the present) (paper|study|work|article)\b|\bthe authors\b", question, re.I):
        raise ValueError(f"the question does not stand on its own: {question!r}")
    return question, answer


def build(client, target=100, out=print, parquet=None, http=None, candidates=None):
    """Write the Fresh set (fresh.csv, abstracts.json) to its study folder; returns its rows."""
    DQ.use_dataset("fresh")
    folder = DQ.study_dir()
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "fresh-build.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"papers": {}}
    http = http or httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": "dblp-explorer-research"})
    meter = DQ.Meter()

    papers = sample_papers(candidates or target * 4, parquet)
    if not papers:
        raise SystemExit(f"no papers from {FIRST_YEAR} on in the dblp dump")
    out(f"{len(papers)} candidate papers from {FIRST_YEAR} on (newest year in the sample: "
        f"{max(p[2] for p in papers)})")
    entries = {key: dict(RAG.ids_from_ee(ee), title=(title or "").rstrip(".")) for key, title, _, ee in papers}
    abstracts = state.setdefault("abstracts", {})
    RAG.fetch_pool_abstracts(entries, abstracts, http, out)
    # Semantic Scholar's corpus id and dblp key, so retrieval can recognise the paper under either key
    ids = state.setdefault("s2", {})
    todo = [k for k in entries if k not in ids and entries[k].get("doi")]
    for i in range(0, len(todo), 400):
        chunk = todo[i:i + 400]
        r = DQ._get(http, DQ.S2_BATCH, method="POST", params={"fields": "corpusId,externalIds"},
                    json={"ids": [f"DOI:{entries[k]['doi']}" for k in chunk]})
        for k, paper in zip(chunk, r.json() if r.status_code == 200 else [None] * len(chunk)):
            ids[k] = {"corpus_id": (paper or {}).get("corpusId"),
                      "dblp": ((paper or {}).get("externalIds") or {}).get("DBLP")}

    rows, started = [], time.time()
    for key, title, year, _ in papers:
        if len(rows) >= target:
            break
        abstract = entries[key].get("abstract") or ""
        record = state["papers"].setdefault(key, {"title": title, "year": year})
        if len(abstract.split()) < MIN_WORDS:
            record["skipped"] = "no abstract" if not abstract else "abstract too short"
            continue
        if "question" not in record and "skipped" not in record:
            step = DQ.judge_call(client, meter, GENERATOR, [{"role": "system", "content": GEN_SYSTEM},
                                                            {"role": "user", "content": f"Abstract:\n{abstract}"}])
            try:
                record["question"], record["answer"] = parse_pair(step.get("content"))
            except ValueError as e:
                record["skipped"] = str(e)[:200]
        if "question" in record and "check" not in record:
            got = DQ.answer(client, meter, CHECKER, DQ.messages_for("oracle", record["question"], abstract))
            record["check"] = DQ.judge(client, meter, "gpt-4.1", record["question"], record["answer"], got)
            record["check"]["answer"] = got
        if "question" in record and record["check"]["score"] == 2:
            s2 = ids.get(key) or {}
            rows.append({"id": f"fq{len(rows) + 1}", "question": record["question"], "answer": record["answer"],
                         "dblp_key": key, "semantic_scholar_id": str(s2.get("corpus_id") or "")})
            record["kept_as"] = rows[-1]["id"]
        if len(state["papers"]) % 10 == 0:
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
            out(f"  {len(rows)} kept of {len(state['papers'])} papers read, {time.time() - started:.0f}s, ${meter.cost()}")
    state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")

    with open(folder / "fresh.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DQ.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # the oracle and retrieval read the source abstract from here, as they do for DBLP-QA
    cached = {r["id"]: {"title": entries[r["dblp_key"]]["title"], "abstract": entries[r["dblp_key"]]["abstract"],
                        "source": "fresh-build", "doi": entries[r["dblp_key"]].get("doi"),
                        "dblp_alias": (ids.get(r["dblp_key"]) or {}).get("dblp")} for r in rows}
    (folder / "abstracts.json").write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")

    reasons = {}
    for rec in state["papers"].values():
        why = rec.get("skipped") or ("kept" if rec.get("kept_as") else
                                     "its abstract did not answer it" if rec.get("check") else "not reached")
        why = why if why in ("kept", "no abstract", "abstract too short", "its abstract did not answer it",
                             "not reached") else "unusable question"
        reasons[why] = reasons.get(why, 0) + 1
    out(f"\nDBLP-QA-Fresh: {len(rows)} questions (target {target}) -> {folder / 'fresh.csv'}")
    out("papers read: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + f"   ${meter.cost()}")
    if len(rows) < target:
        out(f"fewer than {target}: re-run with a larger --candidates to read more papers (the rest is cached)")
    return rows
