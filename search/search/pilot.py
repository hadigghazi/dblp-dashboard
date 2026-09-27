"""
Is a different embedding model actually better on THIS data?

Re-embedding 5.36M titles costs money and a day, and MTEB averages are computed on web text, not on
the titles of computer-science papers - a domain bge-small may well handle better than its average
suggests. So before committing: embed a sample of the corpus with the candidate model, and score the
same queries against the same papers with both.

The trick that makes it cheap is scoring within the sample. A rank among 200,000 papers is not a
rank among 5.36M, but both models are ranked against exactly the same 200,000, so the comparison is
fair even though the absolute numbers are not comparable with a full run. Two hundred thousand
titles cost about forty cents with text-embedding-3-large.

The queries come from the file the assistant's paraphrase test writes, so the pilot is answering the
question that motivated it, on the same questions.
"""
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

from . import config, embed as E, store as S, vectors as V

log = logging.getLogger("dblp.search.pilot")


def load_cases(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    return [{"key": c["key"], "query": c["query"], "title": c.get("title")} for c in cases]


def sample_rows(con, n_sample, keys, seed=7):
    """`n_sample` papers, with the ones being searched for guaranteed to be among them - otherwise
    the experiment measures nothing."""
    con.execute("CREATE OR REPLACE TEMP TABLE pilot_key (key VARCHAR)")
    con.executemany("INSERT INTO pilot_key VALUES (?)", [(k,) for k in keys])
    rows = con.execute("""
        WITH wanted AS (SELECT p.row, p.pid, p.key, p.title FROM x.paper p JOIN pilot_key k USING (key)),
        rest AS (
            SELECT p.row, p.pid, p.key, p.title FROM x.paper p
            WHERE p.key NOT IN (SELECT key FROM pilot_key)
            QUALIFY row_number() OVER (ORDER BY hash(p.pid::BIGINT * 1000003 + ?)) <= ?)
        SELECT * FROM wanted UNION ALL SELECT * FROM rest""",
                     [int(seed), max(0, int(n_sample) - len(keys))]).fetchall()
    return [{"row": int(r[0]), "pid": int(r[1]), "key": r[2], "title": r[3]} for r in rows]


def _ranks(doc_vectors, query_vectors, target_index):
    """The target's rank for each query, by cosine against the sampled papers."""
    sims = doc_vectors @ query_vectors.T            # (papers, queries)
    out = []
    for q in range(sims.shape[1]):
        column = sims[:, q]
        target = column[target_index[q]]
        out.append(int((column > target).sum()) + 1)  # how many papers beat it, plus itself
    return out


def _summary(ranks, ks=(1, 5, 10, 20, 100)):
    n = len(ranks) or 1
    out = {"queries": len(ranks), "median_rank": int(np.median(ranks)) if ranks else None}
    for k in ks:
        out[f"acc@{k}"] = round(sum(1 for r in ranks if r <= k) / n, 4)
    out["mrr"] = round(sum(1 / r for r in ranks) / n, 4)
    return out


def run(con, fingerprint, cases, candidate_model, candidate_dim, n_sample=200_000, seed=7):
    """Score `cases` against a sample of the corpus with the current vectors and with a candidate
    model's. Returns both summaries and what the candidate cost."""
    papers = sample_rows(con, n_sample, [c["key"] for c in cases], seed)
    by_key = {p["key"]: i for i, p in enumerate(papers)}
    cases = [c for c in cases if c["key"] in by_key]
    target_index = [by_key[c["key"]] for c in cases]
    queries = [c["query"] for c in cases]
    log.info("pilot: %s papers, %s queries", f"{len(papers):,}", len(cases))

    # ---- the incumbent, read from the index it already wrote
    t0 = time.time()
    n_all = con.execute("SELECT count(*) FROM x.paper").fetchone()[0]
    stored = V.open_vectors(fingerprint, n_all, "r")
    current_docs = np.asarray(stored[[p["row"] for p in papers], :], dtype=np.float32)
    del stored
    local = E.make_encoder()
    current_queries = np.vstack([local.encode_query(q) for q in queries]).astype(np.float32)
    current = _ranks(current_docs, current_queries, target_index)
    log.info("current model scored in %.0fs", time.time() - t0)

    # ---- the candidate
    t1 = time.time()
    encoder = E.make_encoder(candidate_model, candidate_dim)
    candidate_docs = encoder.encode_docs([p["title"] for p in papers])
    candidate_queries = encoder.encode_docs(queries)
    candidate = _ranks(candidate_docs, candidate_queries, target_index)
    log.info("candidate scored in %.0fs", time.time() - t1)

    better = sum(1 for a, b in zip(candidate, current) if a < b)
    worse = sum(1 for a, b in zip(candidate, current) if a > b)
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "ranks are within the sample, so they are comparable between the two models but not "
                "with a full-corpus run",
        "sample": len(papers), "queries": len(cases),
        "current": {"model": config.MODEL_NAME, "dim": config.EMBED_DIM, **_summary(current)},
        "candidate": {"model": candidate_model, "dim": candidate_dim, **_summary(candidate)},
        "per_query": {"candidate_better": better, "current_better": worse,
                      "same": len(cases) - better - worse},
        "candidate_tokens": getattr(encoder, "tokens", None),
        "seconds_waiting_for_rate_limit": round(getattr(encoder, "waited", 0.0), 1),
        "candidate_cost_usd": encoder.cost_usd() if hasattr(encoder, "cost_usd") else None,
        "projected_full_corpus_usd": (round(encoder.cost_usd() * n_all / max(1, len(papers)), 2)
                                      if hasattr(encoder, "cost_usd") else None),
        "projected_full_corpus_minutes": (round(getattr(encoder, "tokens", 0) * n_all
                                                / max(1, len(papers)) / config.API_TOKENS_PER_MINUTE, 1)
                                          if hasattr(encoder, "cost_usd") else None),
        "seconds": round(time.time() - t0, 1),
    }


def save(payload, fingerprint):
    d = config.MODELS_DIR / "search-eval"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"pilot-{fingerprint}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
