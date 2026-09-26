"""
Self-retrieval evaluation: for a sample of indexed papers, replace a distinctive word in the title
with a synonym and check whether the paper still comes back for that altered query.

This targets exactly the case exact-word search cannot handle by construction - a substituted word
is no longer a substring of the title - so it directly measures the thing hybrid search is for,
without needing external relevance judgments. BM25 can partially recover through any words that
were *not* substituted; dense embeddings are the only signal that can recover through the
substituted word itself. Titles containing none of the table's words cannot be used and are
reported as such, so the numbers stay honest about what was actually tested.
"""
import logging
import re

import numpy as np

from . import bm25 as B, config, fuse as F, store as S, vectors as V

log = logging.getLogger("dblp.search.evaluate")

SYNONYMS = {
    "neural": "deep learning", "graph": "network", "algorithm": "method", "clustering": "grouping",
    "prediction": "forecasting", "optimization": "tuning", "classification": "categorization",
    "detection": "identification", "analysis": "study", "efficient": "fast", "robust": "reliable",
    "survey": "review", "learning": "training", "model": "framework", "approach": "technique",
    "framework": "system", "evaluation": "assessment", "recognition": "identification",
    "generation": "synthesis", "representation": "embedding", "adaptive": "flexible",
    "scalable": "large-scale", "novel": "new", "improved": "enhanced", "comparative": "comparison",
    "distributed": "decentralised", "secure": "protected", "privacy": "confidentiality",
    "wireless": "radio", "autonomous": "self-driving",
}
_PATTERNS = [(re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE), s) for w, s in SYNONYMS.items()]


def substitute(title):
    """(query, changed): the title with the first known word replaced by its synonym."""
    out = title
    for pattern, sub in _PATTERNS:
        if pattern.search(out):
            return pattern.sub(sub, out, count=1), True
    return title, False


def _rank_of(ranking, pid):
    for r, (p, _) in enumerate(ranking, start=1):
        if p == pid:
            return r
    return None


def _summary(ranks, n, ks=(1, 5, 10)):
    r = np.asarray([v for v in ranks if v is not None], dtype=float)
    out = {"papers": int(n), "found": int(len(r))}
    for k in ks:
        out[f"acc@{k}"] = round(float((r <= k).sum() / n), 4) if n else None
    out["mrr"] = round(float((1.0 / r).sum() / n), 4) if n else None
    return out


def evaluate(con, fingerprint, encoder, n_papers=None, seed=None):
    n_papers = n_papers or config.EVAL_PAPERS
    seed = seed if seed is not None else config.SEED
    rows = con.execute("""
        SELECT pid, title FROM x.paper
        QUALIFY row_number() OVER (ORDER BY hash(pid::BIGINT * 1000003 + ?)) <= ?""", [seed, n_papers]).fetchall()

    cases = []
    for pid, title in rows:
        q, changed = substitute(title)
        if changed:
            cases.append((pid, q))
    log.info("evaluation: %s of %s sampled titles contain a table word", f"{len(cases):,}", f"{len(rows):,}")
    if not cases:
        return {"sampled": len(rows), "substitutable": 0}

    bm25_ranks, dense_ranks, hybrid_ranks = [], [], []
    weights, coverages = [], []      # what the fusion made of the word ranking on THIS population
    for pid, q in cases:
        stats = {}
        sparse = B.search(con, q, stats=stats)
        qvec = encoder.encode_query(q)
        dense_hits = V.search(con, fingerprint, qvec)
        weight = S.sparse_weight(stats)
        weights.append(weight)
        coverages.append(stats.get("coverage"))
        fused = F.rrf(sparse, dense_hits, weights=(weight, 1.0))
        bm25_ranks.append(_rank_of(sparse, pid))
        dense_ranks.append(_rank_of(dense_hits, pid))
        hybrid_ranks.append(_rank_of(fused, pid))

    n = len(cases)
    return {
        "sampled": len(rows), "substitutable": n,
        "bm25_only": _summary(bm25_ranks, n),
        "dense_only": _summary(dense_ranks, n),
        "hybrid": _summary(hybrid_ranks, n),
        "mean_sparse_weight": round(sum(weights) / len(weights), 3) if weights else None,
        "mean_word_match_coverage": (round(sum(c for c in coverages if c is not None)
                                           / max(1, len([c for c in coverages if c is not None])), 3)),
        "example_substitutions": [f"{w} -> {s}" for w, s in list(SYNONYMS.items())[:8]],
    }
