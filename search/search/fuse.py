"""Reciprocal rank fusion: combine two rankings of the same items into one, using only their
ranks (not the raw scores, which live on different, incomparable scales - BM25 vs. cosine)."""
from . import config


def rrf(*rankings, k=None, weights=None):
    """rankings: each an iterable of (pid, score), best first. Returns [(pid, fused_score)], best
    first; a pid missing from a ranking simply contributes nothing from it.

    `weights` says how much each ranking is worth for this query. Equal weights - the default,
    and the usual presentation of RRF - assume both rankings are informative, which is false
    whenever one of them had nothing to match on."""
    k = k or config.RRF_K
    weights = weights or [1.0] * len(rankings)
    fused = {}
    for ranking, weight in zip(rankings, weights):
        if not weight:
            continue
        for rank, (pid, _) in enumerate(ranking, start=1):
            fused[pid] = fused.get(pid, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])
