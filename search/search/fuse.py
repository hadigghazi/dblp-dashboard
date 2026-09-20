"""Reciprocal rank fusion: combine two rankings of the same items into one, using only their
ranks (not the raw scores, which live on different, incomparable scales - BM25 vs. cosine)."""
from . import config


def rrf(*rankings, k=None):
    """rankings: each an iterable of (pid, score), best first. Returns [(pid, fused_score)], best
    first; a pid missing from a ranking simply contributes nothing from it."""
    k = k or config.RRF_K
    fused = {}
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking, start=1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])
