"""
Evaluation on the papers of a later year than the statistics and the ranker were built from.

For every test paper: where does its real venue land in the ranking? Accuracy@k and MRR over ALL
test papers - a paper whose venue is outside the class set, or not among the candidates, counts as
a miss, and both shares are reported as the ceilings they are. Each scorer alone (Naive Bayes, the
centroid, the authors' history, plain popularity) is ranked on the same candidates, and the numbers
are split by whether any author had a history at all: history is the strong signal when it exists;
the title has to carry the rest.
"""
import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

log = logging.getLogger("dblp.ml.venues.evaluate")

KS = (1, 3, 5, 10)


def true_ranks(qid, score, y, seed=0):
    """qid -> rank of the true series under `score` (ties broken at random); absent = not a candidate."""
    qid, score, y = np.asarray(qid), np.asarray(score, dtype=np.float64), np.asarray(y)
    if not len(y):
        return {}
    jitter = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((jitter, -score, qid))
    q_s, y_s = qid[order], y[order]
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    pos = np.arange(len(q_s)) - np.repeat(starts, np.diff(np.r_[starts, len(q_s)])) + 1
    hit = y_s == 1
    return {int(q): int(r) for q, r in zip(q_s[hit], pos[hit])}


def summarize(ranks, n, ks=KS):
    """Accuracy@k and MRR over n papers, given the ranks of those whose venue was found."""
    r = np.asarray(list(ranks.values()), dtype=float)
    out = {"papers": int(n), "found": int(len(r))}
    for k in ks:
        out[f"acc@{k}"] = round(float((r <= k).sum() / n), 4) if n else None
    out["mrr"] = round(float((1.0 / r).sum() / n), 4) if n else None
    return out


def popularity_ranks(con):
    """Rank of each query's true series in the class set ordered by recent output."""
    return {int(q): int(r) for q, r in con.execute("""
        SELECT q.qid, r.rnk FROM q
        JOIN (SELECT sid, row_number() OVER (ORDER BY recent DESC, papers DESC, sid) AS rnk FROM cls) r
          ON r.sid = q.true_sid""").fetchall()}


def evaluate(con, info, y, p):
    """Everything reported for one test set. `info`/`y`/`p` come from the `pair` table; the query
    table `q` and the class set `cls` must be in scope."""
    q_all = con.execute("""
        SELECT q.qid, (q.true_sid IN (SELECT sid FROM cls))::INT AS covered,
               (EXISTS (SELECT 1 FROM pair p WHERE p.qid = q.qid AND p.from_history = 1))::INT AS has_history
        FROM q""").fetchnumpy()
    qids = np.asarray(q_all["qid"]).astype(int)
    covered = np.asarray(q_all["covered"]).astype(bool)
    has_hist = np.asarray(q_all["has_history"]).astype(bool)
    n = len(qids)

    rankers = {
        "model": true_ranks(info["qid"], p, y),
        "naive_bayes": true_ranks(info["qid"], info["nb"], y),
        "centroid": true_ranks(info["qid"], info["cen"], y),
        "history": true_ranks(info["qid"],
                              info["hist_papers"] * 1000.0 - info["hist_age"] + info["log_recent"], y),
        "popularity": popularity_ranks(con),
    }
    in_cand = set(rankers["model"])
    out = {
        "papers": int(n),
        "covered": {"papers": int(covered.sum()), "share": round(float(covered.mean()), 4) if n else None},
        "in_candidates": {"papers": len(in_cand),
                          "share": round(len(in_cand) / n, 4) if n else None,
                          "share_of_covered": round(len(in_cand) / max(int(covered.sum()), 1), 4)},
        "with_author_history": {"papers": int(has_hist.sum()), "share": round(float(has_hist.mean()), 4) if n else None},
        "rankers": {name: summarize(r, n) for name, r in rankers.items()},
    }
    for label, mask in (("with_history", has_hist), ("without_history", ~has_hist)):
        sel = set(qids[mask].tolist())
        out[label] = {name: summarize({q: r for q, r in rk.items() if q in sel}, len(sel)) for name, rk in rankers.items()}
    if len(np.unique(y)) == 2:
        out["pairwise"] = {"pairs": int(len(y)), "roc_auc": round(float(roc_auc_score(y, p)), 4),
                           "average_precision": round(float(average_precision_score(y, p)), 4)}
    return out


CALIBRATION_BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]


def calibration(y, p):
    rows = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({"from": lo, "to": min(hi, 1.0), "pairs": int(m.sum()),
                     "came_true": round(float(y[m].mean()), 4), "mean_score": round(float(p[m].mean()), 4)})
    return rows


def came_true_rate(calib, score):
    """The measured rate for the score's bin; if no test pair reached that bin, the nearest bin's."""
    if not calib:
        return None
    for row in calib:
        if row["from"] <= score < row["to"] or (score >= 1.0 and row["to"] >= 1.0):
            return row["came_true"]
    nearest = min(calib, key=lambda r: min(abs(score - r["from"]), abs(score - r["to"])))
    return nearest["came_true"]
