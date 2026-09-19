"""
Evaluation on a later snapshot than the one trained on, for anchors the model never saw.

Two views, because they answer different questions:
  * pooled - over every candidate pair: ROC-AUC and average precision. Says whether the scores
    separate future co-authors from the rest at all.
  * per anchor - the task as a user meets it: for one author, is the right person near the top of
    the list? MRR, Hits@k, Precision@k, Recall@k, averaged over anchors who did gain a new
    distance-2 co-author.
Every number is reported next to the classic heuristics (common neighbours, Jaccard, Adamic-Adar,
resource allocation, preferential attachment) scored on the same candidate sets.
"""
import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import HEURISTICS

log = logging.getLogger("dblp.ml.links.evaluate")

KS = (5, 10)


def pooled(y, score):
    out = {"pairs": int(len(y)), "positives": int(y.sum()),
           "positive_rate": round(float(y.mean()), 5) if len(y) else None}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y, score)), 4)
        out["average_precision"] = round(float(average_precision_score(y, score)), 4)
    return out


def ranking(u, score, y, ks=KS, seed=0):
    """
    Per-anchor ranking quality, over anchors with at least one positive. Ties (frequent for integer
    heuristics like common neighbours) are broken at random, so a heuristic is not credited for
    the order rows happened to arrive in.
    """
    u, score, y = np.asarray(u), np.asarray(score, dtype=np.float64), np.asarray(y)
    if not len(y):
        return {"anchors": 0}
    jitter = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((jitter, -score, u))         # by anchor, then best score first
    u_s, y_s = u[order], y[order]
    starts = np.flatnonzero(np.r_[True, u_s[1:] != u_s[:-1]])
    ends = np.r_[starts[1:], len(u_s)]
    mrr, hits, prec, rec = [], {k: [] for k in ks}, {k: [] for k in ks}, {k: [] for k in ks}
    for s, e in zip(starts, ends):
        ys = y_s[s:e]
        n_pos = int(ys.sum())
        if n_pos == 0:
            continue
        first = int(np.argmax(ys)) + 1
        mrr.append(1.0 / first)
        for k in ks:
            top = ys[:k]
            hits[k].append(float(top.any()))
            prec[k].append(float(top.sum() / k))
            rec[k].append(float(top.sum() / n_pos))
    out = {"anchors": int(len(starts)), "anchors_with_new_link": len(mrr),
           "mrr": round(float(np.mean(mrr)), 4) if mrr else None}
    for k in ks:
        out[f"hits@{k}"] = round(float(np.mean(hits[k])), 4) if mrr else None
        out[f"precision@{k}"] = round(float(np.mean(prec[k])), 4) if mrr else None
        out[f"recall@{k}"] = round(float(np.mean(rec[k])), 4) if mrr else None
    return out


def baselines(info, y):
    """Each heuristic as a ranker, pooled and per anchor."""
    return {h: {"pooled": pooled(y, info[h]), "ranking": ranking(info["u"], info[h], y)} for h in HEURISTICS}


CALIBRATION_BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001]


def calibration(y, p):
    """How often pairs scored in each range actually became co-authors: what a shown score means."""
    rows = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({"from": lo, "to": min(hi, 1.0), "pairs": int(m.sum()),
                     "came_true": round(float(y[m].mean()), 4),
                     "mean_score": round(float(p[m].mean()), 4)})
    return rows


def came_true_rate(calib, score):
    for row in calib:
        if row["from"] <= score < row["to"] or (score >= 1.0 and row["to"] >= 1.0):
            return row["came_true"]
    return None


def evaluate(u, score, y, info):
    return {
        "pooled": pooled(y, score),
        "ranking": ranking(u, score, y),
        "baselines": baselines(info, y),
        "calibration": calibration(y, score),
    }
