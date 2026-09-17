"""
Evaluation, held out by name block.

Three things are measured, because pairwise accuracy alone would flatter the model: the pairwise
decision, the clustering it produces (B-cubed and ARI, the standard measures in this literature),
and the practical task - given a paper whose author is unassigned, does it land on the right person.
Each is compared with the co-author-overlap heuristic, which is a strong baseline here and the thing
a hand-written rule would do.

Two separate cuts come out of this, because they answer different questions:
  * `cluster` - where to cut the hierarchy when splitting a block. Tuned for B-cubed F1.
  * `assign`  - how sure to be before showing "this looks like Wei Wang 0007" to a person.
                Calibrated to a precision target, since a wrong name is worse than no name.
"""
import logging

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (adjusted_rand_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import features as F

log = logging.getLogger("dblp.ml.evaluate")

MIN_MARGIN = 0.05   # how far the best candidate must beat the runner-up before we name it


def pairwise_metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int8)
    out = {
        "pairs": int(len(y)),
        "positive_rate": round(float(y.mean()), 4) if len(y) else None,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        out["average_precision"] = round(float(average_precision_score(y, p)), 4)
    return out


def bcubed(true_labels, pred_labels):
    """B-cubed precision/recall/F: per item, how pure and how complete its predicted cluster is."""
    true_labels, pred_labels = np.asarray(true_labels), np.asarray(pred_labels)
    n = len(true_labels)
    if n == 0:
        return {}
    precisions, recalls = np.zeros(n), np.zeros(n)
    for i in range(n):
        same_pred = pred_labels == pred_labels[i]
        same_true = true_labels == true_labels[i]
        both = np.count_nonzero(same_pred & same_true)
        precisions[i] = both / np.count_nonzero(same_pred)
        recalls[i] = both / np.count_nonzero(same_true)
    p, r = float(precisions.mean()), float(recalls.mean())
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0.0}


def block_probabilities(con, model, base_name, features=None):
    """All within-block pair probabilities, plus the block's papers and their true people."""
    con.execute("CREATE OR REPLACE TEMP TABLE inst AS SELECT * FROM inst_all WHERE base_name = ?", [base_name])
    if con.execute("SELECT count(*) FROM inst").fetchone()[0] < 4:
        return None
    F.build_pairs(con, sampled=False)
    X, y, info = F.matrix(con, feature_names=features)
    if not len(y):
        return None
    p = model.predict_proba(X)[:, 1]
    pids = np.asarray(sorted(set(info["pid_a"].tolist()) | set(info["pid_b"].tolist())))
    index = {int(pid): i for i, pid in enumerate(pids)}
    truth = dict(con.execute("SELECT pid, person_id FROM inst").fetchall())
    prob = np.zeros((len(pids), len(pids)), dtype=np.float32)
    shared = np.zeros_like(prob)
    for a, b, pr, sh in zip(info["pid_a"], info["pid_b"], p, info["shared_ids"]):
        i, j = index[int(a)], index[int(b)]
        prob[i, j] = prob[j, i] = pr
        shared[i, j] = shared[j, i] = sh
    return {
        "pids": pids, "index": index, "prob": prob, "shared": shared,
        "true": np.asarray([truth[int(pid)] for pid in pids]),
        "X": X, "y": y, "p": p, "info": info,
    }


def cluster(prob, threshold):
    d = 1.0 - prob
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2, 0, 1)   # exactly symmetric, as squareform demands
    if len(d) < 2:
        return np.zeros(len(d), dtype=int)
    return fcluster(linkage(squareform(d, checks=False), method="average"),
                    t=1.0 - threshold, criterion="distance")


def _assignment_decisions(block, scorer):
    """
    Hold out one paper per person (those with 3+ papers) and score it against every candidate.
    Returns one decision per held-out paper: its best score, the margin over the runner-up, and
    whether the best candidate was the right person. Both evaluation and calibration use this.
    """
    true = block["true"]
    decisions = []
    for person in np.unique(true):
        owned = np.nonzero(true == person)[0]
        if len(owned) < 3:
            continue
        held, kept = owned[0], owned[1:]
        scored = []
        for candidate in np.unique(true):
            cand_idx = np.nonzero(true == candidate)[0]
            cand_idx = kept if candidate == person else cand_idx[cand_idx != held]
            if len(cand_idx):
                scored.append((scorer(block, held, cand_idx), candidate))
        if not scored:
            continue
        scored.sort(reverse=True)
        best_score, best_person = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        decisions.append({"score": float(best_score), "margin": float(best_score - runner_up),
                          "correct": bool(best_person == person)})
    return decisions


def model_scorer(block, held, cand_idx):
    return float(block["prob"][held, cand_idx].mean())


def overlap_scorer(block, held, cand_idx):
    # the hand-written rule: how many co-authors this paper shares with the candidate's papers,
    # with the candidate's paper count as the tie-breaker when nothing is shared
    return float(block["shared"][held, cand_idx].mean() * 1000 + len(cand_idx))


def _load_blocks(con, model, blocks, features, max_blocks):
    con.execute("CREATE OR REPLACE TEMP TABLE inst_all AS SELECT * FROM inst")
    out = []
    for base_name in blocks:
        if len(out) >= max_blocks:
            break
        block = block_probabilities(con, model, base_name, features)
        if block is not None:
            out.append((base_name, block))
    con.execute("CREATE OR REPLACE TEMP TABLE inst AS SELECT * FROM inst_all")
    return out


def tune_cluster_threshold(con, model, blocks, features=None, max_blocks=60):
    """The cut that maximises B-cubed F1 on validation blocks - not the pairwise F1 cut, which
    over-splits because it optimises a different question."""
    loaded = _load_blocks(con, model, blocks, features, max_blocks)
    if not loaded:
        return 0.5, []
    grid = np.round(np.arange(0.20, 0.91, 0.05), 2)
    curve = []
    for t in grid:
        scores, weights = [], []
        for _, block in loaded:
            b3 = bcubed(block["true"], cluster(block["prob"], float(t)))
            scores.append(b3["f1"])
            weights.append(len(block["pids"]))
        curve.append({"threshold": float(t), "b3_f1": round(float(np.average(scores, weights=weights)), 4)})
    best = max(curve, key=lambda r: r["b3_f1"])
    log.info("cluster threshold %.2f (validation B3 F1 %.3f on %d blocks)",
             best["threshold"], best["b3_f1"], len(loaded))
    return best["threshold"], curve


def calibrate_assignment(con, model, blocks, features=None, target_precision=0.95, max_blocks=60):
    """
    Pick the score a suggestion must reach before it is shown, so that shown suggestions are right
    about `target_precision` of the time. Reports the coverage that costs: staying silent on the rest
    is the point - a wrong name is worse than no name.
    """
    loaded = _load_blocks(con, model, blocks, features, max_blocks)
    decisions = [d for _, block in loaded for d in _assignment_decisions(block, model_scorer)]
    if not decisions:
        return {"threshold": 0.5, "precision": None, "coverage": None, "decisions": 0}
    scores = np.asarray([d["score"] for d in decisions])
    margins = np.asarray([d["margin"] for d in decisions])
    correct = np.asarray([d["correct"] for d in decisions])
    curve = []
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        shown = (scores >= t) & (margins >= MIN_MARGIN)
        if shown.sum() < 20:
            continue
        curve.append({"threshold": float(t),
                      "precision": round(float(correct[shown].mean()), 4),
                      "coverage": round(float(shown.mean()), 4),
                      "shown": int(shown.sum())})
    passing = [r for r in curve if r["precision"] >= target_precision]
    chosen = min(passing, key=lambda r: r["threshold"]) if passing else (
        max(curve, key=lambda r: r["precision"]) if curve else
        {"threshold": 0.5, "precision": None, "coverage": None})
    log.info("assignment threshold %.2f (precision %s, coverage %s of %d decisions)",
             chosen["threshold"], chosen.get("precision"), chosen.get("coverage"), len(decisions))
    return {**chosen, "target_precision": target_precision, "decisions": len(decisions), "curve": curve}


def evaluate_blocks(con, model, thresholds, blocks, features=None, max_blocks=150):
    """Clustering and assignment quality on whole held-out blocks."""
    cluster_t = thresholds["cluster"] if isinstance(thresholds, dict) else thresholds
    assign_t = thresholds.get("assign", cluster_t) if isinstance(thresholds, dict) else thresholds
    loaded = _load_blocks(con, model, blocks, features, max_blocks)
    rows = []
    for base_name, block in loaded:
        pred = cluster(block["prob"], cluster_t)
        b3 = bcubed(block["true"], pred)
        b3_overlap = bcubed(block["true"], cluster((block["shared"] > 0).astype(np.float32), 0.5))
        model_dec = _assignment_decisions(block, model_scorer)
        overlap_dec = _assignment_decisions(block, overlap_scorer)
        rows.append({
            "block": base_name, "papers": len(block["pids"]),
            "true_people": int(len(np.unique(block["true"]))), "predicted_clusters": int(len(np.unique(pred))),
            "b3_f1": b3["f1"], "b3_precision": b3["precision"], "b3_recall": b3["recall"],
            "b3_f1_overlap_baseline": b3_overlap["f1"],
            "ari": float(adjusted_rand_score(block["true"], pred)),
            "decisions": model_dec, "overlap_decisions": overlap_dec,
        })
    if not rows:
        return {"blocks": 0}

    weights = np.asarray([r["papers"] for r in rows], dtype=float)
    mean = lambda key: round(float(np.average([r[key] for r in rows], weights=weights)), 4)
    all_dec = [d for r in rows for d in r["decisions"]]
    all_overlap = [d for r in rows for d in r["overlap_decisions"]]
    shown = [d for d in all_dec if d["score"] >= assign_t and d["margin"] >= MIN_MARGIN]
    summary = {
        "blocks": len(rows),
        "papers": int(weights.sum()),
        "cluster_threshold": cluster_t,
        "b3_f1": mean("b3_f1"), "b3_precision": mean("b3_precision"), "b3_recall": mean("b3_recall"),
        "b3_f1_overlap_baseline": mean("b3_f1_overlap_baseline"),
        "ari": mean("ari"),
        "cluster_count_ratio": round(mean("predicted_clusters") / mean("true_people"), 3) if mean("true_people") else None,
        "assignment": {
            "held_out_papers": len(all_dec),
            # forced choice: always name the best candidate, whatever the score
            "top1_accuracy": round(float(np.mean([d["correct"] for d in all_dec])), 4) if all_dec else None,
            "top1_accuracy_overlap_baseline":
                round(float(np.mean([d["correct"] for d in all_overlap])), 4) if all_overlap else None,
            # what a user would actually be shown, at the calibrated cut
            "assign_threshold": assign_t,
            "shown": len(shown),
            "coverage": round(len(shown) / len(all_dec), 4) if all_dec else None,
            "precision_when_shown": round(float(np.mean([d["correct"] for d in shown])), 4) if shown else None,
        },
        "hardest_blocks": [{k: v for k, v in r.items() if k not in ("decisions", "overlap_decisions")}
                           for r in sorted(rows, key=lambda r: r["b3_f1"])[:5]],
    }
    return summary


def baseline_pairwise(y, info, threshold=0.5):
    """The rule 'same person if they share at least one co-author'."""
    p = (np.asarray(info["shared_ids"]) > 0).astype(np.float32)
    return pairwise_metrics(y, p, threshold)
