"""
Evaluation, held out by name block.

Three things are measured, because pairwise accuracy alone would flatter the model: the pairwise
decision, the clustering it produces (B-cubed and ARI, the standard measures in this literature),
and the practical task - given a paper whose author is unassigned, does it land on the right person.
Each is compared with the co-author-overlap heuristic, which is a strong baseline here and the thing
a hand-written rule would do.
"""
import logging

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (adjusted_rand_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import features as F

log = logging.getLogger("dblp.ml.evaluate")


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


def _block_probabilities(con, model, base_name):
    """All within-block pair probabilities, plus the block's papers and their true people."""
    con.execute("CREATE OR REPLACE TEMP TABLE inst AS SELECT * FROM inst_all WHERE base_name = ?", [base_name])
    n_inst = con.execute("SELECT count(*) FROM inst").fetchone()[0]
    if n_inst < 4:
        return None
    F.build_pairs(con, sampled=False)
    X, y, info = F.matrix(con)
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


def _cluster(prob, threshold):
    d = 1.0 - prob
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2, 0, 1)   # exactly symmetric, as squareform demands
    if len(d) < 2:
        return np.zeros(len(d), dtype=int)
    return fcluster(linkage(squareform(d, checks=False), method="average"),
                    t=1.0 - threshold, criterion="distance")


def _assignment(block, scorer):
    """Hold out one paper per person (those with 3+), assign it to the best candidate person."""
    true, pids = block["true"], block["pids"]
    correct = total = 0
    for person in np.unique(true):
        owned = np.nonzero(true == person)[0]
        if len(owned) < 3:
            continue
        held, kept = owned[0], owned[1:]
        best_person, best_score = None, -np.inf
        for candidate in np.unique(true):
            cand_idx = np.nonzero(true == candidate)[0]
            cand_idx = cand_idx[cand_idx != held]
            if candidate == person:
                cand_idx = kept
            if not len(cand_idx):
                continue
            score = scorer(block, held, cand_idx)
            if score > best_score:
                best_person, best_score = candidate, score
        total += 1
        correct += int(best_person == person)
    return correct, total


def _model_scorer(block, held, cand_idx):
    return float(block["prob"][held, cand_idx].mean())


def _overlap_scorer(block, held, cand_idx):
    # the hand-written rule: how many co-authors this paper shares with the candidate's papers,
    # with the candidate's paper count as the tie-breaker when nothing is shared
    return float(block["shared"][held, cand_idx].mean() * 1000 + len(cand_idx))


def evaluate_blocks(con, model, threshold, blocks, max_blocks=150):
    """Clustering and assignment quality on whole held-out blocks."""
    con.execute("CREATE OR REPLACE TEMP TABLE inst_all AS SELECT * FROM inst")
    rows, done = [], 0
    for base_name in blocks:
        if done >= max_blocks:
            break
        block = _block_probabilities(con, model, base_name)
        if block is None:
            continue
        done += 1
        pred = _cluster(block["prob"], threshold)
        b3 = bcubed(block["true"], pred)
        overlap_pred = _cluster((block["shared"] > 0).astype(np.float32), 0.5)
        b3_overlap = bcubed(block["true"], overlap_pred)
        m_correct, m_total = _assignment(block, _model_scorer)
        o_correct, _ = _assignment(block, _overlap_scorer)
        rows.append({
            "block": base_name, "papers": len(block["pids"]),
            "true_people": int(len(np.unique(block["true"]))), "predicted_clusters": int(len(np.unique(pred))),
            "b3_f1": b3["f1"], "b3_precision": b3["precision"], "b3_recall": b3["recall"],
            "b3_f1_overlap_baseline": b3_overlap["f1"],
            "ari": float(adjusted_rand_score(block["true"], pred)),
            "assign_correct": m_correct, "assign_total": m_total, "assign_correct_overlap": o_correct,
        })
    con.execute("CREATE OR REPLACE TEMP TABLE inst AS SELECT * FROM inst_all")
    if not rows:
        return {"blocks": 0}
    weights = np.asarray([r["papers"] for r in rows], dtype=float)
    mean = lambda key: round(float(np.average([r[key] for r in rows], weights=weights)), 4)
    assign_total = sum(r["assign_total"] for r in rows)
    summary = {
        "blocks": len(rows),
        "papers": int(weights.sum()),
        "b3_f1": mean("b3_f1"), "b3_precision": mean("b3_precision"), "b3_recall": mean("b3_recall"),
        "b3_f1_overlap_baseline": mean("b3_f1_overlap_baseline"),
        "ari": mean("ari"),
        "cluster_count_ratio": mean("predicted_clusters") / mean("true_people") if mean("true_people") else None,
        "assignment": {
            "held_out_papers": assign_total,
            "top1_accuracy": round(sum(r["assign_correct"] for r in rows) / assign_total, 4) if assign_total else None,
            "top1_accuracy_overlap_baseline":
                round(sum(r["assign_correct_overlap"] for r in rows) / assign_total, 4) if assign_total else None,
        },
        "hardest_blocks": sorted(rows, key=lambda r: r["b3_f1"])[:5],
    }
    return summary


def baseline_pairwise(y, info, threshold=0.5):
    """The rule 'same person if they share at least one co-author'."""
    p = (np.asarray(info["shared_ids"]) > 0).astype(np.float32)
    return pairwise_metrics(y, p, threshold)
