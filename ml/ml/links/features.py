"""
Pair features for link prediction, all computed in SQL by graph.build_pairs; this module fixes the
model's column order and assembles the matrix.

The first group is the classic neighbourhood heuristics (Liben-Nowell & Kleinberg): they are both
features and the baselines the model has to beat. The rest is what those heuristics ignore - when
the bridge between the two people was last active, how active each of them is now, how far into
their careers they are, and whether they publish in the same venues.
"""
import logging

import numpy as np

log = logging.getLogger("dblp.ml.links.features")

HEURISTICS = ["cn", "jaccard", "aa", "ra", "pa"]

FEATURES = [
    "cn", "jaccard", "aa", "ra", "pa",
    "deg_u", "deg_v",
    "bridge_age", "cn_recent", "bridge_strength",
    "papers_u", "papers_v", "recent_u", "recent_v",
    "age_u", "age_v", "idle_u", "idle_v",
    "shared_venues", "n_venues_u", "n_venues_v",
]

INFO = ["u", "v", "y", "bucket"] + HEURISTICS


def matrix(con, where="TRUE", params=(), feature_names=None):
    """(X, y, info) for the rows of `pair` matching `where`. `info` keeps ids and heuristic scores
    for grouped evaluation and baselines."""
    names = feature_names or FEATURES
    cols = ", ".join(dict.fromkeys(INFO + list(names)))
    d = con.execute(f"SELECT {cols} FROM pair WHERE {where}", list(params)).fetchnumpy()
    if not len(d["y"]):
        return np.empty((0, len(names)), dtype=np.float32), np.empty(0, dtype=np.int8), {}
    X = np.column_stack([np.asarray(d[f], dtype=np.float32) for f in names]).astype(np.float32)
    y = np.asarray(d["y"], dtype=np.int8)
    info = {k: np.asarray(d[k]) for k in INFO if k != "y"}
    return X, y, info
