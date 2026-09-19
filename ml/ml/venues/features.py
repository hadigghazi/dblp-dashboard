"""Column order for the ranker and matrix assembly; the scores are computed in SQL (candidates.py)."""
import numpy as np

FEATURES = [
    "nb_gap", "nb_rank", "cen", "cen_gap", "cen_rank", "n_shared",
    "hist_papers", "hist_authors", "hist_share", "hist_age", "from_history", "from_content",
    "log_prior", "log_recent", "series_idle", "is_journal",
    "n_authors", "n_used",
]

# what the baselines rank by, kept next to the ids
INFO = ["qid", "sid", "y", "nb", "cen", "hist_papers", "hist_age", "log_recent", "log_prior"]


def matrix(con, where="TRUE", params=(), feature_names=None):
    names = feature_names or FEATURES
    cols = ", ".join(dict.fromkeys(INFO + list(names)))
    d = con.execute(f"SELECT {cols} FROM pair WHERE {where}", list(params)).fetchnumpy()
    if not len(d["y"]):
        return np.empty((0, len(names)), dtype=np.float32), np.empty(0, dtype=np.int8), {}
    X = np.column_stack([np.asarray(d[f], dtype=np.float32) for f in names]).astype(np.float32)
    y = np.asarray(d["y"], dtype=np.int8)
    info = {k: np.asarray(d[k]) for k in INFO if k != "y"}
    return X, y, info
