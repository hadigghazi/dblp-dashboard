"""
Pair features. Every feature is computed in DuckDB from the serving tables, then assembled into a
matrix here. The question each pair asks is: were these two papers written by the same person, given
that they carry the same name?

The features follow the author-name-disambiguation literature (Han et al.; Louppe et al.): shared
co-authors dominate, with venue, time, title and name-form as support. Nothing here uses a label.
"""
import logging

import numpy as np

from . import config

log = logging.getLogger("dblp.ml.features")

# Raw columns SQL produces per pair. Order is irrelevant here; FEATURES below fixes the model's order.
PAIR_SQL = """
CREATE OR REPLACE TEMP TABLE pair AS
WITH j AS (
    SELECT a.base_name,
           a.pid AS pid_a, b.pid AS pid_b,
           a.person_id AS person_a, b.person_id AS person_b,
           (a.person_id = b.person_id)::INT AS y,
           (hash(a.base_name) % 10)::INT AS bucket,
           abs(coalesce(a.year, 0) - coalesce(b.year, 0)) AS year_gap,
           (a.year IS NULL OR b.year IS NULL)::INT AS year_missing,
           (a.sid = b.sid)::INT AS same_sid,
           (a.venue IS NOT DISTINCT FROM b.venue)::INT AS same_venue,
           (a.key_prefix = b.key_prefix)::INT AS same_prefix,
           (a.is_preprint OR b.is_preprint)::INT AS any_preprint,
           abs(a.n_authors - b.n_authors) AS team_diff,
           least(a.n_authors, b.n_authors) AS team_min,
           (a.used_name = b.used_name)::INT AS same_name_form,
           len(list_intersect(a.other_ids, b.other_ids)) AS shared_ids,
           len(a.other_ids) AS n_ids_a, len(b.other_ids) AS n_ids_b,
           len(list_intersect(a.other_names, b.other_names)) AS shared_names,
           len(a.other_names) AS n_names_a, len(b.other_names) AS n_names_b,
           len(list_intersect(a.toks, b.toks)) AS shared_toks,
           len(a.toks) AS n_toks_a, len(b.toks) AS n_toks_b,
           CASE WHEN a.orcid IS NOT NULL AND b.orcid IS NOT NULL THEN (a.orcid = b.orcid)::INT
                ELSE -1 END AS orcid_match,
           (a.position = 1 AND b.position = 1)::INT AS both_first,
           (a.position = a.n_authors AND b.position = b.n_authors)::INT AS both_last
    FROM inst a
    JOIN inst b ON a.base_name = b.base_name AND a.pid < b.pid
    {restrict}
)
SELECT * FROM j
{sampling}
"""

# Sampling keeps the pair count bounded per block and per label. Positives are rarer than negatives
# inside a block (one person's papers vs everyone else's), so they get their own quota.
SAMPLING = """
QUALIFY row_number() OVER (
    PARTITION BY base_name, y
    ORDER BY hash(CAST(pid_a AS BIGINT) * 1000003 + pid_b)
) <= CASE WHEN y = 1 THEN {pos} ELSE {neg} END
"""

FEATURES = [
    "shared_ids", "ids_jaccard", "has_shared_id",
    "shared_names", "names_jaccard", "has_shared_name",
    "same_sid", "same_venue", "same_prefix",
    "year_gap", "year_missing",
    "shared_toks", "toks_jaccard",
    "orcid_match",
    "same_name_form",
    "team_diff", "team_min", "any_preprint",
    "both_first", "both_last",
    "n_ids_min", "n_names_min",
]


def build_pairs(con, sampled=True, pos=None, neg=None, touching=None):
    """`touching`: person ids; keep only pairs where at least one paper belongs to one of them.
    A bin prediction needs bin-bin and bin-known pairs, never known-known."""
    sampling = SAMPLING.format(pos=pos or config.POS_PAIRS_PER_BLOCK, neg=neg or config.NEG_PAIRS_PER_BLOCK) \
        if sampled else ""
    restrict = ""
    if touching:
        ids = ", ".join(str(int(i)) for i in touching)
        restrict = f"WHERE a.person_id IN ({ids}) OR b.person_id IN ({ids})"
    con.execute(PAIR_SQL.format(sampling=sampling, restrict=restrict))
    n, pos_n, blocks = con.execute(
        "SELECT count(*), coalesce(sum(y), 0), count(DISTINCT base_name) FROM pair").fetchone()
    log.info("pairs: %s (%s positive, %.1f%%) across %s blocks",
             f"{n:,}", f"{pos_n:,}", 100 * pos_n / n if n else 0, f"{blocks:,}")
    return n


def _jaccard(shared, a, b):
    denom = a + b - shared
    out = np.zeros_like(shared, dtype=np.float32)
    np.divide(shared, denom, out=out, where=denom > 0)
    return out


def matrix(con, where="TRUE", params=(), feature_names=None):
    """
    Fetch pairs and assemble (X, y, info). `info` keeps the identifiers for grouped evaluation.
    `feature_names` selects and orders the columns, so a model trained without a feature (e.g. the
    ORCID ablation) is scored with exactly the columns it was trained on.
    """
    d = con.execute(f"SELECT * FROM pair WHERE {where}", list(params)).fetchnumpy()
    if not len(d["y"]):
        width = len(feature_names or FEATURES)
        return np.empty((0, width), dtype=np.float32), np.empty(0, dtype=np.int8), {}
    col = {k: np.asarray(v, dtype=np.float32) for k, v in d.items() if k not in ("base_name",)}
    derived = {
        "ids_jaccard": _jaccard(col["shared_ids"], col["n_ids_a"], col["n_ids_b"]),
        "names_jaccard": _jaccard(col["shared_names"], col["n_names_a"], col["n_names_b"]),
        "toks_jaccard": _jaccard(col["shared_toks"], col["n_toks_a"], col["n_toks_b"]),
        "has_shared_id": (col["shared_ids"] > 0).astype(np.float32),
        "has_shared_name": (col["shared_names"] > 0).astype(np.float32),
        "n_ids_min": np.minimum(col["n_ids_a"], col["n_ids_b"]),
        "n_names_min": np.minimum(col["n_names_a"], col["n_names_b"]),
    }
    col.update(derived)
    names = feature_names or FEATURES
    X = np.column_stack([col[f] for f in names]).astype(np.float32)
    y = np.asarray(d["y"], dtype=np.int8)
    info = {
        "base_name": np.asarray(d["base_name"], dtype=object),
        "pid_a": np.asarray(d["pid_a"]), "pid_b": np.asarray(d["pid_b"]),
        "person_a": np.asarray(d["person_a"]), "person_b": np.asarray(d["person_b"]),
        "bucket": np.asarray(d["bucket"]),
        "shared_ids": np.asarray(d["shared_ids"]),
    }
    return X, y, info
