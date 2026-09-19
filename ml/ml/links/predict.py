"""
Suggestions for one author: the distance-2 candidates today, scored by the model, with the
evidence a reader can check - the shared co-authors, the shared venues - and what a score of that
size has meant historically (the calibration measured on the test snapshot).
"""
import json
import logging

import numpy as np

from .. import data
from . import evaluate as E, features as F, graph as G, model as M

log = logging.getLogger("dblp.ml.links.predict")

MAX_TOP = 50
EXPLAIN_NEIGHBOURS = 5


def current_year(con):
    """The snapshot "now": the year the dump was taken. Features are differences from the snapshot
    year (career age, years idle, bridge age), so it has to be a real year, not "everything"."""
    meta = dict(con.execute("SELECT k, v FROM s._meta").fetchall())
    return int(meta.get("last_full_year") or 2025) + 1


def suggest(con, key, top=10, model=None, features=None, calibration=None, model_dir=None):
    person = data.person_by_key(con, key)
    if person is None:
        return {"error": f"no author page {key}"}
    if person["page_kind"] == "disambiguation":
        return {"error": f"{key} is a disambiguation bin, not a person: it has no co-authors of its own"}
    if model is None:
        model, features, calibration, model_dir = M.load()
    top = max(1, min(int(top), MAX_TOP))

    now = current_year(con)
    G.set_anchor(con, person["person_id"])
    n = G.build_pairs(con, now)
    degree = con.execute("SELECT count(*) FROM n1").fetchone()[0]
    out = {"author": {"key": person["key"], "name": person["name"], "co_authors": int(degree)},
           "snapshot": now, "candidates": int(n), "suggestions": []}
    if n == 0:
        out["note"] = ("no co-authors on record" if degree == 0 else
                       "every co-author of a co-author is already a co-author")
        return _with_model(out, model_dir)

    X, _, info = F.matrix(con, feature_names=features)
    p = model.predict_proba(X)[:, 1]
    order = np.argsort(-p)[:top]
    chosen = [int(info["v"][i]) for i in order]
    who = _people(con, chosen)
    bridges = _bridges(con, chosen)
    venues = _venues(con, person["person_id"], chosen)
    for rank, i in enumerate(order, start=1):
        v = int(info["v"][i])
        out["suggestions"].append({
            "rank": rank, "key": who[v]["key"], "name": who[v]["name"],
            "score": round(float(p[i]), 3),
            "came_true": E.came_true_rate(calibration or [], float(p[i])),
            "common_coauthors": int(info["cn"][i]),
            "via": bridges.get(v, []),
            "shared_venues": venues.get(v, []),
            "papers": who[v]["papers"], "last_year": who[v]["last_year"],
            "heuristics": {h: round(float(info[h][i]), 3) for h in F.HEURISTICS},
        })
    return _with_model(out, model_dir)


def _with_model(out, model_dir):
    if model_dir is not None and (model_dir / M.METRICS_FILE).exists():
        saved = json.loads((model_dir / M.METRICS_FILE).read_text(encoding="utf-8"))
        out["model"] = {"trained_at": saved.get("trained_at"), "dump": saved.get("dump"),
                        "test": {k: saved.get("metrics", {}).get("test", {}).get(k)
                                 for k in ("snapshot", "horizon", "pooled", "ranking")}}
    return out


def _people(con, ids):
    rows = con.execute("""
        SELECT p.person_id, p.key, p.name, n.papers, n.last_year
        FROM s.persons p
        LEFT JOIN (SELECT person_id, sum(papers)::INTEGER AS papers, max(year) AS last_year
                   FROM g.node_year GROUP BY 1) n ON n.person_id = p.person_id
        WHERE p.person_id IN (SELECT unnest(?::INTEGER[]))""", [ids]).fetchall()
    return {r[0]: {"key": r[1], "name": r[2], "papers": r[3], "last_year": r[4]} for r in rows}


def _bridges(con, ids):
    """The shared co-authors behind each suggestion, most recently active first."""
    rows = con.execute("""
        SELECT n2.v, p.key, p.name, least(n1.last, n2.last) AS last
        FROM n1 JOIN n2 ON n2.w = n1.w JOIN s.persons p ON p.person_id = n1.w
        WHERE n2.v IN (SELECT unnest(?::INTEGER[]))
        QUALIFY row_number() OVER (PARTITION BY n2.v ORDER BY least(n1.last, n2.last) DESC, n1.papers + n2.papers DESC) <= ?
        ORDER BY n2.v, last DESC""", [ids, EXPLAIN_NEIGHBOURS]).fetchall()
    out = {}
    for v, key, name, last in rows:
        out.setdefault(v, []).append({"key": key, "name": name, "last_year": last})
    return out


def _venues(con, anchor_id, ids):
    rows = con.execute("""
        WITH mine AS (SELECT sid FROM g.node_venue WHERE person_id = ?),
        theirs AS (SELECT person_id, sid FROM g.node_venue WHERE person_id IN (SELECT unnest(?::INTEGER[]))),
        shared AS (SELECT t.person_id, t.sid FROM theirs t JOIN mine USING (sid)),
        named AS (SELECT sid, mode(venue) AS name FROM s.pubs WHERE sid IN (SELECT sid FROM shared) GROUP BY sid)
        SELECT sh.person_id, sh.sid, nm.name FROM shared sh LEFT JOIN named nm USING (sid) ORDER BY 1, 2""",
        [int(anchor_id), ids]).fetchall()
    out = {}
    for v, sid, name in rows:
        out.setdefault(v, []).append({"sid": sid, "name": name or sid})
    return out
