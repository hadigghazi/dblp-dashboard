"""
Suggestions for a title (and, when known, its authors): the ranked venues with the evidence behind
each - where the title's words point, where the authors have published - what a score of that size
has meant on the test year, and the papers whose titles are closest.

For an existing paper the same ranking is shown with its real venue marked. The statistics then
include that paper's own title (they are built from the whole dump), which flatters it slightly;
the payload says so.
"""
import json
import logging

import numpy as np

from . import candidates as C, evaluate as E, features as F, model as M, store as S

log = logging.getLogger("dblp.ml.venues.predict")

MAX_TOP = 30
MAX_TITLE = 500
MAX_AUTHORS = 20


def _now(con):
    meta = dict(con.execute("SELECT k, v FROM s._meta").fetchall())
    return int(meta.get("last_full_year") or 2025) + 1


def suggest(con, title, author_keys=(), top=10, model=None, features=None, calibration=None, model_dir=None):
    title = (title or "").strip()[:MAX_TITLE]
    if len(title) < 3:
        return {"error": "give a title of at least three characters"}
    if model is None:
        model, features, calibration, model_dir = M.load()
    S.load_stats(con, model_dir)
    people = _people_by_key(con, list(author_keys)[:MAX_AUTHORS])
    C.query_from_text(con, title, [p["person_id"] for p in people], year=_now(con))
    out = {"query": {"title": title, "authors": [{k: p[k] for k in ("key", "name")} for p in people]}}
    return _rank(con, out, top, model, features, calibration, model_dir)


def for_paper(con, key, top=10, model=None, features=None, calibration=None, model_dir=None):
    if model is None:
        model, features, calibration, model_dir = M.load()
    row = con.execute("""
        SELECT p.pid, p.sid, p.year, b.title FROM v.paper p JOIN s.pubs b USING (pid) WHERE p.key = ?""", [key]).fetchone()
    if not row:
        return {"error": f"{key} is not a journal or conference paper the store knows"}
    pid, sid, year, title = row
    S.load_stats(con, model_dir)
    authors = [r[0] for r in con.execute("SELECT person_id FROM v.paper_author WHERE pid = ?", [pid]).fetchall()]
    # history strictly before the paper's year, as in the evaluation; the title statistics are "now"
    C.query_from_text(con, title, authors, year=year, qid=int(pid), true_sid=sid)
    out = {"query": {"key": key, "title": title, "year": year, "authors_known": len(authors)},
           "actual": {"sid": sid, "name": _series_name(con, sid)},
           "note": "title statistics include this paper itself; author history is limited to earlier years"}
    out = _rank(con, out, top, model, features, calibration, model_dir)
    ranks = [s["sid"] for s in out.get("suggestions", [])]
    out["actual"]["rank"] = ranks.index(sid) + 1 if sid in ranks else None
    out["actual"]["in_class_set"] = bool(con.execute("SELECT count(*) FROM cls WHERE sid = ?", [sid]).fetchone()[0])
    return out


def _rank(con, out, top, model, features, calibration, model_dir):
    top = max(1, min(int(top), MAX_TOP))
    n = C.build_pairs(con)
    n_used = con.execute("SELECT coalesce(max(n_used), 0) FROM q_norm").fetchone()[0] if n else 0
    out["tokens_used"] = [r[0] for r in con.execute("SELECT token FROM q_used ORDER BY idf DESC").fetchall()] if n else []
    out["candidates"] = int(n)
    out["suggestions"] = []
    if n == 0:
        out["note"] = "no token of this title is in the vocabulary, and no author has a history"
        out["related"] = []
        return _with_model(out, model_dir)
    X, _, info = F.matrix(con, feature_names=features)
    p = model.predict_proba(X)[:, 1]
    order = np.argsort(-p)[:top]
    names = _series_names(con, [str(info["sid"][i]) for i in order])
    hist = {r[0]: {"papers": r[1], "authors": r[2], "last_year": r[3]} for r in
            con.execute("SELECT sid, hist_papers, hist_authors, hist_last FROM hist").fetchall()}
    cont = {r[0]: {"nb_rank": r[1], "cen_rank": r[2], "shared_tokens": r[3]} for r in
            con.execute("SELECT sid, nb_rank, cen_rank, n_shared FROM content").fetchall()}
    for rank, i in enumerate(order, start=1):
        sid = str(info["sid"][i])
        out["suggestions"].append({
            "rank": rank, "sid": sid, "name": names.get(sid, {}).get("name", sid),
            "kind": names.get(sid, {}).get("kind"), "score": round(float(p[i]), 3),
            "came_true": E.came_true_rate(calibration or [], float(p[i])),
            "content": cont.get(sid), "history": hist.get(sid),
            "recent_papers": names.get(sid, {}).get("recent"),
        })
    out["related"] = C.related_papers(con)
    return _with_model(out, model_dir)


def _with_model(out, model_dir):
    if model_dir is not None and (model_dir / M.METRICS_FILE).exists():
        saved = json.loads((model_dir / M.METRICS_FILE).read_text(encoding="utf-8"))
        test = saved.get("metrics", {}).get("test", {})
        out["model"] = {"trained_at": saved.get("trained_at"), "dump": saved.get("dump"),
                        "test": {k: test.get(k) for k in ("year", "papers", "covered", "in_candidates", "rankers")}}
    return out


def _people_by_key(con, keys):
    if not keys:
        return []
    rows = con.execute("""
        SELECT person_id, key, name FROM s.persons
        WHERE key IN (SELECT unnest(?::VARCHAR[])) AND page_kind <> 'disambiguation'""", [keys]).fetchall()
    return [dict(zip(["person_id", "key", "name"], r)) for r in rows]


def _series_names(con, sids):
    rows = con.execute("""
        SELECT ss.sid, ss.name, ss.kind, ss.recent FROM stats_series ss
        WHERE ss.sid IN (SELECT unnest(?::VARCHAR[]))""", [sids]).fetchall()
    return {r[0]: {"name": r[1], "kind": r[2], "recent": r[3]} for r in rows}


def _series_name(con, sid):
    r = con.execute("SELECT name FROM v.series WHERE sid = ?", [sid]).fetchone()
    return r[0] if r else sid
