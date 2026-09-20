"""
Orchestration: fuse BM25 and dense rankings over the indexed (recent journal/conference) papers,
then top up with an exact-word match over every record kind and year the index does not cover -
the same guarantee the dashboard's plain paper search always gave, so upgrading to hybrid ranking
never finds fewer papers, only ranks the indexed ones better.
"""
import logging

from . import bm25 as B, fuse as F, vectors as V

log = logging.getLogger("dblp.search.search")

MAX_TITLE = 300
KIND_SQL = ("CASE WHEN is_preprint THEN 'preprint' WHEN type = 'article' THEN 'journal' "
            "WHEN type = 'inproceedings' THEN 'conference' ELSE type END")
SELECT = f"pid, key, title, year, venue, {KIND_SQL} AS kind, n_authors, n_unidentified, has_twin, has_oa"
COLUMNS = ["pid", "key", "title", "year", "venue", "kind", "n_authors", "n_unidentified", "has_twin", "has_oa"]


def _lexical(con, text, kind=None, year_from=None, year_to=None, limit=30, exclude=()):
    words = [w for w in text.strip().split() if w][:8]
    if not words or sum(len(w) for w in words) < 3 or limit <= 0:
        return []
    where = ["title ILIKE ?"] * len(words)
    params = [f"%{w}%" for w in words]
    if kind:
        where.append(f"({KIND_SQL}) = ?")
        params.append(kind)
    if year_from is not None:
        where.append("year >= ?")
        params.append(int(year_from))
    if year_to is not None:
        where.append("year <= ?")
        params.append(int(year_to))
    if exclude:
        where.append("pid NOT IN (SELECT unnest(?::INTEGER[]))")
        params.append(list(exclude))
    rows = con.execute(f"""
        SELECT {SELECT} FROM s.pubs WHERE {' AND '.join(where)}
        ORDER BY year DESC NULLS LAST, key LIMIT ?""", params + [limit]).fetchall()
    return [dict(zip(COLUMNS, r)) for r in rows]


def _hydrate(con, pids):
    if not pids:
        return {}
    rows = con.execute(f"""
        SELECT {SELECT} FROM s.pubs WHERE pid IN (SELECT unnest(?::INTEGER[]))""", [list(pids)]).fetchall()
    return {r[0]: dict(zip(COLUMNS, r)) for r in rows}


def search(con, fingerprint, encoder, text, top=20, kind=None, year_from=None, year_to=None, dense=True):
    text = (text or "").strip()[:MAX_TITLE]
    if len(text) < 3:
        return {"error": "type at least 3 characters"}

    sparse = B.search(con, text, kind=kind, year_from=year_from, year_to=year_to)
    dense_hits, dense_error = [], None
    if dense:
        try:
            qvec = encoder.encode_query(text)
            dense_hits = V.search(con, fingerprint, qvec, kind=kind, year_from=year_from, year_to=year_to)
        except FileNotFoundError as e:
            dense_error = str(e)

    fused = F.rrf(sparse, dense_hits)[:top]
    sparse_rank = {pid: r for r, (pid, _) in enumerate(sparse, start=1)}
    dense_rank = {pid: r for r, (pid, _) in enumerate(dense_hits, start=1)}
    meta = _hydrate(con, [pid for pid, _ in fused])

    results = []
    for pid, score in fused:
        m = meta.get(pid)
        if not m:
            continue
        sources = [s for s, present in (("bm25", pid in sparse_rank), ("dense", pid in dense_rank)) if present]
        results.append({**{k: v for k, v in m.items() if k != "pid"}, "score": round(score, 5),
                        "sources": sources, "bm25_rank": sparse_rank.get(pid), "dense_rank": dense_rank.get(pid)})

    remaining = max(0, top - len(results))
    extra = _lexical(con, text, kind, year_from, year_to, limit=remaining, exclude=set(meta.keys()))
    for row in extra:
        results.append({**{k: v for k, v in row.items() if k != "pid"}, "score": None,
                        "sources": ["exact_word"], "bm25_rank": None, "dense_rank": None})

    return {
        "query": text, "results": results[:top],
        "sparse_candidates": len(sparse), "dense_candidates": len(dense_hits),
        "dense_available": dense and dense_error is None, "dense_error": dense_error,
    }
