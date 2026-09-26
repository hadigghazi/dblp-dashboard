"""
Sparse (lexical) ranking: standard Okapi BM25 over the title-token index, restricted to titles
that share at least one query token - the inverted-index join, not a scan of every paper.

Term frequency within a title is treated as binary presence rather than counted: titles are short
(a handful of words), a repeated word is rare, and the length-normalisation term (`b`) already
accounts for title length. This is a common, documented simplification for short-text BM25.
"""
import logging

from . import config, store as S

log = logging.getLogger("dblp.search.bm25")


def search(con, text, top=None, kind=None, year_from=None, year_to=None, stats=None):
    """[(pid, score)], best first, over papers sharing at least one query token.

    `stats`, if given, is filled with how specific the query was - the caller uses it to decide
    how much this ranking is worth next to a semantic one."""
    tokens = S.query_tokens(con, text)
    if stats is not None:
        stats.update(S.query_stats(con))
    if not tokens:
        if stats is not None:
            stats["coverage"] = 0.0
        return []
    where = ["TRUE"]
    params = {"k1": config.BM25_K1, "b": config.BM25_B, "top": top or config.TOP_SPARSE}
    if kind:
        where.append("p.kind = $kind")
        params["kind"] = kind
    if year_from is not None:
        where.append("p.year >= $year_from")
        params["year_from"] = int(year_from)
    if year_to is not None:
        where.append("p.year <= $year_to")
        params["year_to"] = int(year_to)
    rows = con.execute(f"""
        WITH n AS (SELECT count(*)::DOUBLE AS n, avg(len) AS avgdl FROM x.paper p2 JOIN x.doc_len dl USING (pid)),
        hit AS (
            SELECT t.pid,
                   sum(ln(1 + (n.n - d.df + 0.5) / (d.df + 0.5))
                       * ($k1 + 1) / (1 + $k1 * (1 - $b + $b * dl.len / n.avgdl))) AS score
            FROM x.title_token t
            JOIN q_used qu ON qu.token = t.token
            JOIN x.token_df d ON d.token = t.token
            JOIN x.doc_len dl ON dl.pid = t.pid, n
            GROUP BY t.pid)
        SELECT h.pid, h.score FROM hit h JOIN x.paper p ON p.pid = h.pid
        WHERE {' AND '.join(where)}
        -- dblp holds many identical titles (a preprint and its published twin, reissued papers),
        -- which score exactly alike; without a tiebreaker which one lands at rank 1 varies between
        -- runs, and an evaluation asking for one specific record then moves by a point for no reason
        ORDER BY h.score DESC, h.pid LIMIT $top""", params).fetchall()
    out = [(int(pid), float(score)) for pid, score in rows]
    if stats is not None:
        # how much of the query the best word match explains: the signal the fusion weights by
        stats["coverage"] = S.match_coverage(con, out[0][0]) if out else 0.0
    return out
