"""
From queries to candidate (query, series) pairs with features.

A query is a title and, when known, its authors and year. Three scorers propose candidates and are
also the baselines: Naive Bayes and the TF-IDF centroid read the title, the authors' history reads
who has published where before. The union of their tops is the candidate set the ranker orders.

The query tables are `q` (qid, year, true_sid, n_authors), `q_token` (qid, token) and `q_author`
(qid, person_id); `queries_from_papers` fills them from the store, `query_from_text` from a title.
"""
import logging
import re
import time

from . import config, store as S

log = logging.getLogger("dblp.ml.venues.candidates")


def queries_from_papers(con, year, limit, seed=None):
    """A deterministic sample of the store's papers of one year, as queries."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE q AS
        SELECT p.pid AS qid, p.year, p.sid AS true_sid,
               (SELECT count(*) FROM v.paper_author pa WHERE pa.pid = p.pid)::INTEGER AS n_authors
        FROM v.paper p
        WHERE p.year = ?
        QUALIFY row_number() OVER (ORDER BY hash(p.pid::BIGINT * 7919 + ?)) <= ?
    """, [int(year), int(seed if seed is not None else config.SEED), int(limit)])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE q_token AS
        SELECT q.qid, t.token FROM q JOIN v.title_token t ON t.pid = q.qid""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE q_author AS
        SELECT q.qid, pa.person_id FROM q JOIN v.paper_author pa ON pa.pid = q.qid""")
    return con.execute("SELECT count(*) FROM q").fetchone()[0]


def query_from_text(con, title, person_ids=(), year=9999, qid=0, true_sid=None):
    con.execute("CREATE OR REPLACE TEMP TABLE q (qid INTEGER, year INTEGER, true_sid VARCHAR, n_authors INTEGER)")
    con.execute("INSERT INTO q VALUES (?, ?, ?, ?)", [qid, int(year), true_sid, len(person_ids)])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE q_token AS
        WITH ws AS (SELECT {S.words_expr('?')} AS w)
        SELECT ?::INTEGER AS qid, unnest({S.TOKENS_OF_WORDS}) AS token FROM ws""", [title, qid])
    con.execute("CREATE OR REPLACE TEMP TABLE q_author (qid INTEGER, person_id INTEGER)")
    if person_ids:
        con.executemany("INSERT INTO q_author VALUES (?, ?)", [(qid, int(p)) for p in person_ids])


CANDIDATE_SQL = [
    # the query's rarest tokens that the statistics know: the informative ones, and a bound on the join
    ("q_used", """
        CREATE OR REPLACE TEMP TABLE q_used AS
        SELECT qt.qid, qt.token, vc.idf
        FROM q_token qt JOIN vocab vc USING (token)
        QUALIFY row_number() OVER (PARTITION BY qt.qid ORDER BY vc.df, qt.token) <= $max_tokens"""),
    ("q_norm", """
        CREATE OR REPLACE TEMP TABLE q_norm AS
        SELECT qid, count(*)::INTEGER AS n_used, sqrt(sum(idf * idf)) AS nrm FROM q_used GROUP BY 1"""),
    # both content scores in one join over the present tokens
    ("content", """
        CREATE OR REPLACE TEMP TABLE content AS
        WITH present AS (
            SELECT qu.qid, st.sid, sum(st.nb_w) AS nb_present, count(*)::INTEGER AS n_shared,
                   sum(qu.idf * st.cen_w) AS dot
            FROM q_used qu JOIN stats st ON st.token = qu.token
            GROUP BY 1, 2),
        scored AS (
            SELECT p.qid, p.sid, p.n_shared,
                   ss.log_prior + qn.n_used * ss.c_absent + p.nb_present AS nb,
                   p.dot / qn.nrm AS cen
            FROM present p JOIN stats_series ss USING (sid) JOIN q_norm qn USING (qid))
        SELECT *, row_number() OVER (PARTITION BY qid ORDER BY nb DESC) AS nb_rank,
                  row_number() OVER (PARTITION BY qid ORDER BY cen DESC) AS cen_rank,
                  max(nb) OVER (PARTITION BY qid) AS nb_best,
                  max(cen) OVER (PARTITION BY qid) AS cen_best
        FROM scored
        QUALIFY nb_rank <= $top OR cen_rank <= $top"""),
    # the authors' history: series they published in before the query's year
    ("hist", """
        CREATE OR REPLACE TEMP TABLE hist AS
        SELECT qa.qid, av.sid, sum(av.papers)::INTEGER AS hist_papers,
               count(DISTINCT qa.person_id)::INTEGER AS hist_authors, max(av.year) AS hist_last
        FROM q_author qa
        JOIN q ON q.qid = qa.qid
        JOIN v.author_venue av ON av.person_id = qa.person_id AND av.year < q.year
        WHERE av.sid IN (SELECT sid FROM cls)
        GROUP BY 1, 2
        QUALIFY row_number() OVER (PARTITION BY qa.qid ORDER BY hist_papers DESC, hist_last DESC) <= $max_hist"""),
    ("pair", """
        CREATE OR REPLACE TEMP TABLE pair AS
        WITH cand AS (
            SELECT qid, sid FROM content UNION SELECT qid, sid FROM hist)
        SELECT c.qid, c.sid, (c.sid = q.true_sid)::INT AS y,
               coalesce(ct.nb - ct.nb_best, -50.0) AS nb_gap,
               coalesce(ct.nb_rank, 99)::INTEGER AS nb_rank,
               coalesce(ct.cen, 0.0) AS cen,
               coalesce(ct.cen - ct.cen_best, -1.0) AS cen_gap,
               coalesce(ct.cen_rank, 99)::INTEGER AS cen_rank,
               coalesce(ct.n_shared, 0)::INTEGER AS n_shared,
               coalesce(h.hist_papers, 0)::INTEGER AS hist_papers,
               coalesce(h.hist_authors, 0)::INTEGER AS hist_authors,
               CASE WHEN q.n_authors > 0 THEN coalesce(h.hist_authors, 0)::DOUBLE / q.n_authors ELSE 0.0 END AS hist_share,
               coalesce(q.year - h.hist_last, 50)::INTEGER AS hist_age,
               (h.qid IS NOT NULL)::INT AS from_history,
               (ct.qid IS NOT NULL)::INT AS from_content,
               ss.log_prior, ln(ss.recent + 1) AS log_recent, (q.year - ss.last_year)::INTEGER AS series_idle,
               (ss.kind = 'journal')::INT AS is_journal,
               q.n_authors, coalesce(qn.n_used, 0)::INTEGER AS n_used,
               coalesce(ct.nb, -1e9) AS nb
        FROM cand c
        JOIN q ON q.qid = c.qid
        JOIN stats_series ss ON ss.sid = c.sid
        LEFT JOIN content ct ON ct.qid = c.qid AND ct.sid = c.sid
        LEFT JOIN hist h ON h.qid = c.qid AND h.sid = c.sid
        LEFT JOIN q_norm qn ON qn.qid = c.qid"""),
]


def _run(con, sql, params):
    used = {k: v for k, v in params.items() if re.search(rf"\${k}\b", sql)}
    con.execute(sql, used)


def build_pairs(con):
    """From q / q_token / q_author and the statistics in scope, build `pair`."""
    params = {"max_tokens": config.MAX_QUERY_TOKENS, "top": config.TOP_CONTENT, "max_hist": config.MAX_HISTORY}
    t0 = time.time()
    for _, sql in CANDIDATE_SQL:
        _run(con, sql, params)
    n, pos, queries = con.execute("SELECT count(*), coalesce(sum(y), 0), count(DISTINCT qid) FROM pair").fetchone()
    log.info("candidates: %s pairs for %s queries, %s hold the true series, %.1fs",
             f"{n:,}", f"{queries:,}", f"{int(pos):,}", time.time() - t0)
    return int(n)


def related_papers(con, limit=10):
    """The papers whose titles share the most (rarest) tokens with the query: what a search returns."""
    return [dict(zip(["key", "title", "year", "sid", "venue", "score"], r)) for r in con.execute("""
        WITH qt AS (
            SELECT qu.token, qu.idf FROM q_used qu JOIN v.token_df d USING (token) WHERE d.df <= ?),
        hits AS (
            SELECT t.pid, sum(qt.idf) AS score
            FROM qt JOIN v.title_token t ON t.token = qt.token
            GROUP BY t.pid ORDER BY score DESC LIMIT ?)
        SELECT p.key, b.title, p.year, p.sid, sr.name, round(h.score, 2)
        FROM hits h
        JOIN v.paper p ON p.pid = h.pid
        JOIN s.pubs b ON b.pid = h.pid
        LEFT JOIN v.series sr ON sr.sid = p.sid
        WHERE p.pid <> coalesce((SELECT qid FROM q LIMIT 1), -1)
        ORDER BY h.score DESC""", [config.KNN_MAX_DF, limit + 1]).fetchall()][:limit]
