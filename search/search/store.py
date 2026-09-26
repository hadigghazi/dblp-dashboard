"""
The sparse index: eligible papers, their title tokens, and BM25's per-corpus statistics.

Self-contained by design - it does not attach the venue-recommendation store, even though the two
use the same eligibility predicate (journal/conference, not a preprint, recent enough). Each service
builds its own store from `s` so neither can break the other, and search is useful even if the venue
model was never trained. Older papers and other record kinds (preprints, theses, books, www pages)
are not indexed here; the search endpoint falls back to an exact-word match over the full `s.pubs`
for those, same as the dashboard's plain paper search always has.
"""
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger("dblp.search.store")

STOP = ("the a an of in on to by is as at or and for with from into over under via using based toward towards "
        "its their this that these those are can not than versus vs our your who how what when why which where "
        "we it be do does new").split()


def words_expr(col):
    stop = ", ".join(f"'{w}'" for w in STOP)
    return (f"list_filter(regexp_split_to_array(lower({col}), '[^a-z0-9]+'), "
            f"x -> len(x) >= 2 AND x NOT IN ({stop}))")


TOKENS_OF_WORDS = ("list_distinct(list_concat(w, list_transform(range(1, len(w)), i -> w[i] || '_' || w[i + 1])))")

STORE_STEPS = [
    ("papers", """
        CREATE TABLE {x}.paper AS
        SELECT (row_number() OVER (ORDER BY p.pid) - 1)::INTEGER AS row, p.pid, p.key, p.title,
               p.year::INTEGER AS year, p.venue, p.sid,
               CASE WHEN p.key_prefix = 'journals' THEN 'journal' ELSE 'conference' END AS kind
        FROM s.pubs p
        WHERE p.key_prefix IN ('conf', 'journals') AND NOT p.is_preprint
          AND p.sid IS NOT NULL AND p.title IS NOT NULL AND length(p.title) >= 8
          AND p.year >= {first_year}
        ORDER BY pid"""),
    ("title tokens", """
        CREATE TABLE {x}.title_token AS
        WITH w AS (SELECT pid, {words} AS w FROM {x}.paper)
        SELECT token, pid FROM (SELECT pid, unnest({tokens}) AS token FROM w)
        ORDER BY token, pid"""),
    ("token frequencies", """
        CREATE TABLE {x}.token_df AS
        SELECT token, count(*)::INTEGER AS df FROM {x}.title_token GROUP BY 1 ORDER BY 1"""),
    ("title lengths", """
        CREATE TABLE {x}.doc_len AS
        SELECT p.pid, coalesce(t.n, 0)::INTEGER AS len
        FROM {x}.paper p LEFT JOIN (SELECT pid, count(*) AS n FROM {x}.title_token GROUP BY 1) t USING (pid)"""),
]


def store_path(fingerprint) -> Path:
    return config.MODELS_DIR / f"search-store-{fingerprint}.duckdb"


def query_stats(con):
    """How specific the last query_tokens() call was: the rarest token it kept, as an IDF over
    the indexed papers. High means the query named something; low means it used only words that
    hundreds of thousands of papers share."""
    row = con.execute("SELECT count(*), min(df), max(df) FROM q_used").fetchone()
    n = con.execute("SELECT count(*) FROM x.paper").fetchone()[0] or 1
    if not row or not row[0] or not row[1]:
        return {"tokens": 0, "min_df": None, "max_df": None, "idf": 0.0}
    return {"tokens": int(row[0]), "min_df": int(row[1]), "max_df": int(row[2]),
            "idf": math.log(n / row[1])}


def sparse_weight(stats):
    """How much the word ranking is worth for this query, between SPARSE_FLOOR and 1."""
    idf = (stats or {}).get("idf") or 0.0
    if idf >= config.IDF_FULL:
        return 1.0
    if idf <= config.IDF_FLOOR:
        return config.SPARSE_FLOOR
    span = (idf - config.IDF_FLOOR) / (config.IDF_FULL - config.IDF_FLOOR)
    return round(config.SPARSE_FLOOR + span * (1.0 - config.SPARSE_FLOOR), 3)


def vectors_path(fingerprint) -> Path:
    return config.MODELS_DIR / f"search-vectors-{fingerprint}.{config.VECTOR_DTYPE}"


def eval_path(fingerprint) -> Path:
    d = config.MODELS_DIR / "search-eval"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{fingerprint}.json"


def build_store(con, meta) -> Path:
    fp = meta.get("fingerprint", "unknown")
    target = store_path(fp)
    building = target.with_suffix(".duckdb.building")
    building.unlink(missing_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    con.execute(f"ATTACH '{building}' AS xb")
    try:
        for name, sql in STORE_STEPS:
            t = time.time()
            con.execute(sql.format(x="xb", first_year=config.FIRST_YEAR, words=words_expr("title"),
                                   tokens=TOKENS_OF_WORDS))
            log.info("search store: built %s in %.1fs", name, time.time() - t)
        papers, tokens, avgdl = con.execute(
            "SELECT (SELECT count(*) FROM xb.paper), (SELECT count(*) FROM xb.token_df), (SELECT avg(len) FROM xb.doc_len)"
        ).fetchone()
        con.execute("CREATE TABLE xb._meta (k VARCHAR, v VARCHAR)")
        con.executemany("INSERT INTO xb._meta VALUES (?, ?)", [
            ("fingerprint", fp), ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("papers", str(papers)), ("tokens", str(tokens)), ("avgdl", str(avgdl)),
            ("first_year", str(config.FIRST_YEAR)), ("embed_dim", str(config.EMBED_DIM)),
            ("model", config.MODEL_NAME),
        ])
    finally:
        con.execute("DETACH xb")
    building.replace(target)
    log.info("search store %s ready in %.0fs (%s papers, %s tokens)", target.name, time.time() - t_all,
             f"{papers:,}", f"{tokens:,}")
    return target


def attach_store(con, meta, build_if_missing=False):
    fp = meta.get("fingerprint", "unknown")
    path = store_path(fp)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"no search store for dump {fp} in {config.MODELS_DIR}; run `store` first")
        build_store(con, meta)
    con.execute(f"ATTACH '{path}' AS x (READ_ONLY)")
    xmeta = dict(con.execute("SELECT k, v FROM x._meta").fetchall())
    log.info("attached %s (%s papers)", path.name, xmeta.get("papers"))
    return xmeta


def _run(con, sql, params):
    used = {k: v for k, v in params.items() if re.search(rf"\${k}\b", sql)}
    con.execute(sql, used)


def query_tokens(con, text, max_tokens=None, max_postings=None):
    """The query's rarest known tokens, for BM25 - same tokenizer as the index.

    Two limits, not one. The token count bounds the join's width; the cumulative document frequency
    bounds its depth, which is what a query made entirely of common words ("systems for learning
    from data") otherwise blows through: its rarest sixteen tokens can still be a million postings
    each. The rarest tokens are always kept, so a query is never left with nothing to match on."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE q_token AS
        WITH w AS (SELECT {words_expr('?')} AS w)
        SELECT unnest({TOKENS_OF_WORDS}) AS token FROM w""", [text])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE q_used AS
        SELECT token, df FROM (
            SELECT qt.token, d.df,
                   row_number() OVER (ORDER BY d.df) AS rank,
                   sum(d.df) OVER (ORDER BY d.df, qt.token ROWS UNBOUNDED PRECEDING) AS postings
            FROM q_token qt JOIN x.token_df d USING (token))
        WHERE rank <= ? AND (postings <= ? OR rank <= ?)""",
                [max_tokens or config.MAX_QUERY_TOKENS, max_postings or config.MAX_POSTINGS,
                 config.MIN_QUERY_TOKENS])
    return [r[0] for r in con.execute("SELECT token FROM q_used ORDER BY df").fetchall()]
