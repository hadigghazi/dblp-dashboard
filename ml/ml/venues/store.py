"""
The venue store and the statistics a snapshot is scored with.

The store is built once per dump next to the models: the eligible papers (journal and conference
papers with a series id), an inverted index of their title tokens, and each author's publishing
history per series and year. Training snapshots and live suggestions read the same file.

Statistics "as of T" are derived from it for any year T: the class set (series with enough papers
and recent activity), and per (token, series) the two content scores' weights - a multinomial
Naive Bayes weight and a TF-IDF centroid weight. Nothing after T enters them.
"""
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import config as base
from . import config

log = logging.getLogger("dblp.ml.venues.store")

STOP = ("the a an of in on to by is as at or and for with from into over under via using based toward towards "
        "its their this that these those are can not than versus vs our your who how what when why which where "
        "we it be do does new").split()


def words_expr(col):
    """SQL: the title's lower-cased words, without stop words and one-letter tokens."""
    stop = ", ".join(f"'{w}'" for w in STOP)
    return (f"list_filter(regexp_split_to_array(lower({col}), '[^a-z0-9]+'), "
            f"x -> len(x) >= 2 AND x NOT IN ({stop}))")


TOKENS_OF_WORDS = ("list_distinct(list_concat(w, list_transform(range(1, len(w)), i -> w[i] || '_' || w[i + 1])))")

STORE_STEPS = [
    ("papers", """
        CREATE TABLE {v}.paper AS
        SELECT p.pid, p.key, p.sid, p.year::INTEGER AS year,
               CASE WHEN p.key_prefix = 'journals' THEN 'journal' ELSE 'conference' END AS kind,
               p.n_authors::INTEGER AS n_authors
        FROM s.pubs p
        WHERE p.key_prefix IN ('conf', 'journals') AND NOT p.is_preprint
          AND p.sid IS NOT NULL AND p.title IS NOT NULL AND p.year >= {first_year}
        ORDER BY pid"""),
    ("title tokens", """
        CREATE TABLE {v}.title_token AS
        WITH ws AS (SELECT p.pid, {words} AS w FROM {v}.paper p JOIN s.pubs b USING (pid))
        SELECT token, pid FROM (SELECT pid, unnest({tokens}) AS token FROM ws)
        ORDER BY token, pid"""),
    ("token frequencies", """
        CREATE TABLE {v}.token_df AS
        SELECT token, count(*)::INTEGER AS df FROM {v}.title_token GROUP BY 1 ORDER BY 1"""),
    ("paper authors", """
        CREATE TABLE {v}.paper_author AS
        SELECT sl.pid, sl.person_id
        FROM s.slots sl
        JOIN {v}.paper p ON p.pid = sl.pid
        JOIN s.persons pp ON pp.person_id = sl.person_id AND pp.page_kind <> 'disambiguation'
        ORDER BY pid"""),
    ("author history", """
        CREATE TABLE {v}.author_venue AS
        SELECT pa.person_id, p.sid, p.year, count(*)::INTEGER AS papers
        FROM {v}.paper_author pa JOIN {v}.paper p USING (pid)
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3"""),
    ("series", """
        CREATE TABLE {v}.series AS
        SELECT p.sid, any_value(p.kind) AS kind, mode(b.venue) AS name, count(*)::INTEGER AS papers,
               min(p.year) AS first_year, max(p.year) AS last_year
        FROM {v}.paper p JOIN s.pubs b USING (pid)
        GROUP BY 1 ORDER BY 1"""),
]


def store_path(fingerprint) -> Path:
    return base.MODELS_DIR / f"venues-store-{fingerprint}.duckdb"


def build_store(con, meta) -> Path:
    fp = meta.get("fingerprint", "unknown")
    target = store_path(fp)
    building = target.with_suffix(".duckdb.building")
    building.unlink(missing_ok=True)
    base.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    con.execute(f"ATTACH '{building}' AS vb")
    try:
        for name, sql in STORE_STEPS:
            t = time.time()
            con.execute(sql.format(v="vb", first_year=config.FIRST_YEAR, words=words_expr("b.title"),
                                   tokens=TOKENS_OF_WORDS))
            log.info("venue store: built %s in %.1fs", name, time.time() - t)
        papers, tokens, series = con.execute(
            "SELECT (SELECT count(*) FROM vb.paper), (SELECT count(*) FROM vb.token_df), (SELECT count(*) FROM vb.series)"
        ).fetchone()
        con.execute("CREATE TABLE vb._meta (k VARCHAR, v VARCHAR)")
        con.executemany("INSERT INTO vb._meta VALUES (?, ?)", [
            ("fingerprint", fp), ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("papers", str(papers)), ("tokens", str(tokens)), ("series", str(series)),
            ("first_year", str(config.FIRST_YEAR)),
        ])
    finally:
        con.execute("DETACH vb")
    building.replace(target)
    log.info("venue store %s ready in %.0fs (%s papers, %s tokens, %s series)", target.name,
             time.time() - t_all, f"{papers:,}", f"{tokens:,}", f"{series:,}")
    return target


def attach_store(con, meta, build_if_missing=False):
    """Attach the venue store for this dump read-only as `v`."""
    fp = meta.get("fingerprint", "unknown")
    path = store_path(fp)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"no venue store for dump {fp} in {base.MODELS_DIR}; run `store` first")
        build_store(con, meta)
    con.execute(f"ATTACH '{path}' AS v (READ_ONLY)")
    vmeta = dict(con.execute("SELECT k, v FROM v._meta").fetchall())
    log.info("attached %s (%s papers, %s series)", path.name, vmeta.get("papers"), vmeta.get("series"))
    return vmeta


def years(meta):
    """(T_rank, T_test): the years whose papers train the ranker and are reported on."""
    last_full = int(meta.get("last_full_year") or 2025)
    t_test = config.T_TEST if config.T_TEST is not None else last_full
    return t_test - config.RANK_LAG, t_test


# --------------------------------------------------------------------------- statistics as of T
STATS_SQL = [
    # the class set: series with enough papers by T and a paper in the last ACTIVE_YEARS
    ("cls", """
        CREATE OR REPLACE TEMP TABLE cls AS
        SELECT sid, count(*)::INTEGER AS papers,
               count(*) FILTER (WHERE year >= $T - $active + 1)::INTEGER AS recent, max(year) AS last_year
        FROM v.paper
        WHERE year <= $T
        GROUP BY sid
        HAVING count(*) >= $min_papers AND max(year) >= $T - $active + 1"""),
    # how many papers of each series carry each token
    ("tok_sid", """
        CREATE OR REPLACE TEMP TABLE tok_sid AS
        SELECT t.token, p.sid, count(*)::INTEGER AS n
        FROM v.title_token t
        JOIN v.paper p ON p.pid = t.pid
        WHERE p.year <= $T AND p.sid IN (SELECT sid FROM cls)
          AND t.token IN (SELECT token FROM v.token_df WHERE df >= $min_df)
        GROUP BY 1, 2"""),
    ("vocab", """
        CREATE OR REPLACE TEMP TABLE vocab AS
        SELECT token, sum(n)::INTEGER AS df,
               ln((SELECT sum(papers) FROM cls)::DOUBLE / sum(n)) AS idf
        FROM tok_sid GROUP BY 1"""),
    # Naive Bayes: log P(token | series) = ln(n + a) - ln(N_s + aV); only the part that differs from
    # an absent token is stored, so scoring joins on present tokens alone. TF-IDF centroid: the share
    # of the series' papers carrying the token, times idf, L2-normalised per series.
    ("stats", """
        CREATE OR REPLACE TEMP TABLE stats AS
        WITH w AS (
            SELECT ts.token, ts.sid, ln(ts.n + $alpha) - ln($alpha) AS nb_w,
                   (ts.n::DOUBLE / c.papers) * vc.idf AS tfidf
            FROM tok_sid ts JOIN cls c USING (sid) JOIN vocab vc USING (token)),
        norm AS (SELECT sid, sqrt(sum(tfidf * tfidf)) AS nrm FROM w GROUP BY sid)
        SELECT w.token, w.sid, w.nb_w::FLOAT AS nb_w, (w.tfidf / nrm)::FLOAT AS cen_w
        FROM w JOIN norm USING (sid)
        ORDER BY token, sid"""),
    ("stats_series", """
        CREATE OR REPLACE TEMP TABLE stats_series AS
        WITH tot AS (SELECT sid, sum(n) AS N FROM tok_sid GROUP BY sid),
             vsz AS (SELECT count(*) AS V FROM vocab),
             all_papers AS (SELECT sum(papers) AS D FROM cls)
        SELECT c.sid, c.papers, c.recent, c.last_year, s.kind, s.name,
               ln(c.papers::DOUBLE / all_papers.D) AS log_prior,
               ln($alpha / (tot.N + $alpha * vsz.V)) AS c_absent
        FROM cls c JOIN tot USING (sid) JOIN v.series s USING (sid), vsz, all_papers"""),
]


STATS_NAMES = ("cls", "tok_sid", "vocab", "stats", "stats_series")


def _run(con, sql, params):
    used = {k: val for k, val in params.items() if re.search(rf"\${k}\b", sql)}
    con.execute(sql, used)


def _clear(con):
    """The statistics are temp tables when built and temp views when loaded; either may be in scope,
    and DuckDB will not drop one kind with the other's statement."""
    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE temporary").fetchall()}
    for name in STATS_NAMES:
        con.execute(f"DROP {'VIEW' if name in views else 'TABLE'} IF EXISTS {name}")


def build_stats(con, T):
    """Temp tables cls, vocab, stats, stats_series as of year T."""
    params = {"T": int(T), "active": config.ACTIVE_YEARS, "min_papers": config.MIN_SERIES_PAPERS,
              "min_df": config.MIN_DF, "alpha": config.NB_ALPHA}
    t0 = time.time()
    _clear(con)
    for _, sql in STATS_SQL:
        _run(con, sql, params)
    n_cls, n_stats, n_vocab = con.execute(
        "SELECT (SELECT count(*) FROM cls), (SELECT count(*) FROM stats), (SELECT count(*) FROM vocab)").fetchone()
    log.info("statistics as of %s: %s series, %s tokens, %s (token, series) weights, %.1fs",
             T, f"{n_cls:,}", f"{n_vocab:,}", f"{n_stats:,}", time.time() - t0)
    return {"series": int(n_cls), "tokens": int(n_vocab), "weights": int(n_stats)}


def export_stats(con, out_dir: Path):
    """Persist the current statistics as parquet, for the server: no file lock, atomic replace."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for table in ("stats", "stats_series", "vocab"):
        tmp = out_dir / f"{table}.parquet.tmp"
        con.execute(f"COPY {table} TO '{tmp}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000)")
        tmp.replace(out_dir / f"{table}.parquet")


def load_stats(con, in_dir: Path):
    for table in ("stats", "stats_series", "vocab"):
        if not (in_dir / f"{table}.parquet").exists():
            raise FileNotFoundError(f"no {table}.parquet in {in_dir}; run `train` first")
    _clear(con)
    for table in ("stats", "stats_series", "vocab"):
        path = str(in_dir / f"{table}.parquet").replace("'", "''")
        con.execute(f"CREATE TEMP VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    con.execute("CREATE TEMP VIEW cls AS SELECT sid, papers, recent, last_year FROM stats_series")
