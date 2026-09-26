"""
The leaderboard store.

Almost every question the bot gets is cheap SQL, but a handful - "who has the most papers?", "the
biggest venues", "the most shared names" - are a sort over four million rows. Those are the same
answer for every user until the next dump, so they are precomputed once into
`chat-store-<fingerprint>.duckdb` (a few MB, seconds to build) and read as a lookup. That is the
difference between a two-second answer and a five-second one.

Everything else stays live SQL against the serving tables: precomputing the long tail of possible
filters is neither possible nor necessary.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, data

log = logging.getLogger("dblp.chat.store")

TOP_N = 2000
# Bumped whenever STEPS change: a store built by older code is rebuilt rather than served with a
# table missing. (The first live build silently skipped its facts step, and without this stamp the
# fix would not have reached an existing store.)
VERSION = "2"

# `person_degree` is the api's one optional serving table (the heaviest to build): if it is missing,
# the co-author leaderboard is simply absent and the tool says so.
STEPS = [
    ("authors by papers", """
        CREATE TABLE {c}.top_author_papers AS
        SELECT (row_number() OVER (ORDER BY ps.n_pubs DESC, p.name))::INTEGER AS rank,
               p.key, p.name, ps.n_pubs::INTEGER AS papers,
               c.first_year, c.last_year, c.papers::INTEGER AS journal_conference_papers
        FROM s.person_stats ps
        JOIN s.persons p USING (person_id)
        LEFT JOIN s.career c USING (person_id)
        WHERE p.page_kind <> 'disambiguation'
        QUALIFY rank <= {top}
        ORDER BY rank"""),

    ("authors by co-authors", """
        CREATE TABLE {c}.top_author_coauthors AS
        SELECT (row_number() OVER (ORDER BY d.n_coauthors DESC, p.name))::INTEGER AS rank,
               p.key, p.name, d.n_coauthors::INTEGER AS coauthors, coalesce(ps.n_pubs, 0)::INTEGER AS papers
        FROM s.person_degree d
        JOIN s.persons p USING (person_id)
        LEFT JOIN s.person_stats ps USING (person_id)
        WHERE p.page_kind <> 'disambiguation'
        QUALIFY rank <= {top}
        ORDER BY rank"""),

    ("venues by papers", """
        CREATE TABLE {c}.top_venue AS
        SELECT (row_number() OVER (ORDER BY papers DESC, usual_name))::INTEGER AS rank,
               sid, kind, usual_name AS name, papers::INTEGER AS papers, first_year, last_year,
               round(100 * oa_share, 1) AS pct_oa, round(100 * doi_share, 1) AS pct_doi,
               active_years::INTEGER AS active_years
        FROM s.series
        QUALIFY rank <= {top}
        ORDER BY rank"""),

    ("most shared names", """
        CREATE TABLE {c}.top_name AS
        SELECT (row_number() OVER (ORDER BY count(*) DESC, base_name))::INTEGER AS rank,
               base_name, count(*)::INTEGER AS people
        FROM s.persons WHERE page_kind = 'numbered' AND base_name IS NOT NULL
        GROUP BY base_name
        QUALIFY rank <= 200
        ORDER BY rank"""),

    # The headline numbers, so "how big is dblp" never runs a scan. Deliberately no reference to
    # `src`: that is a VIEW over dblp.parquet, so counting it needs the parquet mounted - the record
    # count is already in the serving database's own metadata.
    ("dataset facts", """
        CREATE TABLE {c}.facts AS
        SELECT 'records' AS k,
               (SELECT max(try_cast(v AS BIGINT)) FROM s._meta WHERE k = 'records')::BIGINT AS v UNION ALL
        SELECT 'publications', (SELECT count(*) FROM s.pubs) UNION ALL
        SELECT 'journal_conference_papers', (SELECT count(*) FROM s.pubs
            WHERE type IN ('article', 'inproceedings') AND NOT is_preprint) UNION ALL
        SELECT 'preprints', (SELECT count(*) FROM s.pubs WHERE is_preprint) UNION ALL
        SELECT 'author_pages', (SELECT count(*) FROM s.persons) UNION ALL
        SELECT 'disambiguation_bins', (SELECT count(*) FROM s.persons WHERE page_kind = 'disambiguation') UNION ALL
        SELECT 'numbered_pages', (SELECT count(*) FROM s.persons WHERE page_kind = 'numbered') UNION ALL
        SELECT 'venue_series', (SELECT count(*) FROM s.series) UNION ALL
        SELECT 'first_year', (SELECT min(year) FROM s.pubs WHERE year IS NOT NULL) UNION ALL
        SELECT 'last_year', (SELECT max(year) FROM s.pubs WHERE year IS NOT NULL)"""),
]


def store_path(fingerprint) -> Path:
    return config.MODELS_DIR / f"chat-store-{fingerprint}.duckdb"


def build(con, meta, path=None):
    """Build the store for the attached dump and leave it attached as `c`."""
    fp = meta.get("fingerprint", "unknown")
    target = Path(path) if path else store_path(fp)
    building = target.with_suffix(".building")
    building.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con.execute(f"ATTACH '{building}' AS c")
    skipped = []
    for name, sql in STEPS:
        t = time.time()
        try:
            con.execute(sql.format(c="c", top=TOP_N))
        except Exception as e:      # a missing optional serving table must not fail the build
            log.warning("skipped leaderboard '%s': %s", name, e)
            skipped.append(name)
            continue
        log.info("built %s in %.1fs", name, time.time() - t)
    con.execute("CREATE TABLE c._meta (k VARCHAR, v VARCHAR)")
    con.executemany("INSERT INTO c._meta VALUES (?, ?)", [
        ("fingerprint", fp),
        ("version", VERSION),
        ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("build_seconds", f"{time.time() - t0:.1f}"),
        ("skipped", ", ".join(skipped)),
        ("top_n", str(TOP_N)),
    ])
    con.execute("CHECKPOINT")
    con.execute("DETACH c")
    building.replace(target)
    log.info("chat store ready in %.0fs -> %s", time.time() - t0, target.name)
    return target


def attach(con, meta, path=None):
    """Attach the store read-only, building it first if this dump has none or if the one on disk was
    built by older code. Returns its metadata."""
    fp = meta.get("fingerprint", "unknown")
    target = Path(path) if path else store_path(fp)
    attached = {r[0] for r in con.execute("SELECT database_name FROM duckdb_databases()").fetchall()}
    if "c" in attached:
        return dict(con.execute("SELECT k, v FROM c._meta").fetchall())
    if target.exists():
        con.execute(f"ATTACH '{target}' AS c (READ_ONLY)")
        found = dict(con.execute("SELECT k, v FROM c._meta").fetchall())
        if found.get("version") == VERSION:
            return found
        log.info("chat store %s was built by version %s (now %s); rebuilding",
                 target.name, found.get("version", "0"), VERSION)
        con.execute("DETACH c")
        target.unlink(missing_ok=True)
    build(con, meta, target)
    con.execute(f"ATTACH '{target}' AS c (READ_ONLY)")
    out = dict(con.execute("SELECT k, v FROM c._meta").fetchall())
    stale = [p for p in config.MODELS_DIR.glob("chat-store-*.duckdb") if p != target]
    for old in stale:
        try:
            old.unlink()
        except OSError:
            pass
    return out


def has_table(con, name) -> bool:
    return bool(con.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE database_name = 'c' AND table_name = ?", [name]).fetchone()[0])


def facts(con) -> dict:
    return {k: int(v) for k, v in con.execute("SELECT k, v FROM c.facts").fetchall()}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    con, meta = data.connect()
    try:
        build(con, meta)
    finally:
        con.close()


if __name__ == "__main__":
    main()
