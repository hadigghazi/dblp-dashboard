"""
DOI and arXiv id <-> dblp key, built once per dump.

The abstract search (content.py) finds papers in OpenAlex and has to name them by their dblp records,
and has to find abstracts for the papers dblp's own title search returns. dblp keeps a record's DOI
and arXiv link only inside its `ee` list in the parquet, and reading 7.8 million records for every
question takes seconds. One scan per dump writes this small database instead, attached read-only as
`x` beside the serving tables (`s`) and the leaderboard store (`c`).

Titles need no index: the serving table `pubs` already has `title_norm`, the same normalisation the
DBLP-QA study used (letters and digits, lower case).

Built by `python -m chat.cli paper-ids`; until it exists the abstract search maps by title only.
"""
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger("dblp.chat.paperids")

VERSION = "1"
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$")

STEPS = [
    ("ids", r"""
        CREATE TABLE x.ids AS
        WITH links AS (
            SELECT key, lower(unnest(ee)) AS u FROM s.src
            WHERE type NOT IN ('www', 'proceedings') AND ee IS NOT NULL),
        found AS (
            SELECT 'doi' AS kind, rtrim(regexp_extract(u, 'doi\.org/(10\..+)$', 1), '.,;') AS id, key
            FROM links WHERE u LIKE '%doi.org/10.%'
            UNION ALL
            SELECT 'arxiv', regexp_replace(rtrim(regexp_extract(u, 'arxiv\.org/abs/(.+)$', 1), '.,;'),
                                           'v[0-9]+$', ''), key
            FROM links WHERE u LIKE '%arxiv.org/abs/%')
        SELECT DISTINCT kind, id, key FROM found WHERE id <> '' ORDER BY id"""),
    ("ids per record", r"""
        CREATE TABLE x.key_ids AS
        SELECT key, min(id) FILTER (WHERE kind = 'doi') AS doi, min(id) FILTER (WHERE kind = 'arxiv') AS arxiv
        FROM x.ids GROUP BY key ORDER BY key"""),
]


def path_for(fingerprint) -> Path:
    return config.MODELS_DIR / f"chat-paperids-{fingerprint}.duckdb"


def build(con, meta, path=None):
    """Write the index for the attached dump (one scan of the parquet) and return its path."""
    fp = meta.get("fingerprint", "unknown")
    target = Path(path) if path else path_for(fp)
    building = target.with_suffix(".building")
    building.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con.execute(f"ATTACH '{building}' AS x")
    try:
        for name, sql in STEPS:
            t = time.time()
            con.execute(sql)
            log.info("built %s in %.1fs", name, time.time() - t)
        counts = dict(con.execute("SELECT kind, count(*) FROM x.ids GROUP BY kind").fetchall())
        con.execute("CREATE TABLE x._meta (k VARCHAR, v VARCHAR)")
        con.executemany("INSERT INTO x._meta VALUES (?, ?)", [
            ("fingerprint", fp), ("version", VERSION),
            ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("build_seconds", f"{time.time() - t0:.1f}"),
            ("dois", str(counts.get("doi", 0))), ("arxiv_ids", str(counts.get("arxiv", 0)))])
        con.execute("CHECKPOINT")
    finally:
        con.execute("DETACH x")
    building.replace(target)
    log.info("paper ids ready in %.0fs -> %s", time.time() - t0, target.name)
    return target


def attached(con) -> bool:
    return "x" in {r[0] for r in con.execute("SELECT database_name FROM duckdb_databases()").fetchall()}


def attach(con, meta, path=None):
    """Attach this dump's index read-only if it has been built; its metadata, or None. Indexes of
    other dumps are deleted once this one is attached."""
    if attached(con):
        return dict(con.execute("SELECT k, v FROM x._meta").fetchall())
    target = Path(path) if path else path_for(meta.get("fingerprint", "unknown"))
    if not target.exists():
        return None
    con.execute(f"ATTACH '{target}' AS x (READ_ONLY)")
    found = dict(con.execute("SELECT k, v FROM x._meta").fetchall())
    if found.get("version") != VERSION:
        log.info("paper ids %s were built by version %s (now %s); build them again",
                 target.name, found.get("version"), VERSION)
        con.execute("DETACH x")
        return None
    for old in config.MODELS_DIR.glob("chat-paperids-*.duckdb"):
        if old != target:
            try:
                old.unlink()
            except OSError:
                pass
    return found


def keys_for(cur, dois=(), arxiv_ids=()):
    """({doi: dblp key}, {arxiv id: dblp key}) for whatever the index holds; empty without it. When a
    DOI belongs to two records, the published version wins over a preprint (as for titles)."""
    wanted = sorted({d.lower() for d in dois if d} | {a.lower() for a in arxiv_ids if a})
    if not wanted:
        return {}, {}
    try:
        rows = cur.execute("SELECT kind, id, key FROM x.ids WHERE id IN (SELECT unnest(?::VARCHAR[]))",
                           [wanted]).fetchall()
    except Exception as e:                      # not built yet for this dump
        log.debug("paper ids unavailable: %s", e)
        return {}, {}
    found = {"doi": {}, "arxiv": {}}
    for kind, ident, key in sorted(rows, key=lambda r: (r[2].startswith("journals/corr/"), r[2])):
        found[kind].setdefault(ident, key)
    return found["doi"], found["arxiv"]


def ids_for(cur, keys):
    """{dblp key: {"doi", "arxiv"}} for the records the index has ids for."""
    if not keys:
        return {}
    try:
        rows = cur.execute("SELECT key, doi, arxiv FROM x.key_ids WHERE key IN (SELECT unnest(?::VARCHAR[]))",
                           [list(keys)]).fetchall()
    except Exception as e:
        log.debug("paper ids unavailable: %s", e)
        return {}
    return {key: {k: v for k, v in (("doi", doi), ("arxiv", arxiv)) if v} for key, doi, arxiv in rows}
