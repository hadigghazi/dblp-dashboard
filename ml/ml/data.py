"""
Reading the data for the ML jobs.

We attach the api's serving database read-only rather than rebuilding those tables: it already holds
pubs / persons / person_names / slots / series, built from dblp.parquet with the analysis's own
predicates. DuckDB allows any number of read-only readers, so this never blocks the api (or the
analysis scripts, which the api deliberately keeps clear of by not opening dblp.duckdb at all).
"""
import logging
from pathlib import Path

import duckdb

from . import config

log = logging.getLogger("dblp.ml.data")


def find_serving_db() -> Path:
    """The serving database the api built for the current dump."""
    candidates = sorted(config.CACHE_DIR.glob("serve-*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if not p.name.endswith(".building")]
    if not candidates:
        raise FileNotFoundError(
            f"no serving database in {config.CACHE_DIR}. Start the api first: it builds one from "
            f"dblp.parquet (a few minutes on the full dump)."
        )
    return candidates[0]


def connect():
    """An in-memory connection with the serving database attached read-only as `s`."""
    path = find_serving_db()
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
    con.execute(f"SET threads = {config.DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order = false")   # large aggregates spill instead of failing
    tmp = config.TMP_DIR
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp}'")
    con.execute(f"ATTACH '{path}' AS s (READ_ONLY)")
    meta = dict(con.execute("SELECT k, v FROM s._meta").fetchall())
    log.info("attached %s (dump fingerprint %s, %s records)", path.name, meta.get("fingerprint"), meta.get("records"))
    return con, meta


# --------------------------------------------------------------------------- blocks
# A "name block" is everyone whose name shares one base form: the 522 "Wei Wang NNNN" pages are one
# block. Disambiguation happens inside a block and never across blocks, so all pairs stay local.
BLOCKS_SQL = """
CREATE OR REPLACE TEMP TABLE block AS
SELECT base_name, count(*) AS people
FROM s.persons
WHERE page_kind = 'numbered' AND base_name IS NOT NULL
GROUP BY base_name
HAVING count(*) >= ?
QUALIFY row_number() OVER (ORDER BY hash(base_name)) <= ?
"""


def build_blocks(con, min_people=None, max_blocks=None):
    con.execute(BLOCKS_SQL, [min_people or config.MIN_PEOPLE_PER_BLOCK, max_blocks or config.MAX_BLOCKS])
    n, people = con.execute("SELECT count(*), coalesce(sum(people), 0) FROM block").fetchone()
    log.info("blocks: %s (holding %s labelled people)", f"{n:,}", f"{people:,}")
    return n


# --------------------------------------------------------------------------- instances
# One row per (person, paper): the unit a pair is formed from. `other_names` and `other_ids` are the
# paper's *co-authors* with the block's own name removed - otherwise every pair in a block would
# share that name and the feature would carry no signal.
INSTANCE_SQL = """
CREATE OR REPLACE TEMP TABLE inst AS
WITH chosen AS (
    SELECT base_name, person_id, pid, position FROM (
        SELECT p.base_name, p.person_id, sl.pid, min(sl.position) AS position,
               row_number() OVER (PARTITION BY p.person_id ORDER BY hash(sl.pid)) AS rn
        FROM s.persons p
        JOIN person_set ps ON ps.person_id = p.person_id
        JOIN s.slots sl ON sl.person_id = p.person_id
        GROUP BY p.base_name, p.person_id, sl.pid
    ) WHERE rn <= ?
),
rec AS (
    SELECT p.pid, p.year, p.sid, p.venue, p.key_prefix, p.n_authors, p.title, p.is_preprint,
           r.authors, r.author_orcids
    FROM s.pubs p
    JOIN s.src r ON r.key = p.key
    WHERE p.pid IN (SELECT pid FROM chosen)
),
ctx AS (
    SELECT pid, list(DISTINCT person_id) FILTER (WHERE person_id IS NOT NULL) AS ids
    FROM s.slots
    WHERE pid IN (SELECT pid FROM chosen)
    GROUP BY pid
)
SELECT c.base_name, c.person_id, c.pid, c.position,
       rec.year, rec.sid, rec.venue, rec.key_prefix, rec.n_authors, rec.is_preprint,
       -- NOTE: dblp writes the assignment into the author string itself (<author>Wei Wang 0001</author>),
       -- so the raw string would leak the label: within a block, identical string <=> same person.
       -- Strip the suffix; what remains is the real signal, the printed name variant.
       lower(regexp_replace(coalesce(rec.authors[c.position], ''), ' [0-9]{4}$', '')) AS used_name,
       rec.author_orcids[c.position] AS orcid,
       coalesce(list_filter(list_transform(rec.authors, x -> lower(regexp_replace(x, ' [0-9]{4}$', ''))),
                            x -> x <> lower(c.base_name)), []) AS other_names,
       coalesce(list_filter(ctx.ids, x -> x <> c.person_id), []) AS other_ids,
       coalesce(list_distinct(list_filter(regexp_split_to_array(lower(coalesce(rec.title, '')), '[^a-z0-9]+'),
                                          x -> length(x) >= 4)), []) AS toks
FROM chosen c
JOIN rec USING (pid)
LEFT JOIN ctx USING (pid)
"""


def build_instances(con, person_ids=None, labelled=True, cap_per_person=None):
    """
    Build the `inst` table. With labelled=True it covers the numbered (editor-verified) people of the
    blocks in `block`; otherwise it covers exactly `person_ids` (used for a disambiguation bin).
    """
    if labelled:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE person_set AS
            SELECT p.person_id FROM s.persons p JOIN block b USING (base_name)
            WHERE p.page_kind = 'numbered'
            QUALIFY row_number() OVER (PARTITION BY p.base_name ORDER BY hash(p.person_id)) <= ?
        """, [config.MAX_PEOPLE_PER_BLOCK])
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE person_set (person_id INTEGER)")
        con.executemany("INSERT INTO person_set VALUES (?)", [(int(i),) for i in person_ids])
    con.execute(INSTANCE_SQL, [cap_per_person or config.MAX_PAPERS_PER_PERSON])
    n, people, blocks = con.execute(
        "SELECT count(*), count(DISTINCT person_id), count(DISTINCT base_name) FROM inst").fetchone()
    log.info("instances: %s paper-author rows, %s people, %s blocks", f"{n:,}", f"{people:,}", f"{blocks:,}")
    return n


def person_by_key(con, key):
    row = con.execute(
        "SELECT person_id, key, name, base_name, page_kind FROM s.persons WHERE key = ?", [key]).fetchone()
    if not row:
        return None
    return dict(zip(["person_id", "key", "name", "base_name", "page_kind"], row))


def largest_bins(con, limit=50, q=None):
    """Disambiguation bins by number of papers sitting on them, optionally filtered by name."""
    where = "AND p.name ILIKE ?" if q else ""
    params = [f"%{q}%", limit] if q else [limit]
    return [dict(zip(["key", "name", "papers", "numbered_pages"], r)) for r in con.execute(f"""
        WITH n AS (SELECT base_name, count(*) AS numbered FROM s.persons
                   WHERE page_kind = 'numbered' GROUP BY base_name)
        SELECT p.key, p.name, count(sl.pid) AS papers, coalesce(n.numbered, 0) AS numbered_pages
        FROM s.persons p
        JOIN s.slots sl ON sl.person_id = p.person_id
        LEFT JOIN n ON n.base_name = p.base_name
        WHERE p.page_kind = 'disambiguation' {where}
        GROUP BY p.key, p.name, n.numbered
        ORDER BY papers DESC LIMIT ?""", params).fetchall()]


def numbered_in_block(con, base_name):
    return [dict(zip(["person_id", "key", "name"], r)) for r in con.execute(
        "SELECT person_id, key, name FROM s.persons WHERE base_name = ? AND page_kind = 'numbered' "
        "ORDER BY name", [base_name]).fetchall()]
