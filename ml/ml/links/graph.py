"""
The co-authorship graph and its snapshots.

The graph is built once per dump into a DuckDB file next to the models (the "graph store"), from the
api's serving tables: one row per (author, co-author, year). Training snapshots and live suggestions
both read it, so the features a suggestion is scored on are exactly the features the model was
trained on - nothing is recomputed differently at serving time.

Disambiguation bins are not nodes. A bin ("Wei Wang") mixes hundreds of people, which makes it the
best-connected node in the whole graph and a co-author of everyone; any local heuristic would route
its predictions through them.

A snapshot at year T sees only rows with year <= T. Candidates for an anchor are its distance-2
neighbours at T (co-authors of co-authors who are not co-authors yet); the label says whether the
pair co-authored within the horizon after T.
"""
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import config as base
from . import config

log = logging.getLogger("dblp.ml.links.graph")

GRAPH_STEPS = [
    # Both directions of every co-author pair, per year, sorted so a lookup by `a` touches few row
    # groups. Papers with 2-50 authors, as in the analysis's person_degree.
    ("edges per year", """
        CREATE TABLE {g}.adj_year AS
        SELECT a.person_id AS a, b.person_id AS b, p.year, count(*)::INTEGER AS papers
        FROM s.slots a
        JOIN s.slots b ON b.pid = a.pid AND b.person_id <> a.person_id
        JOIN s.pubs p ON p.pid = a.pid
        JOIN s.persons pa ON pa.person_id = a.person_id AND pa.page_kind <> 'disambiguation'
        JOIN s.persons pb ON pb.person_id = b.person_id AND pb.page_kind <> 'disambiguation'
        WHERE p.n_authors BETWEEN 2 AND 50 AND p.year IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3"""),

    # New neighbours per author and year: degree at T is a prefix sum.
    ("degree per year", """
        CREATE TABLE {g}.deg_year AS
        SELECT a AS person_id, year, count(*)::INTEGER AS new_neighbours
        FROM (SELECT a, b, min(year) AS year FROM {g}.adj_year GROUP BY a, b)
        GROUP BY 1, 2
        ORDER BY 1, 2"""),

    # Activity per author and year (every publication, single-author ones included).
    ("papers per year", """
        CREATE TABLE {g}.node_year AS
        SELECT sl.person_id, p.year, count(*)::INTEGER AS papers
        FROM s.slots sl
        JOIN s.pubs p ON p.pid = sl.pid
        JOIN s.persons pp ON pp.person_id = sl.person_id AND pp.page_kind <> 'disambiguation'
        WHERE p.year IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2"""),

    # Venue series per author, with the year they first published there.
    ("venues per author", """
        CREATE TABLE {g}.node_venue AS
        SELECT sl.person_id, p.sid, min(p.year) AS first_year, count(*)::INTEGER AS papers
        FROM s.slots sl
        JOIN s.pubs p ON p.pid = sl.pid
        JOIN s.persons pp ON pp.person_id = sl.person_id AND pp.page_kind <> 'disambiguation'
        WHERE p.sid IS NOT NULL AND p.year IS NOT NULL AND p.key_prefix IN ('conf', 'journals')
        GROUP BY 1, 2
        ORDER BY 1, 2"""),
]


def store_path(fingerprint) -> Path:
    return base.MODELS_DIR / f"links-graph-{fingerprint}.duckdb"


def build_store(con, meta) -> Path:
    """Build the graph store for the attached serving database `s`; atomic, so a reader never sees
    a half-written file."""
    fp = meta.get("fingerprint", "unknown")
    target = store_path(fp)
    building = target.with_suffix(".duckdb.building")
    building.unlink(missing_ok=True)
    base.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    con.execute(f"ATTACH '{building}' AS gb")
    try:
        for name, sql in GRAPH_STEPS:
            t = time.time()
            con.execute(sql.format(g="gb"))
            log.info("graph store: built %s in %.1fs", name, time.time() - t)
        counts = {
            "edges": con.execute("SELECT count(*) / 2 FROM gb.adj_year").fetchone()[0],
            "nodes": con.execute("SELECT count(DISTINCT a) FROM gb.adj_year").fetchone()[0],
        }
        con.execute("CREATE TABLE gb._meta (k VARCHAR, v VARCHAR)")
        con.executemany("INSERT INTO gb._meta VALUES (?, ?)", [
            ("fingerprint", fp), ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("edges", str(int(counts["edges"]))), ("nodes", str(int(counts["nodes"]))),
            ("last_full_year", str(meta.get("last_full_year", ""))),
        ])
    finally:
        con.execute("DETACH gb")   # flushes the file
    building.replace(target)
    log.info("graph store %s ready in %.0fs (%s edges, %s nodes)", target.name, time.time() - t_all,
             f"{int(counts['edges']):,}", f"{int(counts['nodes']):,}")
    return target


def attach_store(con, meta, build_if_missing=False):
    """Attach the graph store for this dump read-only as `g`. Returns its metadata."""
    fp = meta.get("fingerprint", "unknown")
    path = store_path(fp)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"no graph store for dump {fp} in {base.MODELS_DIR}; run `graph` first")
        build_store(con, meta)
    con.execute(f"ATTACH '{path}' AS g (READ_ONLY)")
    gmeta = dict(con.execute("SELECT k, v FROM g._meta").fetchall())
    log.info("attached %s (%s edges, %s nodes)", path.name, gmeta.get("edges"), gmeta.get("nodes"))
    return gmeta


def snapshot_years(meta):
    """(T_train, T_test): the test window ends at the last complete year of the dump."""
    last_full = int(meta.get("last_full_year") or 2025)
    t_test = config.T_TEST if config.T_TEST is not None else last_full - config.HORIZON
    t_train = config.T_TRAIN if config.T_TRAIN is not None else t_test - config.HORIZON
    return t_train, t_test


# --------------------------------------------------------------------------- snapshots
def select_anchors(con, T, buckets, limit):
    """Authors with MIN_PAPERS papers by T and a paper in the last RECENT_YEARS, from the given hash
    buckets, sampled deterministically."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE anchor AS
        SELECT person_id
        FROM g.node_year
        WHERE year <= ?
        GROUP BY person_id
        HAVING sum(papers) >= ? AND max(year) >= ?
           AND (hash(person_id) % 10)::INT IN ({", ".join(str(b) for b in sorted(buckets))})
        QUALIFY row_number() OVER (ORDER BY hash(person_id * 31 + ?)) <= ?
    """, [T, config.MIN_PAPERS, T - config.RECENT_YEARS + 1, T, limit])
    return con.execute("SELECT count(*) FROM anchor").fetchone()[0]


def set_anchor(con, person_id):
    con.execute("CREATE OR REPLACE TEMP TABLE anchor AS SELECT ?::INTEGER AS person_id", [int(person_id)])


CANDIDATE_SQL = [
    # anchor -> co-author at T, with when that link was last active and how many papers it carries
    ("n1", """
        CREATE OR REPLACE TEMP TABLE n1 AS
        SELECT x.a AS u, x.b AS w, max(x.year) AS last, sum(x.papers)::INTEGER AS papers
        FROM g.adj_year x JOIN anchor ON anchor.person_id = x.a
        WHERE x.year <= $T
        GROUP BY 1, 2"""),
    # co-author -> its co-authors at T
    ("n2", """
        CREATE OR REPLACE TEMP TABLE n2 AS
        SELECT x.a AS w, x.b AS v, max(x.year) AS last, sum(x.papers)::INTEGER AS papers
        FROM g.adj_year x
        WHERE x.a IN (SELECT DISTINCT w FROM n1) AND x.year <= $T
        GROUP BY 1, 2"""),
    # degree at T of every node the candidates touch
    ("deg", """
        CREATE OR REPLACE TEMP TABLE deg AS
        SELECT person_id, sum(new_neighbours)::INTEGER AS deg
        FROM g.deg_year
        WHERE year <= $T AND person_id IN (
            SELECT person_id FROM anchor UNION SELECT w FROM n1 UNION SELECT v FROM n2)
        GROUP BY 1"""),
    # every distance-2 pair with the classic scores, before any cap
    ("cand_all", """
        CREATE OR REPLACE TEMP TABLE cand_all AS
        SELECT n1.u, n2.v,
               count(*)::INTEGER AS cn,
               sum(1.0 / ln(greatest(d.deg, 2))) AS aa,
               sum(1.0 / greatest(d.deg, 1)) AS ra,
               max(least(n1.last, n2.last)) AS bridge_last,
               count(*) FILTER (WHERE least(n1.last, n2.last) >= $T - 2)::INTEGER AS cn_recent,
               sum(least(n1.papers, n2.papers))::INTEGER AS bridge_strength
        FROM n1
        JOIN n2 ON n2.w = n1.w
        JOIN deg d ON d.person_id = n1.w
        WHERE n2.v <> n1.u
          AND NOT EXISTS (SELECT 1 FROM n1 e WHERE e.u = n1.u AND e.w = n2.v)
        GROUP BY 1, 2"""),
    # the cap: an anchor with thousands of distance-2 neighbours keeps those sharing the most
    # co-authors with it (the baseline's own ranking, so the cap cannot favour the model)
    ("cand", """
        CREATE OR REPLACE TEMP TABLE cand AS
        SELECT * FROM cand_all
        QUALIFY row_number() OVER (PARTITION BY u ORDER BY cn DESC, hash(u * 1000003 + v)) <= $max_cand"""),
    # the label: a joint paper within the horizon after T (empty when T is "now")
    ("future", """
        CREATE OR REPLACE TEMP TABLE future AS
        SELECT x.a AS u, x.b AS v
        FROM g.adj_year x JOIN anchor ON anchor.person_id = x.a
        WHERE x.year > $T AND x.year <= $T + $H
        GROUP BY 1, 2"""),
    ("node", """
        CREATE OR REPLACE TEMP TABLE node AS
        SELECT person_id, sum(papers)::INTEGER AS papers, min(year) AS first_year, max(year) AS last_year,
               sum(papers) FILTER (WHERE year >= $T - 2)::INTEGER AS recent
        FROM g.node_year
        WHERE year <= $T AND person_id IN (SELECT u FROM cand UNION SELECT v FROM cand)
        GROUP BY 1"""),
    ("venues", """
        CREATE OR REPLACE TEMP TABLE venues AS
        SELECT person_id, list(sid) AS sids
        FROM g.node_venue
        WHERE first_year <= $T AND person_id IN (SELECT u FROM cand UNION SELECT v FROM cand)
        GROUP BY 1"""),
    ("pair", """
        CREATE OR REPLACE TEMP TABLE pair AS
        SELECT c.u, c.v, (f.u IS NOT NULL)::INT AS y, (hash(c.u) % 10)::INT AS bucket,
               c.cn, c.aa, c.ra, c.bridge_strength, c.cn_recent,
               ($T - c.bridge_last)::INTEGER AS bridge_age,
               du.deg AS deg_u, dv.deg AS deg_v,
               c.cn / (du.deg + dv.deg - c.cn) AS jaccard,
               du.deg::DOUBLE * dv.deg AS pa,
               nu.papers AS papers_u, nv.papers AS papers_v,
               coalesce(nu.recent, 0) AS recent_u, coalesce(nv.recent, 0) AS recent_v,
               ($T - nu.first_year)::INTEGER AS age_u, ($T - nv.first_year)::INTEGER AS age_v,
               ($T - nu.last_year)::INTEGER AS idle_u, ($T - nv.last_year)::INTEGER AS idle_v,
               len(list_intersect(coalesce(vu.sids, []), coalesce(vv.sids, [])))::INTEGER AS shared_venues,
               len(coalesce(vu.sids, []))::INTEGER AS n_venues_u, len(coalesce(vv.sids, []))::INTEGER AS n_venues_v
        FROM cand c
        LEFT JOIN future f ON f.u = c.u AND f.v = c.v
        JOIN deg du ON du.person_id = c.u
        JOIN deg dv ON dv.person_id = c.v
        JOIN node nu ON nu.person_id = c.u
        JOIN node nv ON nv.person_id = c.v
        LEFT JOIN venues vu ON vu.person_id = c.u
        LEFT JOIN venues vv ON vv.person_id = c.v"""),
]


def _run(con, sql, params):
    """DuckDB rejects named parameters a statement does not use, so pass only the referenced ones."""
    used = {k: v for k, v in params.items() if re.search(rf"\${k}\b", sql)}
    con.execute(sql, used)


def build_pairs(con, T, horizon=None, max_candidates=None):
    """From the `anchor` table, build `pair`: one row per (anchor, distance-2 candidate) at T with
    its features and label (all zero when T is the current year: nothing has happened yet)."""
    params = {"T": int(T), "H": int(horizon if horizon is not None else config.HORIZON),
              "max_cand": int(max_candidates or config.MAX_CANDIDATES)}
    t0 = time.time()
    for name, sql in CANDIDATE_SQL:
        _run(con, sql, params)
    n, pos, anchors = con.execute(
        "SELECT count(*), coalesce(sum(y), 0), count(DISTINCT u) FROM pair").fetchone()
    log.info("snapshot %s: %s candidate pairs for %s anchors, %s positive (%.2f%%), %.1fs",
             T, f"{n:,}", f"{anchors:,}", f"{int(pos):,}", 100 * pos / n if n else 0, time.time() - t0)
    return int(n)


def sample_training_pairs(con, neg_per_anchor=None):
    """Keep every positive and a bounded number of negatives per anchor."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pair_sampled AS
        SELECT * FROM pair
        QUALIFY row_number() OVER (PARTITION BY u, y ORDER BY hash(u * 1000003 + v))
                <= CASE WHEN y = 1 THEN 1000000000 ELSE ? END
    """, [int(neg_per_anchor or config.NEG_PER_ANCHOR)])
    con.execute("DROP TABLE pair")
    con.execute("ALTER TABLE pair_sampled RENAME TO pair")
    return con.execute("SELECT count(*) FROM pair").fetchone()[0]


def origin_of_new_links(con, T):
    """
    Where the anchors' new co-authors of (T, T+H] were at T: at distance 2 (reachable by this
    method), farther away but already publishing, or newcomers with no paper at all. The distance-2
    share is the ceiling for any local link predictor; `kept` says how many survived the cap.
    """
    rows = con.execute("""
        WITH new_link AS (
            SELECT f.u, f.v FROM future f
            LEFT JOIN n1 ON n1.u = f.u AND n1.w = f.v
            WHERE n1.u IS NULL),
        existed AS (SELECT DISTINCT person_id FROM g.node_year WHERE year <= ?),
        cls AS (
            SELECT nl.u, nl.v,
                   CASE WHEN ca.u IS NOT NULL THEN 'distance 2'
                        WHEN ex.person_id IS NOT NULL THEN 'farther'
                        ELSE 'newcomer' END AS origin,
                   (c.u IS NOT NULL)::INT AS kept
            FROM new_link nl
            LEFT JOIN cand_all ca ON ca.u = nl.u AND ca.v = nl.v
            LEFT JOIN cand c ON c.u = nl.u AND c.v = nl.v
            LEFT JOIN existed ex ON ex.person_id = nl.v)
        SELECT origin, count(*) AS links, sum(kept) AS kept FROM cls GROUP BY 1
    """, [int(T)]).fetchall()
    total = sum(r[1] for r in rows) or 1
    out = {r[0]: {"links": int(r[1]), "share": round(r[1] / total, 4)} for r in rows}
    for origin in ("distance 2", "farther", "newcomer"):
        out.setdefault(origin, {"links": 0, "share": 0.0})
    d2 = next((r for r in rows if r[0] == "distance 2"), None)
    out["distance 2"]["kept_after_cap"] = int(d2[2]) if d2 else 0
    out["new_links"] = int(sum(r[1] for r in rows))
    return out
