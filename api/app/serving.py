"""
The serving database.

The API never opens ~/dblp/dblp.duckdb: DuckDB locks its file, so holding it open would stop the
analysis scripts from writing to it. Instead it reads dblp.parquet (the file dblp.duckdb's `records`
view points at) and builds its own tables into CACHE_DIR, using the same definitions as the analysis
scripts (profile_dblp.py, build_eda_tables.py, eda_0N_*.py). The build runs once per parquet file:
a watcher rebuilds automatically when a new dump is parsed. Every request then runs its own SQL
against these tables.
"""
import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from . import config

log = logging.getLogger("dblp.serving")

STOP_WORDS = ("the and for with from into over under via using based toward towards its their this that "
              "these those are can not than versus vs our your who how what when why which where").split()

# (name, sql, required). Written with the exact predicates of the analysis scripts; comments say which.
STEPS = [
    ("source view", r"""
        CREATE VIEW src AS SELECT * FROM read_parquet('{parquet}')""", True),

    # profile_dblp.py: pubs = records WHERE type NOT IN ('www','proceedings'), plus is_preprint / title_norm.
    # eda_03_venues.py: series id, doi prefix. eda_04_careers.py: surname order flags.
    ("publications", r"""
        CREATE TABLE pubs_base AS
        WITH s AS (
            SELECT *,
                   CASE WHEN n_authors BETWEEN 2 AND 10 THEN
                       list_transform(authors, lambda a: lower(regexp_extract(regexp_replace(a, ' [0-9]{{4}}$', ''), '([^ ]+)$', 1)))
                   END AS surnames
            FROM src WHERE type NOT IN ('www', 'proceedings'))
        SELECT (row_number() OVER ())::INTEGER AS pid,
               key, type, publtype, year, title, n_authors, n_orcids, has_oa, pages,
               journal, booktitle, school, publisher, mdate,
               (coalesce(journal, '') = 'CoRR' OR coalesce(publtype, '') LIKE 'informal%') AS is_preprint,
               regexp_replace(lower(title), '[^a-z0-9]', '', 'g') AS title_norm,
               split_part(key, '/', 1) AS key_prefix,
               split_part(key, '/', 1) || '/' || split_part(key, '/', 2) AS sid,
               coalesce(journal, booktitle) AS venue,
               regexp_extract(array_to_string(ee, ' '), 'doi\.org/(10\.[0-9]+)/', 1) AS doi_prefix,
               array_to_string(ee, ' ') LIKE '%doi.org/%' AS has_doi,
               regexp_extract(ee[1], '^https?://([^/]+)', 1) AS link_host,
               coalesce(publtype, '') LIKE '%withdrawn%' AS is_withdrawn,
               surnames = list_sort(surnames) AS alphabetical,
               len(list_distinct(surnames)) = n_authors AS distinct_surnames
        FROM s""", True),

    # profile_dblp.py: persons = www records under homepages/. build_eda_tables.py: page_kind and link flags.
    ("author pages", r"""
        CREATE TABLE persons AS
        SELECT (row_number() OVER (ORDER BY key))::INTEGER AS person_id,
               key, publtype, authors AS names, authors[1] AS name, n_authors AS n_names, urls, notes,
               CASE WHEN publtype = 'disambiguation' THEN 'disambiguation'
                    WHEN regexp_matches(authors[1], ' [0-9]{{4}}$') THEN 'numbered'
                    ELSE 'regular' END AS page_kind,
               regexp_replace(authors[1], ' [0-9]{{4}}$', '') AS base_name,
               array_to_string(notes, '|') LIKE '%affiliation%' AS has_affiliation,
               array_to_string(urls, ' ') LIKE '%orcid.org%' AS has_orcid_link,
               array_to_string(urls, ' ') LIKE '%wikidata.org%' AS has_wikidata_link
        FROM src WHERE type = 'www' AND key LIKE 'homepages/%'""", True),

    ("name lookup", r"""
        CREATE TABLE person_names AS SELECT person_id, unnest(names) AS name FROM persons""", True),

    # One row per author slot on every publication (authorships + auth_keyed + careers' slots in one).
    # LEFT JOIN keeps unresolved names; queries that mirror the analysis filter person_id IS NOT NULL.
    ("author slots", r"""
        CREATE TABLE slots AS
        WITH x AS (
            SELECT b.pid, b.year, b.n_authors, b.type, b.is_preprint,
                   unnest(s.authors) AS author, unnest(s.author_orcids) AS orcid,
                   unnest(range(1, s.n_authors + 1)) AS position
            FROM src s JOIN pubs_base b ON b.key = s.key
            WHERE s.type NOT IN ('www', 'proceedings') AND s.n_authors > 0)
        SELECT x.pid, x.year, x.n_authors::SMALLINT AS n_authors, x.position::SMALLINT AS position,
               x.type, x.is_preprint, x.orcid IS NOT NULL AS has_orcid,
               pn.person_id, coalesce(p.page_kind = 'disambiguation', false) AS on_bin
        FROM x
        LEFT JOIN person_names pn ON pn.name = x.author
        LEFT JOIN persons p ON p.person_id = pn.person_id
        ORDER BY pn.person_id""", True),

    # build_eda_tables.py: n_unassigned_authors and has_title_twin.
    ("paper flags", r"""
        CREATE TABLE pubs AS
        WITH twins AS (
            SELECT title_norm FROM pubs_base WHERE length(title_norm) >= 30
            GROUP BY title_norm HAVING bool_or(is_preprint) AND bool_or(NOT is_preprint)),
        unid AS (SELECT pid, count(*) AS n_unidentified FROM slots WHERE on_bin GROUP BY pid)
        SELECT b.*, coalesce(u.n_unidentified, 0) AS n_unidentified, t.title_norm IS NOT NULL AS has_twin
        FROM pubs_base b
        LEFT JOIN unid u ON u.pid = b.pid
        LEFT JOIN twins t ON t.title_norm = b.title_norm
        ORDER BY b.pid""", True),

    ("drop staging", "DROP TABLE pubs_base", True),

    # eda_04_careers.py: career over journal/conference papers, bins excluded.
    ("careers", r"""
        CREATE TABLE career AS
        SELECT person_id, min(year) AS first_year, max(year) AS last_year,
               count(*) AS papers, bool_or(has_orcid) AS any_orcid
        FROM slots
        WHERE person_id IS NOT NULL AND NOT on_bin
          AND type IN ('article', 'inproceedings') AND NOT is_preprint
        GROUP BY person_id""", True),

    # build_eda_tables.py: person_stats.n_pubs (all publication types).
    ("author output", r"""
        CREATE TABLE person_stats AS
        SELECT person_id, count(*) AS n_pubs FROM slots WHERE person_id IS NOT NULL GROUP BY person_id""", True),

    # eda_03_venues.py: venue series.
    ("venue series", r"""
        CREATE TABLE series AS
        SELECT sid, CASE WHEN key_prefix = 'journals' THEN 'journal' ELSE 'conference' END AS kind,
               mode(venue) AS usual_name, count(*) AS papers,
               min(year) AS first_year, max(year) AS last_year, count(DISTINCT year) AS active_years,
               count(DISTINCT venue) AS name_variants,
               avg((doi_prefix <> '')::INT) AS doi_share, avg(has_oa::INT) AS oa_share
        FROM pubs
        WHERE type IN ('article', 'inproceedings') AND NOT is_preprint AND key_prefix IN ('conf', 'journals')
        GROUP BY ALL""", True),

    # eda_01_titles.py: word_year.
    ("title words", r"""
        CREATE TABLE word_year AS
        WITH t AS (
            SELECT year, list_distinct(regexp_split_to_array(lower(title), '[^a-z0-9]+')) AS words
            FROM pubs
            WHERE type IN ('article', 'inproceedings') AND NOT is_preprint
              AND title IS NOT NULL AND year >= 1970)
        SELECT year, word, count(*)::INTEGER AS titles
        FROM (SELECT year, unnest(words) AS word FROM t)
        WHERE length(word) >= 3 AND NOT regexp_matches(word, '^[0-9]+$')
          AND word NOT IN (SELECT unnest({stop}::VARCHAR[]))
        GROUP BY ALL""", True),

    # build_eda_tables.py: person_degree (papers with 2-50 authors). The heaviest step, so it is optional:
    # if it fails, only the "co-authors per author" distribution is unavailable.
    ("co-author counts", r"""
        CREATE TABLE person_degree AS
        WITH small AS (SELECT person_id, pid FROM slots WHERE person_id IS NOT NULL AND n_authors BETWEEN 2 AND 50)
        SELECT a.person_id, count(DISTINCT b.person_id) AS n_coauthors
        FROM small a JOIN small b ON a.pid = b.pid AND a.person_id <> b.person_id
        GROUP BY a.person_id""", False),
]


class NotReady(Exception):
    def __init__(self, status):
        super().__init__("serving database not ready")
        self.status = status


def fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path}|{st.st_size}|{st.st_mtime_ns}|v{config.BUILD_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Serving:
    def __init__(self):
        self._lock = threading.Lock()
        self._con = None
        self.generation = 0
        self.meta = {}
        self.status = {"state": "starting", "message": "Starting up", "step": 0, "steps": len(STEPS)}
        self._building = False
        self.on_ready = []  # callbacks run after each successful (re)load, e.g. cache warm-up
        self.stopping = threading.Event()
        self._threads = []

    # ---- connections -------------------------------------------------------
    def cursor(self):
        con = self._con
        if con is None:
            raise NotReady(dict(self.status))
        return con.cursor()

    def _configure(self, con):
        tmp = config.CACHE_DIR / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
        con.execute(f"SET threads = {config.DUCKDB_THREADS}")
        con.execute(f"SET temp_directory = '{tmp}'")
        con.execute("SET preserve_insertion_order = false")

    def _open(self, path: Path):
        con = duckdb.connect(str(path), read_only=True)
        self._configure(con)
        meta = dict(con.execute("SELECT k, v FROM _meta").fetchall())
        with self._lock:
            self._con = con           # the previous connection is left to the garbage collector,
            self.generation += 1      # so requests still holding a cursor on it finish normally
            self.meta = meta
            self.status = {"state": "ready", "message": "Live", "step": len(STEPS), "steps": len(STEPS),
                           "optional_failed": meta.get("optional_failed", "")}
        for cb in self.on_ready:
            try:
                cb()
            except Exception:
                log.exception("on_ready callback failed")

    # ---- build -------------------------------------------------------------
    def ensure(self):
        """Open the serving database for the current parquet, building it first if needed."""
        if not config.PARQUET.exists():
            self.status = {"state": "error", "message": f"Parquet not found at {config.PARQUET}", "step": 0,
                           "steps": len(STEPS)}
            log.error(self.status["message"])
            return
        fp = fingerprint(config.PARQUET)
        target = config.CACHE_DIR / f"serve-{fp}.duckdb"
        if self.meta.get("fingerprint") == fp and self._con is not None:
            return
        if target.exists():
            try:
                self._open(target)
                log.info("opened existing serving database %s", target.name)
                return
            except Exception:
                log.exception("existing serving database unusable, rebuilding")
                target.unlink(missing_ok=True)
        self.build(fp, target)

    def build(self, fp: str, target: Path):
        if self._building or self.stopping.is_set():
            return
        self._building = True
        building = target.with_suffix(".building")
        building.unlink(missing_ok=True)
        was_ready = self._con is not None
        t_all = time.time()
        timings, optional_failed = [], []
        try:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(str(building))
            self._configure(con)
            for i, (name, sql, required) in enumerate(STEPS, start=1):
                if self.stopping.is_set():
                    raise RuntimeError("shutting down")
                if not was_ready:
                    self.status = {"state": "building", "message": f"Preparing live data: {name}",
                                   "step": i, "steps": len(STEPS)}
                else:
                    self.status = dict(self.status, refreshing=f"{name} ({i}/{len(STEPS)})")
                t = time.time()
                try:
                    con.execute(sql.format(parquet=str(config.PARQUET).replace("'", "''"),
                                           stop="['" + "','".join(STOP_WORDS) + "']"))
                except Exception as e:
                    if required:
                        raise
                    log.exception("optional step failed: %s", name)
                    optional_failed.append(name)
                    continue
                timings.append(f"{name}={time.time() - t:.1f}s")
                log.info("built %s in %.1fs", name, time.time() - t)

            src_stats = con.execute("""
                SELECT count(*), max(mdate), count(*) FILTER (WHERE type = 'www' AND key LIKE 'homepages/%')
                FROM src""").fetchone()
            latest_mdate = str(src_stats[1] or "")
            meta = {
                "fingerprint": fp,
                "built_at": _now(),
                "build_seconds": f"{time.time() - t_all:.0f}",
                "timings": ", ".join(timings),
                "optional_failed": ", ".join(optional_failed),
                "parquet": str(config.PARQUET),
                "parquet_bytes": str(config.PARQUET.stat().st_size),
                "parquet_modified": datetime.fromtimestamp(config.PARQUET.stat().st_mtime, timezone.utc)
                                    .isoformat(timespec="seconds"),
                "records": str(src_stats[0]),
                "latest_mdate": latest_mdate,
                # the dump is taken on the 1st of a month; its year is partial, the one before is complete
                "last_full_year": str(int(latest_mdate[:4]) - 1) if latest_mdate[:4].isdigit() else "2025",
            }
            con.execute("CREATE TABLE _meta (k VARCHAR, v VARCHAR)")
            con.executemany("INSERT INTO _meta VALUES (?, ?)", list(meta.items()))
            con.execute("CHECKPOINT")
            con.close()
            building.replace(target)
            self._open(target)
            for old in config.CACHE_DIR.glob("serve-*.duckdb"):
                if old != target:
                    try:
                        old.unlink()
                    except OSError:
                        pass  # still open on this platform; removed on the next rebuild
            log.info("serving database ready in %.0fs", time.time() - t_all)
        except Exception as e:
            log.exception("build failed")
            building.unlink(missing_ok=True)
            if was_ready:
                self.status = dict(self.status, refreshing=None, refresh_error=str(e))
            else:
                self.status = {"state": "error", "message": f"Build failed: {e}", "step": 0, "steps": len(STEPS)}
        finally:
            self._building = False
            if was_ready and self.status.get("refreshing"):
                self.status = dict(self.status, refreshing=None)

    # ---- background --------------------------------------------------------
    def start_background(self):
        def loop():
            while not self.stopping.is_set():
                try:
                    self.ensure()
                except Exception:
                    log.exception("serving check failed")
                self.stopping.wait(config.WATCH_SECONDS)
        self.spawn(loop, "serving-watch")

    def spawn(self, target, name):
        t = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(t)
        t.start()
        return t

    def shutdown(self, timeout=30):
        """Stop background work and close DuckDB before the interpreter exits (avoids aborts mid-query)."""
        self.stopping.set()
        con = self._con
        if con is not None:
            try:
                con.interrupt()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout)
        self._con = None
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


serving = Serving()
