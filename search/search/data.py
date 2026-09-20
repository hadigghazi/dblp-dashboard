"""
Reading the data. Like ml/, this attaches the api's serving database read-only rather than
rebuilding it: the api already holds pubs built from dblp.parquet with the analysis's own
predicates, and DuckDB allows any number of read-only readers without blocking the api.
"""
import logging
from pathlib import Path

import duckdb

from . import config

log = logging.getLogger("dblp.search.data")


def find_serving_db() -> Path:
    candidates = sorted(config.CACHE_DIR.glob("serve-*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if not p.name.endswith(".building")]
    if not candidates:
        raise FileNotFoundError(
            f"no serving database in {config.CACHE_DIR}. Start the api first: it builds one from "
            f"dblp.parquet (a few minutes on the full dump)."
        )
    return candidates[0]


def connect():
    path = find_serving_db()
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
    con.execute(f"SET threads = {config.DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order = false")
    tmp = config.TMP_DIR
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp}'")
    con.execute(f"ATTACH '{path}' AS s (READ_ONLY)")
    meta = dict(con.execute("SELECT k, v FROM s._meta").fetchall())
    log.info("attached %s (dump fingerprint %s, %s records)", path.name, meta.get("fingerprint"), meta.get("records"))
    return con, meta
