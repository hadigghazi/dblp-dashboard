"""
Reading the data.

Like the ML and search services, the chat service attaches the api's serving database read-only
(`s`) instead of rebuilding anything: it already holds pubs / persons / person_names / slots /
career / person_stats / series / word_year with the analysis's own predicates. Any number of
read-only readers is fine, so this never blocks the api or the analysis scripts.

One connection is shared per process and every query takes its own cursor, so tools can run
concurrently in the thread pool.
"""
import logging
import threading
from pathlib import Path

import duckdb

from . import config

log = logging.getLogger("dblp.chat.data")


def find_serving_db() -> Path:
    candidates = sorted(config.CACHE_DIR.glob("serve-*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if not p.name.endswith(".building")]
    if not candidates:
        raise FileNotFoundError(
            f"no serving database in {config.CACHE_DIR}. Start the api first: it builds one from "
            f"dblp.parquet (a few minutes on the full dump)."
        )
    return candidates[0]


def configure(con):
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
    con.execute(f"SET threads = {config.DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order = false")
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{config.TMP_DIR}'")


def connect(serving_path=None):
    """An in-memory connection with the serving database attached read-only as `s`."""
    path = Path(serving_path) if serving_path else find_serving_db()
    con = duckdb.connect()
    configure(con)
    con.execute(f"ATTACH '{str(path).replace(chr(39), chr(39) * 2)}' AS s (READ_ONLY)")
    meta = dict(con.execute("SELECT k, v FROM s._meta").fetchall())
    log.info("attached %s (dump fingerprint %s)", path.name, meta.get("fingerprint"))
    return con, meta


class Pool:
    """One shared connection, a cursor per call. `reload()` swaps in a new dump's database; cursors
    already handed out keep working on the old one, exactly as the api does."""

    def __init__(self):
        self._lock = threading.Lock()
        self._con = None
        self.meta = {}
        self.serving_path = None
        self.error = None

    def load(self, serving_path=None):
        try:
            con, meta = connect(serving_path)
        except Exception as e:
            self.error = str(e)
            log.warning("no serving database yet: %s", e)
            return False
        with self._lock:
            self._con, self.meta, self.error = con, meta, None
            self.serving_path = serving_path or find_serving_db()
        return True

    def ready(self) -> bool:
        return self._con is not None

    def cursor(self):
        return self.connection().cursor()

    def connection(self):
        """The real connection, for statements that must affect the whole instance (ATTACH)."""
        con = self._con
        if con is None:
            raise RuntimeError(self.error or "the serving database is not attached yet")
        return con

    def fingerprint(self):
        return self.meta.get("fingerprint")

    def last_full_year(self, default=2025):
        try:
            return int(self.meta.get("last_full_year", default))
        except (TypeError, ValueError):
            return default


pool = Pool()
