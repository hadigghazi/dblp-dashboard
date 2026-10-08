"""
The escape hatch, guarded.

Eighteen typed tools cover the questions worth designing for; the long tail ("average team size in
venues whose name contains 'workshop'") is unbounded, so the model may also write SQL. That is only
safe under real constraints, not good intentions:

  * one statement, and it must be a SELECT or a WITH;
  * a keyword denylist that removes every way DuckDB can write, read a file, load an extension or
    reach the network, checked on the statement with comments and strings stripped;
  * the connection is opened READ_ONLY on the serving database, with external access disabled and
    the configuration then locked, so the denylist is a second line of defence rather than the only
    one;
  * the result is wrapped in a row cap, and a watchdog interrupts the query after a few seconds;
  * the SQL is returned with the answer, so a reader can check what produced a number.

A denied statement comes back as a message for the model, not an exception: it can rephrase or fall
back to a typed tool.
"""
import logging
import re
import threading

import duckdb

from . import config

log = logging.getLogger("dblp.chat.sqlguard")

FORBIDDEN = [
    "attach", "detach", "copy", "export", "import", "install", "load", "pragma", "set", "reset",
    "create", "insert", "update", "delete", "drop", "alter", "truncate", "replace", "vacuum",
    "checkpoint", "call", "prepare", "execute", "transaction", "begin", "commit", "rollback",
    "read_csv", "read_parquet", "read_json", "read_text", "read_blob", "glob", "sniff_csv",
    "duckdb_extensions", "httpfs", "postgres_scan", "sqlite_scan", "mysql_scan", "iceberg", "delta",
    "getenv", "shell", "system", "gen_random_uuid",
]
_WORD = re.compile(r"[a-z_][a-z0-9_]*")


def strip_noise(sql: str) -> str:
    """Remove string literals and comments, so a paper title containing the word 'update' is not
    mistaken for a statement (and a comment cannot hide one)."""
    out = re.sub(r"'(?:''|[^'])*'", "''", sql)
    out = re.sub(r'"(?:""|[^"])*"', '""', out)
    out = re.sub(r"--[^\n]*", " ", out)
    out = re.sub(r"/\*.*?\*/", " ", out, flags=re.S)
    return out


def check(sql: str):
    """(ok, message). `message` explains the refusal, phrased for the model."""
    raw = (sql or "").strip().rstrip(";").strip()
    if not raw:
        return False, "empty statement"
    if len(raw) > 4000:
        return False, "statement too long (4000 characters max)"
    bare = strip_noise(raw)
    if ";" in bare:
        return False, "only one statement is allowed; remove the ';'"
    head = bare.lstrip("( \n\t").lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False, "only SELECT (or WITH ... SELECT) is allowed"
    words = set(_WORD.findall(bare.lower()))
    hits = sorted(words & set(FORBIDDEN))
    if hits:
        return False, f"not allowed here: {', '.join(hits)}. This connection is read-only."
    return True, ""


def _interrupt_after(con, seconds):
    timer = threading.Timer(seconds, con.interrupt)
    timer.daemon = True
    timer.start()
    return timer


def run(serving_path, sql: str, row_cap=None, timeout=None):
    """Execute a checked statement on its own read-only connection. Returns
    {ok, columns, rows, row_count, truncated, sql} or {ok: False, error}."""
    ok, message = check(sql)
    if not ok:
        return {"ok": False, "error": message, "sql": sql}
    row_cap = row_cap or config.SQL_ROW_CAP
    timeout = timeout or config.SQL_TIMEOUT
    wrapped = f"SELECT * FROM (\n{sql.strip().rstrip(';')}\n) AS _q LIMIT {int(row_cap) + 1}"
    con = duckdb.connect(str(serving_path), read_only=True)
    timer = None
    try:
        try:
            con.execute(f"SET threads = {config.DUCKDB_THREADS}")
            con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
        except duckdb.Error:
            # older builds, or a call running beside this one already configured and locked the instance
            # DuckDB shares between connections to one file: the denylist and read-only mode still apply
            log.debug("configuration already locked or not lockable")
        timer = _interrupt_after(con, timeout)
        con.execute(wrapped)
        columns = [d[0] for d in con.description]
        rows = con.fetchall()
    except duckdb.InterruptException:
        return {"ok": False, "error": f"query took longer than {timeout:.0f}s and was stopped; "
                                     f"narrow it (add a year filter, or aggregate)", "sql": sql}
    except duckdb.Error as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "sql": sql}
    finally:
        if timer is not None:
            timer.cancel()
        con.close()
    truncated = len(rows) > row_cap
    rows = rows[:row_cap]
    return {"ok": True, "columns": columns, "rows": [dict(zip(columns, r)) for r in rows],
            "row_count": len(rows), "truncated": truncated, "sql": sql.strip()}
