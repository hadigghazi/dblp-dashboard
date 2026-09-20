"""
The dense index: one embedding per eligible paper, memory-mapped rather than loaded whole into
RAM. The VM runs several services in a fixed memory budget, so serving trades a page-fault on first
touch of a vector block for not holding a multi-gigabyte array resident all the time.

Building is resumable and crash-safe: which papers are already embedded lives in a small separate
database (`prog`), committed every few batches, independent of the (also separate) sparse store.
Re-running the same command after an interruption picks up where it left off.
"""
import logging
import time
from pathlib import Path

import numpy as np

from . import config, store as S

log = logging.getLogger("dblp.search.vectors")

DTYPE = np.dtype(config.VECTOR_DTYPE)
BLOCK = 200_000   # rows per chunk when scoring a query against the whole index, bounding peak memory


def progress_path(fingerprint) -> Path:
    return config.MODELS_DIR / f"search-progress-{fingerprint}.duckdb"


def open_vectors(fingerprint, n, mode):
    path = S.vectors_path(fingerprint)
    if mode == "r":
        if not path.exists():
            raise FileNotFoundError(f"no vector file at {path}; run build-index first")
        return np.memmap(path, dtype=DTYPE, mode="r", shape=(n, config.EMBED_DIM))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.memmap(path, dtype=DTYPE, mode="w+", shape=(n, config.EMBED_DIM))
        arr.flush()
        del arr
    return np.memmap(path, dtype=DTYPE, mode="r+", shape=(n, config.EMBED_DIM))


def _attach_progress(con, fingerprint):
    path = progress_path(fingerprint)
    attached = {r[0] for r in con.execute("SELECT database_name FROM duckdb_databases()").fetchall()}
    if "prog" not in attached:
        path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"ATTACH '{path}' AS prog")
    con.execute("CREATE TABLE IF NOT EXISTS prog.done (pid INTEGER PRIMARY KEY)")
    return path


def progress(con, fingerprint):
    """How much of the index is built, without opening the (possibly huge) vector file."""
    n = con.execute("SELECT count(*) FROM x.paper").fetchone()[0]
    _attach_progress(con, fingerprint)
    done = con.execute("SELECT count(*) FROM prog.done").fetchone()[0]
    return {"papers": int(n), "embedded": int(done),
            "share": round(done / n, 4) if n else None, "complete": bool(done >= n)}


def build(con, fingerprint, encoder, batch_size=None, checkpoint_every=None):
    """Encode every not-yet-embedded paper's title, in row order, writing directly into the vector
    file at that paper's row. Interrupting and re-running loses at most one checkpoint's work."""
    batch_size = batch_size or config.ENCODE_BATCH
    checkpoint_every = checkpoint_every or config.CHECKPOINT_EVERY
    n = con.execute("SELECT count(*) FROM x.paper").fetchone()[0]
    vectors = open_vectors(fingerprint, n, "r+")
    _attach_progress(con, fingerprint)

    n_done_start = con.execute("SELECT count(*) FROM prog.done").fetchone()[0]
    n_remaining = n - n_done_start
    log.info("resuming: %s of %s papers already embedded, %s remaining",
             f"{n_done_start:,}", f"{n:,}", f"{n_remaining:,}")
    if n_remaining <= 0:
        return {"papers": int(n), "embedded": int(n_done_start), "new_this_run": 0, "batches": 0, "seconds": 0.0}

    # a DuckDB connection holds one active result at a time: the INSERTs below (on `con`) would
    # otherwise invalidate this SELECT's cursor after the first batch. A child cursor shares the
    # same attached databases but keeps its own result stream.
    reader = con.cursor()
    reader.execute("""
        SELECT p.row, p.pid, p.title FROM x.paper p
        WHERE p.pid NOT IN (SELECT pid FROM prog.done)
        ORDER BY p.row""")
    t0 = time.time()
    n_new, batches = 0, 0
    while True:
        chunk = reader.fetchmany(batch_size)
        if not chunk:
            break
        rows = np.asarray([c[0] for c in chunk], dtype=np.int64)
        pids = [c[1] for c in chunk]
        titles = [c[2] for c in chunk]
        vecs = encoder.encode_docs(titles).astype(DTYPE)
        vectors[rows, :] = vecs
        con.executemany("INSERT INTO prog.done VALUES (?)", [(p,) for p in pids])
        n_new += len(chunk)
        batches += 1
        if batches % checkpoint_every == 0 or n_new >= n_remaining:
            vectors.flush()
            con.execute("CHECKPOINT prog")
            elapsed = time.time() - t0
            rate = n_new / elapsed if elapsed > 0 else 0.0
            eta_min = (n_remaining - n_new) / rate / 60 if rate > 0 else float("inf")
            log.info("embedded %s/%s new this run (%s/%s total), %.1f titles/s, eta %.0fm",
                     f"{n_new:,}", f"{n_remaining:,}", f"{n_done_start + n_new:,}", f"{n:,}", rate, eta_min)
    vectors.flush()
    con.execute("CHECKPOINT prog")
    total = n_done_start + n_new
    log.info("run finished: %s/%s papers embedded (%s new, %.0fs)", f"{total:,}", f"{n:,}", f"{n_new:,}", time.time() - t0)
    return {"papers": int(n), "embedded": int(total), "new_this_run": int(n_new), "batches": batches,
            "seconds": round(time.time() - t0, 1)}


def search(con, fingerprint, qvec, top=None, kind=None, year_from=None, year_to=None, min_sim=0.0):
    """[(pid, cosine)], best first. Scores the whole index in blocks (vectors are L2-normalised, so
    a dot product is a cosine similarity), takes the raw top `TOP_DENSE` among those actually above
    `min_sim`, then applies the kind/year filters against just those - cheap, and generous enough
    that a filter rarely starves the result. The threshold matters once the index is smaller than
    `TOP_DENSE` (never true on the real corpus): without it, "top N" would pad the list with
    zero-or-negative-similarity rows that are not really matches at all."""
    n = con.execute("SELECT count(*) FROM x.paper").fetchone()[0]
    vectors = open_vectors(fingerprint, n, "r")
    q = np.asarray(qvec, dtype=np.float32)
    sims = np.empty(n, dtype=np.float32)
    for start in range(0, n, BLOCK):
        end = min(start + BLOCK, n)
        sims[start:end] = vectors[start:end].astype(np.float32) @ q

    raw_top = min(config.TOP_DENSE, n)
    idx = np.argpartition(-sims, raw_top - 1)[:raw_top] if raw_top < n else np.arange(n)
    idx = idx[np.argsort(-sims[idx])]
    idx = idx[sims[idx] > min_sim]

    where = ["row IN (SELECT unnest(?::BIGINT[]))"]
    params = [idx.tolist()]
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if year_from is not None:
        where.append("year >= ?")
        params.append(int(year_from))
    if year_to is not None:
        where.append("year <= ?")
        params.append(int(year_to))
    meta = {r[0]: r[1] for r in con.execute(
        f"SELECT row, pid FROM x.paper WHERE {' AND '.join(where)}", params).fetchall()}

    out = []
    for i in idx:
        pid = meta.get(int(i))
        if pid is not None:
            out.append((int(pid), float(sims[i])))
            if len(out) >= (top or config.TOP_DENSE):
                break
    return out
