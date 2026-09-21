"""
The dense index: one embedding per eligible paper, stored on disk as float16 and memory-mapped
while *building* (so a multi-hour, resumable job never needs the whole array resident).

Serving is different: the file is small enough (a few GB) to hold in RAM against the VM's budget,
and re-reading + re-casting it from disk block by block on every request measured 15-20s per query
in practice - re-scanning 5.36M rows off disk is not something to repeat per search. So the search
path loads the array once into a process-wide float32 cache on first use and reuses it; a single
BLAS matmul against RAM then answers a query in well under a second. Building still goes through
`open_vectors`/memmap, untouched, so a completed embedding run never needs to be redone by this.

Building is resumable and crash-safe: which papers are already embedded lives in a small separate
database (`prog`), committed every few batches, independent of the (also separate) sparse store.
Re-running the same command after an interruption picks up where it left off.
"""
import logging
import threading
import time
from pathlib import Path

import numpy as np

from . import config, store as S

log = logging.getLogger("dblp.search.vectors")

DTYPE = np.dtype(config.VECTOR_DTYPE)

_cache_lock = threading.Lock()
_cache = {}   # fingerprint -> (n, float32 ndarray), the whole index held in RAM once loaded


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


def _load_full(fingerprint, n):
    """The whole vector index as a resident float32 array, loaded once per process. Only ever
    called once the index is complete (the caller gates on `progress()["complete"]`), so the cached
    array is never a stale partial snapshot. Only one fingerprint's array is kept at a time - in
    production there is only ever one active dump, and this avoids leaking RAM across redeploys."""
    with _cache_lock:
        cached = _cache.get(fingerprint)
        if cached is not None and cached[0] == n:
            return cached[1]
        path = S.vectors_path(fingerprint)
        if not path.exists():
            raise FileNotFoundError(f"no vector file at {path}; run build-index first")
        t0 = time.time()
        mm = np.memmap(path, dtype=DTYPE, mode="r", shape=(n, config.EMBED_DIM))
        arr = np.asarray(mm, dtype=np.float32)   # one-time full read + cast into a real ndarray
        del mm
        log.info("loaded vector index into memory: %s rows, %.2f GB, %.1fs",
                 f"{n:,}", arr.nbytes / 1e9, time.time() - t0)
        _cache.clear()
        _cache[fingerprint] = (n, arr)
        return arr


def warm(con, fingerprint):
    """Pre-load the vector cache if the index is complete, so the first live query doesn't pay for
    it. Safe to call whenever; a no-op if the index isn't ready or is already cached."""
    try:
        if progress(con, fingerprint)["complete"]:
            n = con.execute("SELECT count(*) FROM x.paper").fetchone()[0]
            _load_full(fingerprint, n)
    except FileNotFoundError:
        pass


def search(con, fingerprint, qvec, top=None, kind=None, year_from=None, year_to=None, min_sim=0.0):
    """[(pid, cosine)], best first. Vectors are L2-normalised, so a dot product is a cosine
    similarity; takes the raw top `TOP_DENSE` among those actually above `min_sim`, then applies the
    kind/year filters against just those - cheap, and generous enough that a filter rarely starves
    the result. The threshold matters once the index is smaller than `TOP_DENSE` (never true on the
    real corpus): without it, "top N" would pad the list with zero-or-negative-similarity rows that
    are not really matches at all."""
    n = con.execute("SELECT count(*) FROM x.paper").fetchone()[0]
    vectors = _load_full(fingerprint, n)
    q = np.asarray(qvec, dtype=np.float32)
    sims = vectors @ q

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
