"""
HTTP front for hybrid search: index status (coverage + the self-retrieval evaluation) and paper
search. Runs from the same image as the batch jobs. A new connection attaches the serving database
and the search store per request; results are cached on disk per (fingerprint, query, filters).
"""
import hashlib
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import config, data, embed as E, search as SR, store as S, vectors as V

log = logging.getLogger("dblp.search.server")

encoder = E.Encoder()


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.fingerprint = None
        self.meta = {}
        self.error = None

    def load(self):
        """Fast: attach + read metadata only. Kept quick so it never blocks app startup or the
        periodic watcher on the slow one-time vector load - see `warm_soon`."""
        con, dump_meta = data.connect()
        try:
            self.meta = S.attach_store(con, dump_meta)
            self.fingerprint = self.meta.get("fingerprint")
            self.error = None
            log.info("search store attached, fingerprint %s", self.fingerprint)
        except Exception as e:   # no store yet is a normal state, not a crash
            self.fingerprint, self.error = None, str(e)
            log.warning("no search store yet: %s", e)
        finally:
            con.close()

    def warm_soon(self):
        """Pre-load the vector cache in the background, so a live user's first query never pays for
        it - a no-op once already warm. Runs off the startup/watch path so /search/health stays fast."""
        def run():
            if self.fingerprint is None:
                return
            con, _ = data.connect()
            try:
                V.warm(con, self.fingerprint)
            except Exception:
                log.exception("vector warm-up failed")
            finally:
                con.close()
        threading.Thread(target=run, name="search-warm", daemon=True).start()

    def watch(self):
        def loop():
            while True:
                time.sleep(300)
                self.load()
                self.warm_soon()
        threading.Thread(target=loop, name="search-watch", daemon=True).start()

    def cache_path(self, *parts):
        tag = hashlib.sha1("|".join([str(self.fingerprint), *map(str, parts)]).encode()).hexdigest()[:16]
        d = config.MODELS_DIR / "search-cache"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{tag}.json"


state = State()


@asynccontextmanager
async def lifespan(_app):
    state.load()
    state.warm_soon()
    state.watch()
    yield


app = FastAPI(title="dblp search", lifespan=lifespan, docs_url="/search/docs", openapi_url="/search/openapi.json")


@app.get("/search/health")
def health():
    return {"ok": True, "store": state.fingerprint is not None}


@app.get("/search/status")
def status():
    if state.fingerprint is None:
        return {"available": False, "error": state.error}
    con, dump_meta = data.connect()
    try:
        S.attach_store(con, dump_meta)
        progress = V.progress(con, state.fingerprint)
    finally:
        con.close()
    ev_path = S.eval_path(state.fingerprint)
    evaluation = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    # written by `chat.cli search-eval`: the paraphrase test, which needs a model to write the queries
    para_path = ev_path.parent / f"paraphrase-{state.fingerprint}.json"
    paraphrase = json.loads(para_path.read_text(encoding="utf-8")) if para_path.exists() else None
    return {
        "available": True,
        "dump": {"fingerprint": state.fingerprint, "built_at": state.meta.get("built_at"),
                 "papers": state.meta.get("papers"), "model": state.meta.get("model"),
                 "first_year": state.meta.get("first_year")},
        "index": progress, "evaluation": evaluation, "paraphrase_evaluation": paraphrase,
    }


@app.get("/search/papers")
def papers(q: str = Query(..., min_length=3, max_length=SR.MAX_TITLE), kind: Optional[str] = Query(None),
          frm: Optional[int] = Query(None, alias="from", ge=1900, le=2100),
          to: Optional[int] = Query(None, ge=1900, le=2100),
          top: int = Query(20, ge=1, le=50), refresh: bool = False,
          dense: bool = Query(True, description="set false to answer with words alone, for comparison"),
          sparse: bool = Query(True, description="set false to answer with the embeddings alone")):
    if state.fingerprint is None:
        raise HTTPException(503, detail=f"No search index yet ({state.error}). Run: python -m search.cli store")
    with state.lock:   # one search at a time: a dense query is a full pass over the index
        con, dump_meta = data.connect()
        try:
            S.attach_store(con, dump_meta)
            # the vector file exists (pre-allocated, zero-filled) the moment `build-index` starts,
            # long before it is useful: rather than serve a partial, order-of-build-not-relevance
            # slice of dense results, stay BM25 + exact-word only until embedding is complete, and
            # fold that flag into the cache key so a cached "not ready" answer cannot outlive the
            # build finishing
            complete = V.progress(con, state.fingerprint)["complete"] and dense
            path = state.cache_path(q.strip().lower(), kind, frm, to, top, complete, sparse)
            if refresh:
                path.unlink(missing_ok=True)
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))
                cached["cached"] = True
                return JSONResponse(cached)
            t = time.time()
            out = SR.search(con, state.fingerprint, encoder, q, top=top, kind=kind, year_from=frm,
                            year_to=to, dense=complete, sparse=sparse)
            if not complete:
                out["dense_error"] = ("answered with words alone, as asked" if not dense
                                      else "the embedding index is still building")
        finally:
            con.close()
    if "error" in out:
        raise HTTPException(422, detail=out["error"])
    out["computed_in_seconds"] = round(time.time() - t, 2)
    out["cached"] = False
    try:
        path.write_text(json.dumps(out), encoding="utf-8")
    except OSError as e:
        log.warning("could not cache %s: %s", path, e)
    return out
