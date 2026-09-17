"""
HTTP front for the disambiguation model, so the dashboard can show proposed splits.

Runs from the same image as the batch jobs. It loads the most recent model once, opens the api's
serving database read-only, and answers three questions: is a model available and how good is it,
which bins are the biggest, and how should this bin be split. Splits are cached on disk per
(dump, bin, cap) - a big bin takes tens of seconds to score, and the answer does not change until
the dump or the model does.
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

from . import config, data, model as M, predict as P

log = logging.getLogger("dblp.ml.server")

MAX_ON_DEMAND = 150   # papers per bin when computed on request; the batch job can go higher


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = self.thresholds = self.features = self.model_dir = None
        self.metrics = {}
        self.fingerprint = None
        self.error = None

    def load(self):
        try:
            self.model, self.thresholds, self.features, self.model_dir = M.load()
            self.metrics = json.loads((self.model_dir / M.METRICS_FILE).read_text(encoding="utf-8"))
            self.fingerprint = self.metrics.get("dump", {}).get("fingerprint")
            self.error = None
            log.info("loaded %s", self.model_dir.name)
        except Exception as e:  # no model yet is a normal state, not a crash
            self.model, self.error = None, str(e)
            log.warning("no model loaded: %s", e)

    def watch(self):
        def loop():
            while True:
                time.sleep(300)
                try:
                    _, _, _, d = M.load()
                    if d != self.model_dir:
                        with self.lock:
                            self.load()
                except Exception:
                    pass
        threading.Thread(target=loop, name="model-watch", daemon=True).start()


state = State()


@asynccontextmanager
async def lifespan(_app):
    state.load()
    state.watch()
    yield


app = FastAPI(title="dblp disambiguation", lifespan=lifespan, docs_url="/ml/docs", openapi_url="/ml/openapi.json")


def _model_card():
    m = state.metrics.get("metrics", {}) if state.metrics else {}
    test = m.get("test", {})
    clustering = test.get("clustering", {})
    return {
        "available": state.model is not None,
        "error": state.error,
        "trained_at": state.metrics.get("trained_at") if state.metrics else None,
        "dump": state.metrics.get("dump") if state.metrics else None,
        "thresholds": state.thresholds,
        "assignment": clustering.get("assignment"),
        "clustering": {k: clustering.get(k) for k in ("blocks", "papers", "b3_f1", "b3_f1_overlap_baseline", "ari",
                                                      "cluster_count_ratio")},
        "clustering_bin_like": test.get("clustering_bin_like"),
        "pairwise": test.get("pairwise"),
        "pairwise_overlap_baseline": test.get("pairwise_overlap_baseline"),
        "feature_importance": (m.get("feature_importance") or [])[:8],
        "dataset": m.get("dataset"),
        "dropped_features": m.get("dropped_features", []),
    }


def _cache_path(key, cap):
    tag = hashlib.sha1(f"{state.fingerprint}|{state.model_dir.name}|{key}|{cap}".encode()).hexdigest()[:16]
    d = config.MODELS_DIR / "suggestions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.json"


@app.get("/ml/health")
def health():
    return {"ok": True, "model": state.model is not None}


@app.get("/ml/status")
def status():
    return _model_card()


@app.get("/ml/bins")
def bins(top: int = Query(40, ge=1, le=200), q: Optional[str] = Query(None, max_length=100)):
    con, _ = data.connect()
    try:
        return {"bins": data.largest_bins(con, top, q)}
    finally:
        con.close()


@app.get("/ml/bin")
def split(key: str = Query(..., max_length=200), max_papers: int = Query(MAX_ON_DEMAND, ge=10, le=600),
          refresh: bool = False):
    if state.model is None:
        raise HTTPException(503, detail=f"No trained model yet ({state.error}). Run: python -m ml.cli train")
    max_papers = min(max_papers, MAX_ON_DEMAND)
    path = _cache_path(key, max_papers)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["cached"] = True
        return JSONResponse(cached)
    with state.lock:   # one prediction at a time: each one is a burst of DuckDB + numpy work
        con, _ = data.connect()
        try:
            t = time.time()
            out = P.split_bin(con, key, state.model, state.thresholds, state.features, state.model_dir,
                              max_papers=max_papers)
        finally:
            con.close()
    if "error" in out:
        raise HTTPException(404, detail=out["error"])
    out["computed_in_seconds"] = round(time.time() - t, 1)
    out["cached"] = False
    try:
        path.write_text(json.dumps(out), encoding="utf-8")
    except OSError as e:
        log.warning("could not cache %s: %s", path, e)
    return out
