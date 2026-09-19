"""
HTTP front for the models, so the dashboard can show their output.

Runs from the same image as the batch jobs. Each model is loaded once (and re-checked every five
minutes, so a retrain on the VM reaches the site without a redeploy), the api's serving database is
opened read-only per request, and heavy answers are cached on disk per (dump, model, input) - they
do not change until the dump or the model does.

  /ml/status, /ml/bins, /ml/bin          author disambiguation
  /ml/links/status, /ml/links            co-author link prediction
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
from .links import graph as LG, model as LM, predict as LP

log = logging.getLogger("dblp.ml.server")

MAX_ON_DEMAND = 150   # papers per bin when computed on request; the batch job can go higher


class State:
    """One loaded model. `loader` returns (payload dict, model_dir); payload holds whatever the
    endpoints need."""

    def __init__(self, name, loader, cache_dir):
        self.name, self.loader, self.cache_dir = name, loader, cache_dir
        self.lock = threading.Lock()
        self.payload, self.model_dir, self.metrics, self.error = None, None, {}, None

    @property
    def available(self):
        return self.payload is not None

    @property
    def fingerprint(self):
        return (self.metrics.get("dump") or {}).get("fingerprint") if self.metrics else None

    def load(self):
        try:
            self.payload, self.model_dir = self.loader()
            self.metrics = json.loads((self.model_dir / "metrics.json").read_text(encoding="utf-8"))
            self.error = None
            log.info("%s: loaded %s", self.name, self.model_dir.name)
        except Exception as e:  # no model yet is a normal state, not a crash
            self.payload, self.error = None, str(e)
            log.warning("%s: no model loaded: %s", self.name, e)

    def watch(self):
        def loop():
            while True:
                time.sleep(300)
                try:
                    _, d = self.loader()
                    if d != self.model_dir:
                        with self.lock:
                            self.load()
                except Exception:
                    pass
        threading.Thread(target=loop, name=f"{self.name}-watch", daemon=True).start()

    def cache_path(self, *parts):
        tag = hashlib.sha1("|".join([str(self.fingerprint), self.model_dir.name, *map(str, parts)]).encode()).hexdigest()[:16]
        d = config.MODELS_DIR / self.cache_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{tag}.json"


def _load_disambiguation():
    model, thresholds, features, d = M.load()
    return {"model": model, "thresholds": thresholds, "features": features}, d


def _load_links():
    model, features, calibration, d = LM.load()
    return {"model": model, "features": features, "calibration": calibration}, d


disambiguation = State("disambiguation", _load_disambiguation, "suggestions")
links = State("links", _load_links, "links-suggestions")


@asynccontextmanager
async def lifespan(_app):
    for s in (disambiguation, links):
        s.load()
        s.watch()
    yield


app = FastAPI(title="dblp models", lifespan=lifespan, docs_url="/ml/docs", openapi_url="/ml/openapi.json")


def _cached(state_, path, compute):
    """Serve from the on-disk cache, else compute under the model's lock and store."""
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["cached"] = True
        return JSONResponse(cached)
    with state_.lock:   # one prediction at a time: each one is a burst of DuckDB + numpy work
        t = time.time()
        out = compute()
    if "error" in out:
        raise HTTPException(404, detail=out["error"])
    out["computed_in_seconds"] = round(time.time() - t, 1)
    out["cached"] = False
    try:
        path.write_text(json.dumps(out), encoding="utf-8")
    except OSError as e:
        log.warning("could not cache %s: %s", path, e)
    return out


@app.get("/ml/health")
def health():
    return {"ok": True, "model": disambiguation.available, "links": links.available}


# --------------------------------------------------------------------------- disambiguation
def _model_card():
    m = disambiguation.metrics.get("metrics", {}) if disambiguation.metrics else {}
    test = m.get("test", {})
    clustering = test.get("clustering", {})
    return {
        "available": disambiguation.available,
        "error": disambiguation.error,
        "trained_at": disambiguation.metrics.get("trained_at") if disambiguation.metrics else None,
        "dump": disambiguation.metrics.get("dump") if disambiguation.metrics else None,
        "thresholds": (disambiguation.payload or {}).get("thresholds"),
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
    if not disambiguation.available:
        raise HTTPException(503, detail=f"No trained model yet ({disambiguation.error}). Run: python -m ml.cli train")
    max_papers = min(max_papers, MAX_ON_DEMAND)
    path = disambiguation.cache_path(key, max_papers)
    if refresh:
        path.unlink(missing_ok=True)

    def compute():
        con, _ = data.connect()
        try:
            pl = disambiguation.payload
            return P.split_bin(con, key, pl["model"], pl["thresholds"], pl["features"], disambiguation.model_dir,
                               max_papers=max_papers)
        finally:
            con.close()
    return _cached(disambiguation, path, compute)


# --------------------------------------------------------------------------- link prediction
def _links_card():
    m = links.metrics.get("metrics", {}) if links.metrics else {}
    test = m.get("test", {})
    return {
        "available": links.available,
        "error": links.error,
        "trained_at": links.metrics.get("trained_at") if links.metrics else None,
        "dump": links.metrics.get("dump") if links.metrics else None,
        "dataset": m.get("dataset"),
        "test": {k: test.get(k) for k in ("snapshot", "horizon", "pooled", "ranking")},
        "baselines": test.get("baselines"),
        "calibration": links.metrics.get("calibration") if links.metrics else None,
        "feature_importance": (m.get("feature_importance") or [])[:10],
        "dropped_features": m.get("dropped_features", []),
    }


@app.get("/ml/links/status")
def links_status():
    return _links_card()


@app.get("/ml/links")
def links_suggest(key: str = Query(..., max_length=200), top: int = Query(10, ge=1, le=LP.MAX_TOP),
                  refresh: bool = False):
    if not links.available:
        raise HTTPException(503, detail=f"No trained link model yet ({links.error}). Run: python -m ml.links.cli train")
    path = links.cache_path(key, top)
    if refresh:
        path.unlink(missing_ok=True)

    def compute():
        con, meta = data.connect()
        try:
            LG.attach_store(con, meta)
            pl = links.payload
            return LP.suggest(con, key, top, pl["model"], pl["features"], pl["calibration"], links.model_dir)
        except FileNotFoundError as e:
            return {"error": str(e)}
        finally:
            con.close()
    return _cached(links, path, compute)
