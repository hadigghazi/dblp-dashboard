"""dblp Explorer API: live queries over the dblp dump."""
import logging
import math
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import config, queries as Q, snapshots
from .serving import NotReady, serving

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dblp.api")

@asynccontextmanager
async def lifespan(_app):
    serving.on_ready.append(warm_cache)
    serving.start_background()
    yield
    serving.shutdown()


app = FastAPI(title="dblp Explorer API", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)

# ---------------------------------------------------------------- caching ----
_cache: "OrderedDict[tuple, object]" = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 512
_heavy = threading.BoundedSemaphore(config.HEAVY_QUERY_SLOTS)


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def cached(name, fn, *args, heavy=False, **kwargs):
    """Run fn(cursor, *args) once per (data generation, name, args); results are immutable per dump."""
    key = (serving.generation, name, _freeze(args), _freeze(kwargs))
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    if heavy and not _heavy.acquire(timeout=60):
        raise HTTPException(503, detail="Too many heavy queries running; try again in a moment")
    try:
        cur = serving.cursor()
        try:
            t = time.time()
            result = fn(cur, *args, **kwargs)
            log.info("%s %s %.2fs", name, args, time.time() - t)
        finally:
            cur.close()
    finally:
        if heavy:
            _heavy.release()
    with _cache_lock:
        _cache[key] = result
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return result


def meta():
    return serving.meta


@app.exception_handler(NotReady)
def not_ready(_request, exc: NotReady):
    return JSONResponse(status_code=503, content={"detail": exc.status.get("message", "Not ready"),
                                                  "status": exc.status})


@app.exception_handler(Q.BadRequest)
def bad_request(_request, exc: Q.BadRequest):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _finite(v):
    """JSON has no NaN/Infinity: a degenerate statistic (e.g. a flat series) becomes null."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


def with_meta(data):
    return {"data": _finite(data), "generation": serving.generation,
            "latest_mdate": serving.meta.get("latest_mdate")}


def _periods(text):
    try:
        out = []
        for part in text.split(","):
            a, b = part.split("-")
            a, b = int(a), int(b)
            if not (1900 <= a <= b <= 2100):
                raise ValueError
            out.append((a, b))
        return out
    except ValueError:
        raise Q.BadRequest("periods look like 2001-2005,2011-2015")


def _window(text, name):
    p = _periods(text)
    if len(p) != 1:
        raise Q.BadRequest(f"{name} must be one period like 2011-2015")
    return p[0]


# ---------------------------------------------------------------- status ----
@app.get("/api/status")
def status():
    fits = snapshots.power_law_fits()
    return {
        "status": serving.status,
        "meta": serving.meta,
        "generation": serving.generation,
        "jobs": {
            "network": (snapshots.network().get("source") or None),
            "openalex": (snapshots.openalex().get("source") or None),
            "statistics": fits.get("source") if fits.get("available") else None,
        },
    }


@app.get("/api/health")
def health():
    return {"ok": True, "state": serving.status.get("state")}


# ---------------------------------------------------------------- overview ----
@app.get("/api/overview")
def overview():
    return with_meta(cached("overview", Q.overview, meta(), heavy=True))


# ---------------------------------------------------------------- publishing ----
@app.get("/api/publishing/growth")
def growth(frm: Optional[int] = Query(None, alias="from", ge=1900, le=2100),
           to: Optional[int] = Query(None, ge=1900, le=2100)):
    return with_meta(cached("growth", Q.growth, meta(), frm, to))


@app.get("/api/publishing/teams")
def teams(frm: Optional[int] = Query(None, alias="from", ge=1900, le=2100),
          to: Optional[int] = Query(None, ge=1900, le=2100)):
    return with_meta(cached("teams", Q.teams, meta(), frm, to))


@app.get("/api/publishing/team-boxes")
def team_boxes(frm: int = Query(1970, alias="from", ge=1900, le=2100), to: Optional[int] = Query(None, ge=1900, le=2100),
               step: int = Query(5, ge=1, le=20)):
    return with_meta(cached("team_boxes", Q.team_boxes, meta(), frm, to, step))


@app.get("/api/publishing/metadata")
def metadata(frm: Optional[int] = Query(None, alias="from", ge=1900, le=2100),
             to: Optional[int] = Query(None, ge=1900, le=2100)):
    return with_meta(cached("metadata", Q.metadata_trends, meta(), frm, to))


@app.get("/api/publishing/growth-rates")
def growth_rates(frm: int = Query(1990, alias="from", ge=1950, le=2100),
                 to: int = Query(2023, ge=1950, le=2100),
                 split: int = Query(2005, ge=1950, le=2100)):
    return with_meta(cached("growth_rates", Q.growth_rates, meta(), frm, to, split))


# ---------------------------------------------------------------- identity ----
@app.get("/api/identity/homonyms")
def homonyms(top: int = Query(15, ge=1, le=100)):
    return with_meta(cached("homonyms", Q.homonyms, top))


@app.get("/api/identity/affiliation")
def affiliation():
    return with_meta(cached("affiliation", Q.affiliation))


@app.get("/api/identity/newcomers")
def newcomers(frm: Optional[int] = Query(None, alias="from", ge=1900, le=2100),
              to: Optional[int] = Query(None, ge=1900, le=2100)):
    return with_meta(cached("newcomers", Q.newcomers, meta(), frm, to, heavy=True))


@app.get("/api/identity/cohorts")
def cohorts(frm: int = Query(1975, alias="from", ge=1900, le=2100), to: int = Query(2005, ge=1900, le=2100),
            step: int = Query(5, ge=1, le=20)):
    return with_meta(cached("cohorts", Q.cohorts, meta(), frm, to, step))


@app.get("/api/identity/positions")
def positions(min_authors: int = Query(3, ge=2, le=50)):
    return with_meta(cached("positions", Q.positions, min_authors, heavy=True))


@app.get("/api/identity/unidentified-by-position")
def unidentified_by_position(since: int = Query(2015, ge=1950, le=2100), min_authors: int = Query(3, ge=2, le=50)):
    return with_meta(cached("unid_pos", Q.unidentified_by_position, since, min_authors, heavy=True))


@app.get("/api/identity/alphabetical")
def alphabetical(since: int = Query(2000, ge=1950, le=2100), min_papers: int = Query(2000, ge=50, le=1_000_000),
                 most: int = Query(10, ge=1, le=50), least: int = Query(5, ge=1, le=50)):
    return with_meta(cached("alphabetical", Q.alphabetical, since, min_papers, most, least, heavy=True))


# ---------------------------------------------------------------- network (job output) ----
@app.get("/api/network")
def network():
    return {"data": _finite(snapshots.network())}


# ---------------------------------------------------------------- titles ----
@app.get("/api/titles/terms")
def terms(terms: str = Query("llm,deep,neural,transformer", max_length=300),
          frm: Optional[int] = Query(None, alias="from", ge=1900, le=2100),
          to: Optional[int] = Query(None, ge=1900, le=2100)):
    term_list = tuple(t for t in terms.split(",") if t.strip())
    return with_meta(cached("terms", Q.terms, meta(), term_list, frm, to, heavy=True))


@app.get("/api/titles/style")
def title_style():
    return with_meta(cached("title_style", Q.title_style, meta(), heavy=True))


@app.get("/api/titles/words")
def words(old: str = Query("2011-2015"), new: str = Query("2021-2025"),
          direction: str = Query("rising"), min_titles: int = Query(1500, ge=1, le=1_000_000),
          limit: int = Query(15, ge=1, le=100), skip_noise: bool = True):
    return with_meta(cached("words", Q.words, _window(old, "old"), _window(new, "new"), direction,
                            min_titles, limit, skip_noise))


# ---------------------------------------------------------------- venues ----
@app.get("/api/venues/lifespans")
def lifespans():
    return with_meta(cached("lifespans", Q.lifespans, meta()))


@app.get("/api/venues/concentration")
def concentration(frm: int = Query(1980, alias="from", ge=1950, le=2100), to: Optional[int] = Query(None, ge=1950, le=2100),
                  top: int = Query(10, ge=1, le=100), step: int = Query(5, ge=1, le=10)):
    return with_meta(cached("concentration", Q.concentration, meta(), frm, to, top, step))


@app.get("/api/venues/publishers")
def publishers(periods: str = Query("2001-2005,2011-2015,2021-2025"), top: int = Query(8, ge=1, le=30)):
    return with_meta(cached("publishers", Q.publishers, tuple(_periods(periods)), top))


@app.get("/api/venues/doi-gaps")
def doi_gaps(min_papers: int = Query(2000, ge=10, le=10_000_000), max_doi_pct: float = Query(1.0, ge=0, le=100),
             limit: int = Query(12, ge=1, le=100)):
    return with_meta(cached("doi_gaps", Q.doi_gaps, min_papers, max_doi_pct, limit))


@app.get("/api/venues/search")
def venue_search(q: str = Query("", max_length=100), kind: Optional[str] = Query(None, pattern="^(journal|conference)$"),
                 limit: int = Query(25, ge=1, le=100)):
    return with_meta(cached("venue_search", Q.venue_search, q, kind, limit))


@app.get("/api/venues/detail")
def venue_detail(sid: str = Query(..., max_length=200)):
    data = cached("venue_detail", Q.venue_detail, sid, heavy=True)
    if data is None:
        raise HTTPException(404, detail=f"No venue series {sid!r}")
    return with_meta(data)


@app.get("/api/venues/profile")
def venue_profile(sid: str = Query(..., max_length=200)):
    # the ranks need every series; that table is built once per dump and each sid picks its row
    profiles = cached("venue_profiles", Q.venue_profiles, meta(), heavy=True)
    data = Q.venue_profile(profiles, sid)
    if data is None:
        raise HTTPException(404, detail=f"No venue series {sid!r} with at least {profiles['min_papers']} papers")
    return with_meta(data)


@app.get("/api/venues/treemap")
def venue_treemap(period: str = Query("2021-2025"), publishers: int = Query(10, ge=2, le=20),
                  series: int = Query(12, ge=1, le=40)):
    frm, to = _window(period, "period")
    return with_meta(cached("venue_treemap", Q.venue_treemap, frm, to, publishers, series, heavy=True))


@app.get("/api/venues/scatter")
def venue_scatter(min_papers: int = Query(300, ge=10, le=1_000_000)):
    return with_meta(cached("venue_scatter", Q.venue_scatter, meta(), min_papers))


# ---------------------------------------------------------------- quality ----
@app.get("/api/quality/coverage")
def coverage():
    return with_meta(cached("coverage", Q.coverage, heavy=True))


@app.get("/api/quality/page-formats")
def page_formats():
    return with_meta(cached("page_formats", Q.page_formats, heavy=True))


@app.get("/api/quality/theses")
def theses(limit: int = Query(12, ge=1, le=50)):
    return with_meta(cached("theses", Q.theses, limit))


@app.get("/api/quality/research-data")
def research_data(frm: int = Query(2015, alias="from", ge=1950, le=2100), to: Optional[int] = Query(None, ge=1950, le=2100)):
    return with_meta(cached("research_data", Q.research_data, meta(), frm, to))


# ---------------------------------------------------------------- enrichment (job output) ----
@app.get("/api/enrichment/openalex")
def openalex():
    return {"data": _finite(snapshots.openalex())}


# ---------------------------------------------------------------- tails ----
@app.get("/api/tails")
def tails(panel: str = Query("papers_per_author")):
    fits = snapshots.power_law_fits()
    data = cached("tails", Q.tails, panel, fits.get("fits") if fits.get("available") else None)
    return with_meta({**data, "fit_source": fits.get("source")})


@app.get("/api/tails/joint")
def joint_density(bins_per_decade: int = Query(6, ge=2, le=12)):
    return with_meta(cached("joint_density", Q.joint_density, bins_per_decade, heavy=True))


# ---------------------------------------------------------------- explore ----
@app.get("/api/authors/search")
def author_search(q: str = Query(..., max_length=100), limit: int = Query(30, ge=1, le=100)):
    return with_meta(cached("author_search", Q.author_search, q, limit, heavy=True))


@app.get("/api/authors/detail")
def author_detail(key: str = Query(..., max_length=200)):
    data = cached("author_detail", Q.author_detail, key, heavy=True)
    if data is None:
        raise HTTPException(404, detail=f"No author page {key!r}")
    return with_meta(data)


@app.get("/api/authors/ego")
def author_ego(key: str = Query(..., max_length=200), limit: int = Query(30, ge=5, le=60)):
    data = cached("author_ego", Q.author_ego, key, limit, heavy=True)
    if data is None:
        raise HTTPException(404, detail=f"No author page {key!r}")
    return with_meta(data)


@app.get("/api/papers/search")
def paper_search(q: str = Query(..., max_length=200), kind: Optional[str] = Query(None, max_length=20),
                 frm: Optional[int] = Query(None, alias="from", ge=1900, le=2100),
                 to: Optional[int] = Query(None, ge=1900, le=2100), limit: int = Query(30, ge=1, le=100)):
    return with_meta(cached("paper_search", Q.paper_search, q, kind, frm, to, limit, heavy=True))


@app.get("/api/papers/detail")
def paper_detail(key: str = Query(..., max_length=300)):
    data = cached("paper_detail", Q.paper_detail, key, heavy=True)
    if data is None:
        raise HTTPException(404, detail=f"No publication {key!r}")
    return with_meta(data)


# ---------------------------------------------------------------- startup ----
WARM = [
    ("overview", lambda: overview()), ("growth", lambda: growth(None, None)), ("teams", lambda: teams(None, None)),
    ("metadata", lambda: metadata(None, None)), ("rates", lambda: growth_rates(1990, 2023, 2005)),
    ("homonyms", lambda: homonyms(15)), ("affiliation", lambda: affiliation()),
    ("newcomers", lambda: newcomers(None, None)), ("cohorts", lambda: cohorts(1975, 2005, 5)),
    ("positions", lambda: positions(3)), ("unid", lambda: unidentified_by_position(2015, 3)),
    ("alphabetical", lambda: alphabetical(2000, 2000, 10, 5)),
    ("terms", lambda: terms("llm,deep,neural,transformer", None, None)), ("style", lambda: title_style()),
    ("words", lambda: words("2011-2015", "2021-2025", "rising", 1500, 15, True)),
    ("words-", lambda: words("2011-2015", "2021-2025", "falling", 1500, 15, True)),
    ("lifespans", lambda: lifespans()), ("concentration", lambda: concentration(1980, None, 10, 5)),
    ("publishers", lambda: publishers("2001-2005,2011-2015,2021-2025", 8)),
    ("doi", lambda: doi_gaps(2000, 1.0, 12)), ("coverage", lambda: coverage()),
    ("pages", lambda: page_formats()), ("theses", lambda: theses(12)), ("data", lambda: research_data(2015, None)),
    ("tails", lambda: tails("papers_per_author")), ("tails2", lambda: tails("coauthors_per_author")),
    ("tails3", lambda: tails("papers_per_series")),
    ("venues", lambda: venue_search("", None, 25)),
]


def warm_cache():
    """Run the default query of every page once, so first visits after a (re)build are instant."""
    def run():
        t = time.time()
        for name, fn in WARM:
            if serving.stopping.is_set():
                return
            try:
                fn()
            except Exception as e:
                log.warning("warm-up %s failed: %s", name, e)
        log.info("cache warmed in %.0fs", time.time() - t)
    serving.spawn(run, "warm-cache")
