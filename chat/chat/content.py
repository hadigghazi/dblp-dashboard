"""
Questions about what papers say.

dblp holds titles, not abstracts, so until now Dewey declined "what does this paper propose?" and
"what is X?". The DBLP-QA study (research/dblpqa-paper) measured how to answer them: given the right
abstract every model it tried answers almost perfectly, so what decides an answer is finding that
abstract, and the best way it found was to pool dblp's own title search with OpenAlex's keyword and
semantic search over abstracts and re-rank the pool with BM25 over title and abstract. That is this
tool, built from the study's own functions, so the tool is the pipeline that was measured:

  1. three searches at once: dblp's hybrid title search (top 50), OpenAlex keyword search (the
     question's words OR-ed, top 50) and OpenAlex semantic search (top 50);
  2. OpenAlex works kept only where dblp has the paper: DOI, then arXiv id (paperids), then exact
     normalised title, a published version preferred over its preprint;
  3. abstracts: OpenAlex's own for its hits, one OpenAlex lookup by DOI for dblp's;
  4. BM25 (k1 1.2, b 0.75, idf over the pool) over title and abstract; the top five go to the model,
     numbered, to be cited as [1] to [5].

Version 3 adds Dewey's own abstract index (abstractindex.py): the abstracts of dblp's papers from
OpenAlex's snapshot, searched with BM25 over the whole index as a fourth first stage, and looked up
by key before OpenAlex is asked for a candidate's abstract. The search can also be local only - the
index and dblp's title search, no outside service - which is the closed world RAGScholar had.

Semantic Scholar is not asked: its keyless search refused nearly every request during the study.
Results are cached per query for 30 days, so asking again costs nothing and returns the same papers.
OpenAlex calls are counted per UTC day against a cap; past it the tool still answers from dblp's
titles and says it is degraded. For the evaluation (dblpqa_dewey) a context can carry the study's
frozen pool for the question being asked, which replaces steps 1 to 3: Dewey then ranks exactly the
candidates the fixed pipeline ranked, with whatever query it chose to send.
"""
import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import abstractindex as AI, config, dblpqa_rag as RAG, paperids
from .tools import refusal, result

log = logging.getLogger("dblp.chat.content")

SOURCES = ("dewey-index", "dblp-search", "openalex-search", "openalex-semantic")
SOURCE_NAMES = {"dewey-index": "Dewey's abstract index", "dblp-search": "dblp's title search",
                "openalex-search": "OpenAlex", "openalex-semantic": "OpenAlex"}
NOTE = ("Abstracts come from OpenAlex, not dblp. Answer from them and cite each claim as [n]; if none "
        "addresses the question, say so. A paper missing here is not evidence that none exists.")
ANSWER_V2 = ("Answer from the one that addresses the question most directly and cite each claim as [n]. If "
             "none addresses it, search once more with the question's most distinctive terms. A paper missing "
             "here is not evidence that none exists.")
NOTE_V2 = "Abstracts come from OpenAlex, Semantic Scholar or Crossref, not dblp. " + ANSWER_V2
NOTE_V3 = ("Abstracts come from Dewey's own copy of OpenAlex's abstracts of dblp's papers, or from OpenAlex, "
           "Semantic Scholar or Crossref - not from dblp itself. " + ANSWER_V2)
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
CROSSREF = "https://api.crossref.org/works/"
FALLBACK_TOP = 10          # the best-ranked candidates worth a second abstract source
FALLBACK_CROSSREF = 3

_semantic_lock = threading.Lock()
_semantic_last = [0.0]
_budget_lock = threading.Lock()


# --------------------------------------------------------------------------- the day's OpenAlex calls

def _budget_path():
    return config.MODELS_DIR / "openalex-calls.json"


def take_openalex(n):
    """Count n OpenAlex calls against today's cap; False (and nothing counted) past it."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = _budget_path()
    with _budget_lock:
        try:
            used = json.loads(path.read_text(encoding="utf-8")).get(day, 0)
        except (OSError, ValueError):
            used = 0
        if used + n > config.OPENALEX_CALLS_PER_DAY:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({day: used + n}), encoding="utf-8")
        tmp.replace(path)
        return True


def _semantic_turn(deadline):
    """OpenAlex's semantic search allows one request a second, for every caller in the process."""
    with _semantic_lock:
        wait = _semantic_last[0] + RAG.OPENALEX_GAP - time.time()
        if wait > 0:
            if time.time() + wait > deadline:
                return False
            time.sleep(wait)
        _semantic_last[0] = time.time()
        return True


# --------------------------------------------------------------------------- the cache

def _cache_file(payload):
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return config.MODELS_DIR / "content-cache" / f"{digest}.json"


def _cached(payload):
    path = _cache_file(payload)
    try:
        if time.time() - path.stat().st_mtime < config.CONTENT_CACHE_DAYS * 86400:
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return None


def _keep(payload, value):
    path = _cache_file(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- the searches

def _dblp_search(http, query, timeout):
    r = http.get(f"{config.SEARCH_URL}/search/papers", params={"q": query[:300], "top": config.CONTENT_POOL},
                 timeout=timeout)
    r.raise_for_status()
    return [{"key": h["key"], "title": h.get("title")} for h in r.json().get("results") or [] if h.get("key")]


def _openalex_works(http, params, timeout):
    params = dict(params, **{"per-page": config.CONTENT_POOL,
                             "select": "id,doi,display_name,abstract_inverted_index"})
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY
    r = http.get(config.OPENALEX_URL, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAlex answered {r.status_code}")
    return [{"id": w["id"], "title": w.get("display_name"),
             "doi": (w.get("doi") or "").lower().replace("https://doi.org/", "") or None,
             "abstract": RAG._inverted(w.get("abstract_inverted_index"))}
            for w in (r.json() or {}).get("results") or [] if w.get("id")]


def map_works(cur, works):
    """{OpenAlex id: dblp key} for the works dblp holds: DOI, then arXiv id, then exact normalised
    title of 20+ characters (short titles like "Introduction" are shared by hundreds of records) -
    the study's map_to_dblp, from the index instead of a scan of the parquet."""
    dois = {w["doi"] for w in works if w.get("doi")}
    arxiv = {m.group(1) for d in dois if (m := paperids.ARXIV_DOI.match(d))}
    by_doi, by_arxiv = paperids.keys_for(cur, dois, arxiv)
    titles = sorted({RAG.title_key(w.get("title")) for w in works if len(RAG.title_key(w.get("title"))) >= 20})
    by_title = {}
    if titles:
        for key, norm in cur.execute("SELECT key, title_norm FROM s.pubs WHERE title_norm IN "
                                     "(SELECT unnest(?::VARCHAR[]))", [titles]).fetchall():
            by_title.setdefault(norm, set()).add(key)
    found = {}
    for w in works:
        doi = w.get("doi") or ""
        arx = paperids.ARXIV_DOI.match(doi)
        key = by_doi.get(doi) or (by_arxiv.get(arx.group(1)) if arx else None)
        if not key:
            same = by_title.get(RAG.title_key(w.get("title")))
            # a title shared by a preprint and its published version: the published one
            key = sorted(same, key=lambda k: (k.startswith("journals/corr/"), k))[0] if same else None
        if key:
            found[w["id"]] = key
    return found


def twins(cur, keys):
    """{key: [records with the same normalised title]} - a published paper and its preprint are two
    dblp records, and OpenAlex often has the abstract of only one of them."""
    if not keys:
        return {}
    rows = cur.execute("""
        SELECT a.key, b.key FROM s.pubs a JOIN s.pubs b ON a.title_norm = b.title_norm AND a.key <> b.key
        WHERE a.key IN (SELECT unnest(?::VARCHAR[])) AND length(a.title_norm) >= 20""", [list(keys)]).fetchall()
    found = {}
    for key, twin in rows:
        found.setdefault(key, []).append(twin)
    return found


def fill_abstracts(http, cur, cands, timeout):
    """Abstracts for the candidates that came without one (dblp's own hits), in one OpenAlex lookup
    by DOI - their own, or their preprint's or published twin's. Returns whether the lookup was made."""
    need = [k for k, c in cands.items() if not c.get("abstract")]
    twin_of = twins(cur, need)
    ids = paperids.ids_for(cur, need + [t for ts in twin_of.values() for t in ts])
    by_doi = {}
    for key in need:
        for source in [key] + twin_of.get(key, []):
            got = ids.get(source) or {}
            doi = got.get("doi") or (f"10.48550/arxiv.{got['arxiv']}" if got.get("arxiv") else None)
            # OpenAlex's filter syntax splits on "," and "|", so a DOI holding either cannot be asked for
            if doi and not set(doi) & {",", "|"} and doi.lower() not in by_doi:
                by_doi[doi.lower()] = key
                if source == key:
                    cands[key]["doi"] = cands[key].get("doi") or doi.lower()
    if not by_doi or not take_openalex(1):
        return False
    params = {"filter": "doi:" + "|".join(list(by_doi)[:50]), "per-page": 50,
              "select": "doi,abstract_inverted_index"}
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY
    r = http.get(config.OPENALEX_URL, params=params, timeout=timeout)
    if r.status_code != 200:
        return False
    for work in (r.json() or {}).get("results") or []:
        doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
        text = RAG._inverted(work.get("abstract_inverted_index"))
        if doi in by_doi and text and not cands[by_doi[doi]].get("abstract"):
            cands[by_doi[doi]]["abstract"] = text
    return True


def local_abstracts(cur, cands):
    """Version 3: abstracts from Dewey's own index for the candidates that came without one - their
    own record's, else a preprint's or published twin's - before any outside service is asked.
    Returns how many were found."""
    need = [k for k, c in cands.items() if not c.get("abstract")]
    if not need:
        return 0
    found = AI.lookup(need)
    left = [k for k in need if k not in found]
    twin_of = twins(cur, left) if left else {}
    if twin_of:
        found.update(AI.lookup([t for ts in twin_of.values() for t in ts]))
    filled = 0
    for key in need:
        for source in [key] + twin_of.get(key, []):
            hit = found.get(source)
            if hit and hit.get("abstract"):
                cands[key]["abstract"] = hit["abstract"]
                if source == key:
                    cands[key]["doi"] = cands[key].get("doi") or hit.get("doi")
                filled += 1
                break
    return filled


def wanted_sources():
    """The first-stage searches of the version in force: the index (version 3, once built) and dblp's
    title search, and OpenAlex's two searches unless the search is local only."""
    found = ["dewey-index"] if config.CONTENT_LOCAL_INDEX and AI.available() else []
    found.append("dblp-search")
    if config.CONTENT_SEARCH != "local":
        found += ["openalex-search", "openalex-semantic"]
    return found


def _crossref_abstract(http, doi, timeout):
    r = http.get(CROSSREF + doi, timeout=timeout)
    if r.status_code != 200:
        return None
    raw = ((r.json() or {}).get("message") or {}).get("abstract") or ""
    text = " ".join(re.sub(r"<[^>]+>", " ", raw).split())
    text = re.sub(r"^Abstract\s+", "", text)
    return text if len(text.split()) >= 20 else None


def fallback_abstracts(http, cur, cands, query, deadline):
    """Version 2: abstracts for the best-ranked candidates OpenAlex has none for - Semantic Scholar in
    one batch, then Crossref for a few - as the study fetched its own source abstracts. The paper a
    question is about is often found by its title and has no OpenAlex abstract (a 2026 conference
    paper, say), so without this it is ranked first and read as nothing. Returns how many were found."""
    need = [k for k in rank(query, cands)[:FALLBACK_TOP] if not cands[k].get("abstract")]
    if not need or time.time() > deadline - 1.5:
        return 0
    ids = paperids.ids_for(cur, need)
    s2 = {}
    for key in need:
        got = ids.get(key) or {}
        if got.get("doi"):
            s2[key] = f"DOI:{got['doi']}"
        elif got.get("arxiv"):
            s2[key] = f"ARXIV:{got['arxiv']}"
    found = 0
    if s2:
        try:
            r = http.post(S2_BATCH, params={"fields": "abstract"}, json={"ids": list(s2.values())},
                          timeout=max(0.5, min(3.0, deadline - time.time() - 1.0)))
            if r.status_code == 200:
                for key, paper in zip(s2, r.json() or []):
                    if paper and len((paper.get("abstract") or "").split()) >= 20:
                        cands[key]["abstract"] = paper["abstract"]
                        found += 1
        except Exception as e:                  # a slow or refused source costs the abstract, not the answer
            log.warning("Semantic Scholar abstracts: %s", e)
    left = [k for k in need if not cands[k].get("abstract") and (ids.get(k) or {}).get("doi")][:FALLBACK_CROSSREF]
    if left and time.time() < deadline - 1.0:
        pool = ThreadPoolExecutor(max_workers=len(left))
        try:
            timeout = max(0.5, min(2.5, deadline - time.time() - 0.5))
            futures = {k: pool.submit(_crossref_abstract, http, ids[k]["doi"], timeout) for k in left}
            for key, future in futures.items():
                try:
                    text = future.result(timeout=timeout + 0.5)
                except Exception:
                    text = None
                if text:
                    cands[key]["abstract"] = text
                    found += 1
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    return found


def retrieve(ctx, query, deadline):
    """The live pool: ({dblp key: {title, abstract, doi, found_by}}, {source: status}, degraded)."""
    cur = ctx.cursor()
    status, degraded = {}, None
    stage_end = deadline - 2.0
    sources = wanted_sources()
    local = "dewey-index" in sources
    asks_openalex = any(s.startswith("openalex") for s in sources)
    openalex_ok = asks_openalex and take_openalex(2)
    if asks_openalex and not openalex_ok:
        degraded = ("OpenAlex's daily allowance is used up: Dewey's own index and dblp's title search only"
                    if local else "OpenAlex's daily allowance is used up: titles from dblp's search only, no abstracts")
    keyword = RAG.openalex_query(query, semantic=False)

    def run(source):
        timeout = max(0.5, stage_end - time.time())
        if source == "dewey-index":
            return AI.search(query, config.CONTENT_POOL)
        if source == "dblp-search":
            return _dblp_search(ctx.http, query, timeout)
        if source == "openalex-search":
            return _openalex_works(ctx.http, keyword, timeout) if keyword else []
        if not _semantic_turn(stage_end):
            raise TimeoutError("no free slot for OpenAlex's semantic search in time")
        return _openalex_works(ctx.http, RAG.openalex_query(query, semantic=True), max(0.5, stage_end - time.time()))

    wanted = [s for s in sources if openalex_ok or not s.startswith("openalex")]
    got = {}
    pool = ThreadPoolExecutor(max_workers=len(wanted))
    try:
        futures = {s: pool.submit(run, s) for s in wanted}
        for source, future in futures.items():
            try:
                got[source] = future.result(timeout=max(0.1, stage_end - time.time() + 0.5))
                status[source] = "ok"
            except Exception as e:              # a slow or failing source must not cost the answer
                status[source] = "timeout" if isinstance(e, TimeoutError) or "imeout" in type(e).__name__ \
                    else f"failed: {type(e).__name__}"
                log.warning("abstract search: %s %s", source, status[source])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    for source in sources:
        status.setdefault(source, "skipped")

    cands = {}
    index_hits = got.get("dewey-index") or []
    if index_hits:
        # the index can be older than the dump: a record dblp no longer has is never shown
        known = {k for (k,) in cur.execute("SELECT key FROM s.pubs WHERE key IN (SELECT unnest(?::VARCHAR[]))",
                                           [[h["key"] for h in index_hits]]).fetchall()}
        for hit in index_hits:
            if hit["key"] in known:
                cands[hit["key"]] = {"title": hit.get("title"), "abstract": hit.get("abstract"),
                                     "doi": hit.get("doi"), "found_by": ["dewey-index"],
                                     "dx_rank": len(cands) + 1}
    for hit in got.get("dblp-search") or []:
        cands.setdefault(hit["key"], {"title": hit.get("title"), "found_by": []})["found_by"].append("dblp-search")
    works = [w for s in ("openalex-search", "openalex-semantic") for w in got.get(s) or []]
    oa_map = map_works(cur, works) if works else {}
    for source in ("openalex-search", "openalex-semantic"):
        for w in got.get(source) or []:
            key = oa_map.get(w["id"])
            if not key:
                continue                        # not in dblp: outside what this assistant covers
            entry = cands.setdefault(key, {"title": w.get("title"), "found_by": []})
            if source not in entry["found_by"]:
                entry["found_by"].append(source)
            entry["abstract"] = entry.get("abstract") or w.get("abstract")
            entry["doi"] = entry.get("doi") or w.get("doi")
    if local:
        try:
            local_abstracts(cur, cands)
        except Exception as e:
            log.warning("abstract lookup in the index failed: %s", e)
    if openalex_ok and time.time() < deadline - 0.5:
        try:
            fill_abstracts(ctx.http, cur, cands, max(0.5, deadline - time.time() - 0.3))
        except Exception as e:
            log.warning("abstract lookup by DOI failed: %s", e)
    if config.CONTENT_FALLBACK and cands and config.CONTENT_SEARCH != "local":
        fallback_abstracts(ctx.http, cur, cands, query, deadline)
    return cands, status, degraded


# --------------------------------------------------------------------------- the tool

def _cut(text, limit):
    """At most `limit` characters, ending at a sentence where one is near."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    head = text[:limit]
    stop = head.rfind(". ")
    return (head[:stop + 1] if stop > limit * 0.6 else head.rstrip()) + " [...]"


def _records(cur, keys):
    if not keys:
        return {}
    rows = cur.execute("SELECT key, title, year, venue FROM s.pubs WHERE key IN (SELECT unnest(?::VARCHAR[]))",
                       [list(keys)]).fetchall()
    return {k: {"title": t, "year": y, "venue": v} for k, t, y, v in rows}


def _count_call(ctx):
    """Two abstract searches per answer at most: a third is told to answer from what it has."""
    turn = getattr(ctx, "turn", None)
    if turn is None:
        return True
    with turn["lock"]:
        turn["content_calls"] = turn.get("content_calls", 0) + 1
        return turn["content_calls"] <= config.CONTENT_CALLS_PER_QUESTION


def pool_for(ctx, query, deadline):
    """(candidates, status per source, degraded, cached) for a query: the study's frozen pool when the
    context carries one, else the cache, else the three searches (kept for a month when complete)."""
    frozen = getattr(ctx, "frozen_pool", None)
    if frozen is not None:
        cands = {k: {"title": e.get("title"), "abstract": e.get("abstract"), "doi": e.get("doi"),
                     "found_by": ["frozen pool"]}
                 for k, e in (frozen.get("candidates") or {}).items()}
        return cands, {"frozen pool": "ok"}, None, False
    # v2: abstracts from a preprint's or published twin, a 12 s search limit; v3: Semantic Scholar and
    # Crossref abstracts too (content version 2) - pools of one version are never served to another
    request = {"v": 3 if config.CONTENT_FALLBACK else 2, "q": query.lower()}
    # version 3: a pool from one index (or none) is never served to another, nor a local one to a live one
    index = AI.opened()[1] if config.CONTENT_LOCAL_INDEX else None
    if index:
        request["index"] = index.get("snapshot")
    if config.CONTENT_SEARCH != "live":
        request["search"] = config.CONTENT_SEARCH
    hit = _cached(request)
    if hit is not None:
        return hit["cands"], hit["status"], None, True
    cands, status, degraded = retrieve(ctx, query, deadline)
    # only a complete pool is kept: a degraded or timed-out one would be served for a month
    if not degraded and all(v == "ok" for v in status.values()):
        _keep(request, {"cands": cands, "status": status})
    return cands, status, degraded, False


def rank(query, cands):
    """Every candidate with a title or an abstract, best first: the study's BM25 over title + abstract,
    with the pool's statistics; with CONTENT_FUSION=rrf, fused with the index's own order."""
    docs = {k: RAG.doc_text(c) for k, c in cands.items() if RAG.doc_text(c)}
    if not docs:
        return []
    ranking = RAG.bm25_rank(query, docs)
    if config.CONTENT_FUSION == "rrf":
        index = sorted((k for k in docs if cands[k].get("dx_rank")), key=lambda k: cands[k]["dx_rank"])
        if index:
            return RAG.rrf([ranking, index])
    return ranking


def search_abstracts(ctx, question=None, keys=None, frm=None, to=None):
    query = " ".join((question or "").split())
    if not query and not keys:
        return refusal("say what to look for: the question, in words")
    if not _count_call(ctx):
        return refusal("this answer has already searched abstracts twice",
                       "answer from the abstracts already returned, or say what is missing")
    cur = ctx.cursor()
    t0 = time.time()
    # the agent says how long this call may take (its own per-tool limit and the answer's budget)
    allowed = min(config.CONTENT_DEADLINE, (getattr(ctx, "time_left", None) or config.CONTENT_DEADLINE) - 0.5)
    deadline = t0 + max(1.0, allowed)

    if keys:
        return _by_keys(ctx, cur, keys, deadline)

    frozen = getattr(ctx, "frozen_pool", None)
    index = AI.opened()[1] if config.CONTENT_LOCAL_INDEX else None
    cands, status, degraded, cached = pool_for(ctx, query, deadline)
    ranking = rank(query, cands)
    info = _records(cur, ranking)
    if frm is not None or to is not None:
        lo, hi = (frm if frm is not None else -10 ** 6), (to if to is not None else 10 ** 6)
        ranking = [k for k in ranking if info.get(k, {}).get("year") is not None and lo <= info[k]["year"] <= hi]
    top = ranking[:config.CONTENT_TOP]
    rows = []
    for n, key in enumerate(top, 1):
        c, rec = cands[key], info.get(key, {})
        rows.append({"n": n, "title": rec.get("title") or c.get("title"), "year": rec.get("year"),
                     "venue": rec.get("venue"), "key": key, "doi": c.get("doi"),
                     "found_by": "+".join(c.get("found_by") or []),
                     "abstract": _cut(c.get("abstract"), config.CONTENT_ABSTRACT_CHARS) or "(no abstract available)"})
    with_abstract = sum(1 for c in cands.values() if c.get("abstract"))
    missing = sum(1 for r in rows if r["abstract"] == "(no abstract available)")
    # the frozen pool came from dblp's title search and OpenAlex (and Semantic Scholar): said as before
    names = list(dict.fromkeys(SOURCE_NAMES[s] for s in status if s in SOURCE_NAMES)) \
        or ["dblp's title search", "OpenAlex"]
    origin = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    summary = (f"{len(rows)} abstracts for “{query}”, best first: BM25 over the titles and abstracts of "
               f"{len(cands)} candidates from {origin}"
               + (f" ({missing} of them without an abstract)" if missing else "") + ".")
    if not rows:
        summary = f"No papers found for “{query}”."
    if config.CONTENT_RULE_VERSION >= 2:
        note = NOTE_V3 if "dewey-index" in status else NOTE_V2
    else:
        note = NOTE
    note += f" Degraded: {degraded}." if degraded else ""
    return result(summary, ["n", "title", "year", "venue", "found_by", "key", "doi", "abstract"], rows,
                  note=note, link={"page": "papers", "q": query}, candidates=len(cands),
                  with_abstract=with_abstract, sources=status, degraded=degraded, cached=cached,
                  frozen=frozen is not None, index=(index or {}).get("snapshot") if "dewey-index" in status else None,
                  ms=int(1000 * (time.time() - t0)))


def _by_keys(ctx, cur, keys, deadline):
    """The abstracts of papers already known by key ("what does the second one say?")."""
    keys = [k for k in keys if isinstance(k, str)][:config.CONTENT_TOP]
    info = _records(cur, keys)
    unknown = [k for k in keys if k not in info]
    if unknown:
        return refusal(f"no dblp record has the key {unknown[0]}",
                       "use a key a tool returned, or search_abstracts with the question")
    cands = {k: {"title": info[k]["title"], "found_by": ["key"]} for k in keys}
    try:
        if config.CONTENT_LOCAL_INDEX and AI.available():
            local_abstracts(cur, cands)
        if config.CONTENT_SEARCH != "local" and any(not c.get("abstract") for c in cands.values()):
            fill_abstracts(ctx.http, cur, cands, max(0.5, deadline - time.time()))
    except Exception as e:
        log.warning("abstract lookup by key failed: %s", e)
    rows = [{"n": n, "title": info[k]["title"], "year": info[k]["year"], "venue": info[k]["venue"],
             "found_by": "key", "key": k, "doi": cands[k].get("doi"),
             "abstract": _cut(cands[k].get("abstract"), config.CONTENT_ABSTRACT_CHARS) or "(no abstract available)"}
            for n, k in enumerate(keys, 1)]
    return result(f"The abstracts of {len(rows)} paper(s), from OpenAlex.",
                  ["n", "title", "year", "venue", "found_by", "key", "doi", "abstract"], rows, note=NOTE,
                  link={"page": "papers", "q": info[keys[0]]["title"] or ""})
