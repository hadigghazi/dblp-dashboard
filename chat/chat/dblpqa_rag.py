"""
Realistic retrieval for DBLP-QA: what happens between "no abstract" and "the right abstract".

The oracle run showed that, given the abstract a question was written from, every model scores about
2/2 - generation is solved on this benchmark, so everything that still varies is retrieval. This
module measures it, the way RAGScholar does it and the way its own future work proposes, and tests a
way to keep what retrieval gives without what it takes away.

**The candidate pool.** RAGScholar ran BM25 over a Lucene index of 4.6 million Semantic Scholar
abstracts; that bulk abstract set is no longer fully available (the paper's own footnote), so no one
can rebuild it. Instead each question gets a pool from four first-stage retrievers that need no bulk
download, each kept only where the paper is in dblp, as the paper's corpus was:

  * dblp-search - this project's hybrid search over every dblp title (BM25 + text-embedding-3-large,
    fused), top 50;
  * s2-search - Semantic Scholar's own relevance search, top 100;
  * openalex-search - OpenAlex's keyword search over titles, abstracts and full text, the question's
    words OR-ed (OpenAlex ANDs plain words, which a whole question never satisfies), top 50 - the
    nearest thing available to the paper's BM25 over a whole abstract corpus;
  * openalex-semantic - OpenAlex's embedding search over the title and abstract of every work
    (GTE-large), top 50.

OpenAlex works are mapped to dblp records by DOI, then arXiv id, then exact normalised title. Abstracts
come with the OpenAlex results, otherwise from Semantic Scholar by DOI or arXiv id, then OpenAlex by
DOI; everything is cached.

**The pool is built once and then frozen.** `dblpqa retrieval` builds it and retries any search that
was refused (Semantic Scholar rate-limits freely); `dblpqa rag` never searches, refuses a pool with a
missing search, and records the pool's fingerprint, so runs compared with each other saw the same
candidates.

**The rankers**, applied to the same pool so that only the ranking differs:

  * bm25 - the paper's method, Lucene's formula and defaults (k1 = 1.2, b = 0.75; the paper reports
    b = 2, which BM25 does not allow), over title + abstract;
  * dense - cosine similarity of text-embedding-3-large embeddings (512 dimensions);
  * hybrid - reciprocal rank fusion (k = 60) of the two, which is the paper's stated future work.

The four first-stage orders are scored as well. Pool statistics stand in for corpus statistics in
BM25's idf, the standard compromise when re-ranking a pool, stated wherever the results are.

**Dewey's index** (`dewey-index`) is the closed world rebuilt from open data (abstractindex.py): BM25
over the abstracts of every dblp paper OpenAlex has one for, with whole-index statistics as
RAGScholar's Lucene index had. It is not a re-ranker of the pool but a retriever of its own: its top
50 for each question are kept beside the pool (dewey-index-pools.json, frozen like the pool, never
mixed into it, so every earlier ranking and fingerprint stands), and the source paper is recognised
in them by the same rule.

**Which paper is "the source".** dblp often holds a preprint and its published version under two keys
with one title, and the benchmark names one of them. A candidate counts as the source if it has the
benchmark's key, Semantic Scholar's dblp key for the benchmark's paper, or the same normalised title.

**Selective retrieval.** With realistic retrieval the strong models gained on the questions whose
source was retrieved and lost about as much on the rest: off-topic abstracts pulled them below their
own closed-book answers. Two remedies are measured against plain RAG on the same frozen pool:

  * permissive - the answer prompt says the abstracts may be off-topic and to answer from what the
    model knows if none addresses the question;
  * gated - a separate gpt-4.1-mini call keeps only the abstracts that address the question, and with
    none kept the model answers closed-book. The verdicts are cached, so every answer model is given
    exactly the same filtered context.

Everything here is pure Python on purpose: the chat image has no numpy, and the arithmetic - a few
thousand 512-dimension dot products - does not need it.
"""
import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import httpx

from . import config, dblpqa as DQ

log = logging.getLogger("dblp.chat.dblpqa_rag")

POOL_DBLP = 50
POOL_S2 = 100
POOL_OA = 50
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 512
EMBED_PRICE = 0.13            # $ per 1M tokens
BM25_K1, BM25_B = 1.2, 0.75
RRF_K = 60
RERANKERS = ("bm25", "dense", "hybrid")
# the first-stage retrievers, and the candidate field each one's rank is kept in
FIRST_STAGE = {"dblp-search": "dblp_rank", "s2-search": "s2_rank",
               "openalex-search": "oa_rank", "openalex-semantic": "oas_rank"}
RANKERS = RERANKERS + tuple(FIRST_STAGE)
INDEX_RANKER = "dewey-index"   # Dewey's own abstract index: its own candidates, kept beside the pool
INDEX_POOL = 50
MODES = ("plain", "permissive", "gated")
TOP_K = 5
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_ATTEMPTS = 8
S2_GIVE_UP = 3                # refused searches in a row before Semantic Scholar is left for a later run
OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_KEY = os.environ.get("OPENALEX_API_KEY", "")
OPENALEX_GAP = 1.1            # semantic search allows one request a second
PARQUET = Path(os.environ.get("PARQUET", config.DATA_DIR / "parquet" / "dblp.parquet"))
GATE_MODEL = "gpt-4.1-mini"
GATE_VERSION = 1
GATE_SYSTEM = ("You check search results for a question about computer-science research. You are given "
               "the question and numbered paper abstracts. Say which abstracts contain information that "
               "answers the question; an abstract on the same broad topic that does not address what is "
               "asked does not count. Reply with JSON only: {\"relevant\": [the numbers]}, with an empty "
               "list if none does.")

# Lucene's English stop set plus the question words every DBLP-QA question opens with
STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then "
           "there these they this to was will with what how why which who does do".split())


def tokens(text):
    return [t for t in re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
            if t not in STOP and len(t) > 1]


def norm_title(title):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def title_key(title):
    """dblp's own title normalisation (the api's title_norm): lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def doc_text(entry):
    title = (entry.get("title") or "").strip()
    abstract = (entry.get("abstract") or "").strip()
    return f"{title}. {abstract}" if abstract else title


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --------------------------------------------------------------------------- rankers

def bm25_rank(query, docs, k1=BM25_K1, b=BM25_B):
    """Keys of `docs` ({key: text}) best first, by Lucene's BM25. Ties fall back to key order so a
    re-run ranks identically."""
    q = tokens(query)
    toks = {key: tokens(text) for key, text in docs.items()}
    n = len(toks) or 1
    avgdl = sum(len(t) for t in toks.values()) / n or 1.0
    df = {}
    for t in toks.values():
        for term in set(t):
            df[term] = df.get(term, 0) + 1
    scores = {}
    for key, t in toks.items():
        counts = {}
        for term in t:
            counts[term] = counts.get(term, 0) + 1
        score = 0.0
        for term in q:
            if term in counts:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                tf = counts[term]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avgdl))
        scores[key] = score
    return sorted(scores, key=lambda k: (-scores[k], k))


def dense_rank(qvec, vectors):
    """Keys of `vectors` ({key: unit vector}) best first, by cosine with the query vector."""
    scores = {k: sum(a * b for a, b in zip(qvec, v)) for k, v in vectors.items()}
    return sorted(scores, key=lambda k: (-scores[k], k))


def rrf(rankings, k=RRF_K):
    """Reciprocal rank fusion of several rankings."""
    scores = {}
    for ranking in rankings:
        for i, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + i)
    return sorted(scores, key=lambda key: (-scores[key], key))


def source_rank(ranking, aliases):
    """1-based rank of the first candidate that is the source paper, or None."""
    return next((i for i, key in enumerate(ranking, 1) if key in aliases), None)


def retrieval_metrics(ranks):
    """`ranks`: per question, the source paper's 1-based rank, or None if it was not found."""
    n = len(ranks) or 1
    hit = lambda cut: round(sum(1 for r in ranks if r is not None and r <= cut) / n, 3)
    mrr = sum(1.0 / r for r in ranks if r is not None and r <= 10) / n
    return {"questions": len(ranks), "recall@1": hit(1), "recall@3": hit(3), "recall@5": hit(5),
            "recall@10": hit(10), "mrr@10": round(mrr, 3),
            "ranked_at_all": round(sum(r is not None for r in ranks) / n, 3)}


# --------------------------------------------------------------------------- embeddings

class Embeddings:
    """text-embedding-3-large through the same provider, cached by text so nothing is paid for twice."""

    def __init__(self, cache_path, http=None):
        self.path = Path(cache_path)
        self.cache = _load(self.path)
        self.http = http or httpx.Client(timeout=120)
        self.tokens = 0

    @staticmethod
    def _key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, texts, out=print):
        todo = list(dict.fromkeys(t for t in texts if self._key(t) not in self.cache))
        for i in range(0, len(todo), 100):
            batch = todo[i:i + 100]
            for attempt in range(6):
                r = self.http.post(f"{config.BASE_URL}/embeddings",
                                   headers={"Authorization": f"Bearer {config.API_KEY}"},
                                   json={"model": EMBED_MODEL, "input": [t[:6000] for t in batch],
                                         "dimensions": EMBED_DIMS})
                if r.status_code != 429 and r.status_code < 500:
                    break
                time.sleep(min(30, 2 ** attempt))
            if r.status_code != 200:
                raise RuntimeError(f"embeddings failed ({r.status_code}): {r.text[:300]}")
            payload = r.json()
            self.tokens += (payload.get("usage") or {}).get("total_tokens", 0)
            for text, item in zip(batch, sorted(payload["data"], key=lambda d: d["index"])):
                vec = item["embedding"]
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                self.cache[self._key(text)] = [round(x / norm, 6) for x in vec]
            if (i // 100) % 10 == 9 or i + 100 >= len(todo):
                out(f"  embedded {min(i + 100, len(todo))}/{len(todo)}")
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.cache), encoding="utf-8")
        return {t: self.cache[self._key(t)] for t in texts}

    def cost(self):
        return round(self.tokens / 1e6 * EMBED_PRICE, 4)


# --------------------------------------------------------------------------- dblp records

def ids_from_ee(ee):
    """{"doi", "arxiv"} from a record's `ee` links. CoRR records usually have no DOI, only arXiv."""
    text = " ".join(ee) if isinstance(ee, (list, tuple)) else str(ee or "")
    doi = re.search(r"doi\.org/(10\.[^\s'\"]+)", text)
    arxiv = re.search(r"arxiv\.org/abs/(\S+)", text)
    got = {}
    if doi:
        got["doi"] = doi.group(1).rstrip(".,;")
    if arxiv:
        got["arxiv"] = re.sub(r"v\d+$", "", arxiv.group(1).rstrip(".,;"))
    return got


def _parquet_rows(sql, params, parquet=None):
    parquet = Path(parquet or PARQUET)
    if not parquet.exists():
        return []
    import duckdb
    con = duckdb.connect()
    try:
        return con.execute(sql.format(src=f"read_parquet('{parquet.as_posix()}')"), params).fetchall()
    finally:
        con.close()


def ids_from_dblp(keys, parquet=None):
    """{dblp key: {"doi", "arxiv"}} from the records' `ee` links, one scan of the parquet for all."""
    if not keys:
        return {}
    rows = _parquet_rows("SELECT key, ee FROM {src} WHERE key IN (SELECT unnest(?::VARCHAR[]))",
                         [list(keys)], parquet)
    return {key: ids for key, ee in rows if (ids := ids_from_ee(ee))}


def map_to_dblp(works, parquet=None):
    """{OpenAlex id: dblp key} for the works dblp holds: by DOI, then arXiv id, then exact normalised
    title (long titles only - short ones like "Introduction" are shared by hundreds of records)."""
    urls, titles = set(), set()
    for w in works:
        doi = (w.get("doi") or "").lower()
        if doi:
            urls.add(f"https://doi.org/{doi}")
            arxiv = re.match(r"10\.48550/arxiv\.(.+)$", doi)
            if arxiv:
                urls.add(f"https://arxiv.org/abs/{arxiv.group(1)}")
        if len(title_key(w.get("title"))) >= 20:
            titles.add(title_key(w.get("title")))
    if not urls and not titles:
        return {}
    rows = _parquet_rows("""
        SELECT key, title, ee FROM {src}
        WHERE type NOT IN ('www', 'proceedings')
          AND regexp_replace(lower(title), '[^a-z0-9]', '', 'g') IN (SELECT unnest(?::VARCHAR[]))
        UNION ALL
        SELECT key, title, ee FROM (SELECT key, title, ee, type, unnest(ee) AS u FROM {src})
        WHERE type NOT IN ('www', 'proceedings')
          AND (CASE WHEN lower(u) LIKE '%arxiv.org/abs/%' THEN regexp_replace(lower(u), 'v[0-9]+$', '')
                    ELSE lower(u) END) IN (SELECT unnest(?::VARCHAR[]))""",
                         [sorted(titles) or ["-"], sorted(urls) or ["-"]], parquet)
    by_doi, by_arxiv, by_title = {}, {}, {}
    for key, title, ee in rows:
        for url in ee or []:
            url = url.lower()
            if (m := re.search(r"doi\.org/(10\..+)$", url)):
                by_doi.setdefault(m.group(1).rstrip(".,;"), key)
            if (m := re.search(r"arxiv\.org/abs/(.+)$", url)):
                by_arxiv.setdefault(re.sub(r"v\d+$", "", m.group(1)), key)
        by_title.setdefault(title_key(title), set()).add(key)
    found = {}
    for w in works:
        doi = (w.get("doi") or "").lower()
        arxiv = re.match(r"10\.48550/arxiv\.(.+)$", doi)
        key = by_doi.get(doi) or (by_arxiv.get(arxiv.group(1)) if arxiv else None)
        if not key and len(title_key(w.get("title"))) >= 20:
            same = by_title.get(title_key(w.get("title")))
            # a title shared by a preprint and its published version: the published one
            key = sorted(same, key=lambda k: (k.startswith("journals/corr/"), k))[0] if same else None
        if key:
            found[w["id"]] = key
    return found


# --------------------------------------------------------------------------- the first-stage searches

def _search_dblp(http, row):
    r = http.get(f"{config.SEARCH_URL}/search/papers",
                 params={"q": row["question"][:300], "top": POOL_DBLP}, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"dblp search failed for {row['id']} ({r.status_code}): {r.text[:200]}")
    return [{"key": h["key"], "title": h.get("title")} for h in r.json().get("results") or []]


def _s2_query(question):
    # Semantic Scholar's search matches nothing for hyphenated terms (its documentation says so)
    return " ".join(question.replace("-", " ").split())[:300]


def _search_s2(http, row):
    """(status, hits). Semantic Scholar's unauthenticated pool is shared and refuses freely, so this
    waits longer than the general helper before giving up."""
    for attempt in range(S2_ATTEMPTS):
        r = http.get(S2_SEARCH, params={"query": _s2_query(row["question"]), "limit": POOL_S2,
                                        "fields": "title,abstract,externalIds"}, timeout=60)
        if r.status_code != 429 and r.status_code < 500:
            break
        time.sleep(min(60, 5 * 2 ** attempt))
    if r.status_code != 200:
        return r.status_code, []
    hits = []
    for paper in (r.json() or {}).get("data") or []:
        ids = paper.get("externalIds") or {}
        if ids.get("DBLP"):
            hits.append({"key": ids["DBLP"], "title": paper.get("title"), "abstract": paper.get("abstract"),
                         "doi": ids.get("DOI")})
    return 200, hits


def openalex_query(question, semantic):
    if semantic:
        return {"search.semantic": " ".join(question.split())[:2000]}
    # lower case on purpose: OpenAlex reads upper-case AND/OR/NOT in a query as operators
    words = list(dict.fromkeys(tokens(question)))
    return {"search": " OR ".join(words)} if words else None


def _inverted(index):
    words = sorted((p, w) for w, ps in (index or {}).items() for p in ps)
    return " ".join(w for _, w in words) or None


def _search_openalex(http, row, semantic):
    params = openalex_query(row["question"], semantic)
    if params is None:
        return 200, []
    params.update({"per-page": POOL_OA, "select": "id,doi,display_name,abstract_inverted_index"})
    if OPENALEX_KEY:
        params["api_key"] = OPENALEX_KEY
    r = DQ._get(http, OPENALEX_WORKS, params=params)
    if r.status_code != 200:
        return r.status_code, []
    return 200, [{"id": w.get("id"), "title": w.get("display_name"),
                  "doi": (w.get("doi") or "").lower().replace("https://doi.org/", "") or None,
                  "abstract": _inverted(w.get("abstract_inverted_index"))}
                 for w in (r.json() or {}).get("results") or [] if w.get("id")]


def status_of(pool):
    """Which first-stage searches succeeded for a question (HTTP status per retriever)."""
    status = dict(pool.get("status") or {})
    if "s2_status" in pool:          # a pool from before OpenAlex: dblp search either worked or raised
        status.setdefault("dblp-search", 200)
        status.setdefault("s2-search", pool["s2_status"])
    return status


def incomplete(pools, rows):
    """{question id: [retrievers whose search is missing]} - empty when the pool is whole."""
    missing = {}
    for row in rows:
        status = status_of(pools.get(row["id"]) or {})
        lost = [s for s in FIRST_STAGE if status.get(s) != 200]
        if lost:
            missing[row["id"]] = lost
    return missing


def pool_sha(pools, rows):
    """Fingerprint of what a run is given: every question's candidates and source keys."""
    view = {r["id"]: {"candidates": pools[r["id"]]["candidates"], "aliases": pools[r["id"]]["aliases"]}
            for r in rows if r["id"] in pools}
    return hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- the pool

def _s2_id(entry):
    return f"DOI:{entry['doi']}" if entry.get("doi") else f"ARXIV:{entry['arxiv']}" if entry.get("arxiv") else None


def fetch_pool_abstracts(entries, cache, http, out=print):
    """Fill `abstract` on every entry ({dblp key: entry}) that lacks one: Semantic Scholar by DOI or
    arXiv id in batches, then OpenAlex by DOI. `cache` ({key: abstract or None}) is shared and updated
    in place."""
    need = [k for k, e in entries.items() if not e.get("abstract") and k not in cache]
    if need:
        for key, ids in ids_from_dblp([k for k in need if not _s2_id(entries[k])]).items():
            entries[key].update(ids)
        findable = [k for k in need if _s2_id(entries[k])]
        out(f"  abstracts: {len(need)} to find, {len(findable)} with a DOI or arXiv id")
        for i in range(0, len(findable), 400):
            chunk = findable[i:i + 400]
            r = DQ._get(http, DQ.S2_BATCH, method="POST", params={"fields": "abstract"},
                        json={"ids": [_s2_id(entries[k]) for k in chunk]})
            if r.status_code == 200:
                for k, paper in zip(chunk, r.json()):
                    if paper and paper.get("abstract"):
                        cache[k] = paper["abstract"]
        # OpenAlex's filter syntax splits on "," and "|", so a DOI holding either cannot be asked for
        still = [k for k in findable if not cache.get(k) and entries[k].get("doi")
                 and not set(entries[k]["doi"]) & {",", "|"}]
        for i in range(0, len(still), 50):
            by_doi = {entries[k]["doi"].lower(): k for k in still[i:i + 50]}
            params = {"filter": "doi:" + "|".join(by_doi), "per-page": 50, "select": "doi,abstract_inverted_index"}
            if OPENALEX_KEY:
                params["api_key"] = OPENALEX_KEY
            r = DQ._get(http, OPENALEX_WORKS, params=params)
            if r.status_code != 200:
                continue
            for work in (r.json() or {}).get("results") or []:
                doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
                text = _inverted(work.get("abstract_inverted_index"))
                if doi in by_doi and text:
                    cache[by_doi[doi]] = text
        for k in need:
            cache.setdefault(k, None)
    for k, e in entries.items():
        e["abstract"] = e.get("abstract") or cache.get(k)


def source_aliases(rows, http):
    """{question id: set of dblp keys that are the source paper}: the benchmark's key plus Semantic
    Scholar's dblp key for the benchmark's paper (its CorpusId), which differ when one is a preprint."""
    aliases = {r["id"]: {r["dblp_key"]} for r in rows}
    known = [r for r in rows if r.get("semantic_scholar_id")]
    if not known:
        return aliases
    r = DQ._get(http, DQ.S2_BATCH, method="POST", params={"fields": "externalIds"},
                json={"ids": [f"CorpusId:{row['semantic_scholar_id']}" for row in known]})
    if r.status_code == 200:
        for row, paper in zip(known, r.json()):
            key = ((paper or {}).get("externalIds") or {}).get("DBLP")
            if key:
                aliases[row["id"]].add(key)
    return aliases


def _place_openalex(pool, oa_map):
    """Rank the dblp records among a question's OpenAlex results, per OpenAlex retriever."""
    cands = pool["candidates"]
    for source, works in (pool.get("raw") or {}).items():
        field = FIRST_STAGE[source]
        for entry in cands.values():
            entry.pop(field, None)
        seen = []
        for w in works:
            key = oa_map.get(w["id"])
            if not key or key in seen:
                continue
            seen.append(key)
            entry = cands.setdefault(key, {"title": w.get("title")})
            entry[field] = len(seen)
            entry["abstract"] = entry.get("abstract") or w.get("abstract")
            entry["doi"] = entry.get("doi") or w.get("doi")


def build_pools(rows, oracle, http=None, cache_dir=None, out=print):
    """{question id: {"candidates": {dblp key: {title, abstract, doi, <retriever>_rank}}, "aliases":
    [source keys], "status": {retriever: HTTP status}, "raw": {OpenAlex retriever: works}}}.
    Searches only what is missing, so a re-run finishes a pool a refused search left incomplete."""
    cache_dir = Path(cache_dir or DQ.study_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    pools_path = cache_dir / "pools.json"
    abstracts_path, oa_map_path = cache_dir / "pool-abstracts.json", cache_dir / "openalex-dblp.json"
    pools, abstracts, oa_map = _load(pools_path), _load(abstracts_path), _load(oa_map_path)
    http = http or httpx.Client(follow_redirects=True, timeout=60,
                                headers={"User-Agent": "dblp-explorer-research"})
    missing = incomplete(pools, rows)
    todo = [r for r in rows if r["id"] in missing]
    aliases = source_aliases([r for r in todo if r["id"] not in pools], http)
    openalex_open = True
    # each refused Semantic Scholar search waits minutes before giving up; when its keyless pool is
    # saturated, every search is refused, so after a few in a row the rest is left for a later run
    s2_refusals = 0
    for i, row in enumerate(todo, 1):
        pool = pools.setdefault(row["id"], {"candidates": {}, "aliases": [row["dblp_key"]]})
        status = pool["status"] = status_of(pool)
        pool.pop("s2_status", None)
        pool.pop("s2_results", None)
        pool.setdefault("raw", {})
        pool["aliases"] = sorted(set(pool["aliases"]) | aliases.get(row["id"], set()))
        cands = pool["candidates"]
        if status.get("dblp-search") != 200:
            for rank, hit in enumerate(_search_dblp(http, row), 1):
                cands.setdefault(hit["key"], {"title": hit["title"]})["dblp_rank"] = rank
            status["dblp-search"] = 200
        if status.get("s2-search") != 200 and s2_refusals < S2_GIVE_UP:
            status["s2-search"], hits = _search_s2(http, row)
            s2_refusals = 0 if status["s2-search"] == 200 else s2_refusals + 1
            if s2_refusals == S2_GIVE_UP:
                out(f"  Semantic Scholar refused {S2_GIVE_UP} searches in a row - not asked again in this run "
                    f"(re-run `dblpqa retrieval` later to fill them in)")
            for rank, hit in enumerate(hits, 1):
                entry = cands.setdefault(hit["key"], {"title": hit["title"]})
                entry["s2_rank"] = rank
                entry["abstract"] = entry.get("abstract") or hit["abstract"]
                entry["doi"] = entry.get("doi") or hit["doi"]
            time.sleep(1.0)
        for source, semantic in (("openalex-search", False), ("openalex-semantic", True)):
            if status.get(source) == 200 or not openalex_open:
                continue
            status[source], works = _search_openalex(http, row, semantic)
            if status[source] == 200:
                pool["raw"][source] = works
            elif status[source] == 429:
                # the keyless daily budget is about 100 searches; the rest waits for a later run
                openalex_open = False
                out("  OpenAlex refused a search (429: daily budget spent?) - re-run `dblpqa retrieval` "
                    "later, or set OPENALEX_API_KEY")
            time.sleep(OPENALEX_GAP)
        if i % 10 == 0 or i == len(todo):
            out(f"  pools: {i}/{len(todo)} searched")
            pools_path.write_text(json.dumps(pools, ensure_ascii=False), encoding="utf-8")

    works = [w for p in pools.values() for ws in (p.get("raw") or {}).values() for w in ws]
    unmapped = list({w["id"]: w for w in works if w["id"] not in oa_map}.values())
    if unmapped:
        found = map_to_dblp(unmapped)
        for w in unmapped:
            oa_map[w["id"]] = found.get(w["id"])
        oa_map_path.write_text(json.dumps(oa_map), encoding="utf-8")
        out(f"  OpenAlex: {len(unmapped)} new works, {len(found)} of them in dblp")
    for pool in pools.values():
        _place_openalex(pool, oa_map)

    every = {}
    for pool in pools.values():
        for key, entry in pool["candidates"].items():
            every.setdefault(key, entry)
    fetch_pool_abstracts(every, abstracts, http, out)
    for pool in pools.values():
        for key, entry in pool["candidates"].items():
            entry["abstract"] = entry.get("abstract") or every[key].get("abstract")
    # the source paper is shown with the very abstract the oracle condition used, so a hit here means
    # the same text the oracle gave; a title match counts as the source (preprint vs published version)
    for row in rows:
        pool = pools.get(row["id"])
        if not pool:
            continue
        gold = (oracle or {}).get(row["id"]) or {}
        alias = set(pool["aliases"])
        title = norm_title(gold.get("title"))
        for key, entry in pool["candidates"].items():
            if key in alias or (title and norm_title(entry.get("title")) == title):
                alias.add(key)
                if gold.get("abstract"):
                    entry["abstract"] = gold["abstract"]
        pool["aliases"] = sorted(alias)
    pools_path.write_text(json.dumps(pools, ensure_ascii=False), encoding="utf-8")
    abstracts_path.write_text(json.dumps(abstracts, ensure_ascii=False), encoding="utf-8")
    return pools


def rank_pools(rows, pools, embeddings, out=print):
    """{ranker: {question id: [dblp keys best first]}} over each question's own pool."""
    texts = {r["id"]: {k: doc_text(e) for k, e in pools[r["id"]]["candidates"].items() if doc_text(e)}
             for r in rows if r["id"] in pools}
    every = [t for docs in texts.values() for t in docs.values()] + [r["question"] for r in rows]
    vecs = embeddings.get(every, out)
    ranked = {name: {} for name in RANKERS}
    for row in rows:
        qid = row["id"]
        cands = (pools.get(qid) or {}).get("candidates") or {}
        for name, field in FIRST_STAGE.items():
            ranked[name][qid] = sorted((k for k, e in cands.items() if e.get(field)), key=lambda k: cands[k][field])
        docs = texts.get(qid) or {}
        if not docs:
            for name in RERANKERS:
                ranked[name][qid] = []
            continue
        b = bm25_rank(row["question"], docs)
        d = dense_rank(vecs[row["question"]], {k: vecs[t] for k, t in docs.items()})
        ranked["bm25"][qid], ranked["dense"][qid], ranked["hybrid"][qid] = b, d, rrf([b, d])
    return ranked


def evaluate_retrieval(rows, pools, rankings):
    """Where the source paper lands under each ranker - no model, no judge."""
    report = {"rankers": {}}
    for name, per_q in rankings.items():
        ranks = [source_rank(per_q.get(r["id"]) or [], set(pools[r["id"]]["aliases"])) for r in rows]
        report["rankers"][name] = dict(retrieval_metrics(ranks),
                                       ranks={r["id"]: rank for r, rank in zip(rows, ranks)})
    sizes = [len(pools[r["id"]]["candidates"]) for r in rows]
    with_abstract = [sum(1 for e in pools[r["id"]]["candidates"].values() if e.get("abstract")) for r in rows]
    in_pool = [any(k in pools[r["id"]]["candidates"] for k in pools[r["id"]]["aliases"]) for r in rows]
    missing = incomplete(pools, rows)
    n = len(rows) or 1
    report["pool"] = {"mean_size": round(sum(sizes) / n, 1),
                      "share_with_abstract": round(sum(with_abstract) / max(1, sum(sizes)), 3),
                      "source_in_pool": round(sum(in_pool) / n, 3),
                      "failed": {s: sum(1 for lost in missing.values() if s in lost) for s in FIRST_STAGE},
                      "complete": not missing, "sha256": pool_sha(pools, rows)}
    return report


def print_retrieval(report, out=print):
    p = report["pool"]
    out(f"pool {p['sha256']}: {p['mean_size']} candidates per question on average, "
        f"{p['share_with_abstract']:.0%} with an abstract; the source paper is in the pool for "
        f"{p['source_in_pool']:.0%} of questions")
    failed = {s: n for s, n in p["failed"].items() if n}
    out("every search succeeded - the pool is complete" if not failed else
        "INCOMPLETE - searches missing: " + ", ".join(f"{s} {n}" for s, n in failed.items())
        + " (run `dblpqa retrieval` again to retry them)")
    out(f"\n{'ranker':18s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR@10':>7s} {'found':>6s}")
    for name, m in report["rankers"].items():
        out(f"{name:18s} {m['recall@1']:6.2f} {m['recall@3']:6.2f} {m['recall@5']:6.2f} "
            f"{m['recall@10']:6.2f} {m['mrr@10']:7.3f} {m['ranked_at_all']:6.2f}")
    out(f"(found = the source paper anywhere in that ranker's list; embeddings ${report.get('embedding_cost_usd', 0)})")
    index = report.get("index")
    if index:
        share = index["source_in_index"]
        out(f"dewey-index = BM25 over Dewey's own index ({index.get('documents') or 0:,} abstracts, OpenAlex snapshot "
            f"{index['snapshot']}), its own top {INDEX_POOL}; the source paper has an abstract in it for "
            f"{'?' if share is None else f'{share:.0%}'} of questions")


# --------------------------------------------------------------------------- contexts

def pool_of(pools, qid, ranker):
    """What a ranker chose from: the question's frozen pool, or for Dewey's index its own top 50."""
    pool = pools[qid]
    if ranker == INDEX_RANKER:
        return pool.get("index") or {"candidates": {}, "aliases": pool.get("aliases") or []}
    return pool


def _blocks(cands, keys):
    return "\n\n".join(f"[{i}] {cands[key].get('title') or ''}\n{cands[key].get('abstract') or '(no abstract available)'}"
                       for i, key in enumerate(keys, 1))


def condition_name(ranker, mode="plain", strategy="cd", k=TOP_K):
    """rag-bm25 for the paper's best strategy (Top-5 Concatenated Documents), the name every earlier run
    has; the paper's other strategies say which they are (see dblpqa.rag_strategy)."""
    name = f"{DQ.RAG_PREFIX}{ranker}"
    if strategy == "single":
        name += f"-a{k}"
    elif strategy == "ca":
        name += f"-ca{k}"
    elif k != TOP_K:
        name += f"-cd{k}"
    return name + ("" if mode == "plain" else f"-{mode}")


def rag_contexts(rows, pools, rankings, ranker, k=TOP_K):
    """The top-k papers as one concatenated context per question - the paper's best strategy (Top-k
    Concatenated Documents) - with where the source paper landed, for the analysis."""
    contexts = {}
    for row in rows:
        ranking = rankings[ranker].get(row["id"]) or []
        pool = pool_of(pools, row["id"], ranker)
        cands = pool["candidates"]
        keys = ranking[:k]
        rank = source_rank(ranking, set(pool["aliases"]))
        contexts[row["id"]] = {"abstract": _blocks(cands, keys) or "(no papers were retrieved)",
                               "source": f"{ranker}@{k}", "retrieved": keys, "source_rank": rank,
                               "source_in_context": rank is not None and rank <= k}
    return contexts


def single_contexts(rows, pools, rankings, ranker, j):
    """The paper's Single-Document strategy: the j-th ranked paper's abstract on its own (A1 ... A5)."""
    contexts = {}
    for row in rows:
        ranking = rankings[ranker].get(row["id"]) or []
        pool = pool_of(pools, row["id"], ranker)
        cands = pool["candidates"]
        keys = ranking[j - 1:j]
        rank = source_rank(ranking, set(pool["aliases"]))
        contexts[row["id"]] = {"abstract": _blocks(cands, keys) or "(no paper was retrieved at this rank)",
                               "source": f"{ranker}@a{j}", "retrieved": keys, "source_rank": rank,
                               "source_in_context": rank == j}
    return contexts


def parse_gate(text, n):
    """The kept abstract numbers from the gate's reply, or ValueError."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        got = json.loads(match.group(0)) if match else None
        numbers = got["relevant"]
        kept = sorted({int(x) for x in numbers})
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError(f"the gate did not return a list: {(text or '')[:200]!r}")
    if any(x < 1 or x > n for x in kept):
        raise ValueError(f"the gate named an abstract that is not there: {kept}")
    return kept


def gate_contexts(client, rows, pools, contexts, gate_model=GATE_MODEL, cache_dir=None, out=print, ranker=None):
    """Keep only the retrieved abstracts the gate says address the question; with none kept the
    context is None and the answer is closed-book. Verdicts are cached by question, retrieved papers
    and gate, so every answer model gets the same filtered context and none is paid for twice."""
    path = Path(cache_dir or DQ.study_dir()) / "gates.json"
    cache = _load(path)
    meter = DQ.Meter()
    gated = {}
    for row in rows:
        qid, ctx = row["id"], contexts[row["id"]]
        keys = ctx["retrieved"]
        pool = pool_of(pools, qid, ranker)
        cands, aliases = pool["candidates"], set(pool["aliases"])
        ck = hashlib.sha1(f"{GATE_VERSION}|{gate_model}|{row['question']}|{'|'.join(keys)}".encode()).hexdigest()
        if keys and ck not in cache:
            step = client.complete([{"role": "system", "content": GATE_SYSTEM},
                                    {"role": "user", "content": f"Question: {row['question']}\n\n"
                                                                f"Abstracts:\n{ctx['abstract']}"}],
                                   model=gate_model, temperature=0)
            meter.add(gate_model, step.get("usage", {}))
            try:
                cache[ck] = {"kept": parse_gate(step.get("content"), len(keys)), "unreadable": False}
            except ValueError:
                # fail open: an unreadable verdict keeps everything, exactly as plain RAG would
                cache[ck] = {"kept": list(range(1, len(keys) + 1)), "unreadable": True}
        verdict = cache.get(ck) or {"kept": [], "unreadable": False}
        kept = [keys[i - 1] for i in verdict["kept"]]
        gated[qid] = dict(ctx, kept=kept, abstract=_blocks(cands, kept) or None,
                          gate_kept_source=any(k in aliases for k in kept),
                          gate_unreadable=verdict["unreadable"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")
    n = len(rows) or 1
    info = {"model": gate_model, "version": GATE_VERSION, "cost_usd": meter.cost(),
            "mean_kept": round(sum(len(g["kept"]) for g in gated.values()) / n, 2),
            "passed_nothing": sum(1 for g in gated.values() if not g["kept"]),
            "unreadable": sum(1 for g in gated.values() if g["gate_unreadable"])}
    out(f"gate {gate_model}: keeps {info['mean_kept']} of the top abstracts on average; passes nothing "
        f"for {info['passed_nothing']} questions (answered closed-book); ${info['cost_usd']}")
    return gated, info


def index_pools(rows, pools, oracle, cache_dir, out=print):
    """{question id: {"candidates": {key: {title, abstract, doi, dx_rank}}, "aliases": [...]}}: the top
    50 of Dewey's abstract index for each question as written. Searched once per index snapshot and
    kept (dewey-index-pools.json), so later runs read them as they read the pool; None without an index
    and nothing kept. As in the pool, the source paper is recognised by key or by title, and shown with
    the abstract the oracle condition used."""
    from . import abstractindex as AI
    path = Path(cache_dir) / "dewey-index-pools.json"
    kept = _load(path)
    meta = AI.info()
    if meta and kept.get("snapshot") != meta["snapshot"]:
        kept = {"snapshot": meta["snapshot"], "documents": meta["documents"], "questions": {}, "in_index": {}}
    if not kept.get("snapshot"):
        return None
    todo = [r for r in rows if r["id"] not in kept["questions"]]
    if todo and not meta:
        out(f"  Dewey's index: {len(todo)} questions were never searched and the index is not here; "
            f"they get no candidates")
    elif todo:
        for row in todo:
            hits = AI.search(row["question"], INDEX_POOL)
            kept["questions"][row["id"]] = [{k: h[k] for k in ("key", "title", "abstract", "doi", "score")}
                                            for h in hits]
            aliases = set((pools.get(row["id"]) or {}).get("aliases") or [row["dblp_key"]])
            kept["in_index"][row["id"]] = bool(AI.lookup(sorted(aliases)))
        path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    found = {}
    for row in rows:
        gold = (oracle or {}).get(row["id"]) or {}
        title = norm_title(gold.get("title"))
        alias = set((pools.get(row["id"]) or {}).get("aliases") or [row["dblp_key"]])
        cands = {}
        for rank, hit in enumerate(kept["questions"].get(row["id"]) or [], 1):
            entry = {"title": hit["title"], "abstract": hit["abstract"], "doi": hit["doi"], "dx_rank": rank}
            if hit["key"] in alias or (title and norm_title(hit["title"]) == title):
                alias.add(hit["key"])
                entry["abstract"] = gold.get("abstract") or entry["abstract"]
            cands.setdefault(hit["key"], entry)
        found[row["id"]] = {"candidates": cands, "aliases": sorted(alias),
                            "in_index": kept.get("in_index", {}).get(row["id"])}
    found["_meta"] = {"snapshot": kept["snapshot"], "documents": kept.get("documents")}
    return found


def _add_index(rows, pools, rankings, report, ipools):
    """Dewey's index as a ranker beside the pool's: its ranking, its retrieval metrics, its pool."""
    meta = ipools.pop("_meta")
    rankings[INDEX_RANKER] = {qid: sorted(p["candidates"], key=lambda k: p["candidates"][k]["dx_rank"])
                              for qid, p in ipools.items()}
    ranks = [source_rank(rankings[INDEX_RANKER].get(r["id"]) or [], set(ipools[r["id"]]["aliases"])) for r in rows]
    report["rankers"][INDEX_RANKER] = dict(retrieval_metrics(ranks), ranks={r["id"]: rk for r, rk in zip(rows, ranks)})
    known = [ipools[r["id"]]["in_index"] for r in rows if ipools[r["id"]]["in_index"] is not None]
    report["index"] = dict(meta, sha256=pool_sha(ipools, rows),
                           source_in_index=round(sum(known) / len(known), 3) if known else None)
    for qid, p in ipools.items():
        pools[qid]["index"] = p


def prepare(rows, out=print, cache_dir=None, http=None, embeddings=None, frozen=False, allow_incomplete=False):
    """Pool, abstracts, embeddings and every ranking; returns (pools, rankings, report). `frozen`
    uses the pool as it is - no searching - and refuses one with a missing search."""
    cache_dir = Path(cache_dir or DQ.study_dir())
    oracle = DQ.fetch_abstracts(rows, cache_dir=cache_dir, http=http, out=out)
    if frozen:
        pools = _load(cache_dir / "pools.json")
        missing = incomplete(pools, rows)
        if missing and not allow_incomplete:
            raise SystemExit(f"the pool is missing searches for {len(missing)} questions "
                             f"(e.g. {next(iter(missing.items()))}); run `dblpqa retrieval` again until it "
                             f"says complete, or pass --allow-incomplete-pool")
        for row in rows:
            pools.setdefault(row["id"], {"candidates": {}, "aliases": [row["dblp_key"]]})
    else:
        pools = build_pools(rows, oracle, http=http, cache_dir=cache_dir, out=out)
    emb = embeddings or Embeddings(cache_dir / "embeddings.json")
    rankings = rank_pools(rows, pools, emb, out)
    report = evaluate_retrieval(rows, pools, rankings)
    ipools = index_pools(rows, pools, oracle, cache_dir, out)
    if ipools:
        _add_index(rows, pools, rankings, report, ipools)
    report["embedding_cost_usd"] = emb.cost()
    report["settings"] = {"pool": {"dblp-search": POOL_DBLP, "s2-search": POOL_S2, "openalex-search": POOL_OA,
                                   "openalex-semantic": POOL_OA},
                          "bm25": {"k1": BM25_K1, "b": BM25_B, "idf": "pool-level"},
                          "dense": {"model": EMBED_MODEL, "dimensions": EMBED_DIMS}, "rrf_k": RRF_K}
    if not frozen:
        (cache_dir / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pools, rankings, report
