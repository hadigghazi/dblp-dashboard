"""
Realistic retrieval for DBLP-QA: what happens between "no abstract" and "the right abstract".

The oracle run showed that, given the abstract a question was written from, every model scores about
2/2 - generation is solved on this benchmark, so everything that still varies is retrieval. This
module measures it, the way RAGScholar does it and the way its own future work proposes.

**The candidate pool.** RAGScholar ran BM25 over a Lucene index of 4.6 million Semantic Scholar
abstracts; that bulk abstract set is no longer fully available (the paper's own footnote), so no one
can rebuild it. Instead each question gets a pool from two first-stage retrievers that need no bulk
abstracts:

  * dblp-search - this project's hybrid search over every dblp title (BM25 + text-embedding-3-large,
    fused), top 50;
  * s2-search - Semantic Scholar's own relevance search, top 100, kept only where the paper has a
    dblp key, so the pool is dblp, as the paper's corpus was.

Abstracts for the pool come from Semantic Scholar by DOI, falling back to OpenAlex, and are cached.

**The rankers**, applied to the same pool so that only the ranking differs:

  * bm25 - the paper's method, Lucene's formula and defaults (k1 = 1.2, b = 0.75; the paper reports
    b = 2, which BM25 does not allow), over title + abstract;
  * dense - cosine similarity of text-embedding-3-large embeddings (512 dimensions);
  * hybrid - reciprocal rank fusion (k = 60) of the two, which is the paper's stated future work.

The two first-stage orders are scored as well, as what a reader would get from either search box.
Pool statistics stand in for corpus statistics in BM25's idf, the standard compromise when re-ranking
a pool, stated wherever the results are.

**Which paper is "the source".** dblp often holds a preprint and its published version under two keys
with one title, and the benchmark names one of them. A candidate counts as the source if it has the
benchmark's key, Semantic Scholar's dblp key for the benchmark's paper, or the same normalised title.

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
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 512
EMBED_PRICE = 0.13            # $ per 1M tokens
BM25_K1, BM25_B = 1.2, 0.75
RRF_K = 60
RERANKERS = ("bm25", "dense", "hybrid")
FIRST_STAGE = ("dblp-search", "s2-search")
RANKERS = RERANKERS + FIRST_STAGE
TOP_K = 5
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_WORKS = "https://api.openalex.org/works"
PARQUET = Path(os.environ.get("PARQUET", config.DATA_DIR / "parquet" / "dblp.parquet"))

# Lucene's English stop set plus the question words every DBLP-QA question opens with
STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then "
           "there these they this to was will with what how why which who does do".split())


def tokens(text):
    return [t for t in re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
            if t not in STOP and len(t) > 1]


def norm_title(title):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def doc_text(entry):
    title = (entry.get("title") or "").strip()
    abstract = (entry.get("abstract") or "").strip()
    return f"{title}. {abstract}" if abstract else title


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
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
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


# --------------------------------------------------------------------------- the pool

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


def ids_from_dblp(keys, parquet=PARQUET):
    """{dblp key: {"doi", "arxiv"}} from the records' `ee` links, one scan of the parquet for all."""
    if not keys or not Path(parquet).exists():
        return {}
    import duckdb
    con = duckdb.connect()
    try:
        rows = con.execute(f"SELECT key, ee FROM read_parquet('{Path(parquet).as_posix()}') "
                           "WHERE key IN (SELECT unnest(?::VARCHAR[]))", [list(keys)]).fetchall()
    finally:
        con.close()
    return {key: ids for key, ee in rows if (ids := ids_from_ee(ee))}


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
            r = DQ._get(http, OPENALEX_WORKS, params={"filter": "doi:" + "|".join(by_doi), "per-page": 50,
                                                      "select": "doi,abstract_inverted_index"})
            if r.status_code != 200:
                continue
            for work in (r.json() or {}).get("results") or []:
                doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
                index = work.get("abstract_inverted_index") or {}
                if doi in by_doi and index:
                    words = sorted((p, w) for w, ps in index.items() for p in ps)
                    cache[by_doi[doi]] = " ".join(w for _, w in words)
        for k in need:
            cache.setdefault(k, None)
    for k, e in entries.items():
        e["abstract"] = e.get("abstract") or cache.get(k)


def source_aliases(rows, http):
    """{question id: set of dblp keys that are the source paper}: the benchmark's key plus Semantic
    Scholar's dblp key for the benchmark's paper (its CorpusId), which differ when one is a preprint."""
    aliases = {r["id"]: {r["dblp_key"]} for r in rows}
    r = DQ._get(http, DQ.S2_BATCH, method="POST", params={"fields": "externalIds"},
                json={"ids": [f"CorpusId:{row['semantic_scholar_id']}" for row in rows]})
    if r.status_code == 200:
        for row, paper in zip(rows, r.json()):
            key = ((paper or {}).get("externalIds") or {}).get("DBLP")
            if key:
                aliases[row["id"]].add(key)
    return aliases


def _s2_query(question):
    # Semantic Scholar's search matches nothing for hyphenated terms (its documentation says so)
    return " ".join(question.replace("-", " ").split())[:300]


def build_pools(rows, oracle, http=None, cache_dir=None, out=print):
    """{question id: {"candidates": {dblp key: {title, abstract, doi, dblp_rank, s2_rank}},
    "aliases": [source keys]}}, built once and cached - the pool is the experiment's fixed input."""
    cache_dir = Path(cache_dir or config.MODELS_DIR / "dblpqa")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pools_path, abstracts_path = cache_dir / "pools.json", cache_dir / "pool-abstracts.json"
    pools = json.loads(pools_path.read_text(encoding="utf-8")) if pools_path.exists() else {}
    abstracts = json.loads(abstracts_path.read_text(encoding="utf-8")) if abstracts_path.exists() else {}
    http = http or httpx.Client(follow_redirects=True, timeout=60,
                                headers={"User-Agent": "dblp-explorer-research"})
    # a Semantic Scholar search that was refused (rate limit) is asked again rather than kept as empty
    todo = [r for r in rows if r["id"] not in pools or pools[r["id"]].get("s2_status") != 200]
    aliases = source_aliases(todo, http) if todo else {}
    for i, row in enumerate(todo, 1):
        cands = {}
        r = http.get(f"{config.SEARCH_URL}/search/papers",
                     params={"q": row["question"][:300], "top": POOL_DBLP}, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"dblp search failed for {row['id']} ({r.status_code}): {r.text[:200]}")
        for rank, hit in enumerate(r.json().get("results") or [], 1):
            cands.setdefault(hit["key"], {"title": hit.get("title")})["dblp_rank"] = rank
        r = DQ._get(http, S2_SEARCH, params={"query": _s2_query(row["question"]), "limit": POOL_S2,
                                             "fields": "title,abstract,externalIds"})
        s2 = ((r.json() or {}).get("data") or []) if r.status_code == 200 else []
        rank = 0
        for paper in s2:
            ids = paper.get("externalIds") or {}
            if not ids.get("DBLP"):
                continue
            rank += 1
            entry = cands.setdefault(ids["DBLP"], {"title": paper.get("title")})
            entry["s2_rank"] = rank
            entry["abstract"] = entry.get("abstract") or paper.get("abstract")
            entry["doi"] = entry.get("doi") or ids.get("DOI")
        pools[row["id"]] = {"candidates": cands, "aliases": sorted(aliases.get(row["id"], {row["dblp_key"]})),
                            "s2_status": r.status_code, "s2_results": len(s2)}
        if i % 10 == 0 or i == len(todo):
            out(f"  pools: {i}/{len(todo)} searched")
            pools_path.write_text(json.dumps(pools, ensure_ascii=False), encoding="utf-8")
        time.sleep(1.0)               # Semantic Scholar's unauthenticated pool is shared and easily upset

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
        for name, field in (("dblp-search", "dblp_rank"), ("s2-search", "s2_rank")):
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
    n = len(rows) or 1
    report["pool"] = {"mean_size": round(sum(sizes) / n, 1),
                      "mean_with_abstract": round(sum(with_abstract) / n, 1),
                      "share_with_abstract": round(sum(with_abstract) / max(1, sum(sizes)), 3),
                      "source_in_pool": round(sum(in_pool) / n, 3),
                      "s2_search_failed": sum(1 for r in rows if pools[r["id"]].get("s2_status") != 200)}
    return report


def print_retrieval(report, out=print):
    p = report["pool"]
    out(f"pool: {p['mean_size']} candidates per question on average, {p['share_with_abstract']:.0%} with "
        f"an abstract; the source paper is in the pool for {p['source_in_pool']:.0%} of questions"
        + (f"; Semantic Scholar search failed for {p['s2_search_failed']}" if p["s2_search_failed"] else ""))
    out(f"\n{'ranker':12s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR@10':>7s}")
    for name, m in report["rankers"].items():
        out(f"{name:12s} {m['recall@1']:6.2f} {m['recall@3']:6.2f} {m['recall@5']:6.2f} "
            f"{m['recall@10']:6.2f} {m['mrr@10']:7.3f}")


def rag_contexts(rows, pools, rankings, ranker, k=TOP_K):
    """The top-k papers as one concatenated context per question - the paper's best strategy (Top-k
    Concatenated Documents) - with where the source paper landed, for the analysis."""
    contexts = {}
    for row in rows:
        ranking = rankings[ranker].get(row["id"]) or []
        cands = pools[row["id"]]["candidates"]
        keys = ranking[:k]
        blocks = [f"[{i}] {cands[key].get('title') or ''}\n{cands[key].get('abstract') or '(no abstract available)'}"
                  for i, key in enumerate(keys, 1)]
        rank = source_rank(ranking, set(pools[row["id"]]["aliases"]))
        contexts[row["id"]] = {"abstract": "\n\n".join(blocks) or "(no papers were retrieved)",
                               "source": f"{ranker}@{k}", "retrieved": keys, "source_rank": rank,
                               "source_in_context": rank is not None and rank <= k}
    return contexts


def prepare(rows, out=print, cache_dir=None, http=None, embeddings=None):
    """Pool, abstracts, embeddings and every ranking, all cached; returns (pools, rankings, report)."""
    cache_dir = Path(cache_dir or config.MODELS_DIR / "dblpqa")
    oracle = DQ.fetch_abstracts(rows, cache_dir=cache_dir, http=http, out=out)
    pools = build_pools(rows, oracle, http=http, cache_dir=cache_dir, out=out)
    emb = embeddings or Embeddings(cache_dir / "embeddings.json")
    rankings = rank_pools(rows, pools, emb, out)
    report = evaluate_retrieval(rows, pools, rankings)
    report["embedding_cost_usd"] = emb.cost()
    report["settings"] = {"pool": {"dblp-search": POOL_DBLP, "s2-search": POOL_S2},
                          "bm25": {"k1": BM25_K1, "b": BM25_B, "idf": "pool-level"},
                          "dense": {"model": EMBED_MODEL, "dimensions": EMBED_DIMS}, "rrf_k": RRF_K}
    (cache_dir / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pools, rankings, report
