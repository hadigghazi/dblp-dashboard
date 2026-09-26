"""
Measuring the thing hybrid search was built for.

The search service measures itself by swapping one title word for a synonym, and on real titles BM25
alone recovers from that almost perfectly (98.05% vs 98.65% hybrid) - so the number is honest but it
does not test the case the embeddings exist for: somebody who does not know the title and describes
the paper in their own words.

That case had no ground truth, which is why it went unmeasured. It does now, because there is a model
in this service: take a paper, ask the model to describe it in a sentence *without reusing its
distinctive words*, and the pair (description, paper) is a labelled query whose answer is known. This
is the standard query-generation trick for retrieval evaluation; its weakness is that the queries are
synthetic, which is stated in the report rather than hidden.

The comparison that matters is the same query run twice: with the embeddings and without them. The
search endpoint takes `dense=false` for exactly this.
"""
import json
import logging
import re
import time
from datetime import datetime, timezone

from . import config

log = logging.getLogger("dblp.chat.searcheval")

PROMPT = (
    "You are writing a search query the way a researcher would when they remember a paper's topic "
    "but not its title.\n"
    "Describe what this paper is about in ONE sentence of at most 20 words.\n"
    "Rules: do not reuse any distinctive word from the title (ordinary words like 'for', 'data', "
    "'model' are fine); do not name the venue or the authors; write it as a description, not a "
    "title; no quotation marks.\n\nTitle: {title}\nDescription:"
)
STOP = set("a an the of for and or in on to with from into over under via using based toward towards "
           "its their this that these those are can not than versus vs our your who how what when why "
           "which where is by at as new".split())


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2 and w not in STOP}


def queries_path(fingerprint, seed, n):
    d = config.MODELS_DIR / "search-eval"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"queries-{fingerprint}-{seed}-{n}.json"


def sample_papers(ctx, n, seed=None):
    """Papers from the same population the search index covers."""
    cur = ctx.cursor()
    cur.execute("""
        SELECT key, title, year FROM s.pubs
        WHERE key_prefix IN ('conf', 'journals') AND NOT is_preprint AND sid IS NOT NULL
          AND title IS NOT NULL AND length(title) >= 30 AND year >= ?
        QUALIFY row_number() OVER (ORDER BY hash(pid::BIGINT * 1000003 + ?)) <= ?""",
                [config.SEARCH_FIRST_YEAR, int(seed if seed is not None else 7), int(n)])
    return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]


def describe(client, title, model=None):
    """One vague query for one title, or None if the model reused the title's own words anyway."""
    out = client.complete([{"role": "user", "content": PROMPT.format(title=title)}],
                          model=model or config.MODEL_FAST)
    text = (out.get("content") or "").strip().strip('"').replace("\n", " ")
    if not text:
        return None, out.get("usage", {})
    overlap = _words(text) & _words(title)
    # a description that keeps a distinctive word is not the case we are trying to measure
    return (None if len(overlap) > 1 else text), out.get("usage", {})


def _rank(results, key):
    for i, row in enumerate(results, start=1):
        if row.get("key") == key:
            return i
    return None


SEARCH_TIMEOUT = 90     # a description is a long query, and the service answers one search at a time


def _search(ctx, query, dense=True, sparse=True, top=20):
    r = ctx.http.get(f"{config.SEARCH_URL}/search/papers",
                     params={"q": query, "top": top, "dense": str(bool(dense)).lower(),
                             "sparse": str(bool(sparse)).lower()},
                     timeout=SEARCH_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _summary(ranks, n, ks=(1, 5, 10, 20)):
    found = [r for r in ranks if r is not None]
    out = {"queries": n, "found": len(found)}
    for k in ks:
        out[f"acc@{k}"] = round(sum(1 for r in found if r <= k) / n, 4) if n else None
    out["mrr"] = round(sum(1 / r for r in found) / n, 4) if n else None
    return out


def build_queries(ctx, client, n_papers, seed, model=None, ledger=None):
    """The descriptions, generated once and kept: a test whose questions change between runs
    cannot answer whether a change to the ranker helped."""
    papers = sample_papers(ctx, n_papers, seed)
    cases, skipped, tokens = [], 0, {"input_tokens": 0, "output_tokens": 0}
    for paper in papers:
        query, usage = describe(client, paper["title"], model)
        tokens["input_tokens"] += usage.get("input_tokens", 0)
        tokens["output_tokens"] += usage.get("output_tokens", 0)
        if ledger is not None:                      # writing the queries costs money like anything else
            ledger.record(model or config.MODEL_FAST, usage)
        if not query:
            skipped += 1
            continue
        cases.append({"key": paper["key"], "title": paper["title"], "query": query})
    return {"sampled": len(papers), "skipped_unusable": skipped, "cases": cases,
            "query_tokens": tokens, "seed": seed, "model": model or config.MODEL_FAST}


def run(ctx, client, n_papers=60, seed=None, model=None, ledger=None, regenerate=False):
    seed = 7 if seed is None else seed
    path = queries_path(ctx.meta.get("fingerprint", "unknown"), seed, n_papers)
    if path.exists() and not regenerate:
        built = json.loads(path.read_text(encoding="utf-8"))
        log.info("reusing %s descriptions from %s", len(built["cases"]), path.name)
    else:
        built = build_queries(ctx, client, n_papers, seed, model, ledger)
        path.write_text(json.dumps(built, indent=2, default=str), encoding="utf-8")
    cases, skipped = built["cases"], built["skipped_unusable"]
    papers, tokens = [None] * built["sampled"], built["query_tokens"]
    log.info("%s usable descriptions from %s papers", len(cases), len(papers))

    hybrid, words, dense_only, failures = [], [], [], []
    weights, coverages = [], []          # did the fusion weighting actually fire on these?
    paired = {"hybrid_better": 0, "embeddings_better": 0, "same": 0}
    t0 = time.time()
    for case in cases:
        try:
            both = _search(ctx, case["query"])
            words_arm = _search(ctx, case["query"], dense=False)
            dense_arm = _search(ctx, case["query"], sparse=False)
        except Exception as e:
            log.warning("search failed for %r: %s", case["query"], e)
            skipped += 1
            continue
        weights.append(both.get("sparse_weight"))
        coverages.append(both.get("word_match_coverage"))
        h = _rank(both.get("results", []), case["key"])
        w = _rank(words_arm.get("results", []), case["key"])
        d = _rank(dense_arm.get("results", []), case["key"])
        hybrid.append(h)
        words.append(w)
        dense_only.append(d)
        # fusion losing to the embeddings alone is the interesting failure: blending in a
        # ranking that found nothing is then costing the answer
        if h is None or (d is not None and d < h):
            failures.append({**case, "hybrid_rank": h, "words_only_rank": w, "dense_only_rank": d})
        # paired, because that is the comparison the summaries cannot make honestly at this size
        if (h or 10 ** 6) < (d or 10 ** 6):
            paired["hybrid_better"] += 1
        elif (d or 10 ** 6) < (h or 10 ** 6):
            paired["embeddings_better"] += 1
        else:
            paired["same"] += 1

    n = len(hybrid) or 1
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "a model describes each paper without reusing its distinctive words; the "
                  "description is then searched for, with the embeddings and without them",
        "caveat": "the queries are generated, not collected from users: they show whether meaning "
                  "survives a paraphrase, not what real users type",
        "queries_from": path.name, "regenerated": bool(regenerate) or not path.exists(),
        "sampled": len(papers), "usable": len(cases), "skipped": skipped,
        "hybrid_vs_embeddings_per_query": paired,
        "hybrid": _summary(hybrid, n), "words_only": _summary(words, n),
        "embeddings_only": _summary(dense_only, n),
        "gain_over_words_acc@10": round((_summary(hybrid, n)["acc@10"] or 0)
                                        - (_summary(words, n)["acc@10"] or 0), 4),
        # negative means fusion is diluting the embeddings with a ranking that found nothing
        "fusion_cost_acc@10": round((_summary(hybrid, n)["acc@10"] or 0)
                                    - (_summary(dense_only, n)["acc@10"] or 0), 4),
        "seconds": round(time.time() - t0, 1),
        "mean_sparse_weight": round(sum(w for w in weights if w is not None)
                                    / max(1, len([w for w in weights if w is not None])), 3),
        "mean_word_match_coverage": round(sum(c for c in coverages if c is not None)
                                          / max(1, len([c for c in coverages if c is not None])), 3),
        "queries_with_no_word_weight": sum(1 for w in weights if not w),
        "query_tokens": tokens,
        "examples": [{"title": c["title"], "query": c["query"]} for c in cases[:5]],
        "hybrid_lost_to_embeddings_alone": failures[:10],
    }


def save(payload, fingerprint):
    d = config.MODELS_DIR / "search-eval"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"paraphrase-{fingerprint}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
