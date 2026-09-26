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
import random
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


def _search(ctx, query, dense, top=20):
    r = ctx.http.get(f"{config.SEARCH_URL}/search/papers",
                     params={"q": query, "top": top, "dense": str(bool(dense)).lower()},
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


def run(ctx, client, n_papers=60, seed=None, model=None, ledger=None):
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
    log.info("%s usable descriptions from %s papers", len(cases), len(papers))

    hybrid, sparse, failures = [], [], []
    t0 = time.time()
    for case in cases:
        try:
            both = _search(ctx, case["query"], dense=True)
            words_only = _search(ctx, case["query"], dense=False)
        except Exception as e:
            log.warning("search failed for %r: %s", case["query"], e)
            skipped += 1
            continue
        h, s = _rank(both.get("results", []), case["key"]), _rank(words_only.get("results", []), case["key"])
        hybrid.append(h)
        sparse.append(s)
        if h is None or (s is not None and s < h):
            failures.append({**case, "hybrid_rank": h, "words_only_rank": s})

    n = len(hybrid) or 1
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "a model describes each paper without reusing its distinctive words; the "
                  "description is then searched for, with the embeddings and without them",
        "caveat": "the queries are generated, not collected from users: they show whether meaning "
                  "survives a paraphrase, not what real users type",
        "sampled": len(papers), "usable": len(cases), "skipped": skipped,
        "hybrid": _summary(hybrid, n), "words_only": _summary(sparse, n),
        "gain_acc@10": round((_summary(hybrid, n)["acc@10"] or 0) - (_summary(sparse, n)["acc@10"] or 0), 4),
        "seconds": round(time.time() - t0, 1),
        "query_tokens": tokens,
        "examples": [{"title": c["title"], "query": c["query"]} for c in cases[:5]],
        "hybrid_missed": failures[:10],
    }


def save(payload, fingerprint):
    d = config.MODELS_DIR / "search-eval"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"paraphrase-{fingerprint}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
