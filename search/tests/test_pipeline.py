import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp(prefix="dblp-search-test-"))
os.environ.update({"CACHE_DIR": str(_tmp / "cache"), "MODELS_DIR": str(_tmp / "models"),
                   "DUCKDB_MEMORY": "1GB", "DUCKDB_THREADS": "2"})

from tests.make_serving import make  # noqa: E402

make(_tmp / "cache")

from search import bm25 as B, config, data, evaluate as EV, fuse as F, search as SR, store as S, vectors as V  # noqa: E402
from tests.fake_encoder import FakeEncoder  # noqa: E402


@pytest.fixture(scope="module")
def con():
    c, meta = data.connect()
    assert meta["fingerprint"] == "testfp0001"
    S.attach_store(c, meta, build_if_missing=True)
    yield c, meta
    c.close()


def test_store_only_indexes_recent_journal_conference_papers(con):
    c, _ = con
    total = c.execute("SELECT count(*) FROM s.pubs").fetchone()[0]
    indexed = c.execute("SELECT count(*) FROM x.paper").fetchone()[0]
    assert 0 < indexed < total   # preprints and pre-2010 papers are excluded
    assert c.execute("SELECT count(*) FROM x.paper WHERE year < ?", [config.FIRST_YEAR]).fetchone()[0] == 0
    kinds = {r[0] for r in c.execute("SELECT DISTINCT kind FROM x.paper").fetchall()}
    assert kinds <= {"journal", "conference"}
    # rows are 0-indexed and contiguous: build() relies on this to index straight into the vector file
    rows = sorted(r[0] for r in c.execute("SELECT row FROM x.paper").fetchall())
    assert rows == list(range(indexed))


def test_bm25_ranks_the_topic_that_shares_vocabulary(con):
    c, _ = con
    hits = B.search(c, "graph clustering community partition")
    assert hits, hits
    sid = c.execute("SELECT sid FROM x.paper WHERE pid = ?", [hits[0][0]]).fetchone()[0]
    assert sid == "conf/aaa"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_respects_kind_and_year_filters(con):
    c, _ = con
    all_hits = B.search(c, "graph clustering spectral vertex")
    filtered = B.search(c, "graph clustering spectral vertex", kind="journal")
    assert filtered != all_hits   # conf/aaa papers are 'conference', so a journal filter changes the set
    kinds = {c.execute("SELECT kind FROM x.paper WHERE pid = ?", [pid]).fetchone()[0] for pid, _ in filtered}
    assert kinds <= {"journal"}
    recent = B.search(c, "graph clustering spectral vertex", year_from=2020)
    years = [c.execute("SELECT year FROM x.paper WHERE pid = ?", [pid]).fetchone()[0] for pid, _ in recent]
    assert all(y >= 2020 for y in years)


def test_rrf_favours_items_ranked_well_by_both_lists():
    a = [(1, 0.9), (2, 0.5), (3, 0.1)]
    b = [(2, 0.9), (1, 0.5), (4, 0.1)]
    fused = F.rrf(a, b, k=1)
    ids = [pid for pid, _ in fused]
    assert fused[0][0] in (1, 2)   # both appear near the top of one list and mid the other
    assert 3 in ids and 4 in ids and len(ids) == 4
    assert 99 not in ids   # an id in neither list contributes nothing and cannot appear


def test_vector_build_is_resumable_and_matches_a_direct_encode(con):
    c, meta = con
    fp = meta["fingerprint"]
    enc = FakeEncoder()
    n = c.execute("SELECT count(*) FROM x.paper").fetchone()[0]

    first = V.build(c, fp, enc, batch_size=5, checkpoint_every=1)
    assert first["embedded"] == n and first["new_this_run"] == n
    prog = V.progress(c, fp)
    assert prog["complete"] and prog["embedded"] == n

    again = V.build(c, fp, enc, batch_size=5, checkpoint_every=1)
    assert again["new_this_run"] == 0 and again["embedded"] == n   # nothing left to do, safe to re-run

    rows = c.execute("SELECT row, title FROM x.paper ORDER BY row").fetchall()
    vectors = V.open_vectors(fp, n, "r")
    direct = enc.encode_docs([t for _, t in rows]).astype(V.DTYPE)
    assert np.allclose(np.asarray(vectors), direct, atol=2e-3)   # float16 round-trip tolerance


def test_dense_search_ranks_the_matching_topic(con):
    c, meta = con
    enc = FakeEncoder()
    hits = V.search(c, meta["fingerprint"], enc.encode_query("reinforcement policy reward bandit"))
    assert hits
    sid = c.execute("SELECT sid FROM x.paper WHERE pid = ?", [hits[0][0]]).fetchone()[0]
    assert sid == "conf/bbb"


def test_search_merges_hybrid_and_lexical_fallback(con):
    c, meta = con
    enc = FakeEncoder()
    out = SR.search(c, meta["fingerprint"], enc, "graph clustering community", top=8)
    assert out["results"] and out["dense_available"]
    assert any("bm25" in r["sources"] or "dense" in r["sources"] for r in out["results"])
    json.dumps(out)

    # a preprint's exact words are findable even though preprints sit outside the index by kind
    preprint_title = c.execute("SELECT title FROM s.pubs WHERE key LIKE 'corr/%' LIMIT 1").fetchone()[0]
    out2 = SR.search(c, meta["fingerprint"], enc, preprint_title, top=5)
    assert any(r["sources"] == ["exact_word"] for r in out2["results"]), out2["results"]


def test_search_rejects_a_short_query(con):
    c, meta = con
    assert "error" in SR.search(c, meta["fingerprint"], FakeEncoder(), "ab")


def test_search_can_skip_dense_explicitly(con):
    """The server does this while the embedding index is still building: the vector file exists
    (pre-allocated) long before it is useful, so it gates on completion, not existence."""
    c, meta = con
    out = SR.search(c, meta["fingerprint"], FakeEncoder(), "graph clustering community", top=5, dense=False)
    assert out["dense_available"] is False and out["dense_candidates"] == 0
    assert out["results"] and all("dense" not in r["sources"] for r in out["results"])


def test_evaluate_runs_and_reports_honest_coverage(con):
    """Every synthetic title includes two filler words, and every filler word is in the synonym
    table, so all sampled titles should be substitutable - this is a property of the fixture, not
    of the evaluation code, and is asserted here so a future fixture change fails loudly."""
    c, meta = con
    out = EV.evaluate(c, meta["fingerprint"], FakeEncoder(), n_papers=100)
    assert out["sampled"] == 100 and out["substitutable"] == 100
    for mode in ("bm25_only", "dense_only", "hybrid"):
        s = out[mode]
        assert s["papers"] == out["substitutable"] and 0 <= s["found"] <= s["papers"]
        for k in ("acc@1", "acc@5", "acc@10"):
            assert s[k] is None or 0.0 <= s[k] <= 1.0


def test_a_query_of_common_words_is_bounded(con):
    """A description has no rare words, so BM25's rarest tokens are still expensive ones. The
    cumulative document frequency is what has to be capped - a paraphrase query timed out in
    production with only the token count limited."""
    c, _ = con
    common = c.execute("SELECT token FROM x.token_df ORDER BY df DESC LIMIT 8").fetchall()
    query = " ".join(t[0].replace("_", " ") for t in common)

    generous = S.query_tokens(c, query, max_postings=10 ** 9)
    tight = S.query_tokens(c, query, max_postings=1)
    assert len(tight) == config.MIN_QUERY_TOKENS, "a query must never be left with nothing to match"
    assert len(tight) <= len(generous)
    assert tight == generous[:len(tight)], "the tokens kept are the rarest ones"


def test_the_word_ranking_s_weight_follows_the_query_s_rarest_word():
    """Stated in IDF, so it is tested in IDF: the thresholds assume a corpus of millions, where a
    word held by 0.1% of papers is distinctive and one held by 5% is not."""
    assert S.sparse_weight({"idf": config.IDF_FULL}) == 1.0
    assert S.sparse_weight({"idf": config.IDF_FULL + 5}) == 1.0
    assert S.sparse_weight({"idf": config.IDF_FLOOR}) == config.SPARSE_FLOOR
    assert S.sparse_weight({"idf": 0.5}) == config.SPARSE_FLOOR
    assert S.sparse_weight({}) == config.SPARSE_FLOOR          # no tokens matched at all
    middle = S.sparse_weight({"idf": (config.IDF_FLOOR + config.IDF_FULL) / 2})
    assert config.SPARSE_FLOOR < middle < 1.0


def test_bigrams_do_not_make_a_vague_query_look_specific(con):
    """The bug this signal was nearly shipped with: "systems learning data" produces the bigrams
    systems_learning and learning_data, both rare, so the rarest token said "specific".

    Only this reading is asserted against the fixture. Its titles are built from small per-topic
    word lists, so no word in it is rare and every query lands on the floor - the thresholds
    themselves are tested directly above, where they are defined."""
    c, _ = con
    common = [t[0] for t in c.execute("SELECT token FROM x.token_df WHERE NOT contains(token, '_') "
                                      "ORDER BY df DESC LIMIT 3").fetchall()]
    S.query_tokens(c, " ".join(common))
    stats = S.query_stats(c)
    assert stats["word_df"] is not None
    assert stats["word_df"] >= (stats["min_df"] or 0), "a word is never rarer than the bigrams it makes"
    assert stats["idf"] == pytest.approx(math.log(
        c.execute("SELECT count(*) FROM x.paper").fetchone()[0] / stats["word_df"]), rel=1e-6)
