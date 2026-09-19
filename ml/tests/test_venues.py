"""Venue recommendation on the synthetic serving database: venues with planted vocabularies, and
authors loyal to their community's venues."""
import json

import numpy as np
import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db)
from ml import data
from ml.venues import candidates as C, cli, config, evaluate as E, features as F, model as M, predict as P, store as S


class Args:
    rank_papers = 2000
    test_papers = 2000
    drop_features = ""
    fingerprint = None
    title = None
    authors = None
    key = None
    top = 10


@pytest.fixture(scope="module")
def con():
    c, meta = data.connect()
    S.attach_store(c, meta, build_if_missing=True)
    yield c, meta
    c.close()


def test_store_holds_tokens_bigrams_and_no_bins(con):
    c, meta = con
    assert c.execute("SELECT count(*) FROM v.paper").fetchone()[0] > 500
    assert c.execute("SELECT count(*) FROM v.title_token WHERE token LIKE '%\\_%' ESCAPE '\\'").fetchone()[0] > 0
    bins = c.execute("""
        SELECT count(*) FROM v.paper_author pa JOIN s.persons p USING (person_id)
        WHERE p.page_kind = 'disambiguation'""").fetchone()[0]
    assert bins == 0
    # token_df is exactly the index's per-token count
    assert c.execute("""
        SELECT count(*) FROM (SELECT token, count(*) AS n FROM v.title_token GROUP BY 1) x
        JOIN v.token_df d USING (token) WHERE d.df <> x.n""").fetchone()[0] == 0
    assert dict(c.execute("SELECT k, v FROM v._meta").fetchall())["fingerprint"] == meta["fingerprint"]


def test_statistics_see_nothing_after_their_year(con):
    c, meta = con
    T = S.years(meta)[0] - 1
    S.build_stats(c, T)
    off = c.execute("""
        SELECT count(*) FROM cls
        JOIN (SELECT sid, count(*) AS n FROM v.paper WHERE year <= ? GROUP BY sid) x USING (sid)
        WHERE cls.papers <> x.n""", [T]).fetchone()[0]
    assert off == 0
    assert c.execute("SELECT max(last_year) FROM cls").fetchone()[0] <= T
    assert c.execute("SELECT count(*) FROM stats").fetchone()[0] > 0
    # every series' centroid is unit length
    norms = c.execute("SELECT sid, sum(cen_w * cen_w) FROM stats GROUP BY 1").fetchall()
    assert all(abs(n - 1.0) < 1e-3 for _, n in norms), norms[:3]


def test_candidates_use_only_earlier_history_and_are_bounded(con):
    c, meta = con
    year = S.years(meta)[1]
    S.build_stats(c, year - 1)
    assert C.queries_from_papers(c, year, 500) > 20
    assert C.build_pairs(c) > 0
    assert c.execute("SELECT count(*) FROM hist h JOIN q USING (qid) WHERE h.hist_last >= q.year").fetchone()[0] == 0
    per_query = c.execute("SELECT max(n) FROM (SELECT qid, count(*) AS n FROM pair GROUP BY 1)").fetchone()[0]
    assert per_query <= 2 * config.TOP_CONTENT + config.MAX_HISTORY
    assert c.execute("SELECT max(n) FROM (SELECT qid, count(*) AS n FROM q_used GROUP BY 1)").fetchone()[0] <= config.MAX_QUERY_TOKENS
    X, y, info = F.matrix(c)
    assert X.shape[1] == len(F.FEATURES) and np.isfinite(X).all() and y.sum() > 0


def test_rank_metrics_on_a_known_case():
    qid = np.array([1, 1, 1, 2, 2, 3, 3])
    y = np.array([0, 1, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.5, 0.1, 0.9, 0.5, 0.9, 0.5])
    r = E.true_ranks(qid, score, y)
    assert r == {1: 2, 2: 1}                       # query 3's venue is not a candidate
    s = E.summarize(r, n=3, ks=(1, 2))
    assert s == {"papers": 3, "found": 2, "acc@1": round(1 / 3, 4), "acc@2": round(2 / 3, 4), "mrr": round((0.5 + 1) / 3, 4)}


def test_train_learns_planted_vocabulary_and_loyalty(con):
    c, meta = con
    assert cli.train(c, meta, Args()) == 0
    saved = json.loads((M.model_dir(meta["fingerprint"]) / M.METRICS_FILE).read_text(encoding="utf-8"))
    test = saved["metrics"]["test"]
    assert test["year"] == S.years(meta)[1]
    assert test["covered"]["share"] > 0.5 and test["in_candidates"]["share"] > 0.5
    r = test["rankers"]
    assert r["model"]["acc@1"] > r["popularity"]["acc@1"]
    assert r["model"]["mrr"] >= max(r[k]["mrr"] for k in ("naive_bayes", "centroid", "history")) - 0.05, r
    assert r["model"]["acc@1"] > 0.4, r
    assert "with_history" in test and "without_history" in test
    assert saved["calibration"] and saved["metrics"]["feature_importance"][0]["feature"] in F.FEATURES
    assert saved["metrics"]["serving_statistics"]["series"] > 0
    for name in ("stats", "stats_series", "vocab"):
        assert (M.model_dir(meta["fingerprint"]) / f"{name}.parquet").exists()


def test_a_title_in_a_venues_vocabulary_lands_there(con):
    c, _ = con
    out = P.suggest(c, "Reward-driven planning for bandit agents under a policy", top=5)
    assert "error" not in out, out
    assert out["suggestions"][0]["sid"] == "conf/aaa", out["suggestions"]
    assert out["suggestions"][0]["content"]["nb_rank"] == 1
    assert out["related"] and sum(r["sid"] == "conf/aaa" for r in out["related"]) >= len(out["related"]) // 2
    assert out["tokens_used"]
    json.dumps(out)


def test_authors_history_moves_the_ranking(con):
    c, _ = con
    # a title with no vocabulary of any venue: only the authors' history and popularity can speak
    key = c.execute("""
        SELECT p.key FROM s.persons p JOIN v.author_venue av USING (person_id)
        WHERE p.page_kind = 'regular' GROUP BY p.key ORDER BY sum(av.papers) DESC LIMIT 1""").fetchone()[0]
    without = P.suggest(c, "Robust efficient adaptive secure system", top=5)
    with_h = P.suggest(c, "Robust efficient adaptive secure system", author_keys=[key], top=5)
    assert with_h["query"]["authors"][0]["key"] == key
    top = with_h["suggestions"][0]
    assert top["history"] and top["history"]["papers"] > 0
    assert without["suggestions"][0]["history"] is None


def test_for_paper_marks_the_real_venue(con):
    c, _ = con
    key = c.execute("SELECT key FROM v.paper WHERE key LIKE 'conf/aaa/l%' ORDER BY year DESC, key LIMIT 1").fetchone()[0]
    out = P.for_paper(c, key, top=5)
    assert "error" not in out, out
    assert out["actual"]["sid"] == "conf/aaa" and out["actual"]["in_class_set"]
    assert out["actual"]["rank"] == 1, out["suggestions"]
    assert all(r["key"] != key for r in out["related"])
    assert "error" in P.for_paper(c, "conf/nope/x")
    json.dumps(out)


def test_came_true_rate_lookup():
    calib = [{"from": 0.0, "to": 0.5, "came_true": 0.1}, {"from": 0.5, "to": 1.0, "came_true": 0.7}]
    assert E.came_true_rate(calib, 0.2) == 0.1 and E.came_true_rate(calib, 1.0) == 0.7
    assert E.came_true_rate([], 0.2) is None
