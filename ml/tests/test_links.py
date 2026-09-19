"""Co-author link prediction on the synthetic serving database's planted link world."""
import json

import numpy as np
import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db)
from ml import data
from ml.links import cli, config, evaluate as E, features as F, graph as G, model as M, predict as P


class Args:
    train_anchors = 500
    test_anchors = 500
    drop_features = ""
    key = None
    top = 10
    fingerprint = None


@pytest.fixture(scope="module")
def con():
    c, meta = data.connect()
    G.attach_store(c, meta, build_if_missing=True)
    yield c, meta
    c.close()


def test_graph_store_has_no_bins_and_both_directions(con):
    c, meta = con
    bins = c.execute("""
        SELECT count(*) FROM g.adj_year x JOIN s.persons p ON p.person_id = x.a OR p.person_id = x.b
        WHERE p.page_kind = 'disambiguation'""").fetchone()[0]
    assert bins == 0
    one_way = c.execute("""
        SELECT count(*) FROM g.adj_year x
        LEFT JOIN g.adj_year r ON r.a = x.b AND r.b = x.a AND r.year = x.year
        WHERE r.a IS NULL""").fetchone()[0]
    assert one_way == 0
    gmeta = dict(c.execute("SELECT k, v FROM g._meta").fetchall())
    assert gmeta["fingerprint"] == meta["fingerprint"] and int(gmeta["edges"]) > 100


def test_snapshot_years_leave_the_windows_disjoint(con):
    _, meta = con
    t_train, t_test = G.snapshot_years(meta)
    assert t_test == int(meta["last_full_year"]) - config.HORIZON
    assert t_train + config.HORIZON <= t_test


def test_candidates_are_distance_two_non_coauthors_with_correct_labels(con):
    c, meta = con
    T = G.snapshot_years(meta)[0]
    assert G.select_anchors(c, T, set(range(10)), 1000) > 20
    assert G.build_pairs(c, T) > 0
    already = c.execute("""
        SELECT count(*) FROM pair p JOIN g.adj_year x ON x.a = p.u AND x.b = p.v WHERE x.year <= ?""", [T]).fetchone()[0]
    assert already == 0
    assert c.execute("SELECT min(cn) FROM pair").fetchone()[0] >= 1
    # every label is exactly "a joint paper in (T, T+H]"
    wrong = c.execute("""
        SELECT count(*) FROM pair p
        LEFT JOIN (SELECT DISTINCT a, b FROM g.adj_year WHERE year > ? AND year <= ?) f ON f.a = p.u AND f.b = p.v
        WHERE p.y <> (f.a IS NOT NULL)::INT""", [T, T + config.HORIZON]).fetchone()[0]
    assert wrong == 0
    pos = c.execute("SELECT sum(y) FROM pair").fetchone()[0]
    assert pos > 0


def test_nothing_from_after_the_snapshot_reaches_the_features(con):
    """Degrees, activity and bridge ages must be computed as of T: recomputing them straight from
    the serving tables with year <= T has to give the same numbers."""
    c, meta = con
    T = G.snapshot_years(meta)[0]
    G.select_anchors(c, T, set(range(10)), 1000)
    G.build_pairs(c, T)
    mismatch = c.execute("""
        WITH truth AS (
            SELECT a.person_id, count(DISTINCT b.person_id) AS deg
            FROM s.slots a JOIN s.slots b ON b.pid = a.pid AND b.person_id <> a.person_id
            JOIN s.pubs p ON p.pid = a.pid
            JOIN s.persons pa ON pa.person_id = a.person_id AND pa.page_kind <> 'disambiguation'
            JOIN s.persons pb ON pb.person_id = b.person_id AND pb.page_kind <> 'disambiguation'
            WHERE p.n_authors BETWEEN 2 AND 50 AND p.year <= ?
            GROUP BY 1)
        SELECT count(*) FROM pair p JOIN truth t ON t.person_id = p.v WHERE t.deg <> p.deg_v""", [T]).fetchone()[0]
    assert mismatch == 0
    for col in ("bridge_age", "idle_u", "idle_v", "age_u", "age_v"):
        assert c.execute(f"SELECT min({col}) FROM pair").fetchone()[0] >= 0, col
    X, y, info = F.matrix(c)
    assert X.shape[1] == len(F.FEATURES) and np.isfinite(X).all()


def test_origin_of_new_links_adds_up(con):
    c, meta = con
    T = G.snapshot_years(meta)[1]
    G.select_anchors(c, T, set(range(10)), 1000)
    G.build_pairs(c, T)
    origin = G.origin_of_new_links(c, T)
    assert origin["new_links"] > 0
    assert abs(sum(origin[k]["share"] for k in ("distance 2", "farther", "newcomer")) - 1) < 1e-3
    assert 0 <= origin["distance 2"]["kept_after_cap"] <= origin["distance 2"]["links"]


def test_ranking_metrics_on_a_known_case():
    u = np.array([1, 1, 1, 2, 2, 2, 3, 3])
    y = np.array([0, 1, 0, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.5, 0.1, 0.9, 0.5, 0.1, 0.9, 0.5])
    r = E.ranking(u, score, y, ks=(1, 2))
    assert r["anchors"] == 3 and r["anchors_with_new_link"] == 2   # anchor 3 gained nothing: not scored
    assert r["mrr"] == round((1 / 2 + 1 / 1) / 2, 4)
    assert r["hits@1"] == 0.5 and r["hits@2"] == 1.0
    assert r["precision@2"] == 0.5 and r["recall@2"] == 1.0


def test_train_learns_the_planted_closure_and_recency(con):
    c, meta = con
    assert cli.train(c, meta, Args()) == 0
    saved = json.loads((M.model_dir(meta["fingerprint"]) / M.METRICS_FILE).read_text(encoding="utf-8"))
    test = saved["metrics"]["test"]
    assert test["snapshot"] == G.snapshot_years(meta)[1]
    assert test["pooled"]["positives"] > 0 and test["ranking"]["anchors_with_new_link"] > 0
    assert test["pooled"]["roc_auc"] > 0.6, test["pooled"]
    # the heuristics see the closure signal too; the model must at least keep up with the best of them
    best = max(b["pooled"].get("roc_auc", 0) for b in test["baselines"].values())
    assert test["pooled"]["roc_auc"] >= best - 0.03, (test["pooled"], test["baselines"])
    assert set(test["baselines"]) == set(F.HEURISTICS)
    assert saved["calibration"] and all(0 <= r["came_true"] <= 1 for r in saved["calibration"])
    assert saved["metrics"]["feature_importance"][0]["feature"] in F.FEATURES
    assert saved["metrics"]["dataset"]["train"]["new_links_origin"]["new_links"] > 0


def test_predict_suggests_active_people_who_are_not_yet_coauthors(con):
    c, meta = con
    # the busiest recent authors of the link world; some may already know everyone within reach
    keys = [r[0] for r in c.execute("""
        SELECT p.key FROM s.persons p JOIN g.node_year n USING (person_id)
        WHERE p.page_kind = 'regular' AND n.year >= 2023
        GROUP BY p.key ORDER BY sum(n.papers) DESC, p.key LIMIT 8""").fetchall()]
    out = next((o for o in (P.suggest(c, k, top=5) for k in keys) if o.get("suggestions")), None)
    assert out is not None, "no author of the link world got a suggestion"
    key = out["author"]["key"]
    assert out["author"]["co_authors"] > 0
    json.dumps(out)
    anchor = c.execute("SELECT person_id FROM s.persons WHERE key = ?", [key]).fetchone()[0]
    coauthors = {r[0] for r in c.execute("SELECT b FROM g.adj_year WHERE a = ?", [anchor]).fetchall()}
    for s in out["suggestions"]:
        v = c.execute("SELECT person_id FROM s.persons WHERE key = ?", [s["key"]]).fetchone()[0]
        assert v not in coauthors and v != anchor
        assert s["via"] and s["common_coauthors"] >= len(s["via"]) and 0 <= s["score"] <= 1
        assert s["came_true"] is None or 0 <= s["came_true"] <= 1
    assert [s["score"] for s in out["suggestions"]] == sorted((s["score"] for s in out["suggestions"]), reverse=True)
    assert out["model"]["test"]["ranking"]["mrr"] is not None
    # the live snapshot is the dump's year, so year-difference features stay in the range trained on
    assert out["snapshot"] == int(meta["last_full_year"]) + 1
    age, idle, bridge = c.execute("SELECT max(age_u), max(idle_v), max(bridge_age) FROM pair").fetchone()
    assert max(age, idle, bridge) < 100, (age, idle, bridge)


def test_predict_rejects_a_bin_and_an_unknown_key(con):
    c, _ = con
    assert "disambiguation bin" in P.suggest(c, "homepages/bin/0")["error"]
    assert "error" in P.suggest(c, "homepages/nope/9")


def test_came_true_rate_lookup():
    calib = [{"from": 0.0, "to": 0.1, "came_true": 0.01}, {"from": 0.1, "to": 0.5, "came_true": 0.2},
             {"from": 0.5, "to": 1.0, "came_true": 0.6}]
    assert E.came_true_rate(calib, 0.05) == 0.01
    assert E.came_true_rate(calib, 0.3) == 0.2
    assert E.came_true_rate(calib, 1.0) == 0.6
    assert E.came_true_rate([], 0.3) is None
    # a score beyond every bin the test reached takes the nearest bin, rather than nothing
    assert E.came_true_rate(calib[:2], 0.95) == 0.2
