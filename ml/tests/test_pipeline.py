import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp(prefix="dblp-ml-test-"))
os.environ.update({"CACHE_DIR": str(_tmp / "cache"), "MODELS_DIR": str(_tmp / "models"),
                   "DUCKDB_MEMORY": "1GB", "DUCKDB_THREADS": "2"})

from tests.make_serving import make  # noqa: E402

make(_tmp / "cache")

from ml import cli, config, data, evaluate as E, features as F, model as M, predict as P  # noqa: E402


class Args:
    min_people = 3
    max_blocks = 500
    eval_blocks = 12
    tune_blocks = 8
    target_precision = 0.9
    drop_features = ""
    key = None
    fingerprint = None
    max_papers = 300


@pytest.fixture(scope="module")
def con():
    c, meta = data.connect()
    assert meta["fingerprint"] == "testfp0001"
    yield c
    c.close()


def test_dataset_has_both_classes_and_a_block_split(con):
    stats = cli.build_dataset(con, Args())
    assert stats["blocks"] >= 20
    assert 0 < stats["positives"] < stats["pairs"]
    # every split non-empty, and no block appears in two splits
    assert stats["train"] > 0 and stats["val"] > 0 and stats["test"] > 0
    overlap = con.execute("""
        SELECT count(*) FROM (
            SELECT base_name FROM pair WHERE bucket IN (0,1,2,3,4,5,6)
            INTERSECT
            SELECT base_name FROM pair WHERE bucket IN (8,9))""").fetchone()[0]
    assert overlap == 0


def test_features_are_finite_and_named(con):
    X, y, info = F.matrix(con)
    assert X.shape[1] == len(F.FEATURES)
    assert X.shape[0] == len(y) == len(info["pid_a"])
    import numpy as np
    assert np.isfinite(X).all()


def test_name_form_does_not_leak_the_label(con):
    """dblp encodes the assignment in the author string ("Wei Wang 0001"), so within a block an
    identical string would mean "same person" by construction. The suffix must be stripped."""
    leaked = con.execute(
        "SELECT count(*) FROM inst WHERE regexp_matches(used_name, ' [0-9]{4}$')").fetchone()[0]
    assert leaked == 0
    # and the feature must not be able to reconstruct the label on its own
    import numpy as np
    _, y, _ = F.matrix(con)
    d = con.execute("SELECT (a.used_name = b.used_name)::INT AS same, (a.person_id = b.person_id)::INT AS y "
                    "FROM inst a JOIN inst b ON a.base_name = b.base_name AND a.pid < b.pid").fetchnumpy()
    agreement = float(np.mean(d["same"] == d["y"]))
    assert agreement < 0.99, f"name form agrees with the label {agreement:.3f} of the time"
    assert len(y)


def test_model_learns_the_planted_signal_and_beats_the_baseline(con):
    Xtr, ytr, _ = F.matrix(con, "bucket IN (0,1,2,3,4,5,6)")
    Xte, yte, ite = F.matrix(con, "bucket IN (8,9)")
    model = M.train(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    metrics = E.pairwise_metrics(yte, p, 0.5)
    baseline = E.baseline_pairwise(yte, ite)
    assert metrics["roc_auc"] > 0.85, metrics
    assert metrics["f1"] >= baseline["f1"], (metrics, baseline)


def test_clustering_and_assignment_are_measured(con):
    model = M.train(*F.matrix(con, "bucket IN (0,1,2,3,4,5,6)")[:2])
    blocks = [r[0] for r in con.execute(
        "SELECT DISTINCT base_name FROM inst WHERE (hash(base_name) % 10)::INT IN (8,9)").fetchall()]
    summary = E.evaluate_blocks(con, model, {"cluster": 0.5, "assign": 0.7}, blocks, max_blocks=10)
    assert summary["blocks"] > 0
    for key in ("b3_f1", "b3_precision", "b3_recall", "ari"):
        assert 0.0 <= summary[key] <= 1.0, summary
    assert summary["b3_f1"] > 0.4, summary
    a = summary["assignment"]
    assert a["held_out_papers"] > 0 and 0.0 <= a["top1_accuracy"] <= 1.0


def test_bcubed_is_correct_on_a_known_case():
    perfect = E.bcubed([0, 0, 1, 1], [5, 5, 7, 7])
    assert perfect == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    everything_merged = E.bcubed([0, 0, 1, 1], [1, 1, 1, 1])
    assert everything_merged["precision"] == 0.5 and everything_merged["recall"] == 1.0
    everything_split = E.bcubed([0, 0, 1, 1], [1, 2, 3, 4])
    assert everything_split["precision"] == 1.0 and everything_split["recall"] == 0.5


def test_train_command_saves_artifacts_and_metrics(con):
    assert cli.train(con, Args()) == 0
    d = M.model_dir("testfp0001")
    assert (d / M.MODEL_FILE).exists()
    saved = json.loads((d / M.METRICS_FILE).read_text(encoding="utf-8"))
    assert saved["dump"]["fingerprint"] == "testfp0001"
    assert set(saved["thresholds"]) == {"cluster", "linkage", "cluster_bin", "linkage_bin",
                                        "assign", "pairwise"}
    assert saved["metrics"]["test"]["clustering_bin_like"]["blocks"] > 0
    assert saved["metrics"]["test"]["clustering"]["assignment"]["assign_threshold"] > 0
    assert saved["metrics"]["test"]["pairwise"]["roc_auc"] > 0.85
    assert saved["metrics"]["test"]["clustering"]["blocks"] > 0
    assert saved["metrics"]["feature_importance"][0]["feature"] in F.FEATURES
    # the co-author features should carry most of the signal in this fixture
    top = [f["feature"] for f in saved["metrics"]["feature_importance"][:4]]
    assert any("ids" in f or "names" in f for f in top), top


def test_predict_splits_a_bin(con):
    out = P.split_bin(con, "homepages/bin/0")
    assert out["bin"]["name"] == "Bin One"
    assert out["papers"] == 10
    assert out["clusters"], out
    assert sum(c["size"] for c in out["clusters"]) == out["papers"]
    assert out["numbered_people_in_block"] == 2
    json.dumps(out)   # the whole payload must be serialisable for an API to return it
    # the bin's papers were planted around two numbered people, so we expect a suggestion
    assert any(c["suggested_person"] for c in out["clusters"]), out


def test_predict_rejects_a_non_bin(con):
    out = P.split_bin(con, "homepages/00/1")
    assert "error" in out and "not a disambiguation bin" in out["error"]
    assert "error" in P.split_bin(con, "homepages/nope/9")


def test_model_load_reports_a_missing_model_clearly():
    with pytest.raises(FileNotFoundError, match="no model for dump"):
        M.load("nosuchfingerprint")


def test_two_cuts_are_tuned_separately(con):
    """The clustering cut answers a different question than the pairwise cut, so it is tuned on
    B-cubed; the assignment cut is calibrated to a precision target instead."""
    model = M.train(*F.matrix(con, "bucket IN (0,1,2,3,4,5,6)")[:2])
    val = [r[0] for r in con.execute(
        "SELECT DISTINCT base_name FROM inst WHERE (hash(base_name) % 10)::INT IN (7)").fetchall()]
    blocks = val or [r[0] for r in con.execute("SELECT DISTINCT base_name FROM inst LIMIT 8").fetchall()]
    t, curve = E.tune_cluster_threshold(con, model, blocks, max_blocks=8)
    assert 0.2 <= t <= 0.9 and curve
    assert max(curve, key=lambda r: r["b3_f1"])["threshold"] == t
    cal = E.calibrate_assignment(con, model, blocks, target_precision=0.9, max_blocks=8)
    assert 0.3 <= cal["threshold"] <= 0.95
    if cal["precision"] is not None:
        assert 0.0 <= cal["precision"] <= 1.0 and 0.0 <= cal["coverage"] <= 1.0


def test_clusters_pointing_at_one_person_are_merged():
    """The clustering over-splits by design, so several groups may name the same page."""
    def cl(size, key, score):
        return {"size": size, "papers": [{"year": 2020 + size, "title": key or "x"}],
                "suggested_person": {"key": key, "name": key, "score": score, "margin": 0.2} if key else None,
                "best_candidate_below_threshold": None, "looks_new": key is None}
    merged = P.merge_by_suggestion([cl(6, "a", 0.7), cl(4, "a", 0.9), cl(3, "b", 0.8), cl(2, None, 0)])
    assert len(merged) == 3
    a = next(c for c in merged if (c["suggested_person"] or {}).get("key") == "a")
    assert a["size"] == 10 and a["merged_from"] == 2
    assert a["suggested_person"]["score"] == 0.9                  # strongest evidence
    assert a["suggested_person"]["weakest_member_score"] == 0.7   # ... and the weakest, in the open
    assert sum(c["size"] for c in merged) == 15                   # no paper lost or duplicated


def test_a_marginal_group_is_not_absorbed_by_a_confident_one():
    """A group that barely cleared the bar must not borrow a confident group's score by pointing
    at the same page. It stays separate, still labelled, for a human to judge."""
    def cl(size, key, score):
        return {"size": size, "papers": [{"year": 2020, "title": "x"}],
                "suggested_person": {"key": key, "name": key, "score": score, "margin": 0.2},
                "best_candidate_below_threshold": None, "looks_new": False}
    merged = P.merge_by_suggestion([cl(6, "a", 0.95), cl(4, "a", 0.35), cl(2, "a", 0.8)])
    sizes = sorted(c["size"] for c in merged)
    assert sizes == [4, 8]                                        # 0.95 and 0.8 merge; 0.35 does not
    strong = next(c for c in merged if c["size"] == 8)
    assert strong["suggested_person"]["weakest_member_score"] == 0.8
    weak = next(c for c in merged if c["size"] == 4)
    assert weak["suggested_person"]["score"] == 0.35 and "merged_from" not in weak


def test_a_near_tie_is_not_named():
    """Two candidates almost level means we say nothing rather than guess."""
    links = {1: 0.9, 2: 0.9}
    accepted, rejected = P._suggestion([(1, 0.80), (2, 0.78)], links, assign_threshold=0.6)
    assert accepted is None and rejected["person_id"] == 1
    accepted, _ = P._suggestion([(1, 0.80), (2, 0.40)], links, assign_threshold=0.6)
    assert accepted["person_id"] == 1
    accepted, rejected = P._suggestion([(1, 0.50)], links, assign_threshold=0.6)
    assert accepted is None and rejected["score"] == 0.5


def test_complete_linkage_refuses_to_chain():
    """A resembles B, B resembles C, A does not resemble C: average linkage merges all three,
    complete linkage does not. That chaining is what produced a 16-paper cluster spanning
    lattice QCD and cleanroom airflow on a real bin."""
    import numpy as np
    # distances: A-B 0.2, B-C 0.2, A-C 0.7, cut at 1 - 0.4 = 0.6
    # average links {A,B} to C at (0.7 + 0.2) / 2 = 0.45 -> merges; complete uses max = 0.7 -> refuses
    prob = np.array([[1.0, 0.8, 0.3],
                     [0.8, 1.0, 0.8],
                     [0.3, 0.8, 1.0]], dtype=np.float32)
    assert len(set(E.cluster(prob, 0.4, "average").tolist())) == 1
    assert len(set(E.cluster(prob, 0.4, "complete").tolist())) > 1


def test_bin_like_tuning_thins_each_person_down(con):
    """Tuning must run on the distribution prediction sees: many people, one or two papers each."""
    import numpy as np
    model = M.train(*F.matrix(con, "bucket IN (0,1,2,3,4,5,6)")[:2])
    blocks = [r[0] for r in con.execute("SELECT DISTINCT base_name FROM inst LIMIT 10").fetchall()]
    tuned = E.tune_for_bins(con, model, blocks, max_blocks=10)
    assert tuned["linkage"] in ("average", "complete")
    assert 0.3 <= tuned["threshold"] <= 0.95
    assert tuned["blocks"] > 0 and tuned["curve"]
    # the thinned blocks really are thinner than the originals
    rng = np.random.default_rng(3)
    loaded = E._load_blocks(con, model, blocks, None, 4)
    for _, block in loaded:
        thin = E._subsample_like_a_bin(block, 2, rng)
        if thin:
            counts = [int((thin["true"] == p).sum()) for p in np.unique(thin["true"])]
            assert max(counts) <= 2


def test_three_outcomes_are_distinguished():
    """A hopeless best candidate means "no page yet", not "uncertain" - with hundreds of numbered
    pages in a block something always scores highest, so the floor matters."""
    links = {1: 0.9, 2: 0.9}
    accepted, uncertain = P._suggestion([(1, 0.05), (2, 0.01)], links, assign_threshold=0.3)
    assert accepted is None and uncertain is None            # looks new
    accepted, uncertain = P._suggestion([(1, 0.40), (2, 0.38)], links, assign_threshold=0.3)
    assert accepted is None and uncertain["person_id"] == 1  # plausible, not proven
    accepted, uncertain = P._suggestion([(1, 0.80), (2, 0.10)], links, assign_threshold=0.3)
    assert accepted["person_id"] == 1 and uncertain is None   # named


def test_best_of_many_noisy_candidates_is_not_named():
    """With hundreds of candidates, the top of a noisy pile looks like a match. It must stand out
    from the pile (outlier test) and rest on at least one strong pair, or stay unnamed."""
    pile = [(i, 0.30 + 0.01 * (i % 5)) for i in range(2, 300)]     # 298 look-alikes at 0.30-0.34
    ranked = sorted([(1, 0.42)] + pile, key=lambda t: -t[1])
    links = {i: 0.45 for i, _ in ranked}
    accepted, uncertain = P._suggestion(ranked, links, assign_threshold=0.3)
    assert accepted is None and uncertain["person_id"] == 1     # 0.42 is not an outlier at z=3
    # a real match: far above the pile, and with one confident pairwise link
    ranked = sorted([(1, 0.85)] + pile, key=lambda t: -t[1])
    links = {i: 0.45 for i, _ in ranked}
    links[1] = 0.93
    accepted, _ = P._suggestion(ranked, links, assign_threshold=0.3)
    assert accepted["person_id"] == 1 and accepted["z"] > 3
    # far above the pile on average but no single strong pair: name coincidence, stay silent
    links[1] = 0.5
    accepted, uncertain = P._suggestion(ranked, links, assign_threshold=0.3)
    assert accepted is None and uncertain["max_link"] == 0.5


def test_bin_like_evaluation_reports_its_floor(con):
    """On thinned blocks most people have one paper, so "never merge" already scores high on
    B-cubed. The metric is only interpretable next to that floor."""
    model = M.train(*F.matrix(con, "bucket IN (0,1,2,3,4,5,6)")[:2])
    blocks = [r[0] for r in con.execute("SELECT DISTINCT base_name FROM inst LIMIT 12").fetchall()]
    out = E.evaluate_bin_like(con, model, {"cluster": 0.5, "cluster_bin": 0.6, "linkage_bin": "average"},
                              blocks, max_blocks=12)
    assert out["blocks"] > 0
    for k in ("b3_f1", "b3_f1_all_singletons", "b3_f1_overlap_baseline", "share_of_people_with_one_paper"):
        assert 0.0 <= out[k] <= 1.0, out
    assert out["beats_doing_nothing_by"] == round(out["b3_f1"] - out["b3_f1_all_singletons"], 4)
