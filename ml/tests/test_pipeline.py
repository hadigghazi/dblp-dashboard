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
    assert set(saved["thresholds"]) == {"cluster", "assign", "pairwise"}
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
    assert a["suggested_person"]["score"] == 0.9      # keeps the strongest evidence
    assert sum(c["size"] for c in merged) == 15       # no paper lost or duplicated


def test_a_near_tie_is_not_named():
    """Two candidates almost level means we say nothing rather than guess."""
    accepted, rejected = P._suggestion([(1, 0.80), (2, 0.78)], assign_threshold=0.6)
    assert accepted is None and rejected["person_id"] == 1
    accepted, _ = P._suggestion([(1, 0.80), (2, 0.40)], assign_threshold=0.6)
    assert accepted["person_id"] == 1
    accepted, rejected = P._suggestion([(1, 0.50)], assign_threshold=0.6)
    assert accepted is None and rejected["score"] == 0.5
