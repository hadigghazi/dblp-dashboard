"""The pairwise model: given two papers carrying the same name, P(same person)."""
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from . import config
from .features import FEATURES

log = logging.getLogger("dblp.ml.model")

MODEL_FILE = "pairwise.joblib"
METRICS_FILE = "metrics.json"


def train(X, y, seed=None):
    """Gradient boosting on the pair features. CPU-only, seconds on a few hundred thousand pairs."""
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=seed if seed is not None else config.SEED,
    )
    model.fit(X, y)
    log.info("trained on %s pairs, %s boosting iterations", f"{len(y):,}", model.n_iter_)
    return model


def tune_threshold(p, y):
    """The probability cut that maximises pairwise F1 on the validation blocks."""
    if not len(y) or len(np.unique(y)) < 2:
        return 0.5
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(np.int8), zero_division=0) for t in grid]
    best = float(grid[int(np.argmax(scores))])
    log.info("threshold %.2f (validation F1 %.3f)", best, max(scores))
    return best


def model_dir(fingerprint):
    """Artifacts are keyed by the dump's fingerprint, so a model is always traceable to its data."""
    d = config.MODELS_DIR / f"disambiguation-{fingerprint}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(model, threshold, metrics, meta):
    d = model_dir(meta.get("fingerprint", "unknown"))
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES}, d / MODEL_FILE)
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dump": {k: meta.get(k) for k in ("fingerprint", "records", "latest_mdate", "parquet")},
        "threshold": threshold,
        "features": FEATURES,
        "metrics": metrics,
    }
    (d / METRICS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("saved %s and %s", d / MODEL_FILE, d / METRICS_FILE)
    return d


def load(fingerprint=None):
    """Load the model for a fingerprint, or the most recently trained one."""
    if fingerprint:
        d = config.MODELS_DIR / f"disambiguation-{fingerprint}"
        if not (d / MODEL_FILE).exists():
            raise FileNotFoundError(f"no model for dump {fingerprint} in {config.MODELS_DIR}; run `train` first")
    else:
        dirs = sorted(config.MODELS_DIR.glob("disambiguation-*"),
                      key=lambda p: (p / MODEL_FILE).stat().st_mtime if (p / MODEL_FILE).exists() else 0,
                      reverse=True)
        dirs = [x for x in dirs if (x / MODEL_FILE).exists()]
        if not dirs:
            raise FileNotFoundError(f"no trained model in {config.MODELS_DIR}; run `train` first")
        d = dirs[0]
    bundle = joblib.load(d / MODEL_FILE)
    if bundle["features"] != FEATURES:
        raise ValueError("the saved model expects different features than this code builds; retrain")
    return bundle["model"], bundle["threshold"], d


def importances(model, X, y, n_repeats=3, seed=None):
    """Permutation importance: which features actually carry the signal."""
    from sklearn.inspection import permutation_importance
    if len(y) > 40000:  # enough for a stable ranking, and keeps this to a few seconds
        idx = np.random.default_rng(seed or config.SEED).choice(len(y), 40000, replace=False)
        X, y = X[idx], y[idx]
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed or config.SEED,
                               scoring="average_precision")
    order = np.argsort(r.importances_mean)[::-1]
    return [{"feature": FEATURES[i], "drop_in_average_precision": round(float(r.importances_mean[i]), 4)}
            for i in order]
