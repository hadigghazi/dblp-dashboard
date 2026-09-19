"""The ranker: given a paper and a candidate series, P(this is where it was published)."""
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import config as base
from . import config
from .features import FEATURES

log = logging.getLogger("dblp.ml.venues.model")

MODEL_FILE = "ranker.joblib"
METRICS_FILE = "metrics.json"


def train(X, y, seed=None):
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=seed if seed is not None else config.SEED,
    )
    model.fit(X, y)
    log.info("trained on %s pairs (%.2f%% positive), %s boosting iterations",
             f"{len(y):,}", 100 * float(y.mean()), model.n_iter_)
    return model


def model_dir(fingerprint):
    d = base.MODELS_DIR / f"venues-{fingerprint}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(model, metrics, meta, features=None, calibration=None):
    features = features or FEATURES
    d = model_dir(meta.get("fingerprint", "unknown"))
    joblib.dump({"model": model, "features": features, "calibration": calibration or []}, d / MODEL_FILE)
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dump": {k: meta.get(k) for k in ("fingerprint", "records", "latest_mdate", "parquet", "last_full_year")},
        "features": features,
        "calibration": calibration or [],
        "metrics": metrics,
    }
    (d / METRICS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("saved %s and %s", d / MODEL_FILE, d / METRICS_FILE)
    return d


def load(fingerprint=None):
    if fingerprint:
        d = base.MODELS_DIR / f"venues-{fingerprint}"
        if not (d / MODEL_FILE).exists():
            raise FileNotFoundError(f"no venue model for dump {fingerprint} in {base.MODELS_DIR}; run `train` first")
    else:
        dirs = [x for x in base.MODELS_DIR.glob("venues-*") if (x / MODEL_FILE).exists()]
        if not dirs:
            raise FileNotFoundError(f"no trained venue model in {base.MODELS_DIR}; run `train` first")
        d = max(dirs, key=lambda p: (p / MODEL_FILE).stat().st_mtime)
    bundle = joblib.load(d / MODEL_FILE)
    unknown = [f for f in bundle["features"] if f not in FEATURES]
    if unknown:
        raise ValueError(f"the saved model expects features this code no longer builds: {unknown}; retrain")
    return bundle["model"], bundle["features"], bundle.get("calibration", []), d


def importances(model, X, y, feature_names, n_repeats=3, seed=None):
    from sklearn.inspection import permutation_importance
    if len(y) > 60000:
        idx = np.random.default_rng(seed or config.SEED).choice(len(y), 60000, replace=False)
        X, y = X[idx], y[idx]
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed or config.SEED,
                               scoring="average_precision")
    order = np.argsort(r.importances_mean)[::-1]
    return [{"feature": feature_names[i], "drop_in_average_precision": round(float(r.importances_mean[i]), 4)}
            for i in order]
