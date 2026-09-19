"""
Command line for co-author link prediction.

    python -m ml.links.cli graph        # build the graph store for the current dump
    python -m ml.links.cli dataset      # build both snapshots, print their shape and the ceiling
    python -m ml.links.cli train        # build, train, evaluate on the later snapshot, save
    python -m ml.links.cli evaluate     # re-run evaluation with the saved model
    python -m ml.links.cli predict --key homepages/s/JurgenSchmidhuber
"""
import argparse
import json
import logging
import sys
import time

import numpy as np

from .. import data
from . import config, evaluate as E, features as F, graph as G, model as M, predict as P

log = logging.getLogger("dblp.ml.links")


def _snapshot(con, T, buckets, n_anchors, sample_negatives=False):
    t = time.time()
    anchors = G.select_anchors(con, T, buckets, n_anchors)
    pairs = G.build_pairs(con, T)
    origin = G.origin_of_new_links(con, T)
    kept = G.sample_training_pairs(con) if sample_negatives else pairs
    positives = con.execute("SELECT coalesce(sum(y), 0) FROM pair").fetchone()[0]
    return {"snapshot": T, "horizon": config.HORIZON, "anchors": int(anchors), "candidate_pairs": int(pairs),
            "pairs_used": int(kept), "positives": int(positives), "new_links_origin": origin,
            "seconds": round(time.time() - t, 1)}


def build_dataset(con, meta, args, sample_negatives=True):
    t_train, t_test = G.snapshot_years(meta)
    out = {"train": _snapshot(con, t_train, config.TRAIN_BUCKETS | config.VAL_BUCKETS,
                              args.train_anchors, sample_negatives)}
    con.execute("CREATE OR REPLACE TEMP TABLE pair_train AS SELECT * FROM pair")
    out["test"] = _snapshot(con, t_test, config.TEST_BUCKETS, args.test_anchors)
    con.execute("CREATE OR REPLACE TEMP TABLE pair_test AS SELECT * FROM pair")
    log.info("dataset: %s", json.dumps(out))
    return out


def _use(con, table):
    con.execute(f"CREATE OR REPLACE TEMP TABLE pair AS SELECT * FROM {table}")


def train(con, meta, args):
    dataset = build_dataset(con, meta, args)
    drop = {f.strip() for f in (args.drop_features or "").split(",") if f.strip()}
    unknown = drop - set(F.FEATURES)
    if unknown:
        log.error("unknown feature(s) to drop: %s", ", ".join(sorted(unknown)))
        return 1
    active = [f for f in F.FEATURES if f not in drop]

    _use(con, "pair_train")
    tr = ", ".join(str(b) for b in sorted(config.TRAIN_BUCKETS))
    va = ", ".join(str(b) for b in sorted(config.VAL_BUCKETS))
    Xtr, ytr, _ = F.matrix(con, f"bucket IN ({tr})", feature_names=active)
    Xva, yva, iva = F.matrix(con, f"bucket IN ({va})", feature_names=active)
    if not len(ytr) or len(np.unique(ytr)) < 2:
        log.error("no usable training pairs (need anchors who gained a distance-2 co-author)")
        return 1
    model = M.train(Xtr, ytr)

    metrics = {"dataset": dataset, "dropped_features": sorted(drop)}
    if len(yva):
        metrics["validation"] = {"pooled": E.pooled(yva, model.predict_proba(Xva)[:, 1])}

    _use(con, "pair_test")
    Xte, yte, ite = F.matrix(con, feature_names=active)
    calibration = []
    if len(yte):
        pte = model.predict_proba(Xte)[:, 1]
        metrics["test"] = {"snapshot": dataset["test"]["snapshot"], "horizon": config.HORIZON,
                           **E.evaluate(ite["u"], pte, yte, ite)}
        calibration = metrics["test"].pop("calibration")
        metrics["feature_importance"] = M.importances(model, Xte, yte, active)
    else:
        metrics["feature_importance"] = M.importances(model, Xtr, ytr, active)

    if drop:
        meta = dict(meta, fingerprint=f"{meta.get('fingerprint', 'unknown')}-without-{'-'.join(sorted(drop))}")
    d = M.save(model, metrics, meta, active, calibration)
    print(json.dumps({**metrics, "calibration": calibration}, indent=2))
    print("artifacts:", d, file=sys.stderr)
    return 0


def evaluate_only(con, meta, args):
    model, features, _, d = M.load(args.fingerprint)
    dataset = build_dataset(con, meta, args, sample_negatives=False)
    _use(con, "pair_test")
    Xte, yte, ite = F.matrix(con, feature_names=features)
    out = {"model_dir": str(d), "features": features, "dataset": dataset}
    if len(yte):
        pte = model.predict_proba(Xte)[:, 1]
        out["test"] = {"snapshot": dataset["test"]["snapshot"], "horizon": config.HORIZON,
                       **E.evaluate(ite["u"], pte, yte, ite)}
    print(json.dumps(out, indent=2))
    return 0


def predict(con, args):
    out = P.suggest(con, args.key, top=args.top)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 1 if "error" in out else 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="ml.links.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["graph", "dataset", "train", "evaluate", "predict"])
    ap.add_argument("--key", help="an author page's key, for predict")
    ap.add_argument("--top", type=int, default=10, help="suggestions to return, for predict")
    ap.add_argument("--fingerprint", help="use the model trained on this dump")
    ap.add_argument("--train-anchors", type=int, default=config.TRAIN_ANCHORS)
    ap.add_argument("--test-anchors", type=int, default=config.TEST_ANCHORS)
    ap.add_argument("--drop-features", default="",
                    help="comma-separated features to train without, e.g. shared_venues (ablation)")
    args = ap.parse_args(argv)
    if args.command == "predict" and not args.key:
        ap.error("predict needs --key")

    con, meta = data.connect()
    try:
        if args.command == "graph":
            path = G.build_store(con, meta)
            print(json.dumps({"graph_store": str(path)}, indent=2))
            return 0
        G.attach_store(con, meta, build_if_missing=True)
        if args.command == "dataset":
            print(json.dumps(build_dataset(con, meta, args, sample_negatives=False), indent=2))
            return 0
        if args.command == "train":
            return train(con, meta, args)
        if args.command == "evaluate":
            return evaluate_only(con, meta, args)
        return predict(con, args)
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
