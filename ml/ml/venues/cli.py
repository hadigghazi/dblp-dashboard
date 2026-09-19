"""
Command line for venue recommendation.

    python -m ml.venues.cli store        # build the venue store for the current dump
    python -m ml.venues.cli dataset      # build the ranker's and the test year's candidate pairs
    python -m ml.venues.cli train        # build, train the ranker, evaluate on the test year, save
    python -m ml.venues.cli evaluate     # re-run the evaluation with the saved model
    python -m ml.venues.cli predict --title "Graph neural networks for traffic forecasting" [--authors key,key]
    python -m ml.venues.cli predict --key conf/nips/VaswaniSPUJGKP17
"""
import argparse
import json
import logging
import sys
import time

import numpy as np

from .. import data
from . import candidates as C, config, evaluate as E, features as F, model as M, predict as P, store as S

log = logging.getLogger("dblp.ml.venues")


def _snapshot(con, year, limit, seed):
    """Statistics as of year-1, candidates for a sample of the year's papers."""
    t = time.time()
    stats = S.build_stats(con, year - 1)
    queries = C.queries_from_papers(con, year, limit, seed)
    pairs = C.build_pairs(con)
    positives = con.execute("SELECT coalesce(sum(y), 0) FROM pair").fetchone()[0]
    return {"year": int(year), "statistics_up_to": int(year - 1), **stats, "queries": int(queries),
            "pairs": int(pairs), "pairs_with_true_series": int(positives), "seconds": round(time.time() - t, 1)}


def build_dataset(con, meta, args):
    t_rank, t_test = S.years(meta)
    out = {"rank": _snapshot(con, t_rank, args.rank_papers, config.SEED)}
    con.execute("CREATE OR REPLACE TEMP TABLE pair_rank AS SELECT * FROM pair")
    out["test"] = _snapshot(con, t_test, args.test_papers, config.SEED + 1)
    con.execute("CREATE OR REPLACE TEMP TABLE pair_test AS SELECT * FROM pair")
    log.info("dataset: %s", json.dumps(out))
    return out


def train(con, meta, args):
    dataset = build_dataset(con, meta, args)
    drop = {f.strip() for f in (args.drop_features or "").split(",") if f.strip()}
    unknown = drop - set(F.FEATURES)
    if unknown:
        log.error("unknown feature(s) to drop: %s", ", ".join(sorted(unknown)))
        return 1
    active = [f for f in F.FEATURES if f not in drop]

    # the ranker learns only from papers whose series it could have found
    con.execute("CREATE OR REPLACE TEMP TABLE pair AS SELECT * FROM pair_rank "
                "WHERE qid IN (SELECT qid FROM pair_rank WHERE y = 1)")
    Xtr, ytr, _ = F.matrix(con, feature_names=active)
    if not len(ytr) or len(np.unique(ytr)) < 2:
        log.error("no usable training pairs")
        return 1
    model = M.train(Xtr, ytr)

    # the test statistics and query tables are the ones in scope (built last by build_dataset)
    con.execute("CREATE OR REPLACE TEMP TABLE pair AS SELECT * FROM pair_test")
    Xte, yte, ite = F.matrix(con, feature_names=active)
    metrics = {"dataset": dataset, "dropped_features": sorted(drop)}
    calibration = []
    if len(yte):
        pte = model.predict_proba(Xte)[:, 1]
        metrics["test"] = {"year": dataset["test"]["year"], **E.evaluate(con, ite, yte, pte)}
        calibration = E.calibration(yte, pte)
        metrics["feature_importance"] = M.importances(model, Xte, yte, active)
    else:
        metrics["feature_importance"] = M.importances(model, Xtr, ytr, active)

    if drop:
        meta = dict(meta, fingerprint=f"{meta.get('fingerprint', 'unknown')}-without-{'-'.join(sorted(drop))}")
    # the statistics the server scores with: everything the dump holds
    now = int(meta.get("last_full_year") or 2025) + 1
    metrics["serving_statistics"] = {"up_to": now, **S.build_stats(con, now)}
    S.export_stats(con, M.model_dir(meta.get("fingerprint", "unknown")))
    d = M.save(model, metrics, meta, active, calibration)
    print(json.dumps({**metrics, "calibration": calibration}, indent=2))
    print("artifacts:", d, file=sys.stderr)
    return 0


def evaluate_only(con, meta, args):
    model, features, _, d = M.load(args.fingerprint)
    t_rank, t_test = S.years(meta)
    dataset = {"test": _snapshot(con, t_test, args.test_papers, config.SEED + 1)}
    Xte, yte, ite = F.matrix(con, feature_names=features)
    out = {"model_dir": str(d), "features": features, "dataset": dataset}
    if len(yte):
        out["test"] = {"year": t_test, **E.evaluate(con, ite, yte, model.predict_proba(Xte)[:, 1])}
    print(json.dumps(out, indent=2))
    return 0


def predict(con, args):
    if args.key:
        out = P.for_paper(con, args.key, top=args.top)
    else:
        keys = [k.strip() for k in (args.authors or "").split(",") if k.strip()]
        out = P.suggest(con, args.title, keys, top=args.top)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 1 if "error" in out else 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="ml.venues.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["store", "dataset", "train", "evaluate", "predict"])
    ap.add_argument("--title", help="a title, for predict")
    ap.add_argument("--authors", help="comma-separated author page keys, for predict")
    ap.add_argument("--key", help="an existing paper's key, for predict")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--fingerprint", help="use the model trained on this dump")
    ap.add_argument("--rank-papers", type=int, default=config.RANK_PAPERS)
    ap.add_argument("--test-papers", type=int, default=config.TEST_PAPERS)
    ap.add_argument("--drop-features", default="", help="comma-separated features to train without (ablation)")
    args = ap.parse_args(argv)
    if args.command == "predict" and not (args.title or args.key):
        ap.error("predict needs --title or --key")

    con, meta = data.connect()
    try:
        if args.command == "store":
            print(json.dumps({"venue_store": str(S.build_store(con, meta))}, indent=2))
            return 0
        S.attach_store(con, meta, build_if_missing=True)
        if args.command == "dataset":
            print(json.dumps(build_dataset(con, meta, args), indent=2))
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
