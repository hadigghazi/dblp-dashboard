"""
Command line for the disambiguation pipeline.

    python -m ml.cli dataset     # build the labelled pair dataset, print its shape
    python -m ml.cli train       # build, train, tune, evaluate, save model + metrics
    python -m ml.cli evaluate    # re-run evaluation with the saved model
    python -m ml.cli predict --key homepages/35/7092
"""
import argparse
import json
import logging
import sys
import time

import numpy as np

from . import config, data, evaluate as E, features as F, model as M, predict as P

log = logging.getLogger("dblp.ml")


def _buckets(sql_col, buckets):
    return f"{sql_col} IN ({', '.join(str(b) for b in sorted(buckets))})"


def build_dataset(con, args):
    t = time.time()
    data.build_blocks(con, min_people=args.min_people, max_blocks=args.max_blocks)
    data.build_instances(con)
    F.build_pairs(con)
    stats = con.execute(f"""
        SELECT count(*) AS pairs, sum(y) AS positives, count(DISTINCT base_name) AS blocks,
               count(*) FILTER (WHERE {_buckets('bucket', config.TRAIN_BUCKETS)}) AS train,
               count(*) FILTER (WHERE {_buckets('bucket', config.VAL_BUCKETS)}) AS val,
               count(*) FILTER (WHERE {_buckets('bucket', config.TEST_BUCKETS)}) AS test
        FROM pair""").fetchone()
    names = ["pairs", "positives", "blocks", "train", "val", "test"]
    out = {k: int(v or 0) for k, v in zip(names, stats)}
    out["seconds"] = round(time.time() - t, 1)
    log.info("dataset: %s", json.dumps(out))
    return out


def train(con, args):
    dataset = build_dataset(con, args)
    Xtr, ytr, _ = F.matrix(con, _buckets("bucket", config.TRAIN_BUCKETS))
    Xva, yva, iva = F.matrix(con, _buckets("bucket", config.VAL_BUCKETS))
    Xte, yte, ite = F.matrix(con, _buckets("bucket", config.TEST_BUCKETS))
    if not len(ytr) or len(np.unique(ytr)) < 2:
        log.error("no usable training pairs (need blocks with several people who each have papers)")
        return 1

    model = M.train(Xtr, ytr)
    threshold = M.tune_threshold(model.predict_proba(Xva)[:, 1], yva) if len(yva) else 0.5

    metrics = {"dataset": dataset, "threshold": threshold}
    if len(yte):
        pte = model.predict_proba(Xte)[:, 1]
        metrics["test"] = {
            "pairwise": E.pairwise_metrics(yte, pte, threshold),
            "pairwise_overlap_baseline": E.baseline_pairwise(yte, ite),
        }
    if len(yva):
        metrics["validation"] = {"pairwise": E.pairwise_metrics(yva, model.predict_proba(Xva)[:, 1], threshold)}

    test_blocks = [r[0] for r in con.execute(
        f"SELECT DISTINCT base_name FROM inst WHERE {_buckets('(hash(base_name) % 10)::INT', config.TEST_BUCKETS)} "
        "ORDER BY hash(base_name)").fetchall()]
    log.info("evaluating clustering and assignment on %s held-out blocks", len(test_blocks))
    if "test" not in metrics:
        metrics["test"] = {}
    metrics["test"]["clustering"] = E.evaluate_blocks(con, model, threshold, test_blocks, args.eval_blocks)
    metrics["feature_importance"] = M.importances(model, Xte if len(yte) else Xtr, yte if len(yte) else ytr)

    meta = dict(con.execute("SELECT k, v FROM s._meta").fetchall())
    d = M.save(model, threshold, metrics, meta)
    print(json.dumps(metrics, indent=2))
    print(f"\nartifacts: {d}")
    return 0


def evaluate_only(con, args):
    model, threshold, d = M.load(args.fingerprint)
    data.build_blocks(con, min_people=args.min_people, max_blocks=args.max_blocks)
    data.build_instances(con)
    F.build_pairs(con)
    Xte, yte, ite = F.matrix(con, _buckets("bucket", config.TEST_BUCKETS))
    out = {"model_dir": str(d), "threshold": threshold}
    if len(yte):
        out["pairwise"] = E.pairwise_metrics(yte, model.predict_proba(Xte)[:, 1], threshold)
        out["pairwise_overlap_baseline"] = E.baseline_pairwise(yte, ite)
    blocks = [r[0] for r in con.execute(
        f"SELECT DISTINCT base_name FROM inst WHERE {_buckets('(hash(base_name) % 10)::INT', config.TEST_BUCKETS)} "
        "ORDER BY hash(base_name)").fetchall()]
    out["clustering"] = E.evaluate_blocks(con, model, threshold, blocks, args.eval_blocks)
    print(json.dumps(out, indent=2))
    return 0


def predict(con, args):
    out = P.split_bin(con, args.key, max_papers=args.max_papers)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 1 if "error" in out else 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="ml.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["dataset", "train", "evaluate", "predict"])
    ap.add_argument("--key", help="a disambiguation bin's key, for predict")
    ap.add_argument("--fingerprint", help="use the model trained on this dump")
    ap.add_argument("--min-people", type=int, default=config.MIN_PEOPLE_PER_BLOCK,
                    help="only blocks with at least this many labelled people")
    ap.add_argument("--max-blocks", type=int, default=config.MAX_BLOCKS)
    ap.add_argument("--eval-blocks", type=int, default=150, help="held-out blocks to cluster (runtime)")
    ap.add_argument("--max-papers", type=int, default=P.MAX_BIN_PAPERS, help="cap on a bin's papers, for predict")
    args = ap.parse_args(argv)

    if args.command == "predict" and not args.key:
        ap.error("predict needs --key")

    con, meta = data.connect()
    try:
        if args.command == "dataset":
            print(json.dumps(build_dataset(con, args), indent=2))
            return 0
        if args.command == "train":
            return train(con, args)
        if args.command == "evaluate":
            return evaluate_only(con, args)
        return predict(con, args)
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
