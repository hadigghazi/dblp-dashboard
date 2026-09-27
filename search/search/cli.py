"""
Command line for hybrid paper search.

    python -m search.cli store                          # build the sparse index for the current dump
    python -m search.cli build-index                    # (re)build/resume the embeddings; safe to interrupt
    python -m search.cli status                          # index build progress
    python -m search.cli evaluate                        # self-retrieval evaluation, written for /search/status
    python -m search.cli search --q "graph neural networks for traffic forecasting"
    python -m search.cli pilot --candidate openai:text-embedding-3-large --dim 512
                                                    # is another model better on THIS data?
"""
import argparse
import json
import logging
import pathlib
import sys

from . import (config, data, embed as E, evaluate as EV, pilot as PI, search as SR, store as S,
               vectors as V)

log = logging.getLogger("dblp.search")


def _newest(directory, pattern):
    found = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="search.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",
                    choices=["store", "build-index", "status", "evaluate", "search", "pilot"])
    ap.add_argument("--q", help="a query, for search")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--kind", choices=["journal", "conference"])
    ap.add_argument("--eval-papers", type=int, default=config.EVAL_PAPERS)
    ap.add_argument("--candidate", help="a model to compare against the current one, "
                                        "e.g. openai:text-embedding-3-large")
    ap.add_argument("--dim", type=int, default=512, help="width to ask the candidate for")
    ap.add_argument("--sample", type=int, default=200_000,
                    help="how many papers to score both models against")
    ap.add_argument("--queries", help="the query file to reuse "
                                      "(default: the assistant's saved paraphrase set)")
    args = ap.parse_args(argv)
    if args.command == "search" and not args.q:
        ap.error("search needs --q")
    if args.command == "pilot" and not args.candidate:
        ap.error("pilot needs --candidate")

    con, meta = data.connect()
    try:
        if args.command == "store":
            path = S.build_store(con, meta)
            print(json.dumps({"search_store": str(path)}, indent=2))
            return 0

        S.attach_store(con, meta, build_if_missing=True)
        fp = meta["fingerprint"]

        if args.command == "build-index":
            out = V.build(con, fp, E.make_encoder())
            print(json.dumps(out, indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(V.progress(con, fp), indent=2))
            return 0
        if args.command == "evaluate":
            out = EV.evaluate(con, fp, E.make_encoder(), n_papers=args.eval_papers)
            S.eval_path(fp).write_text(json.dumps(out), encoding="utf-8")
            print(json.dumps(out, indent=2))
            return 0
        if args.command == "pilot":
            queries = (pathlib.Path(args.queries) if args.queries else
                       _newest(config.MODELS_DIR / "search-eval", "queries-*.json"))
            if queries is None:
                sys.exit("no query file: run the assistant's `search-eval` first, or pass --queries")
            cases = PI.load_cases(queries)
            log.info("pilot against %s (%s queries)", queries.name, len(cases))
            out = PI.run(con, fp, cases, args.candidate, args.dim, n_sample=args.sample)
            path = PI.save(out, fp)
            print(json.dumps(out, indent=2))
            print(f"\nwritten to {path}")
            return 0

        out = SR.search(con, fp, E.make_encoder(), args.q, top=args.top, kind=args.kind)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1 if "error" in out else 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
