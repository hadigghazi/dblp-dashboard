"""
Command line for hybrid paper search.

    python -m search.cli store                          # build the sparse index for the current dump
    python -m search.cli build-index                    # (re)build/resume the embeddings; safe to interrupt
    python -m search.cli status                          # index build progress
    python -m search.cli evaluate                        # self-retrieval evaluation, written for /search/status
    python -m search.cli search --q "graph neural networks for traffic forecasting"
"""
import argparse
import json
import logging
import sys

from . import config, data, embed as E, evaluate as EV, search as SR, store as S, vectors as V

log = logging.getLogger("dblp.search")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="search.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["store", "build-index", "status", "evaluate", "search"])
    ap.add_argument("--q", help="a query, for search")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--kind", choices=["journal", "conference"])
    ap.add_argument("--eval-papers", type=int, default=config.EVAL_PAPERS)
    args = ap.parse_args(argv)
    if args.command == "search" and not args.q:
        ap.error("search needs --q")

    con, meta = data.connect()
    try:
        if args.command == "store":
            path = S.build_store(con, meta)
            print(json.dumps({"search_store": str(path)}, indent=2))
            return 0

        S.attach_store(con, meta, build_if_missing=True)
        fp = meta["fingerprint"]

        if args.command == "build-index":
            out = V.build(con, fp, E.Encoder())
            print(json.dumps(out, indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(V.progress(con, fp), indent=2))
            return 0
        if args.command == "evaluate":
            out = EV.evaluate(con, fp, E.Encoder(), n_papers=args.eval_papers)
            S.eval_path(fp).write_text(json.dumps(out), encoding="utf-8")
            print(json.dumps(out, indent=2))
            return 0
        out = SR.search(con, fp, E.Encoder(), args.q, top=args.top, kind=args.kind)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1 if "error" in out else 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
