"""
The co-authorship network as a published dataset, and its centrality measures.

  python -m ml.network.cli export                    # the canonical graph, bins excluded
  python -m ml.network.cli export --with-bins        # everything, for comparison
  python -m ml.network.cli export --scope no-preprints --expect-edges 22200244
  python -m ml.network.cli scopes --expect-edges 22200244   # which definition gives that number?
  python -m ml.network.cli centrality                # degree, betweenness, closeness, eigenvector
  python -m ml.network.cli centrality --resume        # finish a run whose measures already succeeded
  python -m ml.network.cli exact-betweenness          # what the estimate and the shortcuts cost

`--expect-edges` is worth using, with the matching scope: the dashboard's network analysis computed
22,200,244 edges from every record type except preprints, which is `--scope no-preprints`, so that
combination proves the join is right. The default scope keeps the preprints as well, which is a
larger graph - 24.4 million edges - and the difference between the two is the point rather than a
discrepancy.

`centrality` reads the exported files rather than the database, so it measures exactly the dataset
somebody would download. It measures the full-corpus export unless `--in` says otherwise, and it is
the long-running job here: budget half an hour, and run it detached.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .. import config as base, data
from . import config, export as EX

log = logging.getLogger("dblp.ml.network")


def out_dir(meta, with_bins, scope):
    name = f"network-{meta.get('fingerprint', 'unknown')}"
    if scope != "all":
        name += f"-{scope}"
    return base.MODELS_DIR / (f"{name}-withbins" if with_bins else name)


def centrality_out_dir(source: Path):
    """Named after the export it measures: `network-<fingerprint>` -> `centrality-<fingerprint>`."""
    prefix = "network-"
    tail = source.name[len(prefix):] if source.name.startswith(prefix) else source.name
    return base.MODELS_DIR / f"centrality-{tail}"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="ml.network.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "scopes", "centrality", "exact-betweenness"])
    ap.add_argument("--with-bins", action="store_true",
                    help="count disambiguation pages as people (they are not; for comparison only)")
    ap.add_argument("--max-authors", type=int, default=config.MAX_AUTHORS)
    ap.add_argument("--scope", choices=sorted(config.SCOPES), default=config.SCOPE,
                    help="which records make an edge (default: every type)")
    ap.add_argument("--expect-edges", type=int, default=None)
    ap.add_argument("--probe", action="store_true",
                    help="also measure every other definition and put the table in the datasheet")
    ap.add_argument("--out", default=None)
    # centrality only
    ap.add_argument("--in", dest="source", default=None,
                    help="the exported dataset to measure (default: the newest full-corpus export)")
    ap.add_argument("--threads", type=int, default=base.DUCKDB_THREADS,
                    help="threads for the graph algorithms; this VM shares its cores with the site")
    ap.add_argument("--seed", type=int, default=base.SEED)
    ap.add_argument("--betweenness-samples", type=int, default=None,
                    help="sampled sources for betweenness; its error is entirely this number")
    ap.add_argument("--closeness-samples", type=int, default=None)
    ap.add_argument("--spot-check", type=int, default=None,
                    help="exact shortest-path runs used to measure the closeness error")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the measures from a previous run's scores.npz and only redo the "
                         "joining and the files, which takes about a minute")
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep the decompressed edge list, which is a few GB")
    # exact-betweenness only
    ap.add_argument("--from", dest="measured", default=None,
                    help="the finished centrality directory to compare against")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="how large a subgraph exact Brandes may be asked to finish")
    ap.add_argument("--ego-random", type=int, default=None)
    ap.add_argument("--ego-top", type=int, default=None)
    args = ap.parse_args(argv)
    config.MAX_AUTHORS = args.max_authors

    if args.command == "centrality":
        # imported here rather than at the top: `export` needs no graph library, and a missing or
        # broken native wheel must not stop the dataset itself from being built
        from . import centrality as CE
        source = Path(args.source) if args.source else CE.find_export()
        target = Path(args.out) if args.out else centrality_out_dir(source)
        payload = CE.run(source, target, threads=args.threads, seed=args.seed,
                         betweenness_samples=args.betweenness_samples or CE.BETWEENNESS_SAMPLES,
                         closeness_samples=args.closeness_samples or CE.CLOSENESS_SAMPLES,
                         keep_temp=args.keep_temp, spot_check=args.spot_check, resume=args.resume)
        print(json.dumps({k: v for k, v in payload.items() if k not in ("top", "rules", "dump")},
                         indent=2))
        print("\nmost between: " + ", ".join(
            f"{r['name']} ({r['value']:.2e})" for r in payload["top"]["betweenness"][:5]))
        print(f"\nwritten to {target}")
        return 0

    if args.command == "exact-betweenness":
        from . import centrality as CE, exact as XB
        source = Path(args.source) if args.source else CE.find_export()
        measured = Path(args.measured) if args.measured else centrality_out_dir(source)
        target = Path(args.out) if args.out else measured
        payload = XB.run(source, measured, target, max_nodes=args.max_nodes or XB.MAX_EXACT_NODES,
                         ego_random=args.ego_random or XB.EGO_RANDOM,
                         ego_top=args.ego_top or XB.EGO_TOP,
                         threads=args.threads, seed=args.seed, keep_temp=args.keep_temp)
        print(json.dumps(payload, indent=2))
        print(f"\nwritten to {target}")
        return 0

    con, meta = data.connect()
    try:
        if args.command == "scopes":
            rows = EX.probe(con, with_bins=args.with_bins, target=args.expect_edges)
            print(json.dumps(rows, indent=2))
            return 0
        target = args.out or out_dir(meta, args.with_bins, args.scope)
        payload = EX.export(con, meta, target, with_bins=args.with_bins,
                            expect_edges=args.expect_edges, scope=args.scope,
                            probe_scopes=args.probe)
    finally:
        con.close()
    print(json.dumps({k: v for k, v in payload.items() if k not in ("rules", "dump")}, indent=2))
    print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
