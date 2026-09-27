"""
The co-authorship network as a published dataset.

  python -m ml.network.cli export                    # the canonical graph, bins excluded
  python -m ml.network.cli export --with-bins        # everything, for comparison
  python -m ml.network.cli export --scope journal-conference --expect-edges 22200244

`--expect-edges` is worth using, with the matching scope: the analysis job computed 22,200,244 from
journal and conference papers, so that combination proves the join is right. The default scope is
every record type, which is a larger graph - 24.4 million edges - and the difference between the two
is the point rather than a discrepancy.
"""
import argparse
import json
import logging
import sys

from .. import config as base, data
from . import config, export as EX

log = logging.getLogger("dblp.ml.network")


def out_dir(meta, with_bins, scope):
    name = f"network-{meta.get('fingerprint', 'unknown')}"
    if scope != "all":
        name += f"-{scope}"
    return base.MODELS_DIR / (f"{name}-withbins" if with_bins else name)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="ml.network.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export"])
    ap.add_argument("--with-bins", action="store_true",
                    help="count disambiguation pages as people (they are not; for comparison only)")
    ap.add_argument("--max-authors", type=int, default=config.MAX_AUTHORS)
    ap.add_argument("--scope", choices=sorted(config.SCOPES), default=config.SCOPE,
                    help="which records make an edge (default: every type)")
    ap.add_argument("--expect-edges", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    config.MAX_AUTHORS = args.max_authors

    con, meta = data.connect()
    try:
        target = args.out or out_dir(meta, args.with_bins, args.scope)
        payload = EX.export(con, meta, target, with_bins=args.with_bins,
                            expect_edges=args.expect_edges, scope=args.scope)
    finally:
        con.close()
    print(json.dumps({k: v for k, v in payload.items() if k not in ("rules", "dump")}, indent=2))
    print(f"\nwritten to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
