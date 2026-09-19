"""Settings for the link-prediction jobs, overridable through environment variables."""
import os

HORIZON = int(os.environ.get("LINKS_HORIZON", "2"))          # years ahead a new link may appear

# Snapshot years. None = derived from the dump: test ends at the last complete year, train is one
# horizon earlier, so the two label windows never overlap.
T_TRAIN = int(os.environ["LINKS_T_TRAIN"]) if os.environ.get("LINKS_T_TRAIN") else None
T_TEST = int(os.environ["LINKS_T_TEST"]) if os.environ.get("LINKS_T_TEST") else None

# Who gets predictions for: authors with a body of work and recent activity at the snapshot.
MIN_PAPERS = int(os.environ.get("LINKS_MIN_PAPERS", "3"))
RECENT_YEARS = int(os.environ.get("LINKS_RECENT_YEARS", "3"))

# Anchors are sampled: every one of them brings all its distance-2 candidates (hundreds each).
TRAIN_ANCHORS = int(os.environ.get("LINKS_TRAIN_ANCHORS", "20000"))
TEST_ANCHORS = int(os.environ.get("LINKS_TEST_ANCHORS", "4000"))
MAX_CANDIDATES = int(os.environ.get("LINKS_MAX_CANDIDATES", "2000"))   # per anchor, top by common neighbours
NEG_PER_ANCHOR = int(os.environ.get("LINKS_NEG_PER_ANCHOR", "300"))    # training only; positives are all kept

# Anchors are split by hash so no author is an anchor in two snapshots.
TRAIN_BUCKETS = set(range(0, 7))   # snapshot T_TRAIN
VAL_BUCKETS = {7}                  # snapshot T_TRAIN, held out for early stopping checks
TEST_BUCKETS = {8, 9}              # snapshot T_TEST, reported metrics

SEED = 7
