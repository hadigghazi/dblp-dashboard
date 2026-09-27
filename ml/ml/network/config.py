"""Settings for the network export, overridable through environment variables."""
import os

# An edge exists between two authors who share a paper with this many authors. The upper bound is
# the standard treatment of hyperauthorship: one 200-author paper would contribute 19,900 edges of
# a single clique, and those cliques dominate betweenness and closeness. The export reports exactly
# how many papers the cap removes, so the choice is in the datasheet rather than hidden.
MIN_AUTHORS = int(os.environ.get("NETWORK_MIN_AUTHORS", "2"))
MAX_AUTHORS = int(os.environ.get("NETWORK_MAX_AUTHORS", "50"))

# A venue is a "ground-truth community" only if enough of its authors are in the graph - the same
# idea as SNAP's com-DBLP, whose communities are publication venues.
MIN_COMMUNITY = int(os.environ.get("NETWORK_MIN_COMMUNITY", "3"))
TOP_COMMUNITIES = int(os.environ.get("NETWORK_TOP_COMMUNITIES", "5000"))

NAME = os.environ.get("NETWORK_NAME", "dblp-coauthor")
