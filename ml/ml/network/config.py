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

# Which records make an edge. "all" is the whole corpus - journal, conference, preprint, book,
# chapter, thesis, data - and is the default because a co-authorship is a co-authorship. The
# dashboard's own network job excludes preprints only, which is "no-preprints" below; that setting
# reproduces its numbers. The difference between the definitions is measured into the datasheet
# rather than argued about.
SCOPE = os.environ.get("NETWORK_SCOPE", "all")
SCOPES = {
    "all": "TRUE",
    "journal-conference": "b.type IN ('article', 'inproceedings') AND NOT b.is_preprint",
    # dblp marks a preprint two ways: the CoRR journal, and a publtype beginning "informal". These
    # separate the two, because an analysis that excluded only one of them counts a different graph.
    "journal-conference-no-corr": "b.type IN ('article', 'inproceedings') "
                                  "AND coalesce(b.journal, '') <> 'CoRR'",
    "journal-conference-with-preprints": "b.type IN ('article', 'inproceedings')",
    "by-key-prefix": "b.key_prefix IN ('conf', 'journals') AND NOT b.is_preprint",
    # What the dashboard's network job actually counts, per its own header: "papers with 2-50
    # authors, preprints excluded" - every record type, minus preprints. Four definitions were
    # guessed at before reading that line.
    "no-preprints": "NOT b.is_preprint",
}

# What each scope means in a sentence, for the datasheet. A reader of the files has no access to
# the SQL above, and "scope = by-key-prefix" tells them nothing on its own.
SCOPE_LABELS = {
    "all": "all (journal, conference, preprint, book, chapter, thesis, data)",
    "journal-conference": "journal and conference papers only",
    "journal-conference-no-corr": "journal and conference papers, the CoRR preprint server excluded",
    "journal-conference-with-preprints": "journal and conference papers, preprints included",
    "by-key-prefix": "records whose dblp key begins conf/ or journals/, preprints excluded",
    "no-preprints": "every record type except preprints",
}

NAME = os.environ.get("NETWORK_NAME", "dblp-coauthor")
