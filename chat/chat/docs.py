"""
The only part of this service that is retrieval over text.

Questions about the *data itself* - "what is a disambiguation bin?", "does this include preprints?",
"how accurate is the venue model?", "how fresh is this?" - are not answerable by SQL, and they are
exactly the questions a newcomer asks first. The corpus is the project's own definitions: a few
dozen short chunks, so ranking is idf-weighted token overlap computed in plain Python. No embedding
model, no vector store, nothing to keep in sync: at this size a bigger machine would be a decoration.

The model cards are written here as prose but their *numbers* are fetched live from the ml and
search services, so a retrained model never leaves a stale figure in an answer.
"""
import math
import re
from collections import Counter

CHUNKS = [
    ("What dblp is", """
     dblp is a bibliography of computer science, published as one XML dump. It is two datasets in
     one: publication records, and a registry of author pages that decides which person wrote what.
     The dashboard reads a snapshot of that dump (about 12.9 million records), converted to a
     parquet file, and queries it live. Every number is computed from the snapshot at request time,
     not copied from a report."""),

    ("Record types", """
     A record has a type: article (journal paper), inproceedings (conference paper), www (an author
     page, or a cross-reference), proceedings (a whole volume), phdthesis, mastersthesis,
     incollection (a book chapter), book, and data (a research-data record). "Publication" in this
     dashboard means every record except www and proceedings."""),

    ("Journal and conference papers", """
     Most charts count only journal and conference papers: type article or inproceedings, and not a
     preprint. That excludes theses, books, chapters, proceedings volumes and data records, which
     follow dblp's harvesting rather than research output."""),

    ("Preprints", """
     A record counts as a preprint if its journal is CoRR (that is arXiv, as dblp indexes it) or its
     publtype starts with "informal". Preprints are excluded from trend charts unless a chart says
     otherwise, because arXiv growth would otherwise look like publishing growth."""),

    ("Author pages, and the three kinds", """
     An author page is a www record whose key starts with homepages/. Each page is one identity.
     There are three kinds. A regular page is an ordinary person. A numbered page has a name ending
     in four digits, like "Wei Wang 0001": dblp's editors created it to separate namesakes, so a
     numbered page is a verified single person. A disambiguation page - a "bin" - is a bare name
     holding papers by many different people that nobody has separated yet."""),

    ("Disambiguation bins", """
     A disambiguation bin is an author page marked publtype="disambiguation": a bare name like "Wei
     Wang" carrying papers by many different people. There are about 33,700 of them, and roughly
     17.5% of papers (26.8% of papers in the 2020s) have at least one author sitting on one. Bins are
     excluded from anything that treats a page as a person: careers, co-author counts, the network,
     author leaderboards. A bin is not a person, and in the co-authorship graph it would otherwise be
     the best-connected node in computer science."""),

    ("Unidentified authors", """
     An author slot is "unidentified" when the name on the paper resolves to a disambiguation bin:
     dblp knows a name was there but not which person it was. Papers report how many such slots they
     have. Middle authors are lost far more often than first or last authors."""),

    ("Venue series", """
     A venue is identified by its series key - the first two segments of a record key, like conf/cvpr
     or journals/tit - not by the venue name string, which fragments across spellings and volumes.
     "Series" therefore means a conference or a journal across all its years. A series' usual name is
     the most common name string it used."""),

    ("Author slots", """
     The unit behind most author statistics is the slot: one row per (paper, author position). A
     paper with five authors is five slots. Slots keep the position, so first-author and last-author
     shares are computable; a slot whose name matches no author page stays unresolved rather than
     being dropped."""),

    ("Preprint and published twins", """
     dblp often holds the same work twice - once as a CoRR preprint and once as the published paper -
     without linking them. The dashboard detects those twins by comparing normalised titles of at
     least 30 characters. Recent twins are undercounted, because many preprints are not published
     yet."""),

    ("How fresh the data is", """
     The dump is a snapshot, taken on the first of a month. Its own year is therefore incomplete, so
     the dashboard's default "last year" is the last complete year, and a chart that would end on a
     partial year says so. The api rebuilds its serving tables automatically when a new dump
     appears."""),

    ("What the dashboard computes live, and what it does not", """
     Every chart is live SQL except three, which need minutes of compute or an external API and are
     read from the analysis job's latest output: the co-authorship network page (igraph over about 26
     million co-author pairs, Leiden communities, sampled shortest paths), the OpenAlex comparison
     (700 random DOIs per kind), and the power-law fits on the long-tail page. Those cards are
     labelled with the job's date."""),

    ("What dblp does NOT contain", """
     dblp has no citation counts, no abstracts, no full text, no author affiliations on papers, no
     funding data, no journal impact factors, no download or altmetric numbers, no best-paper
     awards, no peer-review data, and no demographic information about authors: no gender, no
     country, no institution per paper. Affiliation exists only as a note on some author pages, and
     mostly on numbered pages, because editors add it to tell namesakes apart. Questions that need
     any of these cannot be answered from this data, and the honest answer is to say so - optionally
     pointing at OpenAlex, which the dashboard samples for exactly this reason. Abstracts are the one
     exception the assistant can work around: see "Questions about what papers say"."""),

    ("Questions about what papers say", """
     What a paper proposes, what a method is, why something is needed: dblp holds titles, so these are
     answered from abstracts. The search_abstracts tool pools three searches - dblp's own title search
     and OpenAlex's keyword and semantic search over abstracts - keeps the papers dblp has, ranks their
     titles and abstracts with BM25, and gives the five best to the model, which cites them as [1] to
     [5]. This is the configuration the project's DBLP-QA study found best. The abstracts come from
     OpenAlex, some papers have none there, and the question's words are sent to OpenAlex to search."""),

    ("Why citation questions cannot be answered", """
     The parquet has an n_cites column, but dblp does not publish citation counts, so it is not a
     usable measure and no chart uses it. "Most cited paper", "h-index", "impact factor" and
     "influential author" are therefore out of scope. The nearest answerable questions are about
     volume (papers per author, per venue) and about connections (co-authors, communities)."""),

    ("The OpenAlex check", """
     To measure what dblp is missing, 700 random DOIs per publication kind were looked up in
     OpenAlex. It reports how many dblp papers are found there, how many then have an abstract or an
     institution, how well the two catalogues agree on year, title and author count, and which
     research fields dblp papers fall into - only about half are primarily computer science, because
     dblp's scope is "published in a CS venue", not "is CS research". Each figure is accurate to
     about four percentage points."""),

    ("Author disambiguation model", """
     A supervised model that splits a disambiguation bin into groups of papers and names the
     numbered page each group resembles. Labels come from dblp's own numbered pages, which editors
     verified. It scores pairs of papers carrying the same name - shared co-authors, shared venue,
     year distance, title words, ORCID - then clusters, then matches clusters to known people.
     Evaluation holds out whole name blocks and always reports a co-author-overlap heuristic as the
     baseline. The live numbers are on the Disambiguation page; the site never shows a suggestion
     without them."""),

    ("Co-author link prediction model", """
     Ranks an author's distance-2 neighbours - co-authors of co-authors who are not co-authors yet -
     by the probability of a joint paper within two years. It is trained and tested on real time
     splits: the graph as it stood at the end of one year, and who actually collaborated in the two
     years after. Features are the classic neighbourhood heuristics plus recency and activity, and
     those heuristics are the baselines. Only about a quarter of new co-authors come from distance 2
     at all - half come from farther away and the rest are newcomers with no paper yet - which is the
     ceiling for any method of this kind, and the site says so. Scores are shown as measured
     came-true rates, not raw probabilities."""),

    ("Venue recommendation model", """
     Given a title and its authors, ranks the journal and conference series where it might be
     published. Three cheap scorers propose candidates and double as baselines - Naive Bayes and a
     TF-IDF centroid over title words and bigrams, plus the authors' own publication history - and a
     gradient-boosting ranker orders them. Statistics always come from years before the paper being
     scored, so nothing leaks. The strongest single signal is where the authors published before, not
     the title."""),

    ("Hybrid paper search", """
     Paper search fuses two rankings: BM25 over title words and bigrams, and cosine similarity over
     sentence-embedding vectors of every indexed title, combined by reciprocal rank fusion. So a
     query that paraphrases a title - different words, same meaning - can still rank it first. The
     embedding index covers journal and conference papers from 2010 on, about 5.4 million; anything
     older or of another kind is still found by an exact-word match run in the same request, so
     hybrid search never returns fewer results than the plain one. It is measured by swapping a
     title word for a synonym and checking whether the paper still comes back."""),

    ("How to read a long-tail chart", """
     The long-tail page plots the share of entities with at least x of something, on log-log axes. A
     straight line would mean a power law; these curves bend down, and likelihood-ratio tests prefer
     a lognormal. The Gini coefficient and the top-1% share summarise the inequality: both are
     computed on all data, not a sample. Authors with zero of something cannot sit on a log axis and
     are reported as the share left out."""),

    ("How to read the venue profile radar", """
     Each axis is a percentile rank among all venue series with at least fifty papers, so the six
     dimensions share one scale and the median series is a regular dashed hexagon at the 50th
     percentile. A percentile hides magnitude, so the table under the chart carries the raw values.
     Recent growth is bounded between -100% and +100% by construction, so a young series cannot post
     a meaningless spike."""),

    ("Counting rules that change answers", """
     Three choices change almost every number, so a good answer states them. Are preprints included?
     Usually not. Are disambiguation bins counted as people? Never. Does a paper count once, or once
     per author? Per-author statistics count slots, so a five-author paper contributes to five
     careers. "Papers per author" from the author registry counts every record type; the career
     table counts only journal and conference papers."""),
]

LIMITS = (
    "dblp has no citations, no full text, no per-paper affiliations, no impact factors, no awards and "
    "no author demographics (gender, country, institution). Questions that need those cannot be "
    "answered from this data. dblp has no abstracts either: search_abstracts takes them from OpenAlex "
    "for the papers it finds, and some papers have none there."
)
# what the assistant said before it could read abstracts, and says again with the tool switched off
LIMITS_WITHOUT_ABSTRACTS = (
    "dblp has no citations, no abstracts, no full text, no per-paper affiliations, no impact "
    "factors, no awards and no author demographics (gender, country, institution). Questions that "
    "need those cannot be answered from this data."
)


def limits():
    from . import config
    return LIMITS if config.CONTENT_TOOL else LIMITS_WITHOUT_ABSTRACTS

_TOKEN = re.compile(r"[a-z0-9']+")
STOP = set("the a an of in on to by is are and or for with from into over under via using this that "
           "these those it its as at be do does can not what which who how when why".split())


def _tokens(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


_DOCS = [(title, " ".join(body.split())) for title, body in CHUNKS]
_BAGS = [Counter(_tokens(f"{title} {body}")) for title, body in _DOCS]
_DF = Counter(t for bag in _BAGS for t in bag)
_IDF = {t: math.log(len(_DOCS) / df) + 1.0 for t, df in _DF.items()}


def search(question, top=3):
    """The chunks most like the question, by idf-weighted overlap. Deterministic and instant."""
    q = Counter(_tokens(question or ""))
    if not q:
        return []
    scored = []
    for (title, body), bag in zip(_DOCS, _BAGS):
        length = sum(bag.values()) ** 0.5 or 1.0
        score = sum(_IDF.get(t, 1.0) * min(n, bag.get(t, 0)) for t, n in q.items()) / length
        if score > 0:
            scored.append((score, title, body))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"title": t, "text": b, "score": round(s, 3)} for s, t, b in scored[:top]]


def titles():
    return [t for t, _ in _DOCS]
