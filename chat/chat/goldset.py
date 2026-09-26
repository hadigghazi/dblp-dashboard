"""
The gold set: what "it works" means for this assistant.

A chatbot over data has three failure modes, and a demo hides all three: it reaches for the wrong
tool, it invents a number, and it answers a question the data cannot answer. Each case below pins
one of those down - the tools that MUST be involved, and for the out-of-scope block, that the answer
refuses. `evaluate.py` runs them against the live service and reports tool-choice accuracy and
refusal accuracy.

`any_of` means at least one of these tools has to be called; `all_of` means every one. Resolution
steps are listed explicitly, because "papers of this person" is only correct if a name was turned
into a key first.
"""

CASES = [
    # ---- lookup and entity facts
    dict(q="Which author has the most papers in dblp?", any_of=["top_authors"]),
    # resolve_author already returns each candidate's record count, from the same count(*) over author
    # slots that author_profile reports, so a plain "how many papers" is correctly answered in one
    # call. Requiring a second one would score a wasted round as the right behaviour.
    dict(q="How many papers does Jürgen Schmidhuber have?", all_of=["resolve_author"]),
    # ...but a count with filters cannot come from the resolver, so this case keeps that path honest
    dict(q="How many journal papers did Jürgen Schmidhuber publish since 2020?",
         all_of=["resolve_author"], any_of=["count_papers", "author_papers"]),
    dict(q="What venues does Yoshua Bengio publish in most?", all_of=["resolve_author"],
         any_of=["author_profile"]),
    dict(q="Show me the paper 'Attention is all you need'", any_of=["paper_detail", "search_papers"]),
    dict(q="Tell me about CVPR", all_of=["resolve_venue"], any_of=["venue_profile"]),

    # ---- superlatives and rankings
    dict(q="Who are the ten most connected authors?", any_of=["top_authors"]),
    dict(q="What are the biggest conferences in dblp?", any_of=["top_venues"]),
    dict(q="Who publishes most in NeurIPS?", all_of=["resolve_venue"], any_of=["top_authors"]),
    dict(q="Which names are shared by the most people?", any_of=["most_shared_names"]),

    # ---- counting
    dict(q="How many papers were published in 2024?", any_of=["count_papers", "papers_timeseries"]),
    dict(q="How many preprints are in dblp?", any_of=["count_papers", "dataset_facts"]),
    dict(q="How many papers have more than 50 authors?", any_of=["count_papers", "run_sql"]),
    dict(q="How big is dblp?", any_of=["dataset_facts"]),

    # ---- trends
    dict(q="Has the number of authors per paper changed since 1990?", any_of=["papers_timeseries"]),
    dict(q="Is 'blockchain' still rising in paper titles?", any_of=["title_terms", "rising_words"]),
    dict(q="Compare 'llm' and 'transformer' in titles over the last ten years", any_of=["title_terms"]),
    dict(q="Which title words are disappearing?", any_of=["rising_words"]),
    dict(q="How has open access changed for journal papers?", any_of=["papers_timeseries"]),

    # ---- semantic search
    dict(q="Find papers about learning robot manipulation from a few demonstrations",
         any_of=["search_papers"]),
    dict(q="What work is there on privacy in federated learning since 2022?", any_of=["search_papers"]),

    # ---- relational / multi-hop
    dict(q="Who are Geoffrey Hinton's co-authors that also publish at ICML?",
         all_of=["resolve_author"], any_of=["coauthors", "authors_in_both"]),
    dict(q="Have Yann LeCun and Yoshua Bengio written a paper together?",
         any_of=["pair_papers", "coauthors"]),
    dict(q="Which people publish in both STOC and CVPR?", any_of=["authors_in_both", "run_sql"]),

    # ---- identity
    dict(q="How many different people are called Wei Wang?", any_of=["namesakes", "resolve_author"]),
    dict(q="What is a disambiguation bin?", any_of=["docs_lookup"]),
    dict(q="How accurate is the disambiguation model?", any_of=["model_cards"]),

    # ---- predictions
    dict(q="Where should a paper called 'Contrastive pretraining for medical image segmentation' be "
           "submitted?", any_of=["predict_venue"]),
    dict(q="Who is Jürgen Schmidhuber likely to publish with next?", all_of=["resolve_author"],
         any_of=["predict_coauthors"]),

    # ---- data quality / meta
    dict(q="Which large venues almost never carry a DOI?", any_of=["run_sql", "top_venues"]),
    dict(q="How fresh is this data?", any_of=["docs_lookup", "dataset_facts"]),
    dict(q="Does this include preprints?", any_of=["docs_lookup"]),

    # ---- out of scope: the answer must say the data cannot support it
    dict(q="What is the most cited paper in dblp?", refuses=True),
    dict(q="What is Geoffrey Hinton's h-index?", refuses=True),
    dict(q="Which country publishes the most papers?", refuses=True),
    dict(q="Show me the abstract of that paper", refuses=True),
    dict(q="Which journal has the highest impact factor?", refuses=True),
    dict(q="How many women publish at NeurIPS?", refuses=True),
]

# Phrases that count as an honest refusal. The answer must say the data cannot support the question,
# not merely hedge.
REFUSAL_MARKERS = [
    "does not", "doesn't", "no citation", "not in the data", "not available", "cannot", "can't",
    "has no", "not contain", "no abstract", "no affiliation", "not something dblp", "outside",
    "not recorded", "no information",
]
