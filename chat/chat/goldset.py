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

`succeed` lists tools that must have returned a result, not a refusal - called is not the same as
worked, and a guessed key calls the right tool and gets nothing back.

`args` pins arguments as well: {tool: {parameter: value or [values]}}. It exists because the right
tool with the wrong argument - "most connected" answered with betweenness - is a wrong answer that a
tool-choice check passes.
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
    # written before the network tools existed, when only top_authors could answer it. Both can now,
    # and both are right - provided central_authors is asked for degree, not for brokers.
    dict(q="Who are the ten most connected authors?", any_of=["top_authors", "central_authors"],
         args={"central_authors": {"metric": "degree"}}),
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

    # ---- harder shapes: filters on a superlative, comparisons, ambiguity, exact records
    dict(q="Who published most at CVPR since 2020?", all_of=["resolve_venue"], any_of=["top_authors"]),
    # Third time this pattern appeared, so the lesson is the test's, not the model's: resolve_venue
    # returns each series' paper count, so "which is bigger" is answered by resolving both and
    # comparing - any further call would be a wasted round. The case below is the one that genuinely
    # needs more, because a trend is not in the resolver.
    dict(q="Is NeurIPS bigger than ICML?", all_of=["resolve_venue"]),
    dict(q="Has NeurIPS grown faster than ICML since 2015?", all_of=["resolve_venue"],
         any_of=["papers_timeseries", "venue_profile"]),
    dict(q="How many journal papers came out in 2015?", any_of=["count_papers", "papers_timeseries"]),
    dict(q="How many papers have exactly two authors?", any_of=["count_papers", "run_sql"]),
    dict(q="Which venues are the most open access?", any_of=["top_venues", "run_sql"]),
    dict(q="Show me the record conf/nips/VaswaniSPUJGKP17", any_of=["paper_detail"]),
    dict(q="What do Yang Liu's papers look like?", any_of=["resolve_author", "namesakes"]),
    dict(q="When did conference papers overtake journal papers?", any_of=["papers_timeseries"]),
    dict(q="What is the average team size in theory conferences?",
         any_of=["run_sql", "papers_timeseries", "top_venues"]),

    # ---- follow-ups: the previous turn is the only place the subject is named
    dict(turns=["Who has the most papers in dblp?", "How many co-authors does he have?"],
         any_of=["coauthors", "author_profile"]),
    dict(turns=["Tell me about CVPR", "How has it grown since 2015?"],
         any_of=["papers_timeseries", "venue_profile", "count_papers"]),
    dict(turns=["How many papers were published in 2024?", "And in 2014?"],
         any_of=["count_papers", "papers_timeseries"]),
    dict(turns=["What does Jürgen Schmidhuber publish?", "Only the journal papers since 2020, please"],
         any_of=["author_papers", "count_papers"]),
    dict(turns=["Which venues are the biggest?", "Who publishes most in the first one?"],
         any_of=["top_authors"]),
    # a follow-up that picks one of several pages the previous answer found. It needs that page's key,
    # which used to be lost between turns: the model guessed one and asked the user to identify the
    # page instead. Two different people have dblp pages under this name.
    dict(turns=["How many papers does Hussein Hazimeh have?",
                "The second one - give me some details about him"],
         all_of=["author_profile"], succeed=["author_profile"]),
    dict(turns=["Tell me about Yoshua Bengio", "Who are his most frequent co-authors?"],
         any_of=["coauthors", "author_profile"], succeed=["coauthors"]),

    # ---- the co-authorship network. "Central" and "prolific" are different questions with
    # different answers - on this dump only 58 of the 100 most-between authors are also among the 100
    # most-connected - so a model that answers one with the other is wrong in a way that reads fine.
    dict(q="Who is the most central author in computer science?", any_of=["central_authors"]),
    dict(q="Which authors bridge the most research communities?", any_of=["central_authors"],
         args={"central_authors": {"metric": "betweenness"}}),
    dict(q="How central is Yoshua Bengio in the co-authorship network?",
         all_of=["resolve_author"], any_of=["author_centrality"]),
    dict(q="How many degrees of separation are there between computer scientists?",
         any_of=["network_shape"]),
    dict(q="Is the dblp co-authorship network a small world?", any_of=["network_shape"]),
    # the trap: this one must NOT go to central_authors
    dict(q="Which author has published the most papers?", any_of=["top_authors"]),
    dict(turns=["Who is the most central author by betweenness?", "And how many papers do they have?"],
         any_of=["central_authors", "resolve_author", "author_profile"]),

    # a name in another script: dblp writes names in Latin, so it has to be transliterated first
    dict(q="كم عدد الأوراق المنشورة ليوشوا بنجيو؟", all_of=["resolve_author"], succeed=["resolve_author"]),
    # one author's papers on a topic: the model reached for a title filter that did not exist yet
    dict(q="What has Jürgen Schmidhuber published about LSTM?", any_of=["author_papers", "search_papers"],
         succeed=["author_papers"]),

    # ---- out of scope: the answer must say the data cannot support it
    dict(q="What is the most cited paper in dblp?", refuses=True),
    dict(q="What is Geoffrey Hinton's h-index?", refuses=True),
    dict(q="Which country publishes the most papers?", refuses=True),
    dict(q="Show me the abstract of that paper", refuses=True),
    dict(q="Which journal has the highest impact factor?", refuses=True),
    dict(q="How many women publish at NeurIPS?", refuses=True),
    dict(q="What is NeurIPS's acceptance rate?", refuses=True),          # no submission data exists
    dict(q="Which authors work at Google?", refuses=True),               # no per-paper affiliation
    dict(q="Which paper won best paper at CVPR 2020?", refuses=True),    # no awards
    dict(q="How many times was 'Attention is All you Need' downloaded?", refuses=True),
]

# Phrases that count as an honest refusal. The answer must say the data cannot support the question,
# not merely hedge.
REFUSAL_MARKERS = [
    "does not", "doesn't", "no citation", "not in the data", "not available", "cannot", "can't",
    "has no", "not contain", "no abstract", "no affiliation", "not something dblp", "outside",
    "not recorded", "no information",
]
