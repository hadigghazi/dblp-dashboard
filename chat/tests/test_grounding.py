"""
The grounding lint: catch invented arithmetic, without flagging good prose.

The cases below are the real ones. The first live run flagged six answers, and five were the lint's
fault: it built its sources from the UI events, which carry summary/rows/note but not `meta` - where
a record's page range, an author's top venues and a model card's accuracy actually live. It now sees
the whole payload the model was handed, plus the tool arguments and the question.
"""
from chat import grounding


def payload(result, name="t", **arguments):
    return [{"name": name, "arguments": arguments, "result": result}]


def test_a_number_straight_from_a_tool_is_grounded():
    out = grounding.check("H. Vincent Poor has 3,351 records.",
                          payload({"summary": "H. Vincent Poor: 3,351 records 1977–2026"}))
    assert out["ok"] and out["ungrounded"] == []


def test_a_number_from_a_table_row_is_grounded():
    assert grounding.check("Wei Wang covers 522 people.",
                           payload({"rows": [{"base_name": "Wei Wang", "people": 522}]}))["ok"]


def test_a_number_that_lives_only_in_meta_is_grounded():
    """author_profile's venues, paper_detail's page range and model_cards' accuracy are all in meta."""
    assert grounding.check("He publishes most at NIPS (114 papers).",
                           payload({"summary": "x", "meta": {"top_venues": [{"papers": 114}]}}))["ok"]
    assert grounding.check("The paper spans pages 5998-6008.",
                           payload({"meta": {"paper": {"pages": "5998-6008"}}}))["ok"]


def test_a_number_echoed_from_the_question_is_grounded():
    out = grounding.check("There are 985 papers with more than 50 authors.",
                          payload({"summary": "985 records match"}, min_authors=51),
                          question="How many papers have more than 50 authors?")
    assert out["ok"]


def test_a_share_quoted_as_a_percentage_is_grounded():
    assert grounding.check("top-1 accuracy of about 95.46%",
                           payload({"meta": {"cards": {"top1": 0.9546}}}))["ok"]


def test_rounding_and_scale_words_are_allowed():
    src = payload({"summary": "3,351 records"})
    assert grounding.check("about 3,400 records", src)["ok"]
    assert grounding.check("roughly 3.4 thousand records", src)["ok"]


def test_years_and_small_counts_are_not_flagged():
    assert grounding.check("Between 1977 and 2026 he worked with 5 groups.",
                           payload({"summary": "nothing numeric"}))["ok"]


def test_arithmetic_the_model_did_itself_is_caught():
    """The NeurIPS-vs-ICML risk: two per-year rows, and a total nobody computed."""
    src = payload({"rows": [{"year": 2023, "papers": 3500}, {"year": 2024, "papers": 3700}]})
    out = grounding.check("NeurIPS published 7,200 papers over those two years.", src)
    assert not out["ok"] and 7200.0 in out["ungrounded"]


def test_an_invented_figure_is_caught():
    out = grounding.check("Roughly 48,000 authors publish there every year.",
                          payload({"summary": "985 records"}))
    assert not out["ok"] and 48000.0 in out["ungrounded"]


def test_an_answer_with_no_numbers_passes():
    assert grounding.check("dblp does not record citations.", payload({"summary": "x"}))["ok"]


def test_a_number_beside_a_non_ascii_character_is_still_found():
    """Serialised with escapes, the dash in "2–50 authors" became \\u2013 and swallowed the 50."""
    from chat import grounding
    got = grounding.check("Co-authorship counts papers with 2 to 50 authors.",
                          [{"name": "coauthors", "result": {"note": "papers with 2–50 authors"}}])
    assert got["ok"], got


def test_a_small_decimal_rounded_to_two_places_is_grounded():
    """"0.03%" for a share of 0.0284% is honest rounding, as "about 3,400" is for 3,351."""
    from chat import grounding
    got = grounding.check("It was 0.03% of titles in 2016.",
                          [{"name": "title_terms", "result": {"rows": [{"year": 2016, "pct": 0.0284}]}}])
    assert got["ok"], got


def test_rounding_does_not_excuse_a_different_number():
    from chat import grounding
    got = grounding.check("It was 0.05% of titles in 2016.",
                          [{"name": "title_terms", "result": {"rows": [{"year": 2016, "pct": 0.0284}]}}])
    assert not got["ok"]
