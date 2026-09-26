"""The grounding lint: it has to catch invented arithmetic without flagging good prose."""
from chat import grounding


def events(*payloads):
    return [{"type": "tool", "name": "t", "summary": p} if isinstance(p, str) else
            {"type": "tool", "name": "t", **p} for p in payloads]


def test_a_number_straight_from_a_tool_is_grounded():
    out = grounding.check("H. Vincent Poor has 3,351 records.",
                          events("H. Vincent Poor: 3,351 records 1977–2026"))
    assert out["ok"] and out["ungrounded"] == []


def test_a_number_from_a_table_row_is_grounded():
    out = grounding.check("Wei Wang covers 522 people.",
                          events({"rows": [{"base_name": "Wei Wang", "people": 522}]}))
    assert out["ok"]


def test_rounding_is_allowed():
    src = events("3,351 records")
    assert grounding.check("about 3,400 records", src)["ok"]
    assert grounding.check("roughly 3.4 thousand records", src)["ok"]


def test_years_and_small_counts_are_not_flagged():
    out = grounding.check("Between 1977 and 2026 he worked with 5 groups.", events("nothing numeric"))
    assert out["ok"]


def test_arithmetic_the_model_did_itself_is_caught():
    """The NeurIPS-vs-ICML risk: two per-year series, and a total nobody computed."""
    src = events({"rows": [{"year": 2023, "papers": 3500}, {"year": 2024, "papers": 3700}]})
    out = grounding.check("NeurIPS published 7,200 papers over those two years.", src)
    assert not out["ok"] and 7200.0 in out["ungrounded"]


def test_an_invented_figure_is_caught():
    out = grounding.check("Roughly 48,000 authors publish there every year.", events("nothing numeric"))
    assert not out["ok"] and 48000.0 in out["ungrounded"]


def test_an_answer_with_no_numbers_passes():
    assert grounding.check("dblp does not record citations.", events("x"))["ok"]
