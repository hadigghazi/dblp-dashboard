"""
The suggested questions are a promise: click one and it works. So every one of them has to be a
question the gold set verifies - word for word, and for a suggested follow-up, the whole exchange.
A suggestion that failed would be the product demonstrating its own failure to a first-time visitor.
"""
from chat import goldset
from chat.server import EXAMPLES

SINGLE = {case["q"] for case in goldset.CASES if "q" in case}
EXCHANGES = {tuple(case["turns"]) for case in goldset.CASES if "turns" in case}


def test_every_suggestion_is_a_verified_gold_case():
    unverified = []
    for example in EXAMPLES:
        if example.get("then"):
            if (example["q"], example["then"]) not in EXCHANGES:
                unverified.append(f"{example['q']} -> {example['then']}")
        elif example["q"] not in SINGLE:
            unverified.append(example["q"])
    assert not unverified, f"suggested but not verified by the gold set: {unverified}"


def test_there_are_hard_ones_and_they_say_why():
    hard = [e for e in EXAMPLES if e["level"] == "hard"]
    assert len(hard) >= 6, "the panel shows three hard ones and a way to see others"
    for example in hard:
        assert example.get("why"), f"a hard suggestion needs a line saying why: {example['q']}"
        assert len(example["why"]) <= 80, f"keep the reason to one short line: {example['q']}"


def test_the_honest_refusal_is_among_them():
    """Refusing what dblp cannot answer is one of the hardest things for a chatbot to get right, and
    worth showing rather than hiding."""
    refusals = {case["q"] for case in goldset.CASES if case.get("refuses")}
    assert any(e["q"] in refusals for e in EXAMPLES)


def test_levels_are_only_basic_or_hard():
    assert {e["level"] for e in EXAMPLES} == {"basic", "hard"}
