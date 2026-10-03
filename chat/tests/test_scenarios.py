"""
The scenario runner's own checks. The scenarios themselves need a model and the real dump, so they run
on the VM; what is tested here is the part that decides pass or fail without one.
"""
from chat import evaluate as E
from chat import scenarios as SC

SUBJECT = {"key": "homepages/s/2", "name": "Sam Same 0002", "papers": 25}
OTHERS = ["homepages/s/1", "homepages/s/bin"]


def test_the_namesake_is_the_wrong_person():
    calls = [{"tool": "author_profile", "arguments": {"key": "homepages/s/1"}}]
    ok, why = SC.right_person(calls, "", SUBJECT, OTHERS)
    assert not ok and "wrong person" in why


def test_the_subject_and_a_coauthor_are_fine():
    calls = [{"tool": "pair_papers", "arguments": {"key_a": "homepages/s/2", "key_b": "homepages/a/Ada"}}]
    assert SC.right_person(calls, "", SUBJECT, OTHERS) == (True, "")


def test_a_lookup_of_someone_else_entirely_is_not_the_subject():
    calls = [{"tool": "author_profile", "arguments": {"key": "homepages/a/Ada"}}]
    ok, why = SC.right_person(calls, "", SUBJECT, OTHERS)
    assert not ok and "own page" in why


def test_a_plain_count_is_judged_by_the_number_it_quotes():
    """The name lookup answers "how many papers" on its own, so there may be no key to check."""
    assert SC.right_person([], "Sam Same 0002 has 25 records.", SUBJECT, OTHERS)[0]
    assert not SC.right_person([], "Sam Same has 37 records.", SUBJECT, OTHERS)[0]


def test_the_pages_are_found_by_name_at_run_time(ctx):
    subject, others = SC.pages(ctx, "Sam Same 0002")
    assert subject["key"] == "homepages/s/2"
    assert {p["key"] for p in others} == {"homepages/s/1", "homepages/s/bin"}
    assert SC.pages(ctx, "Nobody 0009") == (None, [])


def test_every_scenario_is_well_formed():
    known = {"q", "turns", "any_of", "all_of", "succeed", "args", "refuses", "subject", "note"}
    for name, suite in SC.SUITES.items():
        assert suite["subject"] and suite["cases"], name
        for case in suite["cases"]:
            assert set(case) <= known, case
            assert ("q" in case) != ("turns" in case), case
            assert case.get("refuses") or case.get("any_of") or case.get("all_of"), case


def test_scenarios_use_the_same_case_format_as_the_gold_set():
    """So any scenario that proves itself can be moved into the gold set unchanged."""
    case = SC.SUITES["instructor"]["cases"][0]
    assert E.arguments_ok(case, []) == (True, [])
