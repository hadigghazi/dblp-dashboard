"""
Grounding lint: did every number in the answer come from a lookup?

The system prompt forbids stating a figure no tool produced, and the gold set checks that a tool was
called - neither checks that the sentence agrees with the table. The gap is where a model quietly
does arithmetic: asked whether one venue is bigger than another it can add up two per-year series
and report a total nobody computed, which is right until it is not.

So: pull every number out of the answer, and look for it in what the tools actually returned.
Rounding is allowed (a tool says 3,351 and the answer may say "about 3,400"), because insisting on
exact digits would flag good writing. What stays flagged is a number with no plausible source at all.

This is a measurement, not a gate: `evaluate` reports the share of answers whose numbers are all
accounted for, and names the ones that are not, so a regression is visible.
"""
import json
import re

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")
SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}
SMALL = 24          # ordinals, list lengths, "the top 10": never worth flagging
YEARS = range(1900, 2101)


def _floats(text):
    out = []
    for match in NUMBER.finditer(text or ""):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        tail = (text[match.end():match.end() + 12] or "").strip().lower()
        scaled = next((value * factor for word, factor in SCALE.items() if tail.startswith(word)), None)
        # "3.4 million" is one claim about 3,400,000 - the bare 3.4 is not a second claim
        out.append(value if scaled is None else scaled)
    return out


def _sources(payloads, question=None):
    """Every number the model was given: the whole tool result (summary, rows, note AND meta - the
    page range of a record and a model card's accuracy live only there), the arguments it called the
    tool with, and the question itself, because echoing "more than 50 authors" back is not a claim
    the tools have to support."""
    found = set(_floats(question or ""))
    for entry in payloads:
        # ensure_ascii=False: escaped, the dash in "2–50 authors" became \u2013 and its digits
        # swallowed the 50
        for value in _floats(json.dumps(entry, default=str, ensure_ascii=False)):
            found.add(value)
    return found


def _accounted_for(value, sources):
    if value in sources:
        return True
    if value == int(value) and abs(value) <= SMALL:
        return True                                  # counts of listed things, ranks, "top 5"
    if value == int(value) and int(value) in YEARS:
        return True                                  # years are everywhere in this data
    for source in sources:
        if source == 0:
            continue
        for candidate in (source, source * 100, source / 100):
            # a share and its percentage are one claim: a card reporting 0.8388 is quoted as 83.88%
            if candidate and 0.995 <= value / candidate <= 1.005:
                return True
        ratio = value / source
        for digits in (1, 2, 3):                     # "about 3,400" for 3,351
            if round(source, -max(0, len(str(int(abs(source)))) - digits)) == value:
                return True
        for places in (0, 1, 2, 3):                  # "0.03%" for 0.0284%: rounded to two places
            if round(source, places) == value:
                return True
    return False


def check(answer, payloads, question=None):
    """{ok, numbers, ungrounded}. `payloads` are the full tool results the model was handed."""
    sources = _sources(payloads, question)
    numbers = _floats(answer)
    ungrounded = sorted({v for v in numbers if not _accounted_for(v, sources)})
    return {"ok": not ungrounded, "numbers": len(numbers), "ungrounded": ungrounded,
            "sources": len(sources)}
