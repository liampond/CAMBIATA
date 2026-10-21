"""Guard the scope and reversibility of an author-approved score treatment."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "apply_corpus_review", Path(__file__).resolve().parents[1] / "scripts/apply_corpus_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

BEFORE = '<measure number="2"><note><pitch><step>G</step><octave>4</octave></pitch></note></measure>'
AFTER = BEFORE.replace("</pitch>", "</pitch><duration>4</duration>")
CASE = {"id": "test", "measure": 2, "current_xml": BEFORE, "proposed_xml": AFTER,
        "proposal": "Fill the missing duration."}


def test_only_reviewed_measure_changes_and_original_bytes_can_be_restored():
    prefix = '<score-partwise><part id="P1">\n  <measure number="1"><direction/></measure>\n  '
    suffix = '\n  <measure number="3"><note><rest/><duration>4</duration></note></measure></part></score-partwise>'
    original = prefix + BEFORE + suffix
    changed, operation = review.apply_measure(original, CASE)
    assert changed.startswith(prefix) and changed.endswith(suffix)
    assert review.rollback(changed, [operation]) == original
    assert review.apply_measure(changed, CASE) == (changed, None)


def test_refuses_changed_source_pitch():
    with pytest.raises(ValueError, match="differs from the reviewed"):
        review.apply_measure(BEFORE.replace("<step>G</step>", "<step>A</step>"), CASE)


def test_refuses_ambiguous_target_measure():
    with pytest.raises(ValueError, match="exactly one"):
        review.apply_measure(BEFORE + BEFORE, CASE)


def test_refuses_rollback_over_unexpected_edit():
    changed, operation = review.apply_measure(BEFORE, CASE)
    with pytest.raises(ValueError, match="rollback mismatch"):
        review.rollback(changed.replace("<duration>4", "<duration>8"), [operation])
