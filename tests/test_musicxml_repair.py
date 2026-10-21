"""Conservation and refusal tests for offline repair of archived research data."""
import xml.etree.ElementTree as ET

import pytest

from llm_fux.utils.musicxml_repair import RepairError, repair_musicxml


PITCH = "<pitch><step>C</step><alter>1</alter><octave>4</octave></pitch>"
FIELDS = "<duration>4</duration><voice>5</voice><type>whole</type><staff>2</staff>"
NOTE = f"<note>{PITCH}{FIELDS}</note>"


def score(notes):
    return ('<?xml version="1.0"?>'
            '<score-partwise version="4.0"><part-list>'
            '<score-part id="P1"><part-name>Test</part-name></score-part></part-list>'
            '<part id="P1"><measure number="1">' + notes +
            '</measure><measure number="2">' + NOTE +
            '</measure></part></score-partwise>')


def normalized(xml):
    return ET.tostring(ET.fromstring(xml))


@pytest.mark.parametrize("broken", [
    NOTE.replace("</pitch>", "</pitch></pitch>"),
    NOTE.replace("<duration>", "</duration>", 1),
    NOTE.replace("</pitch>", "</pitch></duration>"),
    NOTE + "</note>",
    f"<note>{PITCH}</note>{FIELDS}",
    f"<note>{PITCH}</note>{FIELDS}</note>",
])
def test_repairs_tags_without_changing_music_or_losing_final_measure(broken):
    repaired, changes, inventory = repair_musicxml(score(broken))
    assert changes
    assert normalized(repaired) == normalized(score(NOTE))
    assert inventory["measures"] == 2
    assert inventory["pitched_notes"] == 2
    again, log, _ = repair_musicxml(repaired)
    assert again == repaired and log == []


def test_valid_file_preserves_bytes_comments_grace_notes_and_ties():
    valid = score(
        "<!-- do not repair this example: </pitch></pitch> -->"
        f"<note><grace/>{PITCH}<type>eighth</type></note>"
        f'<note>{PITCH}<duration>4</duration><tie type="start"/>'
        '<voice>5</voice><type>whole</type><staff>2</staff>'
        '<notations><tied type="start"/></notations></note>')
    fixed, log, _ = repair_musicxml(valid)
    assert fixed == valid and not log


@pytest.mark.parametrize("broken", [
    f"<note>{PITCH}</note>",  # No duration value exists to recover.
    f"<note>{PITCH}</note><note>{FIELDS}</note>",  # Two ambiguous note containers.
    NOTE.replace("<duration>4</duration>", "<duration>0</duration>"),
    NOTE.replace("</note>", "<duration>8</duration></note>"),
    NOTE.replace("</pitch>", "</unknown>"),
    NOTE.replace("</pitch>", "</pitch><unrecognized/>"),
])
def test_refuses_missing_values_ambiguous_notes_and_unknown_tag_errors(broken):
    with pytest.raises(RepairError):
        repair_musicxml(score(broken))


def test_refuses_measures_detached_from_part():
    bad = score(NOTE).replace(
        '</measure><measure number="2">',
        '</measure></part><measure number="2">')
    bad = ''.join(bad.rsplit('</part>', 1))
    ET.fromstring(bad)  # Well-formed XML, but the second measure is outside the part.
    with pytest.raises(RepairError):
        repair_musicxml(bad)


def test_refuses_internal_entities():
    with pytest.raises(RepairError):
        repair_musicxml('<!DOCTYPE score-partwise [<!ENTITY x "4">]>' + score(NOTE))
