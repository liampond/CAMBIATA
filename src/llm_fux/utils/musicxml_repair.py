"""Conservative, deterministic repairs of observed MusicXML tag defects.

This is an offline repair utility, not part of the generation pipeline. It changes
only redundant/misplaced closing tags or a misspelled duration opening tag.
Unknown defects raise RepairError. Musical values are never inferred or changed.
It does not claim full MusicXML schema validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
import xml.etree.ElementTree as ET


class RepairError(ValueError):
    """A repair would require a decision outside the supported tag patterns."""


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    name: str
    closing: bool
    empty: bool
    raw: str


# Tokenization only identifies candidate edits. Strict XML parsing is mandatory
# afterward. Comments, CDATA, declarations and external DOCTYPEs are opaque.
_LEX = re.compile(
    r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|"
    r"""<!DOCTYPE(?:[^>"']|"[^"]*"|'[^']*')*>|"""
    r"""</?[A-Za-z_][\w.:-]*(?:[^<>"']|"[^"]*"|'[^']*')*>""",
    re.S,
)
_NAME = re.compile(r"</?([A-Za-z_][\w.:-]*)")
_TAIL = {
    "duration", "tie", "instrument", "footnote", "level", "voice", "type",
    "dot", "accidental", "time-modification", "stem", "notehead",
    "notehead-text", "staff", "beam", "notations", "lyric", "play", "listen",
}
_MEASURE_CHILDREN = {
    "note", "backup", "forward", "direction", "attributes", "harmony",
    "figured-bass", "print", "sound", "listening", "barline", "grouping",
    "link", "bookmark",
}


def _tokens(text: str) -> list[Token]:
    if "<!ENTITY" in text or re.search(r"<!DOCTYPE[^>]*\[", text):
        raise RepairError("Internal DTD/entity declarations are unsupported.")
    result = []
    last = 0
    for match in _LEX.finditer(text):
        if "<" in text[last:match.start()]:
            raise RepairError("Unrecognized markup; refusing lexical recovery.")
        raw = match.group()
        if not raw.startswith(("<?", "<!")):
            name = _NAME.match(raw).group(1)
            if ":" in name or re.search(r"\bxmlns\s*=", raw):
                raise RepairError("Namespaced MusicXML requires a separate policy.")
            result.append(Token(match.start(), match.end(), name,
                                raw.startswith("</"), raw.endswith("/>"), raw))
        last = match.end()
    if "<" in text[last:]:
        raise RepairError("Unterminated markup.")
    return result


def _payload(text: str) -> list[str]:
    """All non-whitespace text outside markup, in its original order."""
    return [s.strip() for s in _LEX.split(text) if s.strip()]


def _opening_tags(tokens: list[Token], names: set[str]) -> list[str]:
    return [t.raw for t in tokens if not t.closing and t.name in names]


def _adjacent(text: str, left: Token, right: Token) -> bool:
    return not text[left.end:right.start].strip()


def _tail_end(text: str, tokens: list[Token], start: int) -> tuple[int, bool]:
    """Read complete note-tail elements; stop at an unambiguous boundary."""
    i = start
    end = tokens[start].start
    while i < len(tokens):
        t = tokens[i]
        if t.closing or t.name not in _TAIL:
            break
        if text[end:t.start].strip():
            raise RepairError("Unexpected content between detached note fields.")
        stack = [] if t.empty else [t.name]
        end = t.end
        i += 1
        while stack and i < len(tokens):
            c = tokens[i]
            if c.closing:
                if c.name != stack[-1]:
                    raise RepairError("Malformed detached note field.")
                stack.pop()
            elif not c.empty:
                stack.append(c.name)
            end = c.end
            i += 1
        if stack:
            raise RepairError("Unclosed detached note field.")
    if i == len(tokens):
        raise RepairError("Detached note fields lack a measure boundary.")
    boundary = tokens[i]
    if text[end:boundary.start].strip():
        raise RepairError("Unexpected content after detached note fields.")
    existing_close = boundary.closing and boundary.name == "note"
    if not (existing_close or (boundary.closing and boundary.name == "measure")
            or (not boundary.closing and boundary.name in _MEASURE_CHILDREN)):
        raise RepairError("Ambiguous boundary after detached note fields.")
    return end, existing_close


def _next_repair(text: str, tokens: list[Token]) -> dict | None:
    stack: list[tuple[Token, list[str]]] = []
    for i, token in enumerate(tokens):
        prev = tokens[i - 1] if i else None
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        parent = stack[-1][0].name if stack else None
        before_pitch = (
            parent == "note" and prev is not None and prev.closing
            and prev.name == "pitch" and _adjacent(text, prev, token))
        rule, patches = None, []
        if token.closing:
            if token.name == "pitch" and before_pitch:
                rule = "remove_duplicate_pitch_close"
                patches = [(token.start, token.end, "")]
            elif token.name == "duration" and before_pitch and nxt:
                if nxt.closing and nxt.name == "duration" and re.fullmatch(
                        r"\s*\d+(?:\.\d+)?\s*", text[token.end:nxt.start]):
                    rule = "restore_duration_opening_tag"
                    patches = [(token.start, token.end, "<duration>")]
                elif (not nxt.closing and nxt.name == "duration"
                      and _adjacent(text, token, nxt)):
                    rule = "remove_stray_duration_close"
                    patches = [(token.start, token.end, "")]
            elif token.name == "note" and parent == "note" and nxt:
                children = stack[-1][1]
                if ("pitch" in children and "duration" not in children
                        and "grace" not in children and nxt.name == "duration"
                        and not nxt.closing and _adjacent(text, token, nxt)):
                    end, existing_close = _tail_end(text, tokens, i + 1)
                    rule = "reattach_note_fields"
                    patches = [(token.start, token.end, "")]
                    if not existing_close:
                        patches.append((end, end, "</note>"))
            elif (token.name == "note" and parent == "measure" and prev
                  and prev.closing and prev.name == "note"
                  and _adjacent(text, prev, token)):
                rule = "remove_duplicate_note_close"
                patches = [(token.start, token.end, "")]
            if rule:
                measure = next((t.raw for t, _ in reversed(stack)
                                if t.name == "measure"), None)
                return {
                    "rule": rule, "line": text.count("\n", 0, token.start) + 1,
                    "measure_tag": measure,
                    # Positions refer to the text immediately before this step.
                    "patches": [
                        {"start": a, "end": b, "before": text[a:b], "after": value}
                        for a, b, value in patches
                    ],
                }
            if not stack or token.name != parent:
                raise RepairError(
                    f"Unrecognized closing tag {token.raw} at line "
                    f"{text.count(chr(10), 0, token.start) + 1}.")
            stack.pop()
        else:
            if stack:
                stack[-1][1].append(token.name)
            if not token.empty:
                stack.append((token, []))
    if stack:
        raise RepairError(f"Unclosed {stack[-1][0].name} element.")
    return None


def structural_inventory(text: str) -> dict:
    """Validate score/note containment and count explicitly encoded objects."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RepairError(f"Strict XML parse failed: {exc}") from exc
    if root.tag != "score-partwise":
        raise RepairError("Only score-partwise is supported.")
    parents = {child: parent for parent in root.iter() for child in parent}
    parts = root.findall("part")
    if not parts:
        raise RepairError("Score has no parts.")
    for element in root.iter():
        parent = parents.get(element)
        if element.tag == "measure" and (parent is None or parent.tag != "part"):
            raise RepairError("A measure is outside its part.")
        if element.tag == "note" and (parent is None or parent.tag != "measure"):
            raise RepairError("A note is outside its measure.")
        if element.tag == "part" and parent is not root:
            raise RepairError("A part is outside the score.")
    for part in parts:
        if not part.findall("measure"):
            raise RepairError("A part has no measures.")
        for measure in part.findall("measure"):
            for child in measure:
                if child.tag not in _MEASURE_CHILDREN:
                    raise RepairError(f"Detached {child.tag} in a measure.")
            for note in measure.findall("note"):
                allowed = _TAIL | {"grace", "cue", "chord", "pitch", "rest", "unpitched"}
                if any(child.tag not in allowed for child in note):
                    raise RepairError("Unsupported child in a note.")
                if sum(len(note.findall(t)) for t in ("pitch", "rest", "unpitched")) != 1:
                    raise RepairError("A note has missing/ambiguous pitch or rest.")
                durations = note.findall("duration")
                grace = note.find("grace") is not None
                if (grace and durations) or (not grace and len(durations) != 1):
                    raise RepairError("A non-grace note needs exactly one duration.")
                if durations:
                    try:
                        positive = Fraction(durations[0].text) > 0
                    except (ValueError, TypeError, ZeroDivisionError):
                        positive = False
                    if not positive:
                        raise RepairError("Note duration is not positive.")
                for field in ("voice", "type", "staff"):
                    if len(note.findall(field)) > 1:
                        raise RepairError(f"Duplicate note {field} field.")
                if note.find("staff") is not None:
                    try:
                        if int(note.findtext("staff")) <= 0:
                            raise ValueError
                    except (ValueError, TypeError):
                        raise RepairError("Invalid explicit staff number.") from None
    return {
        "parts": len(parts),
        "measures_by_part": [
            {"id": p.get("id"), "numbers": [m.get("number") for m in p.findall("measure")]}
            for p in parts
        ],
        "measures": sum(len(p.findall("measure")) for p in parts),
        "notes": len(root.findall("./part/measure/note")),
        "pitched_notes": len(root.findall("./part/measure/note/pitch")),
        "rests": len(root.findall("./part/measure/note/rest")),
    }


def repair_musicxml(text: str) -> tuple[str, list[dict], dict]:
    """Return repaired text, a replayable edit log, and structural checks.

    No generic XML recovery, reserialization, duration normalization, or musical
    correction is performed. Applying this function again must make no changes.
    """
    original = text
    original_tokens = _tokens(original)
    logs = []
    for _ in range(200):
        tokens = _tokens(text)
        step = _next_repair(text, tokens)
        if step is None:
            break
        for patch in sorted(step["patches"], key=lambda p: p["start"], reverse=True):
            text = text[:patch["start"]] + patch["after"] + text[patch["end"]:]
        logs.append(step)
    else:
        raise RepairError("Repair limit exceeded.")
    inventory = structural_inventory(text)
    if _payload(original) != _payload(text):
        raise RepairError("Repair changed the source text values.")
    protected = {
        "part", "measure", "note", "pitch", "step", "alter", "octave",
        "rest", "unpitched", "voice", "staff", "type", "dot", "tie", "tied",
        "divisions", "backup", "forward", "time-modification", "grace", "chord",
    }
    if _opening_tags(original_tokens, protected) != _opening_tags(_tokens(text), protected):
        raise RepairError("Repair changed musical or structural opening tags.")
    for name, count in (("measure", inventory["measures"]), ("note", inventory["notes"])):
        if sum(t.name == name and not t.closing for t in original_tokens) != count:
            raise RepairError(f"Repair lost or duplicated {name} elements.")
    inventory["text_values_preserved"] = True
    inventory["musical_opening_tags_preserved"] = True
    return text, logs, inventory
