#!/usr/bin/env python3
"""Evaluate a separate repair candidate without replacing the published snapshot.

Requires music21 and the evaluator checkout used to reproduce the saved results.
Writes full candidate results, per-file deltas, and timing/structure concerns.
"""
from __future__ import annotations

import argparse
import collections
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6fcfeaeaf9e4a475916d403d131d6c8aa842c11c"
NOTE_LENGTHS = {
    "whole": Fraction(4), "half": Fraction(2), "quarter": Fraction(1),
    "eighth": Fraction(1, 2), "16th": Fraction(1, 4), "32nd": Fraction(1, 8),
}


def timing_concerns(path: Path) -> list[dict]:
    """Report value conflicts; never choose a duration or fill a missing beat."""
    root = ET.parse(path).getroot()
    concerns = []
    for part in root.findall("part"):
        divisions, expected = Fraction(1), Fraction(4)
        for measure in part.findall("measure"):
            attr = measure.find("attributes")
            if attr is not None:
                divisions = Fraction(attr.findtext("divisions", str(divisions)))
                time = attr.find("time")
                if time is not None and time.find("beats") is not None:
                    expected = Fraction(time.findtext("beats")) * 4 / Fraction(time.findtext("beat-type"))
            cursor, ends = Fraction(0), {}
            for element in measure:
                if element.tag in ("backup", "forward"):
                    delta = Fraction(element.findtext("duration")) / divisions
                    cursor += delta if element.tag == "forward" else -delta
                    if cursor < 0:
                        concerns.append(dict(part=part.get("id"), measure=measure.get("number"),
                                             kind="negative_cursor", value=str(cursor)))
                elif element.tag == "note" and element.find("grace") is None:
                    duration = element.findtext("duration")
                    if duration is None:
                        concerns.append(dict(part=part.get("id"), measure=measure.get("number"),
                                             kind="missing_duration"))
                        continue
                    length = Fraction(duration) / divisions
                    typ = element.findtext("type")
                    if typ in NOTE_LENGTHS and element.find("time-modification") is None:
                        written = NOTE_LENGTHS[typ] * sum(
                            (Fraction(1, 2**i) for i in range(len(element.findall("dot")) + 1)),
                            Fraction(0))
                        if written != length:
                            concerns.append(dict(part=part.get("id"), measure=measure.get("number"),
                                                 kind="type_duration_conflict", type=typ,
                                                 duration=str(length), written=str(written)))
                    key = (element.findtext("staff", "1"), element.findtext("voice", "1"))
                    if element.find("chord") is None:
                        cursor += length
                    ends[key] = max(cursor, ends.get(key, Fraction(0)))
            for (staff, voice), end in ends.items():
                if end != expected:
                    concerns.append(dict(part=part.get("id"), measure=measure.get("number"),
                                         staff=staff, voice=voice, kind="voice_end_vs_meter",
                                         end=str(end), expected=str(expected)))
    return concerns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    args = parser.parse_args()
    candidate, evaluator = args.candidate.resolve(), args.evaluator.resolve()
    commit = subprocess.check_output(["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(evaluator), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != COMMIT or dirty:
        raise ValueError("Use an unmodified checkout of evaluator commit " + COMMIT)
    destination = candidate / "evaluation"
    if destination.exists():
        raise ValueError("Evaluation output exists; refusing to overwrite.")
    sys.path.insert(0, str(evaluator / "src"))
    from counterpoint_spellcheck.checker import Checker
    from music21 import converter
    import music21

    warnings.simplefilter("ignore")
    audit = json.loads((candidate / "repair_log.json").read_text())
    saved = json.loads((ROOT / "Résultat d'analyse/batch_results.json").read_text())
    old_by_path = {r["result"]["file"]: r for r in saved["results"].values()}
    results, deltas, concerns, errors = {}, [], [], []
    for i, record in enumerate(audit["files"], 1):
        old = old_by_path[record["retained"]]
        path = candidate / record["candidate"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["candidate_sha256"]:
            raise ValueError(f"Candidate hash mismatch: {path}")
        try:
            score = converter.parse(str(path), forceSource=True)
            xml_root = ET.parse(path).getroot()
            # All corpus files have equal measure counts in their parts/staves.
            xml_counts = {len(part.findall("measure")) for part in xml_root.findall("part")}
            parsed_counts = {len(part.getElementsByClass("Measure")) for part in score.parts}
            if len(score.parts) != 2 or xml_counts != parsed_counts:
                raise ValueError(f"Import changed part/measure coverage: {xml_counts}, {parsed_counts}")
            new = Checker(species=record["species"], cf_positions=[old["metadata"]["cf_voice"]],
                          split_exercises=True).run(path).to_dict()
            new["file"] = record["candidate"]
            results[record["retained"]] = {"metadata": old["metadata"], "result": new,
                                           "repair_status": record["status"]}
            previous = old["result"]["summary"]
            current = new["summary"]
            delta = {
                **{k: record[k] for k in ("source", "retained", "candidate", "model", "species",
                                          "placement", "cantus", "run", "status")},
                "source_failed_xml": record["source_xml_error"] is not None,
                "saved_violations": previous["total_violations"],
                "candidate_violations": current["total_violations"],
                "delta_violations": current["total_violations"] - previous["total_violations"],
                "saved_measures": previous["measures_analyzed"],
                "candidate_measures": current["measures_analyzed"],
                "delta_measures": current["measures_analyzed"] - previous["measures_analyzed"],
            }
            # Non-edited archived files must reproduce the old counts; the only
            # original->retained differences were the 20 failed XML files.
            if record["status"] != "repaired" and delta["delta_violations"]:
                raise ValueError("Unmodified file unexpectedly changed violation count.")
            deltas.append(delta)
            for issue in timing_concerns(path):
                concerns.append({"candidate": record["candidate"], **issue})
        except Exception as exc:
            errors.append({"file": record["candidate"], "error": repr(exc)})
        if i % 100 == 0:
            print("Evaluated", i, flush=True)
    grouped = []
    for model in ("ChatGPT", "Claude", "Gemini"):
        for species in (None, 1, 2, 3, 4, 5):
            rows = [r for r in deltas if r["model"] == model
                    and (species is None or r["species"] == species)]
            if not rows:
                continue
            total = sum(r["candidate_violations"] for r in rows)
            grouped.append(dict(
                model=model, species=species, runs=len(rows),
                saved_total=sum(r["saved_violations"] for r in rows),
                repair_original_20_only_total=sum(
                    r["candidate_violations"] if r["source_failed_xml"] else r["saved_violations"]
                    for r in rows),
                candidate_total=total, candidate_mean=total / len(rows),
                restored_measures=sum(r["delta_measures"] for r in rows)))
    summary = {
        "python": platform.python_version(), "music21": music21.__version__,
        "evaluator_commit": commit, "evaluated": len(results), "errors": errors,
        "unresolved_structure_files": audit["summary"]["statuses"].get("needs_review", 0),
        "timing_concern_files": len({r["candidate"] for r in concerns}),
        "timing_concerns_by_kind": dict(collections.Counter(r["kind"] for r in concerns)),
        "groups": grouped,
        "author_review_cases": audit["summary"].get("author_review_cases", 0),
        "interpretation": (
            "Separate candidate evaluation, with all 450 runs retained; published results are unchanged. "
            + ("Encoding, import, or timing concerns remain."
               if errors or concerns or audit["summary"]["statuses"].get("needs_review", 0)
               else "No unresolved structural or timing issues were found by these checks.")
            + (" Author-approved reconstructions are logged separately from mechanical tag repairs."
               if audit["summary"].get("author_review_cases") else "")
        ),
    }
    destination.mkdir()
    for name, data in (
        ("summary.json", summary), ("candidate_results.json", results),
        ("timing_concerns.json", concerns), ("deltas.json", deltas),
    ):
        (destination / name).write_text(json.dumps(data, indent=2))
    if deltas:
        with (destination / "deltas.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
            writer.writeheader()
            writer.writerows(deltas)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
