#!/usr/bin/env python3
"""Apply hash-matched author decisions to a NEW copy of the repair candidate.

The reviewed measure XML is pinned by the review dataset ID. Only that measure
is replaced; all bytes outside it remain unchanged. Original scores are read-only.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_fux.utils.musicxml_repair import repair_musicxml


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(xml: str) -> bytes:
    root = ET.fromstring(xml)
    for node in root.iter():
        node.text = node.text.strip() if node.text and node.text.strip() else None
        node.tail = None
    return ET.tostring(root)


def apply_measure(text: str, case: dict) -> tuple[str, dict | None]:
    number = str(int(case["measure"]))
    pattern = rf'<measure\b[^>]*\bnumber="{number}"[^>]*>.*?</measure>'
    matches = list(re.finditer(pattern, text, re.S))
    if len(matches) != 1:
        raise ValueError(f"{case['id']}: expected exactly one target measure.")
    match = matches[0]
    before = match.group()
    if normalized(before) == normalized(case["proposed_xml"]):
        return text, None
    if normalized(before) != normalized(case["current_xml"]):
        raise ValueError(f"{case['id']}: current measure differs from the reviewed version.")
    indentation = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
    if indentation.strip():
        indentation = ""
    # The proposal is one reviewed measure, never a replacement whole score.
    root = ET.fromstring(case["proposed_xml"])
    if root.tag != "measure" or root.get("number") != number:
        raise ValueError("Proposal is not the expected measure.")
    ET.indent(root, space="  ")
    after = ET.tostring(root, encoding="unicode").strip()
    after = after.replace("\n", "\n" + indentation)
    result = text[:match.start()] + after + text[match.end():]
    return result, {
        "rule": "author_reviewed_reconstruction", "case_id": case["id"],
        "measure": int(number), "proposal": case["proposal"],
        "patches": [{"start": match.start(), "end": match.end(),
                     "before": before, "after": after}],
    }


def rollback(text: str, operations: list[dict]) -> str:
    for step in reversed(operations):
        shift, positions = 0, []
        for patch in sorted(step["patches"], key=lambda p: p["start"]):
            start = patch["start"] + shift
            positions.append((start, start + len(patch["after"]), patch))
            shift += len(patch["after"]) - len(patch["before"])
        for start, end, patch in reversed(positions):
            if text[start:end] != patch["after"]:
                raise ValueError("Patch rollback mismatch.")
            text = text[:start] + patch["before"] + text[end:]
    return text


def safe_path(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes its directory: {relative}")
    return path


def build(base: Path, review_data: Path, decisions_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output must be a NEW directory.")
    dataset = json.loads(review_data.read_text())
    decisions = json.loads(decisions_path.read_text())
    dataset_id = sha(json.dumps(dataset["cases"], sort_keys=True).encode())[:16]
    if dataset["dataset_id"] != dataset_id or decisions["dataset_id"] != dataset_id:
        raise ValueError("Review dataset ID mismatch.")
    cases = {c["id"]: c for c in dataset["cases"]}
    choices = {c["id"]: c for c in decisions["cases"]}
    if len(choices) != len(decisions["cases"]) or choices.keys() != cases.keys():
        raise ValueError("Missing, duplicate, or unexpected review cases.")
    by_source = {}
    for case_id, c in cases.items():
        choice = choices[case_id]
        for field in ("title", "measure", "source_files", "original_sha256",
                      "candidate_sha256", "proposal"):
            if choice[field] != c[field]:
                raise ValueError(f"{case_id}: reviewed {field} does not match.")
        if choice["decision"] not in ("repair", "keep"):
            raise ValueError(f"{case_id}: an author decision is still outstanding.")
        for source in c["source_files"]:
            if source in by_source:
                raise ValueError("Overlapping review cases are unsupported.")
            by_source[source] = (c, choice)

    audit = json.loads((base / "repair_log.json").read_text())
    if len(audit["files"]) != 450 or len({r["source"] for r in audit["files"]}) != 450:
        raise ValueError("Expected 450 separate runs.")
    if not by_source.keys() <= {r["source"] for r in audit["files"]}:
        raise ValueError("A reviewed source is absent from the candidate.")
    pending, records = [], []
    for original_record in audit["files"]:
        r = copy.deepcopy(original_record)
        before = safe_path(base, r["candidate"]).read_bytes()
        original = safe_path(ROOT, r["source"]).read_bytes()
        retained = safe_path(ROOT, r["retained"]).read_bytes()
        if (sha(before) != r["candidate_sha256"] or sha(original) != r["source_sha256"]
                or sha(retained) != r["retained_sha256"]):
            raise ValueError(f"Input hash mismatch: {r['source']}")
        r["base_candidate_sha256"] = sha(before)
        r["base_status"] = r["status"]
        r["base_unresolved"] = r["unresolved"]
        r["author_review"] = None
        after = before
        if r["source"] in by_source:
            c, choice = by_source[r["source"]]
            if sha(before) != c["candidate_sha256"] or sha(original) != c["original_sha256"]:
                raise ValueError(f"Reviewed file hash mismatch: {r['source']}")
            r["author_review"] = {
                "case_id": c["id"], "decision": choice["decision"], "note": choice["note"],
                "proposal": c["proposal"], "kind": c["kind"], "dataset_id": dataset_id,
            }
            if choice["decision"] == "repair":
                fixed, operation = apply_measure(before.decode("utf-8"), c)
                if operation is None:
                    raise ValueError("Review unexpectedly has no change.")
                repeated, again = apply_measure(fixed, c)
                if repeated != fixed or again:
                    raise ValueError("Author treatment is not idempotent.")
                after = fixed.encode("utf-8")
                r["operations"].append(operation)
                checked, mechanical, r["inventory"] = repair_musicxml(fixed)
                if checked != fixed or mechanical:
                    raise ValueError("Author treatment leaves a mechanical defect.")
                r["status"], r["unresolved"] = "repaired", None
        if rollback(after.decode("utf-8"), r["operations"]).encode("utf-8") != original:
            raise ValueError(f"Cannot restore the original bytes: {r['source']}")
        r["candidate_sha256"] = sha(after)
        records.append(r)
        pending.append((r, original, before, after))

    summary = {
        "files": len(records),
        "statuses": dict(collections.Counter(r["status"] for r in records)),
        "source_parse_failures": sum(r["source_xml_error"] is not None for r in records),
        "candidate_parse_failures": sum(r["candidate_xml_error"] is not None for r in records),
        "operations": dict(collections.Counter(op["rule"] for r in records for op in r["operations"])),
        "author_review_cases": len(choices),
        "author_review_decisions": dict(collections.Counter(c["decision"] for c in choices.values())),
        "author_review_files": len(by_source),
        "author_modified_files": sum(before != after for _, _, before, after in pending),
        "policy": "Preserve mechanical tag repair; apply only the specific author-approved measure reconstructions.",
        "dataset_id": dataset_id, "decisions_sha256": sha(decisions_path.read_bytes()),
        "base_repair_log_sha256": sha((base / "repair_log.json").read_bytes()),
        "schema_validation": False,
    }
    output.mkdir(parents=True)
    shutil.copyfile(decisions_path, output / "author_decisions.json")
    shutil.copyfile(review_data, output / "review_cases.json")
    shutil.copyfile(base / "archived_originals.zip", output / "archived_originals.zip")
    for r, original, before, after in pending:
        target = safe_path(output, r["candidate"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        for name, start in (("diffs", original), ("author_diffs", before)):
            if start == after:
                continue
            diff = output / name / Path(r["candidate"]).relative_to("scores").with_suffix(".patch")
            diff.parent.mkdir(parents=True, exist_ok=True)
            diff.write_text("".join(difflib.unified_diff(
                start.decode("utf-8").splitlines(True), after.decode("utf-8").splitlines(True),
                fromfile=r["source"] if name == "diffs" else "base/" + r["candidate"],
                tofile=r["candidate"])), encoding="utf-8")
    (output / "repair_log.json").write_text(json.dumps({"summary": summary, "files": records}, indent=2))
    with (output / "manifest.csv").open("w", newline="") as handle:
        fields = ["source", "retained", "candidate", "model", "species", "placement", "cantus",
                  "run", "source_sha256", "retained_sha256", "base_candidate_sha256",
                  "candidate_sha256", "base_status", "status", "unresolved"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--review-data", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.review_data, args.decisions, args.output)
