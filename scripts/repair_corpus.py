#!/usr/bin/env python3
"""Build a separate, auditable repair candidate for the fixed 450-run corpus.

Usage: python scripts/repair_corpus.py --output /new/candidate/directory
Requires only the standard library. Original files and saved results are read-only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_fux.utils.musicxml_repair import RepairError, repair_musicxml


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archived_path(root: Path, retained: str) -> Path:
    relative = Path(retained)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "response_standardized":
        raise ValueError(f"Unsafe retained path: {retained}")
    path = root / "outputs/response" / Path(*relative.parts[1:])
    return path.with_name(path.name[:-4]) if path.name.endswith(".musicxml.xml") else path


def parse_error(data: bytes) -> str | None:
    try:
        ET.fromstring(data)
        return None
    except ET.ParseError as exc:
        return str(exc)


def build(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output must be a NEW directory; existing files are never overwritten.")
    with (root / "response_standardized/manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 450 or len({r["path"] for r in manifest}) != 450:
        raise ValueError("Expected the fixed 450-file corpus.")
    pending, records = [], []
    for row in manifest:
        retained = root / row["path"]
        original = archived_path(root, row["path"])
        saved_bytes, source = retained.read_bytes(), original.read_bytes()
        if sha(saved_bytes) != row["sha256"]:
            raise ValueError(f"Retained corpus hash mismatch: {retained}")
        original_text = source.decode("utf-8")
        before_error = parse_error(source)
        try:
            fixed, operations, inventory = repair_musicxml(original_text)
            again, repeat_log, repeat_inventory = repair_musicxml(fixed)
            if again != fixed or repeat_log or repeat_inventory != inventory:
                raise ValueError(f"Repair is not idempotent: {original}")
            # Reversing every patch must restore the precise original bytes.
            restored = fixed
            for step in reversed(operations):
                shift = 0
                final_positions = []
                for patch in sorted(step["patches"], key=lambda p: p["start"]):
                    start = patch["start"] + shift
                    final_positions.append((start, start + len(patch["after"]), patch))
                    shift += len(patch["after"]) - len(patch["before"])
                for start, end, patch in reversed(final_positions):
                    if restored[start:end] != patch["after"]:
                        raise ValueError("Patch rollback mismatch.")
                    restored = restored[:start] + patch["before"] + restored[end:]
            if restored.encode("utf-8") != source:
                raise ValueError("Repair log does not restore original bytes.")
            status, unresolved = ("repaired" if operations else "unchanged"), None
        except RepairError as exc:
            # An unresolved file stays byte-identical to the archived response.
            # Partial repairs are not silently retained.
            fixed, operations, inventory = original_text, [], None
            status, unresolved = "needs_review", str(exc)
        candidate = fixed.encode("utf-8")
        relative = Path("scores") / Path(*Path(row["path"]).parts[1:])
        record = {
            "source": str(original.relative_to(root)), "retained": row["path"],
            "candidate": relative.as_posix(),
            "model": row["model"], "species": int(row["species"]),
            "placement": row["placement"], "cantus": row["cantus"], "run": int(row["run"]),
            "source_sha256": sha(source), "retained_sha256": sha(saved_bytes),
            "candidate_sha256": sha(candidate), "status": status,
            "source_xml_error": before_error, "candidate_xml_error": parse_error(candidate),
            "unresolved": unresolved, "operations": operations, "inventory": inventory,
        }
        records.append(record)
        pending.append((original, source, relative, candidate))
    # All decisions are computed before creating the destination.
    output.mkdir(parents=True)
    with zipfile.ZipFile(output / "archived_originals.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for original, source, relative, candidate in pending:
            info = zipfile.ZipInfo(str(original.relative_to(root)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, source)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(candidate)
            if candidate != source:
                diff = output / "diffs" / relative.relative_to("scores").with_suffix(".patch")
                diff.parent.mkdir(parents=True, exist_ok=True)
                diff.write_text("".join(difflib.unified_diff(
                    source.decode("utf-8").splitlines(True),
                    candidate.decode("utf-8").splitlines(True),
                    fromfile=str(original.relative_to(root)),
                    tofile=str(relative))), encoding="utf-8")
            if original.read_bytes() != source:
                raise ValueError(f"Source changed during processing: {original}")
    summary = {
        "files": len(records),
        "statuses": dict(collections.Counter(r["status"] for r in records)),
        "source_parse_failures": sum(r["source_xml_error"] is not None for r in records),
        "candidate_parse_failures": sum(r["candidate_xml_error"] is not None for r in records),
        "operations": dict(collections.Counter(
            op["rule"] for r in records for op in r["operations"])),
        "policy": "Tag-only repair; no inferred or changed musical values. Unresolved files unchanged.",
        "schema_validation": False,
    }
    (output / "repair_log.json").write_text(json.dumps(
        {"summary": summary, "files": records}, indent=2), encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="") as handle:
        fields = ["source", "retained", "candidate", "model", "species", "placement",
                  "cantus", "run", "source_sha256", "retained_sha256",
                  "candidate_sha256", "status", "source_xml_error",
                  "candidate_xml_error", "unresolved"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)


if __name__ == "__main__":
    main()
