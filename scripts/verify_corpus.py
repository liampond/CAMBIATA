#!/usr/bin/env python3
"""Verify the fixed 450-score snapshot using only the Python standard library."""

from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import unicodedata
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ChatGPT", "Claude", "Gemini")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root=ROOT):
    corpus = root / "response_standardized"
    # Accommodate both NFC and NFD Unicode filenames across filesystems.
    analysis = next(
        p for p in root.iterdir()
        if unicodedata.normalize("NFC", p.name) == "Résultat d'analyse"
    )
    with (corpus / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((analysis / "batch_results.json").read_text(encoding="utf-8"))
    validation = json.loads((analysis / "validation_report.json").read_text(encoding="utf-8"))
    results = report["results"]
    paths = [row["path"] for row in rows]
    source_keys = [row["source_key"] for row in rows]
    require(len(rows) == len(set(paths)) == len(set(source_keys)) == 450,
            "Expected 450 distinct manifest paths and source keys")
    require(set(source_keys) == set(results) == set(validation),
            "Manifest and reports do not cover exactly the same scores")
    files = {
        p.relative_to(root).as_posix() for p in corpus.rglob("*")
        if p.is_file() and p.suffix in {".xml", ".musicxml"}
    }
    require(set(paths) == files, "Manifest does not match the score files on disk")
    require(report["summary"] == {
        "total_files": 450, "valid_files": 450, "parse_errors": 0,
        "structurally_invalid": 0, "checker_errors": 0,
    } and report["checker_errors"] == {}, "Unexpected validation/checker summary")

    cells = Counter()
    totals = defaultdict(int)
    model_counts = Counter()
    for row in rows:
        relative = Path(row["path"])
        parts = relative.parts
        require(len(parts) == 6 and parts[0] == "response_standardized"
                and not relative.is_absolute() and ".." not in parts,
                f"Invalid score path: {relative}")
        path = root / relative
        content = path.read_bytes()
        require(len(content) == int(row["bytes"]), f"Size mismatch: {relative}")
        require(hashlib.sha256(content).hexdigest() == row["sha256"],
                f"Hash mismatch: {relative}")
        ElementTree.fromstring(content)
        require(row["source_key"] == "test_data/response/" + "/".join(parts[1:]),
                f"Source key mismatch: {relative}")
        require(validation[row["source_key"]]["category"] == "valid",
                f"Score not marked valid: {relative}")
        record = results[row["source_key"]]
        meta, result = record["metadata"], record["result"]
        species, run = int(row["species"]), int(row["run"])
        require(meta == {
            "llm": row["model"], "species": species,
            "direction": row["placement"], "note": row["cantus"], "run": run,
            "cf_voice": "lower" if row["placement"] == "Above" else "upper",
        }, f"Metadata mismatch: {relative}")
        require(parts[1] == row["model"] and int(parts[2][0]) == species
                and parts[3] == row["guide_version"]
                and parts[4] == f'{row["placement"]}_{row["cantus"]}'
                and parts[5] in {f"run_{run}.musicxml", f"run_{run}.musicxml.xml"},
                f"Path metadata mismatch: {relative}")
        require(result["file"] == row["path"], f"Result path mismatch: {relative}")
        violations = sum(len(ex["violations"]) for ex in result["exercises"])
        require(all(ex["violation_count"] == len(ex["violations"])
                    for ex in result["exercises"]), f"Exercise count mismatch: {relative}")
        require(violations == result["summary"]["total_violations"] == int(row["violations"]),
                f"Violation count mismatch: {relative}")
        cells[(row["model"], species, row["placement"], row["cantus"])] += 1
        totals[row["model"]] += violations
        model_counts[row["model"]] += 1

    expected_cells = set(product(MODELS, range(1, 6), ("Above", "Below"), "ACEFG"))
    require(set(cells) == expected_cells and set(cells.values()) == {3},
            "Corpus is not balanced at three runs per model/configuration")
    require(dict(totals) == {"ChatGPT": 597, "Claude": 1338, "Gemini": 505},
            "Saved violation totals have changed")
    print("Verified 450 score hashes, XML files, and report entries across 150 cells.")
    print(f"Saved violations: {sum(totals.values()):,}")
    for model in MODELS:
        count = model_counts[model]
        print(f"{model}: n={count}, violations={totals[model]}, mean={totals[model] / count:.6f}")


if __name__ == "__main__":
    try:
        verify()
    except (ValueError, KeyError, OSError, StopIteration, ElementTree.ParseError) as exc:
        print(f"Corpus verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
