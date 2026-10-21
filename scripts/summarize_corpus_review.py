#!/usr/bin/env python3
"""Regenerate manuscript metric cells and compare all three corpus stages."""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ChatGPT", "Claude", "Gemini"]
FAMILIES = [
    ("Melodic contour", set("U11 U12 U13".split())),
    ("Parallel motion", set("S1-8 S1-9 S2-5 S2-6 S3-7 S3-8 S4-7 S5-14".split())),
    ("Dissonance treatment", set("U17 S2-1 S2-2 S3-1 S3-2 S3-3 S3-4 S4-1 S4-2 S4-3 S4-4 S4-5 S4-9 S5-6 S5-8".split())),
    ("Cadential", set("U3 U4 U5 U10 S1-3 S1-4 S2-4 S4-11 S5-15".split())),
]


def metrics(rows, labels):
    output = []

    def values(model=None, species=None, placement=None, cantus=None):
        return [r["result"]["summary"]["total_violations"] for r in rows if all(
            value is None or r["metadata"][key] == value
            for key, value in [("llm", model), ("species", species),
                               ("direction", placement), ("note", cantus)])]

    def add(metric, model, species, placement, cantus, statistic, value, dp=2, percent=False):
        output.append(dict(
            metric=metric, model=model or "All", species=species or "All",
            placement=placement or "All", cantus=cantus or "All",
            statistic=statistic, value=value,
            formatted=f"{value:.{dp}f}" + ("%" if percent else "")))

    # First 332 rows reproduce the cell definitions and order of the prior
    # manuscript table audit. All SDs use the population convention.
    for model in MODELS:
        for species in [None, 1, 2, 3, 4, 5]:
            vals = values(model, species)
            add("M1", model, species, None, None, "mean", statistics.mean(vals))
            add("M1", model, species, None, None, "SD across runs", statistics.pstdev(vals))
        for species in [None, 1, 2, 3, 4, 5]:
            vals = values(model, species)
            tiers = [("perfect", lambda v: v == 0), ("low", lambda v: 1 <= v <= 2),
                     ("moderate", lambda v: 3 <= v <= 5), ("high", lambda v: v >= 6)]
            for name, predicate in tiers:
                add("M2", model, species, None, None, name,
                    100 * sum(predicate(v) for v in vals) / len(vals), 0, True)
        for placement in ["Above", "Below"]:
            means = [statistics.mean(values(model, species, placement)) for species in range(1, 6)]
            for species, mean in enumerate(means, 1):
                add("M4", model, species, placement, None, "mean", mean)
            add("M4", model, None, placement, None, "SD across species means", statistics.pstdev(means), 3)
        for cantus in "ACEFG":
            vals = values(model, cantus=cantus)
            add("M5", model, None, None, cantus, "mean", statistics.mean(vals))
            add("M5", model, None, None, cantus, "SD across runs", statistics.pstdev(vals))
        add("M5", model, None, None, None, "mean", statistics.mean(values(model)))
        add("M5", model, None, None, None, "SD across cantus means",
            statistics.pstdev([statistics.mean(values(model, cantus=n)) for n in "ACEFG"]), 3)
    for cantus in "ACEFG":
        add("M5", None, None, None, cantus, "mean", statistics.mean(values(cantus=cantus)))
        add("M5", None, None, None, cantus, "SD across model means",
            statistics.pstdev([statistics.mean(values(m, cantus=cantus)) for m in MODELS]))
    add("M5", None, None, None, None, "mean", statistics.mean(values()))
    add("M5", None, None, None, None, "SD across cantus means",
        statistics.pstdev([statistics.mean(values(cantus=n)) for n in "ACEFG"]), 3)
    for species in range(1, 6):
        for model in MODELS + [None]:
            means = [statistics.mean(values(model, species, cantus=n)) for n in "ACEFG"]
            for cantus, mean in zip("ACEFG", means):
                add("M5", model, species, None, cantus, "mean", mean)
            add("M5", model, species, None, None, "mean", statistics.mean(values(model, species)))
            add("M5", model, species, None, None, "SD across cantus means", statistics.pstdev(means), 3)
    assert len(output) == 332
    # Preserve the manuscript's category definitions, resolving by rule name
    # through its historical report. This avoids silently remapping U17/U18.
    for model in MODELS:
        categories = Counter()
        for row in rows:
            if row["metadata"]["llm"] != model:
                continue
            for exercise in row["result"]["exercises"]:
                for violation in exercise["violations"]:
                    codes = labels[violation["rule_name"]]
                    name = next((name for name, family in FAMILIES if family & codes), "Other")
                    categories[name] += 1
        total = sum(values(model))
        assert sum(categories.values()) == total
        for category in [f[0] for f in FAMILIES] + ["Other"]:
            add("M3", model, None, None, None, category + ": count", categories[category], 0)
            add("M3", model, None, None, None, category + ": mean", categories[category] / 150)
            add("M3", model, None, None, None, category + ": share", 100 * categories[category] / total, 0, True)
        for placement in ["Above", "Below"]:
            add("M4", model, None, placement, None, "mean", statistics.mean(values(model, placement=placement)))
    return output


def signature(path):
    result = []
    for part in ET.parse(path).getroot().findall("part"):
        divisions, staves = 1, defaultdict(list)
        for measure in part.findall("measure"):
            divisions = int(measure.findtext("attributes/divisions", str(divisions)))
            for note in measure.findall("note"):
                pitch = note.find("pitch")
                name = ("R" if pitch is None else
                        pitch.findtext("step") + pitch.findtext("alter", "") + pitch.findtext("octave"))
                duration = Fraction(note.findtext("duration", "0")) / divisions
                staves[note.findtext("staff", "1")].append((name, str(duration)))
        result.extend(tuple(staves[staff]) for staff in sorted(staves))
    return tuple(result)


def duplicates(rows, root):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row["metadata"][k] for k in ("llm", "species", "direction", "note"))
        groups[key].append(signature(root / row["result"]["file"]))
    extra = members = triples = 0
    by_model, by_species = Counter(), Counter()
    for key, streams in groups.items():
        counts = Counter(streams)
        n = sum(v - 1 for v in counts.values())
        extra += n
        members += sum(v for v in counts.values() if v > 1)
        triples += sum(v == 3 for v in counts.values())
        by_model[key[0]] += n
        by_species[key[1]] += n
    return dict(additional_copies=extra, runs_with_a_matching_copy=members,
                groups_of_three=triples, additional_by_model=dict(by_model),
                additional_by_species=dict(by_species))


def csv_file(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    destination = args.candidate / "comparison"
    if destination.exists():
        raise ValueError("Comparison output exists; refusing to overwrite.")
    report = (ROOT / "Résultat d'analyse/analysis_report.md").read_text()
    labels = {name: set(code.split("/")) for code, name in re.findall(
        r"^- \[([^]]+)\] .*? — ([^:]+):", report, re.M)}
    saved = json.loads((ROOT / "Résultat d'analyse/batch_results.json").read_text())["results"]
    base = json.loads((args.base / "evaluation/candidate_results.json").read_text())
    reviewed = json.loads((args.candidate / "evaluation/candidate_results.json").read_text())
    stages = {"saved": (saved, ROOT), "tag_only": (base, args.base),
              "author_reviewed": (reviewed, args.candidate)}
    cells, summaries = {}, {}
    for stage, (data, root) in stages.items():
        rows = list(data.values())
        if len(rows) != 450:
            raise ValueError("All 450 runs must be evaluated.")
        cells[stage] = metrics(rows, labels)
        summaries[stage] = dict(duplicates=duplicates(rows, root), models={
            model: dict(
                total=sum(r["result"]["summary"]["total_violations"] for r in rows if r["metadata"]["llm"] == model),
                perfect_runs=sum(r["result"]["summary"]["total_violations"] == 0 for r in rows if r["metadata"]["llm"] == model))
            for model in MODELS})
    changes = []
    for previous, intermediate, current in zip(*[cells[s] for s in stages]):
        dimensions = {k: current[k] for k in current if k not in ("value", "formatted")}
        assert dimensions == {k: previous[k] for k in dimensions} == {k: intermediate[k] for k in dimensions}
        if previous["formatted"] != current["formatted"]:
            changes.append(dict(**dimensions, saved=previous["formatted"],
                                tag_only=intermediate["formatted"], author_reviewed=current["formatted"]))
    per_file = []
    for path, row in reviewed.items():
        old = base[path]["result"]["summary"]["total_violations"]
        new = row["result"]["summary"]["total_violations"]
        if old != new:
            per_file.append(dict(retained=path, tag_only=old, author_reviewed=new, change=new - old))
    destination.mkdir()
    for stage in stages:
        csv_file(destination / f"metrics_{stage}.csv", cells[stage])
    csv_file(destination / "changed_metric_cells.csv", changes)
    csv_file(destination / "author_decision_deltas.csv", per_file)
    summary = dict(stages=summaries, changed_displayed_metric_cells=len(changes),
                   files_with_count_changes_after_author_review=len(per_file),
                   duplicate_definition="Ordered pitch/rest and duration streams per staff within a model/configuration; all runs retained.",
                   m3_definition="Historical manuscript category mapping, matched by rule name; U17/U18 display-label question remains separate.")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    (destination / "m3_rule_name_mapping.json").write_text(json.dumps({
        name: next((name for name, family in FAMILIES if family & codes), "Other")
        for name, codes in labels.items()}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
