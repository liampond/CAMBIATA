# Evaluation snapshot and provenance

These four pre-existing exports were recorded with the 450-score corpus on
2026-10-16:

| File | Contents |
| --- | --- |
| `batch_results.json` | Metadata, summary counts, and individual checker violations for all 450 scores |
| `analysis_report.md` | Human-readable per-score violations and rule labels |
| `validation_report.json` | Validation category for each score |
| `validation_summary.md` | Validation counts by model and species |

The saved reports contain 450 valid files, no parse/structural/checker errors,
and 2,440 violations. These validation results describe the retained, processed
corpus and do not measure parsing success across all original API responses.

| Model label | Runs | Violations | Mean per run |
| --- | ---: | ---: | ---: |
| ChatGPT | 150 | 597 | 3.98 |
| Claude | 150 | 1,338 | 8.92 |
| Gemini | 150 | 505 | 3.366667 |

## Paths and integrity

Original dictionary keys such as `test_data/response/ChatGPT/...` are retained
in both JSON reports. They identify files in the evaluator's original layout.
The corpus [manifest](../response_standardized/manifest.csv) maps each key to an
actual repository-relative score path and its SHA-256 digest.

Only the 450 nested `result.file` values in `batch_results.json` were changed:
machine-specific absolute paths were replaced with the corresponding
`response_standardized/...` paths. No metadata, violation, count, score, or
other report content was changed. The original `batch_results.json` byte digest
before path normalization was:

```text
166da58ee936dfe5e4ded55b516eb6e70fa43f5295e00afec05619f5ffeb2bbc
```

Run `python3 scripts/verify_corpus.py` from the repository root to verify the
snapshot's hashes, coverage, metadata, balance, and saved counts.

## Evaluator repository

The evaluator is maintained separately at
[liampond/counterpoint-spellcheck](https://github.com/liampond/counterpoint-spellcheck).
The checkout inspected on 2026-10-16 was commit
[`6fcfeaeaf9e4a475916d403d131d6c8aa842c11c`](https://github.com/liampond/counterpoint-spellcheck/commit/6fcfeaeaf9e4a475916d403d131d6c8aa842c11c).
It contains the music21 checker, `validate_musicxml.py`, and `batch_analyze.py`;
the latter writes the same JSON/Markdown report structure used here.

That commit is an inspection reference, not a verified identifier of the code
that produced these historical results. The exports do not record the original
evaluator commit or dependency versions. Some assessment labels differ: for
example, this checkout maps `legit-dissonances` to U18, while the saved Markdown
report labels it U17. Use internal rule IDs when reconciling versions.

The upstream batch scripts assume a `test_data/response/` directory inside their
checkout. Its validation script scans `*.musicxml`, whereas 449 of these retained
files end in `.musicxml.xml`. A new evaluation must explicitly handle both
extensions and use the manifest's placement/species metadata. Write any new
results separately so the historical snapshot remains intact.

## Statistics and remaining provenance

The detailed results support recalculating counts, means, dispersion, and tiers
by model, species, placement, and cantus. The inspected evaluator checkout does
not contain the paper's aggregate-table or musical-duplicate analysis scripts.
The category mapping, duplicate criterion, and historical selection/repair
procedure still need to be reconciled when rebuilding the paper's analysis.

Two existing v4 issues should be resolved during that revision: rule-level
passages mix all-run and deduplicated denominators, and M2's displayed percentages
match bins of 0 / 1–2 / 3–5 / 6+, while its prose defines 0 / 1–3 / 4–5 / 6+.
The manuscript and exported reports are preserved here, without silently
changing their findings.
