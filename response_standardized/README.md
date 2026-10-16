# Retained 450-score evaluation corpus

This directory preserves the MusicXML scores associated with the saved evaluation
results and the v4 manuscript. The snapshot was recorded on 2026-10-16 without
changing the score files.

The design contains three retained runs for each combination of:

- Three model labels: ChatGPT, Claude, and Gemini.
- Five species: first through fifth.
- Two counterpoint placements: above and below the cantus firmus.
- Five cantus firmi: A, C, E, F, and G.

There are 150 model/configuration cells, 450 scores, and 150 scores per model.
Original run numbers and extensions are preserved, including `.musicxml.xml`.
Some cells contain run numbers other than 1, 2, and 3; the retained filenames
alone do not establish a chronological selection rule.

## Manifest and verification

[`manifest.csv`](manifest.csv) records each score's repository-relative path,
SHA-256 digest, byte size, original result key, model label, species, guide
version from its directory name, placement, cantus, original run number, and
saved violation count. Hashes identify exact file bytes, not musical identity.
Musically identical runs remain separate observations in this snapshot.

From the repository root, run:

```bash
python3 scripts/verify_corpus.py
```

The verifier checks the files, manifest, validation report, and detailed checker
results against one another. It also checks the balanced design and prints the
saved per-model totals and means. It does not rerun the counterpoint checker or
verify musical-duplicate classifications.

See the [analysis provenance](<../Résultat d'analyse/README.md>) for the evaluator
repository, report formats, and limits of the historical record. These are
retained evaluation scores; they are not an archive of every API attempt or a
guarantee of unmodified, pre-repair model responses. New generation output belongs
in the ignored `outputs/` directory, not in this fixed corpus.
