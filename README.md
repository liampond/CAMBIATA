# LLM-Fux

Framework for analyzing Fux counterpoint using Large Language Models (LLMs).

## Overview

This project provides tools for evaluating how well LLMs understand and apply the rules of species counterpoint as described in Johann Joseph Fux's *Gradus ad Parnassum*. The framework uses MusicXML encoding and supports multiple LLM providers (ChatGPT, Claude, Gemini).

## Quick Start

```bash
# 1. Install
poetry install --extras all

# 2. Set up API keys in .env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# 3. Edit config.yaml to set your test parameters

# 4. Run
poetry run run
```

That's it! Results appear in `outputs/`.

## Features

- **MusicXML encoding**: Standardized music notation format
- **Multiple LLM providers**: OpenAI (ChatGPT), Anthropic (Claude), Google (Gemini)
- **Context comparison**: Test with/without counterpoint guides
- **Simple workflow**: Edit `config.yaml`, run `poetry run run`

## Configuration

Edit `config.yaml` to configure your tests:

```yaml
# Single test
single_run:
  enabled: true
  file: Fux_CantusFirmus_C
  model: claude           # chatgpt, claude, or gemini
  datatype: musicxml      # musicxml
  guide_path: data/guides/Pierre-Guide.md  # or null for no guide

# Or batch test (set single_run.enabled: false)
batch_run:
  enabled: false
  models: [chatgpt, claude, gemini]
  datatypes: [musicxml]
  files: [Fux_CantusFirmus_C]
  contexts: [with, without]
```

Then run: `poetry run run`

## Project Structure

```
llm-fux/
├── config.yaml          # Test configuration
├── data/
│   ├── encoded/         # MusicXML music files
│   ├── guides/          # Context guides for LLMs
│   └── prompts/         # Prompt templates
├── outputs/             # Results organized by model/context/format
├── src/llm_fux/         # Main package code
└── tests/               # Test suite
```

## Documentation

See `docs/LLM-AGENT-GUIDE.md` for detailed usage instructions.

## Research snapshot

The retained 450-score evaluation corpus is versioned in
[`response_standardized/`](response_standardized/README.md), with a manifest of
file hashes and links to the saved checker results. The
[analysis provenance](<Résultat d'analyse/README.md>) documents the exports and
the separate [counterpoint-spellcheck](https://github.com/liampond/counterpoint-spellcheck)
evaluator. Verify the snapshot without API calls or external dependencies:

```bash
python3 scripts/verify_corpus.py
```

The current manuscript, `paper/journal/Paper on Fux v4 MTO.docx`, and its source
appendices are tracked. Historical drafts, duplicate PDF exports, temporary
test outputs, and live generation outputs remain ignored.

## License

See LICENSE file for details.
