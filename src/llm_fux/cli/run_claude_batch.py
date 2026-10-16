#!/usr/bin/env python3
"""Submit Claude API calls via the Message Batches API (50% cost discount).

Builds all prompts locally, submits them as a single Anthropic batch,
polls for completion, and saves results using the standard output structure.

Can be driven from config.yaml (claude_batch section) or CLI arguments.

Usage:
    poetry run run-claude-batch                       # config.yaml driven
    poetry run run-claude-batch --guides all          # all 5 species
    poetry run run-claude-batch --guide data/guides/1_first_species/1vs1_v1.7.txt
    poetry run run-claude-batch --files Above_C,Above_D --guide ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from llm_fux.core.runner import PromptRunner
from llm_fux.core.dispatcher import get_llm
from llm_fux.config.config import DEFAULT_MODELS, get_config, get_max_tokens
from llm_fux.utils.path_utils import (
    find_project_root,
    get_output_path,
    list_file_ids,
)
from llm_fux.utils.text_utils import clean_code_blocks, fix_musicxml_durations, clean_response

logger = logging.getLogger(__name__)

# ── Default file list (all cantus firmi) ──────────────────────────────────
ALL_FILES = [
    "Above_A", "Above_C", "Above_D", "Above_E", "Above_F", "Above_G",
    "Below_A", "Below_C", "Below_D", "Below_E", "Below_F", "Below_G",
]

# ── Guide paths for each species ─────────────────────────────────────────
ALL_GUIDES = {
    "1st-species": "data/guides/1_first_species/Fux Rules- 1vs1 note - v1.7.txt",
    "2nd-species": "data/guides/2_second_species/Cantus 2notes vs 1 - version 1.9.txt",
    "3rd-species": "data/guides/3_third_species/Fux Rules- 3rd species - 4vs1 - v1.9.txt",
    "4th-species": "data/guides/4_fourth_species/Fux Rules- 4th species - Syncopatio - v1.6.txt",
    "5th-species": "data/guides/5_fifth_species/Fux Rules- 5th species - Florish - v2.3.txt",
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit Claude batch API requests (50% discount)"
    )
    parser.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated file IDs (default: all cantus firmi)",
    )
    guide_group = parser.add_mutually_exclusive_group()
    guide_group.add_argument(
        "--guides",
        type=str,
        default=None,
        help="Comma-separated species keys (e.g. 1st-species,3rd-species) or 'all'",
    )
    guide_group.add_argument(
        "--guide",
        type=str,
        default=None,
        help="Single guide file path (e.g. data/guides/1_first_species/1vs1_v1.7.txt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Claude model to use (default: {DEFAULT_MODELS['anthropic']})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max response tokens (default: from config or 200000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status polls (default: 30)",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Resume polling/downloading a previously submitted batch by ID",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args with config.yaml claude_batch section.

    Priority: CLI args > config.yaml claude_batch > defaults
    """
    cfg = get_config().get("claude_batch", {})

    model = args.model or cfg.get("model", DEFAULT_MODELS["anthropic"])
    max_tokens = args.max_tokens or cfg.get("max_tokens", get_max_tokens())
    temperature = args.temperature if args.temperature is not None else cfg.get("temperature", 0.0)
    poll_interval = args.poll_interval if args.poll_interval != 30 else cfg.get("poll_interval", 30)

    # Resolve files
    if args.files:
        files = [f.strip() for f in args.files.split(",") if f.strip()]
    elif cfg.get("files"):
        files = cfg["files"]
    else:
        files = list(ALL_FILES)

    # Resolve guides
    guides: Dict[str, str] = {}
    if args.guide:
        # Single guide path from CLI
        guide_path = Path(args.guide)
        if not guide_path.is_file():
            raise FileNotFoundError(f"Guide file not found: {args.guide}")
        # Derive species key from path
        species_key = _species_key_from_path(args.guide)
        guides[species_key] = args.guide
    elif args.guides:
        if args.guides.lower() == "all":
            guides = dict(ALL_GUIDES)
        else:
            for key in args.guides.split(","):
                key = key.strip()
                if key in ALL_GUIDES:
                    guides[key] = ALL_GUIDES[key]
                else:
                    raise ValueError(
                        f"Unknown species key '{key}'. Valid: {', '.join(ALL_GUIDES)}"
                    )
    elif cfg.get("guides"):
        # Config-driven guide list
        cfg_guides = cfg["guides"]
        if isinstance(cfg_guides, str) and cfg_guides.lower() == "all":
            guides = dict(ALL_GUIDES)
        elif isinstance(cfg_guides, list):
            for entry in cfg_guides:
                if isinstance(entry, str) and entry in ALL_GUIDES:
                    guides[entry] = ALL_GUIDES[entry]
                elif isinstance(entry, dict):
                    guides.update(entry)
                else:
                    guides[entry] = ALL_GUIDES.get(entry, entry)
        elif isinstance(cfg_guides, dict):
            guides = cfg_guides
    elif cfg.get("guide_path"):
        gp = cfg["guide_path"]
        guides[_species_key_from_path(gp)] = gp
    else:
        guides = dict(ALL_GUIDES)

    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "poll_interval": poll_interval,
        "files": files,
        "guides": guides,
    }


def _species_key_from_path(guide_path: str) -> str:
    """Derive a species key like '1st-species' from a guide file path."""
    from llm_fux.utils.path_utils import _extract_species_and_version
    species, version = _extract_species_and_version(guide_path)
    return species


def build_base_dirs() -> Dict[str, Path]:
    root = find_project_root()
    data_dir = root / "data"
    return {
        "encoded": data_dir / "encoded",
        "prompts": data_dir / "prompts",
        "questions": data_dir / "prompts" / "questions",
        "guides": data_dir / "guides",
        "outputs": root / "outputs",
    }


def build_prompts(
    files: List[str],
    guides: Dict[str, str],
    base_dirs: Dict[str, Path],
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Build all prompts without calling the model.

    Returns a list of dicts with keys: custom_id, species, file_id, guide,
    system_prompt, user_prompt, components.
    """
    dummy_model = get_llm("claude")
    tasks: List[Dict[str, Any]] = []

    for species, guide_path in guides.items():
        for file_id in files:
            custom_id = f"{species}__{file_id}"

            runner = PromptRunner(
                model=dummy_model,
                file_id=file_id,
                datatype="musicxml",
                context=True,
                guide=guide_path,
                dataset="",
                base_dirs=base_dirs,
                temperature=temperature,
                max_tokens=max_tokens,
                save=False,
            )

            prompt_input = runner._build_prompt_input()

            tasks.append({
                "custom_id": custom_id,
                "species": species,
                "file_id": file_id,
                "guide": guide_path,
                "system_prompt": prompt_input.system_prompt,
                "user_prompt": prompt_input.user_prompt,
                "components": runner._last_components,
            })

            logger.info("Built prompt for %s / %s", species, file_id)

    return tasks


def submit_batch(
    client: Anthropic,
    tasks: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Submit all tasks as a single Anthropic Message Batch. Returns batch ID."""
    requests = []
    for task in tasks:
        requests.append(
            Request(
                custom_id=task["custom_id"],
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=task["system_prompt"],
                    messages=[{"role": "user", "content": task["user_prompt"]}],
                ),
            )
        )

    logger.info(
        "Submitting batch with %d requests to model %s ...",
        len(requests), model,
    )
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch created: id=%s, status=%s", batch.id, batch.processing_status)
    return batch.id


def poll_batch(client: Anthropic, batch_id: str, poll_interval: int = 30) -> None:
    """Poll until the batch reaches a terminal state."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            "Batch %s — status: %s | succeeded: %d, errored: %d, "
            "expired: %d, canceled: %d, processing: %d",
            batch_id,
            batch.processing_status,
            counts.succeeded,
            counts.errored,
            counts.expired,
            counts.canceled,
            counts.processing,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)


def save_results(
    client: Anthropic,
    batch_id: str,
    tasks: List[Dict[str, Any]],
    base_dirs: Dict[str, Path],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, int]:
    """Stream results from the batch and save to the output directory."""
    task_map = {t["custom_id"]: t for t in tasks}
    outputs_dir = base_dirs["outputs"]

    stats = {"succeeded": 0, "errored": 0, "expired": 0, "canceled": 0}

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        task = task_map.get(custom_id)
        if not task:
            logger.warning("Unknown custom_id in results: %s", custom_id)
            continue

        if result.result.type == "succeeded":
            message = result.result.message
            raw_text = message.content[0].text.strip()
            cleaned = clean_code_blocks(raw_text)
            cleaned = clean_response(cleaned, "musicxml")
            cleaned = fix_musicxml_durations(cleaned)

            # Resolve output path first, then skip if identical content already exists.
            response_path = get_output_path(
                outputs_dir=outputs_dir,
                model_name="Claude",
                file_id=task["file_id"],
                datatype="musicxml",
                context=True,
                guide=task["guide"],
                ext=".musicxml",
                output_type="response",
                temperature=temperature,
            )

            duplicate = None
            for existing in sorted(response_path.parent.glob("run_*.musicxml")):
                if existing == response_path or not existing.is_file():
                    continue
                try:
                    if existing.read_text(encoding="utf-8") == cleaned:
                        duplicate = existing
                        break
                except OSError:
                    continue

            if duplicate:
                logger.info(
                    "Skipping duplicate save for %s / %s; identical to %s",
                    task["species"],
                    task["file_id"],
                    duplicate,
                )
                stats["succeeded"] += 1
                continue

            # Save response
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(cleaned, encoding="utf-8")

            # Save metadata
            metadata_path = get_output_path(
                outputs_dir=outputs_dir,
                model_name="Claude",
                file_id=task["file_id"],
                datatype="musicxml",
                context=True,
                guide=task["guide"],
                ext=".json",
                output_type="metadata",
                temperature=temperature,
            )

            usage = message.usage
            metadata = {
                "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "file_id": task["file_id"],
                "species": task["species"],
                "datatype": "musicxml",
                "context": True,
                "guide_path": task["guide"],
                "model": "ClaudeModel",
                "model_id": message.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "batch_id": batch_id,
                "custom_id": custom_id,
                "response_path": str(response_path),
                "extra_params": {"batch_api": True, "discount": "50%"},
                "token_usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                },
                "component_lengths": {
                    k: len(v) if isinstance(v, str) else sum(len(x) for x in v if x)
                    for k, v in task["components"].items()
                },
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Save readable prompt
            prompt_path = get_output_path(
                outputs_dir=outputs_dir,
                model_name="Claude",
                file_id=task["file_id"],
                datatype="musicxml",
                context=True,
                guide=task["guide"],
                ext=".txt",
                output_type="prompt",
                temperature=temperature,
            )
            prompt_lines = [
                "=" * 80,
                "SYSTEM PROMPT",
                "=" * 80,
                task["system_prompt"],
                "",
                "=" * 80,
                "USER PROMPT",
                "=" * 80,
                task["user_prompt"],
            ]
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text("\n".join(prompt_lines), encoding="utf-8")

            logger.info(
                "Saved %s / %s -> %s", task["species"], task["file_id"], response_path
            )
            stats["succeeded"] += 1

        elif result.result.type == "errored":
            logger.error(
                "ERRORED %s / %s: %s",
                task["species"], task["file_id"], result.result.error,
            )
            stats["errored"] += 1

        elif result.result.type == "expired":
            logger.warning("EXPIRED %s / %s", task["species"], task["file_id"])
            stats["expired"] += 1

        elif result.result.type == "canceled":
            logger.warning("CANCELED %s / %s", task["species"], task["file_id"])
            stats["canceled"] += 1

    return stats


def run_main(argv: list[str] | None = None) -> int:
    """Internal entrypoint returning an exit code."""
    load_dotenv()

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
    )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return 1

    client = Anthropic(api_key=api_key)
    base_dirs = build_base_dirs()

    # Resolve configuration (CLI > config.yaml > defaults)
    try:
        resolved = resolve_config(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Configuration error: %s", e)
        return 2

    model = resolved["model"]
    max_tokens = resolved["max_tokens"]
    temperature = resolved["temperature"]
    poll_interval = resolved["poll_interval"]
    files = resolved["files"]
    guides = resolved["guides"]

    # If resuming a previous batch, skip prompt building and submission
    if args.batch_id:
        batch_id = args.batch_id
        logger.info("Resuming batch %s — rebuilding prompts for result mapping...", batch_id)
        tasks = build_prompts(files, guides, base_dirs, temperature, max_tokens)
    else:
        # 1. Build all prompts
        total = len(guides) * len(files)
        logger.info(
            "Building prompts: %d guide(s) x %d file(s) = %d tasks",
            len(guides), len(files), total,
        )
        tasks = build_prompts(files, guides, base_dirs, temperature, max_tokens)
        logger.info("Built %d prompts", len(tasks))

        # 2. Submit batch
        batch_id = submit_batch(client, tasks, model, max_tokens, temperature)

    # 3. Poll for completion
    logger.info("Polling for batch completion (interval=%ds)...", poll_interval)
    poll_batch(client, batch_id, poll_interval)

    # 4. Save results
    logger.info("Batch complete. Retrieving and saving results...")
    stats = save_results(client, batch_id, tasks, base_dirs, model, temperature, max_tokens)

    logger.info(
        "Done! succeeded: %d, errored: %d, expired: %d, canceled: %d",
        stats["succeeded"], stats["errored"], stats["expired"], stats["canceled"],
    )
    return 0 if stats["errored"] == 0 and stats["expired"] == 0 else 1


def main(argv: list[str] | None = None) -> None:
    """Public CLI entrypoint."""
    sys.exit(run_main(argv))


if __name__ == "__main__":
    main()
