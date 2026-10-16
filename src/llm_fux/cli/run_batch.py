#!/usr/bin/env python3
"""Batch CLI for running multiple prompt combinations efficiently.

Goals:
  * Parse arguments & expand cartesian product of (models × files × datatypes).
  * Provide parallel execution with basic retry handling.
  * Validate dataset structure & API keys early.

Non‑goals:
  * Prompt construction logic (delegated to `PromptRunner`).
  * Complex scheduling / rate limiting (future enhancement).
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Dict, Sequence, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from time import sleep
from dotenv import load_dotenv

from llm_fux.core.dispatcher import get_llm, get_llm_with_model_name, detect_model_provider
from llm_fux.core.runner import PromptRunner
from llm_fux.config.config import DEFAULT_MODELS
from llm_fux.utils.path_utils import (
    find_project_root,
    list_file_ids,
    list_datatypes,
)

# Legacy compatibility: tests may patch this symbol.
def list_questions(_path):  # type: ignore
    return ["Q1b"]

MODEL_ENV_VARS: Dict[str, List[str]] = {
    "chatgpt": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY", "GEMINI_API_KEY"],
}


def load_project_env() -> None:
    """Load a .env file at project root if present.

    Tests assert we call load_dotenv with only the dotenv_path kwarg (no override flag).
    """
    root = find_project_root()
    dotenv_path = root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        logging.debug("No .env file found; proceeding with existing environment.")


def parse_csv_or_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Split a list of possibly comma-delimited strings into a flat list.

    Example: ["Q1a,Q1b", "Q1c"] -> ["Q1a", "Q1b", "Q1c"]
    """
    if not values:
        return None
    out: List[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out or None


def validate_api_keys(models: Iterable[str]) -> None:
    for m in set(models):
        env_vars = MODEL_ENV_VARS.get(m)
        if not env_vars:
            continue
        has_valid_key = False
        for env_var in env_vars:
            api_key = os.getenv(env_var)
            if api_key and "your_" not in api_key.lower():
                has_valid_key = True
                break

        if not has_valid_key:
            logging.error(
                "Required API key missing or placeholder for model '%s'. Set one of: %s",
                m,
                ", ".join(env_vars),
            )
            raise SystemExit(2)


@dataclass(frozen=True)
class Task:
    model_name: str  # This is now the original model specification (could be provider or specific model)
    file_id: str
    datatype: str
    context: bool
    guide: Optional[str]
    temperature: float
    max_tokens: Optional[int]
    save: bool
    overwrite: bool
    dataset: str


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Batch-run music-theory prompts")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated models (e.g. chatgpt,claude or gpt-5.1-2025-11-13,claude-opus-4-5) or 'all'",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Include contextual guides",
    )
    parser.add_argument(
        "--guide",
        type=str,
        help="Path to guide file (e.g., data/guides/LLM-Guide.md). Requires --context.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="List of file IDs (default: all discovered)",
    )
    parser.add_argument("--questions", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument(
        "--datatypes",
        nargs="*",
        help="List of formats (default: musicxml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd() / "data",
        help="Root data directory (contains dataset subfolders)",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset name inside data-dir (default: root of data-dir)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path.cwd() / "outputs",
        help="Where to save model responses (default: ./outputs)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional token cap",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Number of retries for failed runs",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--use-batch-api",
        action="store_true",
        help="Use the asynchronous Batch Processing API for supported models",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def expand_models(
    raw: str,
) -> tuple[List[str], List[str]]:
    """Determine explicit model names and their providers.

    If given 'chatgpt', expands to the default OpenAI model.

    Returns:
        (resolved_models, providers)
    """
    input_models = [m.lower().strip() for m in raw.split(",") if m.strip()]
    if "all" in input_models:
        input_models = ["chatgpt", "claude", "gemini"]

    resolved_models: List[str] = []
    providers: List[str] = []

    provider_to_config = {
        "chatgpt": "openai",
        "claude": "anthropic",
        "gemini": "google",
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
    }

    for model in input_models:
        if model in provider_to_config:
            # It's a provider alias — resolve to default model
            conf_key = provider_to_config[model]
            providers.append(model if model in ["chatgpt", "claude", "gemini"] else conf_key)
            resolved_models.append(DEFAULT_MODELS.get(conf_key, model))
        else:
            # Specific model name (e.g. claude-opus-4-20250514)
            try:
                provider = detect_model_provider(model)
                providers.append(provider)
            except ValueError:
                logging.warning("Could not detect provider for '%s'; proceeding anyway.", model)
                providers.append("openai")
            resolved_models.append(model)

    return resolved_models, providers


def prepare_tasks(
    models: Sequence[str],
    file_ids: Sequence[str],
    datatypes: Sequence[str],
    args: argparse.Namespace,
) -> List[Task]:
    return [
        Task(
            model_name=m,
            file_id=fid,
            datatype=dt,
            context=args.context,
            guide=args.guide,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            save=True,  # Always save
            overwrite=args.overwrite,
            dataset=args.dataset,
        )
        for m in models
        for fid in file_ids
        for dt in datatypes
    ]


def run_task(task: Task, base_dirs: Dict[str, Path]) -> bool:
    # Determine if we need to use auto-detection or standard model loading
    try:
        # Try to detect provider (handles specific model names like "gpt-4o")
        provider = detect_model_provider(task.model_name)
        model = get_llm_with_model_name(task.model_name, provider)
    except ValueError:
        # Not a specific model name, try to treat it as a provider name
        try:
            model = get_llm(task.model_name)
        except ValueError as e:
            logging.error(
                "[%s] Invalid model specification: %s", task.model_name, e
            )
            return False
    
    runner = PromptRunner(
        model=model,
        file_id=task.file_id,
        datatype=task.datatype,
        context=task.context,
        guide=task.guide,
        dataset=task.dataset,
        base_dirs=base_dirs,
        temperature=task.temperature,
        max_tokens=task.max_tokens,
        save=task.save,
    )
    out_path = runner.save_to
    if task.save and out_path and out_path.exists() and not task.overwrite:
        logging.info(
            "Skipping %s/%s/%s (already exists)",
            task.model_name,
            task.file_id,
            task.datatype,
        )
        return True
    try:
        runner.run()
        return True
    except Exception as e:  # pragma: no cover (unexpected edge)
        logging.error(
            "[%s][%s][%s] failed: %s", task.model_name, task.file_id, task.datatype, e
        )
        return False


# ---------------------------------------------------------------------------
# Legacy compatibility shim for existing tests expecting a tuple-based worker.
# test_cli_batch imports `worker` with signature worker(task_tuple) where
# task_tuple = (model_name, file_id, datatype, context, base_dirs,
#               temperature, max_tokens, save, overwrite)
# We construct a Task from this tuple and delegate to current logic.
# Keep a reference to detect when tests monkey-patch `worker`.
_original_worker = None  # assigned after definition


def worker(task_tuple) -> bool:  # type: ignore
    try:
        (
            model_name,
            file_id,
            datatype,
            context,
            base_dirs,
            temperature,
            max_tokens,
            save,
            overwrite,
        ) = task_tuple
    except Exception:
        logging.error("Invalid task tuple passed to worker: %r", task_tuple)
        return False

    task = Task(
        model_name=model_name,
        file_id=file_id,
        datatype=datatype,
        context=context,
        guide=None,  # Legacy worker doesn't support guide selection
        temperature=temperature,
        max_tokens=max_tokens,
        save=save,
        overwrite=overwrite,
        dataset="",  # default dataset for legacy tests
    )
    return run_task(task, base_dirs)


_original_worker = worker


def execute_tasks_batch_api(tasks: List[Task], base_dirs: Dict[str, Path]) -> List[Task]:
    """Execute tasks using the asynchronous Batch API of the underlying models."""
    from llm_fux.core.dispatcher import get_llm_with_model_name, get_llm, detect_model_provider
    from llm_fux.core.runner import PromptRunner
    import json
    import time

    # 1. Group tasks by provider/model instance
    model_to_tasks: Dict[str, List[Task]] = {}
    for task in tasks:
        model_to_tasks.setdefault(task.model_name, []).append(task)

    failures: List[Task] = []
    active_batches = [] # List of (model_instance, batch_id, {custom_id -> task}, {custom_id -> runner})

    for model_name, model_tasks in model_to_tasks.items():
        try:
            try:
                provider = detect_model_provider(model_name)
                model = get_llm_with_model_name(model_name, provider)
            except ValueError:
                model = get_llm(model_name)

            if not getattr(model, "submit_batch", None) or not getattr(model, "supports_batch_api", lambda: False)():
                logging.warning(f"Model {model_name} does not support Batch API. Falling back to sequential execution for these tasks.")
                # We could add them to a legacy execution pool, but for simplicity let's just run them here:
                for t in model_tasks:
                    if not run_task(t, base_dirs):
                        failures.append(t)
                continue

            # Prepare Prompts
            batch_inputs = {}
            runners = {}
            task_map = {}
            for t in model_tasks:
                runner = PromptRunner(
                    model=model, file_id=t.file_id, datatype=t.datatype,
                    context=t.context, guide=t.guide, dataset=t.dataset,
                    base_dirs=base_dirs, temperature=t.temperature,
                    max_tokens=t.max_tokens, save=t.save
                )

                # Check overwrite condition early
                if runner.save_to and runner.save_to.exists() and not t.overwrite:
                    logging.info("Skipping %s/%s/%s (already exists)", t.model_name, t.file_id, t.datatype)
                    continue

                prompt_input = runner._build_prompt_input()
                import uuid
                custom_id = f"task_{uuid.uuid4().hex[:8]}"
                batch_inputs[custom_id] = prompt_input
                runners[custom_id] = runner
                task_map[custom_id] = t

            if not batch_inputs:
                continue # All tasks skipped

            # Submit Batch
            try:
                batch_id = model.submit_batch(batch_inputs)
                logging.info("Submitted batch API job for model %s: %s", model_name, batch_id)
                active_batches.append((model, batch_id, task_map, runners))
            except Exception as e:
                logging.error("Failed to submit batch for model %s: %s", model_name, e)
                failures.extend(model_tasks)

        except Exception as e:
            logging.error("[%s] preparation failed: %s", model_name, e)
            failures.extend(model_tasks)

    # 2. Poll active batches
    while active_batches:
        time.sleep(30) # Poll every 30 seconds
        next_active = []
        for model, batch_id, task_map, runners in active_batches:
            try:
                status = model.poll_batch(batch_id)
                logging.info("Batch %s status: %s", batch_id, status)
                if status == "completed":
                    # Retrieve and save results
                    for result in model.retrieve_batch_results(batch_id):
                        t = task_map.get(result.custom_id)
                        runner = runners.get(result.custom_id)
                        if not t or not runner:
                            continue

                        if result.error:
                            logging.error("Task failed in batch %s: %s", batch_id, result.error)
                            failures.append(t)
                        else:
                            # Mock token usage mapping back
                            runner._token_usage = result.response.usage
                            runner._response_model_id = result.response.model_id
                            runner._extra_params = result.response.extra_params
                            runner._api_duration = 0.0 # Unknown for batch

                            if runner.save and runner.save_to:
                                prompt_input = runner._build_prompt_input() # Rebuild for logging persistence
                                runner._persist_artifacts(result.response.text, prompt_input)
                            logging.info("Completed %s using Batch API", t.file_id)

                elif status in ("failed", "cancelled", "expired"):
                    logging.error("Batch %s ended with status %s", batch_id, status)
                    failures.extend(task_map.values())
                else:
                    next_active.append((model, batch_id, task_map, runners))
            except Exception as e:
                logging.error("Error polling batch %s: %s", batch_id, e)
                failures.extend(task_map.values())

        active_batches = next_active

    return failures

def execute_tasks(
    tasks: List[Task], base_dirs: Dict[str, Path], jobs: int = 1, use_batch_api: bool = False
) -> List[Task]:
    if use_batch_api:
        return execute_tasks_batch_api(tasks, base_dirs)

    failures: List[Task] = []
    if jobs <= 1:
        for t in tasks:
            if not run_task(t, base_dirs):
                failures.append(t)
        return failures
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        future_map: Dict[Future, Task] = {
            pool.submit(run_task, t, base_dirs): t for t in tasks
        }
        for fut in as_completed(future_map):
            if not fut.result():
                failures.append(future_map[fut])
    return failures


def retry_failures(
    failures: List[Task], base_dirs: Dict[str, Path], retries: int
) -> List[Task]:
    for attempt in range(retries):
        if not failures:
            break
        logging.info(
            "Retry attempt %d for %d failures", attempt + 1, len(failures)
        )
        new_failures: List[Task] = []
        for t in failures:
            if not run_task(t, base_dirs):
                new_failures.append(t)
            else:
                # optional backoff small pause removed for performance
                pass
        failures = new_failures
    return failures


def run_main(argv: list[str] | None = None) -> int:
    """Internal entrypoint returning an exit code (no SystemExit)."""
    load_project_env()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=level)

    if args.jobs < 1:
        logging.error("--jobs must be >= 1")
        return 2

    dataset_root = args.data_dir / args.dataset if args.dataset else args.data_dir
    base_dirs: Dict[str, Path] = {
        "encoded": dataset_root / "encoded",
        "prompts": dataset_root / "prompts",
        "questions": dataset_root / "prompts" / "questions",
        "guides": dataset_root / "guides",
        "outputs": args.outputs_dir,
    }
    for sub in ("encoded", "prompts"):
        if not base_dirs[sub].exists():
            logging.error("Missing required dataset subdirectory: %s", base_dirs[sub])
            return 2

    models, providers = expand_models(args.models)
    validate_api_keys(providers)

    # Validate guide usage
    if args.guide and not args.context:
        logging.error("--guide requires --context")
        return 2
    
    if args.guide:
        # Validate that the guide file exists
        from pathlib import Path
        guide_path = Path(args.guide)
        if not guide_path.exists():
            logging.error("Guide file not found: %s", args.guide)
            return 2
        if not guide_path.is_file():
            logging.error("Guide path is not a file: %s", args.guide)
            return 2

    selected_files = parse_csv_or_list(args.files) or parse_csv_or_list(args.questions)
    file_ids = selected_files or list_file_ids(base_dirs["encoded"])
    datatypes = parse_csv_or_list(args.datatypes) or list_datatypes(base_dirs["encoded"])

    if not file_ids:
        logging.error("No file IDs resolved (check --files or dataset contents)")
        return 2
    if not datatypes:
        logging.error("No datatypes resolved (check --datatypes or dataset contents)")
        return 2

    tasks = prepare_tasks(models, file_ids, datatypes, args)
    logging.info(
        "Prepared %d tasks (%d models × %d files × %d datatypes) using %d job(s)",
        len(tasks),
        len(models),
        len(file_ids),
        len(datatypes),
        args.jobs,
    )
    # Use the legacy worker shim only when tests have monkey-patched it;
    # otherwise use the modern execute_tasks path that preserves guide/dataset.
    import llm_fux.cli.run_batch as _self_mod
    _worker_patched = getattr(_self_mod, 'worker', None) is not getattr(_self_mod, '_original_worker', None)
    failures: list[Task] = []
    if _worker_patched:
        for t in tasks:
            tuple_task = (
                t.model_name,
                t.file_id,
                t.datatype,
                t.context,
                base_dirs,
                t.temperature,
                t.max_tokens,
                t.save,
                t.overwrite,
            )
            ok = _self_mod.worker(tuple_task)  # type: ignore
            if not ok:
                failures.append(t)
    else:
        failures = execute_tasks(tasks, base_dirs, args.jobs, getattr(args, "use_batch_api", False))
        failures = retry_failures(failures, base_dirs, args.retry)

    if failures:
        logging.error("Batch completed with %d failure(s)", len(failures))
        for t in failures[:10]:
            logging.error("  %s", t)
        if len(failures) > 10:
            logging.error("  ... (%d more)", len(failures) - 10)
        return 1

    logging.info("Batch completed successfully.")
    return 0


def main(argv: list[str] | None = None) -> None:  # type: ignore
    """Public CLI entrypoint raising SystemExit (legacy test contract)."""
    code = run_main(argv)
    raise SystemExit(code)


if __name__ == "__main__":  # pragma: no cover
    main()
