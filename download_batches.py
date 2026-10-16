"""Download completed OpenAI, Anthropic, and Gemini batch results.

Handles batches whose custom_ids use the ``species__file_id`` format
(e.g. ``3rd-species__Above_F``). Results are saved into the standard
output directory structure.

Each batch contributes at most one run per condition. Runs are skipped if:
  - the batch_id is already recorded in an existing metadata file for that condition
  - the condition already has TARGET_RUNS successful responses

Errors and empty responses are recorded as ``_error_<batch_id[:8]>.json``
in the metadata folder so nothing is silently dropped.

Usage:
    python download_batches.py
"""

import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai

from src.llm_fux.utils.path_utils import _extract_species_and_version
from src.llm_fux.utils.text_utils import clean_code_blocks, fix_musicxml_durations, clean_response

load_dotenv()

OUTPUTS_DIR = Path("outputs")
TARGET_RUNS = 5

SPECIES_GUIDES = {
    "1st-species": "data/guides/1_first_species/Fux Rules- 1vs1 note - Schubert v1.7.txt",
    "2nd-species": "data/guides/2_second_species/Cantus 2notes vs 1 - Schubert version 2.0.txt",
    "3rd-species": "data/guides/3_third_species/Fux Rules- 3rd species - 4vs1 - Schubert v2.0.txt",
    "4th-species": "data/guides/4_fourth_species/Fux Rules- 4th species - Syncopatio - Schubert v1.7.txt",
    "5th-species": "data/guides/5_fifth_species/Fux Rules- 5th species - Florish - Schubert v2.4.txt",
}


def is_species_custom_id(custom_id: str) -> bool:
    return "__" in custom_id and custom_id.split("__")[0] in SPECIES_GUIDES


def parse_custom_id(custom_id: str):
    species, file_id = custom_id.split("__", 1)
    return species, file_id


# ── Path helpers ──────────────────────────────────────────────────────────────

def _output_folders(model_name: str, file_id: str, guide: str) -> tuple[Path, Path]:
    """Return (response_folder, metadata_folder) for a condition."""
    species, version = _extract_species_and_version(guide)
    tail = Path(model_name) / species / version / file_id
    return OUTPUTS_DIR / "response" / tail, OUTPUTS_DIR / "metadata" / tail


def _next_run_number(folder: Path, ext: str) -> int:
    """Next available run_N index in *folder* for files with *ext*."""
    if not folder.exists():
        return 1
    existing = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix == ext:
            m = re.match(r"run_(\d+)", f.stem)
            if m:
                existing.append(int(m.group(1)))
    return max(existing) + 1 if existing else 1


def _success_run_count(response_folder: Path) -> int:
    """Count existing successful run_*.musicxml files."""
    if not response_folder.exists():
        return 0
    return sum(1 for f in response_folder.iterdir()
               if f.is_file() and f.suffix == ".musicxml" and re.match(r"run_\d+", f.stem))


def _saved_batch_ids(metadata_folder: Path) -> set[str]:
    """Return all batch_ids already recorded in run_*.json and _error_*.json files."""
    ids: set[str] = set()
    if not metadata_folder.exists():
        return ids
    for f in metadata_folder.iterdir():
        if not f.is_file() or f.suffix != ".json":
            continue
        try:
            bid = json.loads(f.read_text(encoding="utf-8")).get("batch_id")
            if bid:
                ids.add(bid)
        except Exception:
            pass
    return ids


def _save_error_record(metadata_folder: Path, batch_id: str, metadata: dict) -> None:
    """Save an error/empty record without occupying a run_N slot."""
    metadata_folder.mkdir(parents=True, exist_ok=True)
    (metadata_folder / f"_error_{batch_id[:8]}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── OpenAI batches ────────────────────────────────────────────────────────────

def download_openai_batches():
    client = OpenAI()
    batches = list(client.batches.list(limit=100).data)
    print(f"\n=== OpenAI: {len(batches)} batches found ===")

    total_saved = total_empty = total_error = total_skipped = total_full = 0

    for batch in batches:
        if batch.status != "completed":
            print(f"  Skipping {batch.id} (status={batch.status})")
            continue

        try:
            input_content = client.files.content(batch.input_file_id).text
        except Exception as e:
            print(f"  Could not read input for {batch.id}: {e}")
            continue

        first_custom_id = json.loads(input_content.strip().splitlines()[0]).get("custom_id", "")
        if not is_species_custom_id(first_custom_id):
            print(f"  Skipping {batch.id} (old task_* format)")
            continue

        if not batch.output_file_id:
            print(f"  No output file for {batch.id}")
            continue

        try:
            output_content = client.files.content(batch.output_file_id).text
        except Exception as e:
            print(f"  Could not read output for {batch.id}: {e}")
            continue

        batch_saved = batch_empty = batch_error = batch_skipped = batch_full = 0

        for line in output_content.strip().splitlines():
            if not line:
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            if not is_species_custom_id(custom_id):
                continue

            species, file_id = parse_custom_id(custom_id)
            guide = SPECIES_GUIDES[species]
            response_folder, metadata_folder = _output_folders("ChatGPT", file_id, guide)

            # Skip if this batch already contributed to this condition
            if batch.id in _saved_batch_ids(metadata_folder):
                batch_skipped += 1
                continue

            response = data.get("response", {})
            status_code = response.get("status_code")
            body = response.get("body", {})
            usage = body.get("usage") or {}
            model_id = body.get("model", "unknown")
            choices = body.get("choices", [])
            text = choices[0].get("message", {}).get("content") if choices else None
            finish = choices[0].get("finish_reason") if choices else None

            base_metadata = {
                "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "file_id": file_id,
                "species": species,
                "datatype": "musicxml",
                "context": True,
                "guide_path": guide,
                "model": "ChatGPTModel",
                "model_id": model_id,
                "temperature": 0.0,
                "batch_id": batch.id,
                "custom_id": custom_id,
                "extra_params": {"batch_api": True},
                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

            if status_code == 200 and text:
                # ── Success ──────────────────────────────────────────────────
                run_count = _success_run_count(response_folder)
                if run_count >= TARGET_RUNS:
                    batch_full += 1
                    continue

                cleaned = clean_code_blocks(text)
                cleaned = clean_response(cleaned, "musicxml")
                cleaned = fix_musicxml_durations(cleaned)

                run_n = _next_run_number(response_folder, ".musicxml")
                response_folder.mkdir(parents=True, exist_ok=True)
                metadata_folder.mkdir(parents=True, exist_ok=True)

                response_path = response_folder / f"run_{run_n}.musicxml"
                metadata_path = metadata_folder / f"run_{run_n}.json"

                response_path.write_text(cleaned, encoding="utf-8")
                metadata_path.write_text(
                    json.dumps({**base_metadata, "status": "success", "response_path": str(response_path)},
                               indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                batch_saved += 1

            else:
                # ── Error or empty ───────────────────────────────────────────
                if status_code != 200:
                    reason = f"HTTP {status_code}"
                    batch_error += 1
                else:
                    reason = f"empty content (finish_reason={finish})"
                    batch_empty += 1
                print(f"    {batch.id[:12]} {custom_id}: {reason}")
                _save_error_record(metadata_folder, batch.id,
                                   {**base_metadata, "status": "error", "reason": reason})

        print(f"  {batch.id}: saved={batch_saved}, empty={batch_empty}, "
              f"error={batch_error}, skipped={batch_skipped}, at_target={batch_full}")
        total_saved += batch_saved
        total_empty += batch_empty
        total_error += batch_error
        total_skipped += batch_skipped
        total_full += batch_full

    print(f"\nOpenAI total: saved={total_saved}, empty={total_empty}, "
          f"error={total_error}, skipped={total_skipped}, at_target={total_full}")


# ── Anthropic batches ─────────────────────────────────────────────────────────

def download_anthropic_batches():
    client = Anthropic()
    batches = list(client.messages.batches.list(limit=100))
    print(f"\n=== Anthropic: {len(batches)} batches found ===")

    total_saved = total_empty = total_error = total_skipped = total_full = 0

    for batch in batches:
        if batch.processing_status != "ended":
            print(f"  Skipping {batch.id} (status={batch.processing_status})")
            continue

        try:
            results = list(client.messages.batches.results(batch.id))
        except Exception as e:
            print(f"  Could not read results for {batch.id}: {e}")
            continue

        if not results:
            print(f"  No results for {batch.id}")
            continue

        if not is_species_custom_id(results[0].custom_id):
            print(f"  Skipping {batch.id} (old task_* format, first id={results[0].custom_id})")
            continue

        batch_saved = batch_empty = batch_error = batch_skipped = batch_full = 0

        for result in results:
            custom_id = result.custom_id
            if not is_species_custom_id(custom_id):
                continue

            species, file_id = parse_custom_id(custom_id)
            guide = SPECIES_GUIDES[species]
            response_folder, metadata_folder = _output_folders("Claude", file_id, guide)

            # Skip if this batch already contributed to this condition
            if batch.id in _saved_batch_ids(metadata_folder):
                batch_skipped += 1
                continue

            result_type = result.result.type

            if result_type == "succeeded":
                message = result.result.message
                usage = message.usage
                model_id = message.model
                text_blocks = [b for b in message.content if b.type == "text"] if message.content else []
                raw_text = text_blocks[0].text.strip() if text_blocks else ""

                base_metadata = {
                    "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "file_id": file_id,
                    "species": species,
                    "datatype": "musicxml",
                    "context": True,
                    "guide_path": guide,
                    "model": "ClaudeModel",
                    "model_id": model_id,
                    "temperature": 0.0,
                    "batch_id": batch.id,
                    "custom_id": custom_id,
                    "extra_params": {"batch_api": True, "discount": "50%"},
                    "token_usage": {
                        "prompt_tokens": usage.input_tokens,
                        "completion_tokens": usage.output_tokens,
                        "total_tokens": usage.input_tokens + usage.output_tokens,
                    },
                }

                if raw_text:
                    # ── Success ──────────────────────────────────────────────
                    run_count = _success_run_count(response_folder)
                    if run_count >= TARGET_RUNS:
                        batch_full += 1
                        continue

                    cleaned = clean_code_blocks(raw_text)
                    cleaned = clean_response(cleaned, "musicxml")
                    cleaned = fix_musicxml_durations(cleaned)

                    run_n = _next_run_number(response_folder, ".musicxml")
                    response_folder.mkdir(parents=True, exist_ok=True)
                    metadata_folder.mkdir(parents=True, exist_ok=True)

                    response_path = response_folder / f"run_{run_n}.musicxml"
                    metadata_path = metadata_folder / f"run_{run_n}.json"

                    response_path.write_text(cleaned, encoding="utf-8")
                    metadata_path.write_text(
                        json.dumps({**base_metadata, "status": "success", "response_path": str(response_path)},
                                   indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    batch_saved += 1

                else:
                    reason = "succeeded but empty content"
                    print(f"    {batch.id[:12]} {custom_id}: {reason}")
                    _save_error_record(metadata_folder, batch.id,
                                       {**base_metadata, "status": "error", "reason": reason})
                    batch_empty += 1

            else:
                # errored, canceled, expired, or unknown
                error_info = str(getattr(result.result, "error", result_type))
                base_metadata = {
                    "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "file_id": file_id,
                    "species": species,
                    "datatype": "musicxml",
                    "context": True,
                    "guide_path": guide,
                    "model": "ClaudeModel",
                    "model_id": "unknown",
                    "temperature": 0.0,
                    "batch_id": batch.id,
                    "custom_id": custom_id,
                    "extra_params": {"batch_api": True, "discount": "50%"},
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                print(f"    {batch.id[:12]} {custom_id}: {result_type} — {error_info}")
                _save_error_record(metadata_folder, batch.id,
                                   {**base_metadata, "status": "error", "reason": error_info})
                batch_error += 1

        print(f"  {batch.id}: saved={batch_saved}, empty={batch_empty}, "
              f"error={batch_error}, skipped={batch_skipped}, at_target={batch_full}")
        total_saved += batch_saved
        total_empty += batch_empty
        total_error += batch_error
        total_skipped += batch_skipped
        total_full += batch_full

    print(f"\nAnthropic total: saved={total_saved}, empty={total_empty}, "
          f"error={total_error}, skipped={total_skipped}, at_target={total_full}")


# ── Gemini batches ────────────────────────────────────────────────────────────

def download_gemini_batches():
    import os
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))
    batches = list(client.batches.list())
    print(f"\n=== Gemini: {len(batches)} batches found ===")

    total_saved = total_empty = total_error = total_skipped = total_full = 0

    for batch in batches:
        state = batch.state.name
        if state != "JOB_STATE_SUCCEEDED":
            print(f"  Skipping {batch.name} (state={state})")
            continue

        dest_file = getattr(batch.dest, "file_name", None) if batch.dest else None
        if not dest_file:
            print(f"  No output file for {batch.name}")
            continue

        try:
            raw = client.files.download(file=dest_file)
            output_content = raw.decode("utf-8")
        except Exception as e:
            print(f"  Could not read output for {batch.name}: {e}")
            continue

        first_key = None
        for line in output_content.strip().splitlines():
            if line:
                first_key = json.loads(line).get("key", "")
                break
        if not first_key or not is_species_custom_id(first_key):
            print(f"  Skipping {batch.name} (old task_* format, first key={first_key})")
            continue

        batch_saved = batch_empty = batch_error = batch_skipped = batch_full = 0

        for line in output_content.strip().splitlines():
            if not line:
                continue
            data = json.loads(line)
            custom_id = data.get("key", "")
            if not is_species_custom_id(custom_id):
                continue

            species, file_id = parse_custom_id(custom_id)
            guide = SPECIES_GUIDES[species]
            response_folder, metadata_folder = _output_folders("Gemini", file_id, guide)

            if batch.name in _saved_batch_ids(metadata_folder):
                batch_skipped += 1
                continue

            response_data = data.get("response")
            model_id = "unknown"
            raw_text = ""
            error_reason = None

            if response_data:
                try:
                    candidates = response_data.get("candidates", [])
                    parts = candidates[0]["content"]["parts"] if candidates else []
                    raw_text = "".join(p.get("text", "") for p in parts).strip()
                    model_id = response_data.get("modelVersion", "unknown")
                    usage_meta = response_data.get("usageMetadata", {})
                except Exception as e:
                    error_reason = f"parse error: {e}"
            else:
                error_reason = str(data.get("error", "no response"))

            base_metadata = {
                "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "file_id": file_id,
                "species": species,
                "datatype": "musicxml",
                "context": True,
                "guide_path": guide,
                "model": "GeminiModel",
                "model_id": model_id,
                "temperature": 1.0,
                "batch_id": batch.name,
                "custom_id": custom_id,
                "extra_params": {"batch_api": True},
                "token_usage": {
                    "prompt_tokens": usage_meta.get("promptTokenCount", 0) if not error_reason else 0,
                    "completion_tokens": usage_meta.get("candidatesTokenCount", 0) if not error_reason else 0,
                    "total_tokens": usage_meta.get("totalTokenCount", 0) if not error_reason else 0,
                },
            }

            if error_reason or not raw_text:
                reason = error_reason or "empty content"
                print(f"    {'ERROR' if error_reason else 'EMPTY'} {custom_id}: {reason}")
                _save_error_record(metadata_folder, batch.name,
                                   {**base_metadata, "status": "error", "reason": reason})
                batch_error += 1 if error_reason else 0
                batch_empty += 0 if error_reason else 1
                continue

            run_count = _success_run_count(response_folder)
            if run_count >= TARGET_RUNS:
                batch_full += 1
                continue

            cleaned = clean_code_blocks(raw_text)
            cleaned = clean_response(cleaned, "musicxml")
            cleaned = fix_musicxml_durations(cleaned)

            run_n = _next_run_number(response_folder, ".musicxml")
            response_folder.mkdir(parents=True, exist_ok=True)
            metadata_folder.mkdir(parents=True, exist_ok=True)

            response_path = response_folder / f"run_{run_n}.musicxml"
            metadata_path = metadata_folder / f"run_{run_n}.json"

            response_path.write_text(cleaned, encoding="utf-8")
            metadata_path.write_text(
                json.dumps({**base_metadata, "status": "success", "response_path": str(response_path)},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            batch_saved += 1

        print(f"  {batch.name}: saved={batch_saved}, empty={batch_empty}, "
              f"error={batch_error}, skipped={batch_skipped}, at_target={batch_full}")
        total_saved += batch_saved
        total_empty += batch_empty
        total_error += batch_error
        total_skipped += batch_skipped
        total_full += batch_full

    print(f"\nGemini total: saved={total_saved}, empty={total_empty}, "
          f"error={total_error}, skipped={total_skipped}, at_target={total_full}")


if __name__ == "__main__":
    download_openai_batches()
    download_anthropic_batches()
    download_gemini_batches()
