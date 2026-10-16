"""PromptRunner orchestrates prompt assembly and LLM querying.

Responsibilities:
    * Load the appropriate system/base format prompts, encoded data, guides, and question text.
    * Build a `PromptInput` via `PromptBuilder` with dataset‑specific ordering.
    * Invoke the provided model (`LLMInterface`) and optionally persist response + metadata bundle.

Non‑goals:
    * Network retries / rate limiting (leave to model wrapper).
    * Complex caching (could be layered later if needed).
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import zoneinfo

from llm_fux.models.base import LLMInterface, LLMResponse, PromptInput
from llm_fux.prompts.prompt_builder import PromptBuilder
from llm_fux.utils.path_utils import (
    load_text_file,
    find_encoded_file,
    get_output_path,
)
from llm_fux.utils.text_utils import fix_musicxml_durations, clean_response


class PromptRunner:
    """Build and execute a single prompt run.

    Instances are lightweight; create a new runner per prompt.
    
    Args:
        model: LLM interface to use for queries
        file_id: File identifier (e.g., Fux_CantusFirmus_C)
        datatype: Encoding format (musicxml)
        context: Whether to include guide context
        guide: Path to guide file when context=True
        dataset: Dataset subdirectory name
        base_dirs: Override paths for data directories
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum response tokens
        save: Whether to save outputs (always True)
    """

    _EXT_MAP = {"musicxml": ".musicxml"}

    def __init__(
        self,
        model: LLMInterface,
        file_id: Optional[str] = None,
        datatype: str = "musicxml",
        context: bool = False,
        guide: Optional[str] = None,
        dataset: str = "",
        base_dirs: Optional[Dict[str, Path]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        save: bool = True,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.model: LLMInterface = model
        self.file_id: str = file_id or ""
        if not self.file_id:
            raise ValueError("file_id is required")
        self.datatype: str = datatype.lower().strip()
        if self.datatype not in self._EXT_MAP:
            self.logger.warning("Unrecognized datatype '%s'; proceeding anyway", self.datatype)
        self.context: bool = bool(context)
        self.guide: Optional[str] = guide
        self.dataset: str = dataset
        self.base_dirs: Dict[str, Path] = base_dirs or {}
        self.temperature: float = float(temperature)
        if not (0.0 <= self.temperature <= 1.0):
            self.logger.warning("Temperature %.3f outside [0,1]; model may clamp internally", self.temperature)
        self.max_tokens: Optional[int] = max_tokens
        self.save: bool = bool(save)

        self.save_to: Optional[Path] = None
        if self.save:
            try:
                # Use the same extension as the input encoded file
                ext = f".{self.datatype}"
                # Get the base path for responses
                self.save_to = get_output_path(
                    outputs_dir=self.base_dirs.get("outputs", Path("outputs")),
                    model_name=self._get_clean_model_name(self.model),
                    file_id=self.file_id,
                    datatype=self.datatype,
                    context=self.context,
                    guide=self.guide,
                    dataset=self.dataset,
                    ext=ext,
                    output_type="response",
                    temperature=self.temperature,
                )
            except Exception as e:  # pragma: no cover (rare path issues)
                self.logger.error("Failed to compute output path: %s", e)
                self.save_to = None

    def _get_clean_model_name(self, model) -> str:
        """Get clean model name for folder structure."""
        name = type(model).__name__
        # Remove "Model" suffix and clean up names
        if name.endswith("Model"):
            name = name[:-5]  # Remove "Model"
        # Handle specific cases
        if name == "ChatGPT":
            return "ChatGPT"
        elif name == "Claude":
            return "Claude"
        elif name == "Gemini":
            return "Gemini"
        return name

    # Public API -----------------------------------------------------------------
    def run(self) -> str:
        """Execute the prompt build + model query pipeline.

        Returns
        -------
        str
            Raw response text from the underlying model.
        """
        prompt_input = self._build_prompt_input()
        self.logger.info(
            f"Running {self.file_id} [{self.datatype}] dataset={self.dataset} context={self.context} temp={self.temperature}"
        )
        
        # Capture timing for API call
        start_time = time.time()
        result = self.model.query(prompt_input)
        end_time = time.time()
        api_duration = end_time - start_time
        
        # Store timing info for later use in _persist_artifacts
        self._api_duration = api_duration

        # Unpack LLMResponse — support both new LLMResponse and legacy str returns
        if isinstance(result, LLMResponse):
            response = result.text
            self._token_usage = result.usage
            self._response_model_id = result.model_id
            self._extra_params = result.extra_params
        else:
            response = result
            self._token_usage = None
            self._response_model_id = None
            self._extra_params = {}

        # Log token usage if available
        if self._token_usage:
            self.logger.info(
                "Token usage for %s — prompt: %d, completion: %d, total: %d",
                self.file_id,
                self._token_usage.prompt_tokens,
                self._token_usage.completion_tokens,
                self._token_usage.total_tokens,
            )
        
        # Use f-string to match test expectation that the interpolated id appears directly
        self.logger.info(f"Received response for {self.file_id}")
        if self.save and self.save_to:
            self._persist_artifacts(response, prompt_input)
        return response

    # Internal helpers -----------------------------------------------------------
    def _build_prompt_input(self) -> PromptInput:
        system_prompt = self._load_system_prompt()
        format_prompt = self._load_base_format_prompt()
        encoded_data = self._load_encoded()
        question_text = self._load_question()
        guides = self._load_guides()
        example_text = self._load_example()
        # Stash raw components for later logging
        self._last_components = {
            "system_prompt": system_prompt,
            "format_prompt": format_prompt,
            "encoded_data": encoded_data,
            "guides": guides,
            "example": example_text,
            "question_prompt": question_text,
        }
        # Dataset-specific ordering logic. For the default fux counterpoint dataset
        # the user requested ordering: prompt.md (question), guides, base_<FORMAT>, encoded file.
        # Legacy datasets with explicit names retain previous ordering.
        ordering = None
        section_headers = None
        if self.dataset in ("", "fux-counterpoint"):
            ordering = [
                "question_prompt",  # prompt.md baseline instructions
                "guides",  # contextual guide(s)
                "example",  # Fux reference example
                "format_prompt",  # base_format instructions
                "encoded_data",  # the score / encoded file
            ]
            section_headers = {
                "question_prompt": "Task",
                "guides": "Guide",
                "example": "Reference Example (by Fux)",
                "format_prompt": f"Output Format ({self.datatype.upper()})",
                "encoded_data": f"Encoded {self.datatype.upper()} Source",
            }

        builder = PromptBuilder(
            system_prompt=system_prompt,
            format_specific_user_prompt=format_prompt,
            encoded_data=encoded_data,
            guides=guides,
            question_prompt=question_text,
            temperature=self.temperature,
            model_name=None,
            ordering=ordering,
            section_headers=section_headers,
            example=example_text,
        )
        prompt_input = builder.build()
        if self.max_tokens is not None:
            prompt_input.max_tokens = self.max_tokens
        return prompt_input

    def _load_system_prompt(self) -> str:
        """Load system prompt, preferring .md over .txt."""
        prompts_dir = self.base_dirs.get("prompts", Path(""))
        for ext in ("md", "txt"):
            candidate = prompts_dir / f"system_prompt.{ext}"
            if candidate.exists():
                return load_text_file(candidate)
        return ""

    def _load_base_format_prompt(self) -> str:
        base_dir = self.base_dirs.get("prompts", Path("")) / "base"
        for ext in ("md", "txt"):
            candidate = base_dir / f"base_{self.datatype}.{ext}"
            if candidate.exists():
                return load_text_file(candidate)
        raise FileNotFoundError(f"Base format prompt not found for {self.datatype} in {base_dir}")

    def _load_encoded(self) -> str:
        # Standard layout: encoded/<datatype>/<file_id>.<ext>
        encoded_dir = self.base_dirs.get("encoded", Path("encoded")) / self.datatype
        path = find_encoded_file(self.file_id, self.datatype, encoded_dir, required=True)
        if not path:
            raise FileNotFoundError(f"Encoded file not found: {self.file_id}.{self._EXT_MAP.get(self.datatype, '')} in {encoded_dir}")
        return load_text_file(path)

    # Map guide filename patterns to species names for prompt selection.
    _SPECIES_PATTERNS = [
        ("1vs1", "first_species"),
        ("1_first_species", "first_species"),
        ("2vs1", "second_species"),
        ("2_second_species", "second_species"),
        ("4vs1", "third_species"),
        ("3_third_species", "third_species"),
        ("4th species", "fourth_species"),
        ("4_fourth_species", "fourth_species"),
        ("5th species", "fifth_species"),
        ("5_fifth_species", "fifth_species"),
    ]

    def _detect_species(self) -> str:
        """Detect species from guide path. Returns species name or 'first_species' as default."""
        if self.guide:
            guide_str = str(self.guide).lower()
            for pattern, species in self._SPECIES_PATTERNS:
                if pattern.lower() in guide_str:
                    return species
        return "first_species"

    def _load_question(self) -> str:
        # Determine position (above/below) from file_id
        position = "below" if self.file_id.lower().startswith("below") else "above"

        # Detect species from guide filename/path
        species = self._detect_species()

        prompt_file = f"prompt_{position}_{species}.md"
            
        # Load the selected prompt
        prompt_path = self.base_dirs.get("prompts", Path("")) / prompt_file
        if prompt_path.exists():
            return load_text_file(prompt_path)

        raise FileNotFoundError(
            f"Species-specific prompt not found: {prompt_file} "
            f"(position={position}, species={species})"
        )

    def _load_guides(self) -> List[str]:
        collected: List[str] = []
        if self.context and self.guide:
            # Load guide from specified path
            guide_path = Path(self.guide)
            if guide_path.is_file():
                collected.append(load_text_file(guide_path))
            else:
                self.logger.warning("Guide file not found: %s", self.guide)
        return collected

    # Map species names to example file identifiers.
    _SPECIES_EXAMPLE_FILES = {
        "first_species": "Example_D_1st_species",
        "second_species": "Example_D_2nd_species",
        "third_species": "Example_D_3rd_species",
        "fourth_species": "Example_D_4th_species",
        "fifth_species": "Example_D_5th_species",
    }

    def _load_example(self) -> Optional[str]:
        """Load a Fux reference example for the detected species and datatype.

        Looks for files in data/examples/<datatype>/ matching the species.
        Returns the file content or None if no example is available.
        """
        species = self._detect_species()
        example_name = self._SPECIES_EXAMPLE_FILES.get(species)
        if not example_name:
            return None

        # Determine position (above/below) from file_id to select the right half
        position = "below" if self.file_id.lower().startswith("below") else "above"

        # Look for example file in data/examples/<datatype>/
        data_dir = self.base_dirs.get("encoded", Path("encoded")).parent
        examples_dir = data_dir / "examples" / self.datatype
        ext = self._EXT_MAP.get(self.datatype, ".musicxml")
        example_path = examples_dir / f"{example_name}{ext}"

        if not example_path.is_file():
            self.logger.debug("No example file found at %s", example_path)
            return None

        self.logger.info("Loading %s species example for counterpoint %s: %s", species, position, example_path)
        raw = load_text_file(example_path)

        if self.datatype == "musicxml":
            return self._extract_example_section(raw, position)

        return raw

    def _extract_example_section(self, musicxml_content: str, position: str) -> str:
        """Extract the above or below section from a combined Fux example MusicXML file.

        The files contain two examples separated by a double barline (light-light):
        - Measures 1-11: Counterpoint ABOVE the cantus firmus
        - Measures 12-22: Counterpoint BELOW the cantus firmus

        We return the full file with an annotation indicating which section is the
        relevant example, rather than splitting the XML (which would break the document).
        """
        if position == "above":
            annotation = (
                "This example shows Fux's counterpoint ABOVE the cantus firmus (D) "
                "in the first section (measures 1 through the first double barline). "
                "Use this section as your reference for writing counterpoint above the cantus firmus."
            )
        else:
            annotation = (
                "This example shows Fux's counterpoint BELOW the cantus firmus (D) "
                "in the second section (from the second double barline to the end). "
                "Use this section as your reference for writing counterpoint below the cantus firmus."
            )
        return f"{annotation}\n\n{musicxml_content}"

    def _save_response(self, response: str) -> None:
        if not self.save_to:
            return
        # Clean LLM response: strip code fences, preamble text, etc.
        response = clean_response(response, self.datatype)
        # Post-process MusicXML to fix LLM duration/divisions mismatches.
        if self.datatype == "musicxml":
            response = fix_musicxml_durations(response)

        # If an identical response already exists, avoid creating duplicate run files.
        existing = self._find_duplicate_response(response)
        if existing:
            self.logger.info("Skipping duplicate response save; identical to %s", existing)
            self.actual_response_path = existing
            return

        # Response is already in the correct location from get_output_path with output_type="response"
        self.save_to.parent.mkdir(parents=True, exist_ok=True)
        self.save_to.write_text(response, encoding="utf-8")
        self.logger.info("Saved response to %s", self.save_to)
        # Store the actual path for later reference
        self.actual_response_path = self.save_to

    def _find_duplicate_response(self, normalized_response: str) -> Optional[Path]:
        """Return an existing response path if content is identical, else None."""
        if not self.save_to:
            return None

        response_dir = self.save_to.parent
        suffix = self.save_to.suffix
        for existing in sorted(response_dir.glob(f"run_*{suffix}")):
            if existing == self.save_to or not existing.is_file():
                continue
            try:
                if existing.read_text(encoding="utf-8") == normalized_response:
                    return existing
            except OSError:
                # Ignore unreadable files and continue scanning.
                continue
        return None

    def _persist_artifacts(self, response: str, prompt_input: PromptInput) -> None:
        """Persist response, metadata, and readable prompt file (best effort)."""
        try:
            self._save_response(response)
        except Exception as e:  # pragma: no cover
            self.logger.warning("Failed to save response: %s", e)

        # If response save was deduplicated to an existing run, do not create
        # new metadata/prompt run files that only duplicate prior artifacts.
        if self.save_to and getattr(self, "actual_response_path", self.save_to) != self.save_to:
            self.logger.info(
                "Skipping metadata/prompt save for deduplicated response (%s)",
                self.actual_response_path,
            )
            return

        try:
            self._save_metadata(prompt_input)
        except Exception as e:  # pragma: no cover
            self.logger.warning("Failed to write metadata file: %s", e)
        try:
            self._save_readable_prompt(prompt_input)
        except Exception as e:  # pragma: no cover
            self.logger.warning("Failed to write prompt file: %s", e)

    # ------------------------------------------------------------------
    def _save_metadata(self, prompt_input: PromptInput) -> None:
        """Write a single metadata JSON file combining prompt text, run params, and token usage.

        Replaces the former separate prompt (.txt) and input-bundle (.json) files.
        File saved in: metadata/<model>/<context>/<datatype>/ subfolder.
        """
        if not self.save_to:
            return
        # Get path in metadata structure
        from llm_fux.utils.path_utils import get_output_path
        metadata_path = get_output_path(
            outputs_dir=self.base_dirs.get("outputs", Path("outputs")),
            model_name=self._get_clean_model_name(self.model),
            file_id=self.file_id,
            datatype=self.datatype,
            context=self.context,
            guide=self.guide,
            dataset=self.dataset,
            ext=".json",
            output_type="metadata",
            temperature=self.temperature,
        )

        # Build metadata bundle
        try:
            montreal_tz = zoneinfo.ZoneInfo("America/Montreal")
            timestamp = datetime.now(montreal_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        components = getattr(self, "_last_components", {})
        api_duration = getattr(self, "_api_duration", None)
        token_usage = getattr(self, "_token_usage", None)
        response_model_id = getattr(self, "_response_model_id", None)
        extra_params = getattr(self, "_extra_params", {})
        actual_path = getattr(self, "actual_response_path", self.save_to)

        bundle: Dict[str, Any] = {
            "timestamp": timestamp,
            "file_id": self.file_id,
            "dataset": self.dataset,
            "datatype": self.datatype,
            "context": self.context,
            "guide_path": self.guide if self.guide else None,
            "model": type(self.model).__name__,
            "model_id": response_model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_path": str(actual_path),
        }

        if extra_params:
            bundle["extra_params"] = extra_params

        if api_duration is not None:
            bundle["api_duration_seconds"] = round(api_duration, 2)

        # Token usage section
        if token_usage:
            bundle["token_usage"] = token_usage.to_dict()

        # Component char lengths for quick reference (full text is in the prompt file)
        if components:
            bundle["component_lengths"] = {
                k: len(v) if isinstance(v, str) else sum(len(x) for x in v if x)
                for k, v in components.items()
            }

        bundle_json = json.dumps(bundle, ensure_ascii=False, indent=2)
        metadata_path.write_text(bundle_json, encoding="utf-8")
        self.logger.info("Saved metadata to %s", metadata_path)
        self.metadata_path = metadata_path

    def _save_readable_prompt(self, prompt_input: PromptInput) -> None:
        """Save the complete prompt as a readable text file, faithful to what the LLM sees."""
        if not self.save_to:
            return
        from llm_fux.utils.path_utils import get_output_path
        prompt_path = get_output_path(
            outputs_dir=self.base_dirs.get("outputs", Path("outputs")),
            model_name=self._get_clean_model_name(self.model),
            file_id=self.file_id,
            datatype=self.datatype,
            context=self.context,
            guide=self.guide,
            dataset=self.dataset,
            ext=".txt",
            output_type="prompt",
            temperature=self.temperature,
        )

        lines = [
            "=" * 80,
            "SYSTEM PROMPT",
            "=" * 80,
            prompt_input.system_prompt,
            "",
            "=" * 80,
            "USER PROMPT",
            "=" * 80,
            prompt_input.user_prompt,
        ]

        prompt_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info("Saved readable prompt to %s", prompt_path)
