"""Filesystem and path discovery helpers.

Design goals:
    * Keep IO minimal (avoid repeated directory scans where possible).
    * Provide clear errors for missing required artifacts while allowing
        optional lookups (required=False) to return None gracefully.
    * Support legacy naming conventions while nudging toward the new layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Iterable, Dict

__all__ = [
        "find_project_root",
        "load_text_file",
        "find_encoded_file",
        "find_question_file",
        "list_questions",
        "list_file_ids",
        "list_datatypes",
        "list_guides",
        "ensure_dir",
        "get_output_path",
]

_ROOT_CACHE: Optional[Path] = None
_DATATYPE_EXT: Dict[str, str] = {"musicxml": ".musicxml"}


def _normalize_datatype(datatype: str) -> str:
        key = datatype.lower().strip()
        if key not in _DATATYPE_EXT:
                raise ValueError(f"Unknown datatype '{datatype}'. Valid: {sorted(_DATATYPE_EXT)}")
        return key


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Locate and cache the project root containing ``pyproject.toml``.

    Parameters
    ----------
    start_path: Path | None
        Starting directory (defaults to this file's parent). Accepts a file path.
    """
    global _ROOT_CACHE
    if _ROOT_CACHE and _ROOT_CACHE.exists():  # pragma: no cover (cache branch)
        return _ROOT_CACHE
    if start_path is None:
        start_path = Path(__file__).parent
    start_path = start_path if start_path.is_dir() else start_path.parent
    current = start_path.resolve()
    while True:
        if (current / "pyproject.toml").exists():
            _ROOT_CACHE = current
            return current
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError("Could not locate project root containing pyproject.toml")


def load_text_file(path: Path) -> str:
    """Read UTF‑8 text file returning stripped contents.

    Raises
    ------
    FileNotFoundError
        If the path is not an existing file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Expected file at {path} but none was found")
    return path.read_text(encoding="utf-8").strip()


def find_encoded_file(
    file_id: str,
    datatype: str,
    encoded_dir: Path,
    required: bool = True,
) -> Optional[Path]:
    """Locate encoded music file for ``file_id`` with given ``datatype``.

    Returns the first match (exact first, fallback glob) or None if ``required`` is False.
    Raises ValueError for unsupported datatypes & FileNotFoundError when required and missing.
    """
    key = _normalize_datatype(datatype)
    ext = _DATATYPE_EXT[key]
    candidate = encoded_dir / f"{file_id}{ext}"
    if candidate.exists():
        return candidate
    if encoded_dir.exists():
        matches = list(encoded_dir.rglob(f"*{file_id}{ext}"))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"No encoded file found for {file_id} in {encoded_dir}")
    return None


def find_question_file(
    file_id: str,
    context: bool,
    questions_dir: Path,
    required: bool = True,
) -> Optional[Path]:
    """Locate contextual / non‑contextual question prompt file (legacy datasets).

    Note: This is for legacy datasets that don't have a single prompt.md file.
    Modern datasets should use a single prompt.md file instead.
    """
    suffix = "context" if context else "nocontext"
    candidate = questions_dir / f"{file_id}.{suffix}.txt"
    if candidate.exists():
        return candidate
    if questions_dir.exists():
        pattern = f"*{file_id}*{'Context' if context else 'NoContext'}Prompt.txt"
        matches = list(questions_dir.glob(pattern))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Question file not found for {file_id} in {questions_dir}")
    return None


def list_questions(questions_dir: Path) -> List[str]:
    """Return sorted stems of all ``.txt`` question prompt files (legacy datasets)."""
    if not questions_dir.exists():
        return []
    return sorted({f.stem for f in questions_dir.rglob("*.txt")})


def list_file_ids(encoded_dir: Path) -> List[str]:
    """Return unique filename stems under each datatype subdirectory."""
    if not encoded_dir.exists():
        return []
    ids: Set[str] = set()
    for sub in encoded_dir.iterdir():
        if not sub.is_dir():
            continue
        
        # If known datatype, filter by extension recursively
        if sub.name in _DATATYPE_EXT:
            ext = _DATATYPE_EXT[sub.name]
            for f in sub.rglob(f"*{ext}"):
                if f.is_file():
                    ids.add(f.stem)
        # Fallback: if not a known datatype, we skip it to avoid noise.
        # (The system requires datatypes to be in _DATATYPE_EXT anyway)

    return sorted(ids)


def list_datatypes(encoded_dir: Path) -> List[str]:
    """Return supported datatypes inferred from populated subdirectories."""
    if not encoded_dir.exists():
        return []
    found: Set[str] = set()
    for subdir in encoded_dir.iterdir():
        if subdir.is_dir() and subdir.name in _DATATYPE_EXT:
            try:
                next(subdir.iterdir())  # at least one entry
            except StopIteration:  # empty directory
                continue
            found.add(subdir.name)
    return sorted(found)


def list_guides(guides_dir: Path) -> List[str]:
    """Return relative paths of all guide ``.txt`` and ``.md`` files (recursive)."""
    if not guides_dir.exists():
        return []
    guides = []
    for ext in ["*.txt", "*.md"]:
        for f in guides_dir.rglob(ext):
            if f.is_file():
                guides.append(str(f.relative_to(guides_dir)))
    return sorted(guides)


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if missing (idempotent)."""
    path.mkdir(parents=True, exist_ok=True)


import re


_SPECIES_FOLDER_MAP = {
    "1_first_species": "1st-species",
    "2_second_species": "2nd-species",
    "3_third_species": "3rd-species",
    "4_fourth_species": "4th-species",
    "5_fifth_species": "5th-species",
}


def _extract_species_and_version(guide: str) -> tuple[str, str]:
    """Extract species and version from a guide path.

    Supports both ``v2.3`` and ``version 2.0`` naming conventions.

    Examples::

        "data/guides/5_fifth_species/Fux Rules- 5th species - Florish - v2.3.txt"
            → ("5th-species", "v2.3")
        "data/guides/1_first_species/1vs1_v1.2.txt"
            → ("1st-species", "v1.2")
        "data/guides/2_second_species/Cantus 2notes vs 1 - Schubert version 2.0.txt"
            → ("2nd-species", "v2.0")
    """
    guide_path = Path(guide)
    parent = guide_path.parent.name
    species = _SPECIES_FOLDER_MAP.get(parent, parent)
    version_match = re.search(r'(?:version\s+|v)(\d+\.\d+)', guide_path.stem, re.IGNORECASE)
    version = f"v{version_match.group(1)}" if version_match else "unknown"
    return species, version


def _get_next_run_number(folder: Path, ext: str) -> int:
    """Find the next available run number in *folder* for files named ``run_N<ext>``."""
    if not folder.exists():
        return 1
    existing: list[int] = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix == ext:
            m = re.match(r'run_(\d+)', f.stem)
            if m:
                existing.append(int(m.group(1)))
    return max(existing) + 1 if existing else 1


def get_output_path(
    outputs_dir: Path,
    model_name: str,
    file_id: str,
    datatype: str = "musicxml",
    context: bool = False,
    guide: Optional[str] = None,
    dataset: Optional[str] = None,
    ext: str = ".txt",
    output_type: str = "response",
    temperature: float = 0.0,
) -> Path:
    """Return path for model output file.

    Structure (with context)::

        outputs/<output_type>/<model>/<species>/<guide_version>/<file_id>/run_<N>.<ext>

    Structure (no context)::

        outputs/<output_type>/<model>/no-context/<file_id>/run_<N>.<ext>

    Output types:
        - response: The raw LLM-generated file (e.g. MusicXML)
        - metadata: Run parameters + token usage (JSON)
        - prompt: Readable compiled prompt as sent to the LLM (TXT)
    """
    if not file_id:
        raise ValueError("file_id is required for output path")

    type_folder = outputs_dir / output_type / model_name

    if context and guide:
        species, version = _extract_species_and_version(guide)
        file_folder = type_folder / species / version / file_id
    else:
        file_folder = type_folder / "no-context" / file_id

    ensure_dir(file_folder)
    run_number = _get_next_run_number(file_folder, ext)
    return file_folder / f"run_{run_number}{ext}"
