# conftest.py - Centralized Test Configuration
"""
Shared pytest configuration and fixtures for the LLM-MusicTheory test suite.

This file provides:
- Common fixtures used across multiple test files
- Test configuration and markers
- Mock utilities and test data generators
"""

import os
import sys
import types
import tempfile
from pathlib import Path
from typing import Dict, Any, Iterator, List

import pytest
from unittest.mock import Mock

# Ensure the package is importable in a src/ layout
_TESTS_DIR = Path(__file__).resolve().parent
_ROOT = _TESTS_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_fux.models.base import LLMInterface, PromptInput

__all__ = [
    "mock_llm",
    "temp_project_structure",
    "temp_project_dir",
    "mock_api_keys",
    "sample_prompt_input",
    "generate_test_music_data",
]

_API_KEY_ENV_VARS: List[str] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
]


# ===============================================================================
# SHARED FIXTURES
# ===============================================================================

@pytest.fixture
def mock_llm() -> LLMInterface:
    """Create a mock LLM capturing queries without making external API calls."""

    class MockLLM(LLMInterface):  # type: ignore[misc]
        def __init__(self) -> None:
            self.last_query: PromptInput | None = None
            self.captured_queries: list[PromptInput] = []
            self.response = "Mock test response"

        def query(self, input: PromptInput) -> str:  # noqa: A003
            self.last_query = input
            self.captured_queries.append(input)
            return self.response

        def submit_batch(self, inputs):
            raise NotImplementedError

        def poll_batch(self, batch_id):
            raise NotImplementedError

        def retrieve_batch_results(self, batch_id):
            raise NotImplementedError

    return MockLLM()


@pytest.fixture
def temp_project_structure() -> Iterator[Dict[str, Any]]:
    """Yield a realistic temporary project structure for integration-style tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create main project structure
        src_dir = temp_path / "src" / "llm_music_theory"
        src_dir.mkdir(parents=True)

        data_dir = temp_path / "data" / "RCM6"  # legacy dataset name (formerly LLM-RCM)
        data_dir.mkdir(parents=True)

        # Create encoded music files
        encoded_dir = data_dir / "encoded"
        datatype_dir = encoded_dir / "musicxml"
        datatype_dir.mkdir(parents=True)
        (datatype_dir / "Q1a.musicxml").write_text("<?xml version='1.0'?><score-partwise></score-partwise>")
        (datatype_dir / "Q1b.musicxml").write_text("<?xml version='1.0'?><score-partwise></score-partwise>")
        
        # Create prompt templates
        prompts_dir = data_dir / "prompts"
        base_dir = prompts_dir / "base"
        base_dir.mkdir(parents=True)
        
        # Base format prompts
        (base_dir / "system_prompt.txt").write_text("You are a music theory expert.")
        (base_dir / "base_musicxml.txt").write_text("Analyze this MusicXML notation.")
        
        # Question files
        questions_context_dir = prompts_dir / "questions" / "context"
        questions_nocontext_dir = prompts_dir / "questions" / "no_context"
        questions_context_dir.mkdir(parents=True)
        questions_nocontext_dir.mkdir(parents=True)
        
        (questions_context_dir / "Q1a.txt").write_text("What is the key signature? Consider the context.")
        (questions_context_dir / "Q1b.txt").write_text("Identify the time signature? Consider the context.")
        (questions_nocontext_dir / "Q1a.txt").write_text("What is the key signature?")
        (questions_nocontext_dir / "Q1b.txt").write_text("Identify the time signature?")
        
        # Guide files
        guides_dir = data_dir / "guides"
        guides_dir.mkdir(parents=True)
        (guides_dir / "key_signatures.txt").write_text("Key signatures appear at the beginning of the staff.")
        (guides_dir / "time_signatures.txt").write_text("Time signatures indicate the meter.")
        
        # Create pyproject.toml for project root detection
        (temp_path / "pyproject.toml").write_text("""
[tool.poetry]
name = "test-project"
version = "0.1.0"
""")
        
        # Create outputs directory
        outputs_dir = temp_path / "outputs"
        outputs_dir.mkdir()
        
        yield {
            'root': temp_path,
            'src': src_dir,
            'data': data_dir,
            'base_dirs': {
                "encoded": encoded_dir,
                "prompts": prompts_dir,
                "questions": prompts_dir / "questions",
                "guides": guides_dir,
                "outputs": outputs_dir,
            }
        }


@pytest.fixture
def temp_project_dir(temp_project_structure: Dict[str, Any]) -> Path:
    """Return the temporary project root path (compat alias)."""
    return temp_project_structure['root']


@pytest.fixture
def mock_api_keys(monkeypatch) -> Dict[str, str]:
    """Mock API keys in env and settings to avoid network usage."""
    mock_keys: Dict[str, str] = {
        "openai": "test-openai-key",
        "anthropic": "test-anthropic-key",
        "google": "test-google-key",
    }
    for key, value in mock_keys.items():
        monkeypatch.setenv(f"{key.upper()}_API_KEY", value)
    monkeypatch.setattr("llm_fux.config.config.API_KEYS", mock_keys)
    return mock_keys


@pytest.fixture
def sample_prompt_input() -> PromptInput:
    """Return a sample PromptInput instance used by multiple tests."""
    return PromptInput(
        system_prompt="You are a music theory expert.",
        user_prompt="Analyze this musical excerpt: <?xml version='1.0'?><score-partwise>test</score-partwise>",
        temperature=0.7,
        max_tokens=500,
    )


# ===============================================================================
# MOCK UTILITIES
# ===============================================================================

class MockResponse:
    """Utility class used for simulating minimal OpenAI style responses."""

    def __init__(self, content: str):
        self.content = content
        self.choices = [Mock(message=Mock(content=content))]


def create_mock_openai_client(response_content: str = "Mock OpenAI response") -> Mock:
    """Return a mock OpenAI client with a preconfigured response."""
    mock_client: Mock = Mock()
    mock_response = MockResponse(response_content)
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def create_mock_anthropic_client(response_content: str = "Mock Anthropic response") -> Mock:
    """Return a mock Anthropic client with a preconfigured response."""
    mock_client: Mock = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text=response_content)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ===============================================================================
# TEST CONFIGURATION
# ===============================================================================

# Configure pytest markers
def pytest_configure(config):  # type: ignore[override]
    """Register custom pytest markers for test categorization."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "comprehensive: mark test as a comprehensive/end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring API mocking"
    )


def pytest_collection_modifyitems(config, items):  # type: ignore[override]
    """Auto-apply markers based on nodeid conventions to reduce boilerplate."""
    for item in items:
        # Mark tests by file
        if "test_models" in item.nodeid:
            item.add_marker(pytest.mark.api)
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_comprehensive" in item.nodeid:
            item.add_marker(pytest.mark.comprehensive)
        elif "test_runner" in item.nodeid or "test_prompt" in item.nodeid or "test_path" in item.nodeid:
            item.add_marker(pytest.mark.unit)
            
        # Mark slow tests
        if "comprehensive" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)


def pytest_ignore_collect(collection_path, config):  # type: ignore[override]
    """Ignore legacy duplicate tests to avoid double coverage/conflicts."""
    p = str(collection_path)
    if p.endswith("_fixed.py") or p.endswith("_new.py"):
        return True
    return False


# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch) -> None:
    """Ensure a clean, deterministic test environment (API keys sanitized)."""
    for var in _API_KEY_ENV_VARS:
        if var in os.environ:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(var, f"test-{var.lower()}")


def _ensure_stub_modules() -> None:
    """Install lightweight stub modules for optional third‑party SDKs if missing."""
    if "openai" not in sys.modules:
        openai_mod = types.ModuleType("openai")

        class _OpenAI:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=lambda **kwargs: types.SimpleNamespace(
                            choices=[types.SimpleNamespace(
                                message=types.SimpleNamespace(content="stubbed response")
                            )]
                        )
                    )
                )

        openai_mod.OpenAI = _OpenAI  # type: ignore[attr-defined]
        sys.modules["openai"] = openai_mod

    if "anthropic" not in sys.modules:
        anthropic_mod = types.ModuleType("anthropic")

        class _Anthropic:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                self.messages = types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(
                        content=[types.SimpleNamespace(text="stubbed response")]
                    )
                )

        anthropic_mod.Anthropic = _Anthropic  # type: ignore[attr-defined]
        sys.modules["anthropic"] = anthropic_mod

    google_pkg = sys.modules.get("google")
    if google_pkg is None:
        google_pkg = types.ModuleType("google")
        sys.modules["google"] = google_pkg
    if "google.genai" not in sys.modules:
        genai_mod = types.ModuleType("google.genai")

        class _ModelsNS:  # pragma: no cover
            def generate_content(self, **kwargs):
                return types.SimpleNamespace(text="stubbed response")

        class _Client:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                self.models = _ModelsNS()

        genai_mod.Client = _Client  # type: ignore[attr-defined]
        setattr(google_pkg, "genai", genai_mod)
        sys.modules["google.genai"] = genai_mod


@pytest.fixture(autouse=True)
def stub_external_sdks() -> None:
    """Provide lightweight stubs for optional SDK imports to avoid hard deps."""
    _ensure_stub_modules()


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:  # pragma: no cover
    """Placeholder for deterministic seeding (extend if randomness added)."""
    return None


# ===============================================================================
# TEST DATA GENERATORS
# ===============================================================================

def generate_test_music_data(format_type: str, question_id: str = "Q1a") -> str:
    """Generate minimal synthetic music notation for the given format."""
    if format_type == "musicxml":
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
    <part-list>
        <score-part id="P1">
            <part-name>Test Part</part-name>
        </score-part>
    </part-list>
    <part id="P1">
        <measure number="1">
            <note>
                <pitch>
                    <step>C</step>
                    <octave>4</octave>
                </pitch>
                <duration>4</duration>
                <type>quarter</type>
            </note>
        </measure>
    </part>
</score-partwise>"""
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
