"""
Test path utilities and data loading functions.
"""
from pathlib import Path
import tempfile
import pytest

from llm_fux.utils.path_utils import (
    load_text_file,
    find_encoded_file,
    find_question_file,
    list_questions,
    list_datatypes,
    list_guides,
    get_output_path,
    ensure_dir,
    find_project_root
)


class TestPathUtils:
    """Test utility functions for path handling and file discovery."""

    @pytest.fixture
    def temp_structure(self):
        """Create a temporary directory structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create encoded files
            encoded_dir = temp_path / "encoded" / "test_exam"
            encoded_dir.mkdir(parents=True)
            
            # Create files directly in the encoded_dir (simpler structure)
            (encoded_dir / "Q1a.musicxml").write_text('<?xml version="1.0"?><score-partwise/>')
            (encoded_dir / "Q2b.musicxml").write_text('<?xml version="1.0"?><score-partwise/>')
            
            # Create question files
            questions_dir = temp_path / "questions"
            questions_dir.mkdir()
            
            (questions_dir / "Q1a.context.txt").write_text("Context question")
            (questions_dir / "Q1a.nocontext.txt").write_text("No context question")  
            (questions_dir / "Q2b.nocontext.txt").write_text("Another question")
            
            # Create guides
            guides_dir = temp_path / "guides"
            guides_dir.mkdir()
            (guides_dir / "harmonic_analysis.txt").write_text("Harmonic guide")
            (guides_dir / "form_analysis.txt").write_text("Form guide")
            
            yield temp_path

    def test_load_text_file(self, temp_structure):
        """Test loading text files."""
        test_file = temp_structure / "test.txt"
        test_file.write_text("Test content")
        
        content = load_text_file(test_file)
        assert content == "Test content"

    def test_load_text_file_not_exists(self, temp_structure):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_text_file(temp_structure / "nonexistent.txt")

    def test_find_encoded_file(self, temp_structure):
        """Test finding encoded music files."""
        encoded_dir = temp_structure / "encoded" / "test_exam"
        
        # Find MusicXML file
        musicxml_file = find_encoded_file("Q1a", "musicxml", encoded_dir)
        assert musicxml_file is not None
        assert musicxml_file.name == "Q1a.musicxml"
        assert musicxml_file.exists()

    def test_find_encoded_file_not_found(self, temp_structure):
        """Test finding non-existent encoded file."""
        encoded_dir = temp_structure / "encoded" / "test_exam"
        
        with pytest.raises(FileNotFoundError):
            find_encoded_file("Q99", "musicxml", encoded_dir)

    def test_find_question_file_context(self, temp_structure):
        """Test finding question file with context."""
        questions_dir = temp_structure / "questions"
        
        question_file = find_question_file("Q1a", True, questions_dir)
        assert question_file is not None
        assert question_file.name == "Q1a.context.txt"
        assert question_file.exists()

    def test_find_question_file_no_context(self, temp_structure):
        """Test finding question file without context."""
        questions_dir = temp_structure / "questions"
        
        question_file = find_question_file("Q1a", False, questions_dir)
        assert question_file is not None
        assert question_file.name == "Q1a.nocontext.txt"
        assert question_file.exists()

    def test_find_question_file_not_required(self, temp_structure):
        """Test finding non-existent question file when not required."""
        questions_dir = temp_structure / "questions"
        
        question_file = find_question_file("Q99", True, questions_dir, required=False)
        assert question_file is None

    def test_list_questions(self, temp_structure):
        """Test listing available questions."""
        questions_dir = temp_structure / "questions"
        
        questions = list_questions(questions_dir)
        # The list_questions function gets stems, so:
        # "Q1a.context.txt" -> "Q1a.context"
        # "Q1a.nocontext.txt" -> "Q1a.nocontext"
        # "Q2b.nocontext.txt" -> "Q2b.nocontext"
        expected_stems = {"Q1a.context", "Q1a.nocontext", "Q2b.nocontext"}
        assert expected_stems.issubset(set(questions))

    def test_list_datatypes(self, temp_structure):
        """Test listing available data types."""
        # Use the main encoded dir, not a subdirectory
        encoded_dir = temp_structure / "encoded"
        
        # Create datatype subdirectories like our real structure
        musicxml_dir = encoded_dir / "musicxml"
        musicxml_dir.mkdir(parents=True)
        
        # Add some files to make them valid
        (musicxml_dir / "Q1a.musicxml").write_text('<?xml version="1.0"?><score-partwise/>')
        
        datatypes = list_datatypes(encoded_dir)
        assert "musicxml" in datatypes
        assert len(datatypes) == 1

    def test_list_guides(self, temp_structure):
        """Test listing available guides."""
        guides_dir = temp_structure / "guides"
        
        guides = list_guides(guides_dir)
        assert "harmonic_analysis.txt" in guides
        assert "form_analysis.txt" in guides
        assert len(guides) == 2

    def test_get_output_path(self, temp_structure):
        """Test generating output file paths with context."""
        outputs_dir = temp_structure / "outputs"

        output_path = get_output_path(
            outputs_dir=outputs_dir,
            model_name="TestModel",
            file_id="Q1a",
            datatype="musicxml",
            context=True,
            guide="data/guides/1_first_species/Fux Rules- 1vs1 note - v1.7.txt"
        )

        # Structure: outputs/<output_type>/<model>/<species>/<version>/<file_id>/run_<N>.<ext>
        expected = outputs_dir / "response" / "TestModel" / "1st-species" / "v1.7" / "Q1a" / "run_1.txt"
        assert output_path == expected

    def test_get_output_path_no_context(self, temp_structure):
        """Test generating output file paths without context."""
        outputs_dir = temp_structure / "outputs"

        output_path = get_output_path(
            outputs_dir=outputs_dir,
            model_name="TestModel",
            file_id="Q2b",
            datatype="musicxml",
            context=False
        )

        # Structure: outputs/<output_type>/<model>/no-context/<file_id>/run_<N>.<ext>
        expected = outputs_dir / "response" / "TestModel" / "no-context" / "Q2b" / "run_1.txt"
        assert output_path == expected

    def test_find_project_root(self):
        """Test finding project root directory."""
        # This should find the project root containing pyproject.toml
        root = find_project_root()
        assert root.is_dir()
        assert (root / "pyproject.toml").exists()


class TestDataIntegrity:
    """Test data file integrity and structure."""

    def test_data_directory_exists(self):
        """Test that data directory exists in the project."""
        root = find_project_root()
        data_dir = root / "data"
        assert data_dir.exists(), "data directory missing"
        assert data_dir.is_dir()

    def test_required_subdirectories_exist(self):
        """Test that core subdirectories exist in data directory."""
        root = find_project_root()
        data_dir = root / "data"
        required_dirs = ["encoded", "prompts", "guides"]
        for name in required_dirs:
            d = data_dir / name
            assert d.exists(), f"Missing required directory: {name}"
            assert d.is_dir()

    def test_base_prompts_exist(self):
        """Test that base_musicxml prompt file exists."""
        root = find_project_root()
        base_dir = root / "data" / "prompts" / "base"
        assert base_dir.exists()
        # Only musicxml is now required
        for stem in ["base_musicxml"]:
            # Support both .md and .txt extensions
            md_file = base_dir / f"{stem}.md"
            txt_file = base_dir / f"{stem}.txt"
            assert md_file.exists() or txt_file.exists(), f"Missing required file: {stem}.md or {stem}.txt"

    def test_sample_encoded_files_exist(self):
        """Test that encoded musicxml files exist in data directory."""
        root = find_project_root()
        encoded_dir = root / "data" / "encoded"
        assert encoded_dir.exists()
        # Only musicxml is required now
        for sub in ["musicxml"]:
            d = encoded_dir / sub
            assert d.exists(), f"Missing encoded/{sub} directory"
            # Use rglob to find files in subdirectories (above/below)
            files = list(d.rglob(f"*.{sub}"))
            assert files, f"No {sub} files found"

    def test_file_naming_conventions(self):
        """Basic naming checks for musicxml datatype in data directory."""
        root = find_project_root()
        encoded_dir = root / "data" / "encoded"
        if not encoded_dir.exists():
            pytest.skip("encoded directory missing")
        for datatype_dir in encoded_dir.iterdir():
            if not datatype_dir.is_dir():
                continue
            if datatype_dir.name != "musicxml":
                continue
            # Recursively find all files in datatype directory (handles subdirs like above/below)
            for file_path in datatype_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                # Skip hidden files like .DS_Store
                if file_path.name.startswith("."):
                    continue
                assert file_path.suffix == ".musicxml"
