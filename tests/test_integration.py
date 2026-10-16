"""Integration tests against fux-counterpoint dataset only."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from llm_fux.core.runner import PromptRunner
from llm_fux.models.base import LLMInterface, PromptInput


class MockLLMForIntegration(LLMInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.query_log = []
        self.response = "Integration test response"

    def query(self, input: PromptInput) -> str:
        self.query_log.append({
            "system_prompt": input.system_prompt,
            "user_prompt": input.user_prompt,
            "temperature": input.temperature,
            "max_tokens": getattr(input, "max_tokens", None)
        })
        return self.response

    def submit_batch(self, inputs):
        raise NotImplementedError

    def poll_batch(self, batch_id):
        raise NotImplementedError

    def retrieve_batch_results(self, batch_id):
        raise NotImplementedError


class TestCLIIntegration:
    @pytest.fixture
    def mock_all_models(self):
        return {name: MockLLMForIntegration(name) for name in ["chatgpt", "claude", "gemini"]}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "x", "GOOGLE_API_KEY": "x"})
    def test_prompt_compilation_workflow(self, mock_all_models):
        root = Path(__file__).parent.parent
        data_dir = root / "data"
        if not data_dir.exists():
            pytest.skip("data directory missing")
        # Updated to use musicxml and new file structure
        musicxml_file = data_dir / "encoded" / "musicxml" / "above" / "Above_CantusFirmus_C.musicxml"
        if not musicxml_file.exists():
            pytest.skip(f"cantus firmus file missing: {musicxml_file}")
        base_dirs = {
            "encoded": data_dir / "encoded",
            "prompts": data_dir / "prompts",
            "questions": data_dir / "prompts",
            "guides": data_dir / "guides",
            "outputs": root / "outputs",
        }
        llm = mock_all_models["chatgpt"]
        resp = PromptRunner(
            model=llm,
            file_id="Above_CantusFirmus_C",
            datatype="musicxml",
            context=True,
            base_dirs=base_dirs,
            temperature=0.0,
            save=False,
        ).run()
        assert resp == "Integration test response"
        assert llm.query_log

    def test_missing_encoded_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_dirs = {k: tmp_path / k for k in ["encoded", "prompts", "guides", "outputs", "questions"]}
            for p in base_dirs.values():
                p.mkdir(parents=True, exist_ok=True)
            class Dummy(LLMInterface):
                def query(self, input: PromptInput) -> str:
                    return "x"
                def submit_batch(self, inputs):
                    raise NotImplementedError
                def poll_batch(self, batch_id):
                    raise NotImplementedError
                def retrieve_batch_results(self, batch_id):
                    raise NotImplementedError
            dummy = Dummy()
            with pytest.raises(FileNotFoundError):
                PromptRunner(
                    model=dummy,
                    file_id="NoFile",
                    datatype="musicxml",
                    context=False,
                    base_dirs=base_dirs,
                    temperature=0.0,
                    save=False
                ).run()

    def test_missing_question_file(self):
        """Test handling of missing question files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create encoded file but no question file
            encoded_dir = temp_path / "encoded" / "test"
            encoded_dir.mkdir(parents=True)
            (encoded_dir / "Q1a.musicxml").write_text('<?xml version="1.0"?><score-partwise/>')
            
            prompts_base = temp_path / "prompts" / "base"
            prompts_base.mkdir(parents=True)
            (prompts_base / "system_prompt.txt").write_text("System")
            (prompts_base / "base_musicxml.txt").write_text("Format: MusicXML")
            
            base_dirs = {
                "encoded": temp_path / "encoded",
                "prompts": temp_path / "prompts",
                "questions": temp_path / "prompts",
                "guides": temp_path / "guides",
                "outputs": temp_path / "outputs",
            }
            
            mock_llm = MockLLMForIntegration("test")
            
            runner = PromptRunner(
                model=mock_llm,
                file_id="Q1a",
                datatype="musicxml",
                context=False,
                base_dirs=base_dirs,
                temperature=0.0,
                save=False
            )
            
            with pytest.raises(FileNotFoundError):
                runner.run()
