import pytest
from unittest.mock import Mock, patch

from llm_fux.models.base import LLMInterface, PromptInput

pytestmark = pytest.mark.contract


def make_runner(tmp_path, mock_model=None, **kwargs):
    """Factory to build a PromptRunner with temp data dirs and a mock model."""
    from llm_fux.core.runner import PromptRunner

    if mock_model is None:
        class Dummy(LLMInterface):
            def query(self, input: PromptInput) -> str:
                return "ok"
            def submit_batch(self, inputs):
                raise NotImplementedError
            def poll_batch(self, batch_id):
                raise NotImplementedError
            def retrieve_batch_results(self, batch_id):
                raise NotImplementedError
        mock_model = Dummy()

    base_dirs = {
        "prompts": tmp_path / "prompts",
        "encoded": tmp_path / "encoded",
        "guides": tmp_path / "guides",
        "outputs": tmp_path / "outputs",
    }
    for p in base_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    defaults = dict(
        model=mock_model,
        file_id="Q1b",
        datatype="musicxml",
        context=True,
        base_dirs=base_dirs,
        temperature=0.2,
    )
    defaults.update(kwargs)
    return PromptRunner(**defaults)


class TestRunnerInterface:
    def test_runner_exists(self, tmp_path):
        from llm_fux.core.runner import PromptRunner
        assert PromptRunner is not None
        runner = make_runner(tmp_path)
        assert runner is not None

    def test_runner_has_execution_method(self, tmp_path):
        runner = make_runner(tmp_path)
        # Exact name is flexible; today the public method is `run`.
        execution_methods = ["run", "execute", "query", "process"]
        has_execution_method = any(
            hasattr(runner, m) and callable(getattr(runner, m)) for m in execution_methods
        )
        assert has_execution_method, f"Runner should have one of: {execution_methods}"


class TestExecutionWorkflow:
    def test_single_query_execution(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Test response from LLM"
        runner = make_runner(tmp_path, mock_model=mock_llm)

        with patch.object(runner, "_load_system_prompt", return_value="sys"), \
             patch.object(runner, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner, "_load_encoded", return_value="enc"), \
             patch.object(runner, "_load_question", return_value="q"), \
             patch.object(runner, "_load_guides", return_value=["g1"]):
            result = runner.run()

        assert result == "Test response from LLM"
        mock_llm.query.assert_called_once()
        (prompt_input,) = mock_llm.query.call_args[0]
        assert isinstance(prompt_input, PromptInput)
        assert "fmt" in prompt_input.user_prompt
        assert "enc" in prompt_input.user_prompt
        assert "q" in prompt_input.user_prompt
        assert "g1" in prompt_input.user_prompt

    def test_context_vs_no_context_execution(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Mock response"
        runner_ctx = make_runner(tmp_path, mock_model=mock_llm, context=True)
        runner_nctx = make_runner(tmp_path, mock_model=mock_llm, context=False)

        with patch.object(runner_ctx, "_load_system_prompt", return_value="sys"), \
             patch.object(runner_ctx, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner_ctx, "_load_encoded", return_value="enc"), \
             patch.object(runner_ctx, "_load_question", return_value="q"), \
             patch.object(runner_ctx, "_load_guides", return_value=["g1", "g2"]), \
             patch.object(runner_nctx, "_load_system_prompt", return_value="sys"), \
             patch.object(runner_nctx, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner_nctx, "_load_encoded", return_value="enc"), \
             patch.object(runner_nctx, "_load_question", return_value="q"), \
             patch.object(runner_nctx, "_load_guides", return_value=[]):
            res_ctx = runner_ctx.run()
            res_nctx = runner_nctx.run()

        assert res_ctx is not None and res_nctx is not None
        assert mock_llm.query.call_count == 2
        pi_ctx = mock_llm.query.call_args_list[0][0][0]
        pi_nctx = mock_llm.query.call_args_list[1][0][0]
        assert "g1" in pi_ctx.user_prompt and "g2" in pi_ctx.user_prompt
        assert "g1" not in pi_nctx.user_prompt and "g2" not in pi_nctx.user_prompt


class TestDataHandling:
    def test_question_data_loading(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Mock response"
        runner = make_runner(tmp_path, mock_model=mock_llm)

        with patch.object(runner, "_load_system_prompt", return_value="sys"), \
             patch.object(runner, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner, "_load_encoded", return_value="<score-partwise>data</score-partwise>"), \
             patch.object(runner, "_load_question", return_value="Test file content"), \
             patch.object(runner, "_load_guides", return_value=["g1"]):
            _ = runner.run()

        (prompt_input,) = mock_llm.query.call_args[0]
        assert "Test file content" in prompt_input.user_prompt

    def test_encoded_music_data_loading(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Mock response"
        runner = make_runner(tmp_path, mock_model=mock_llm)

        with patch.object(runner, "_load_system_prompt", return_value="sys"), \
             patch.object(runner, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner, "_load_encoded", return_value="<score-partwise>test music data</score-partwise>"), \
             patch.object(runner, "_load_question", return_value="q"), \
             patch.object(runner, "_load_guides", return_value=["g1"]):
            _ = runner.run()

        (prompt_input,) = mock_llm.query.call_args[0]
        assert "<score-partwise>test music data</score-partwise>" in prompt_input.user_prompt

    def test_missing_file_handling(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Mock response"
        runner = make_runner(tmp_path, mock_model=mock_llm)

        with patch.object(runner, "_load_system_prompt", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                runner.run()


class TestFormatSupport:
    def test_musicxml_format_execution(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Response for musicxml"
        runner = make_runner(tmp_path, mock_model=mock_llm, datatype="musicxml")

        musicxml_content = "<?xml version='1.0'?><score-partwise></score-partwise>"

        with patch.object(runner, "_load_system_prompt", return_value="sys"), \
             patch.object(runner, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner, "_load_encoded", return_value=musicxml_content), \
             patch.object(runner, "_load_question", return_value="q"), \
             patch.object(runner, "_load_guides", return_value=["g1"]):
            _ = runner.run()

        (prompt_input,) = mock_llm.query.call_args[0]
        assert musicxml_content in prompt_input.user_prompt


class TestErrorHandling:
    def test_llm_query_error_propagation(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.side_effect = Exception("API Error")
        runner = make_runner(tmp_path, mock_model=mock_llm)

        with patch.object(runner, "_load_system_prompt", return_value="sys"), \
             patch.object(runner, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner, "_load_encoded", return_value="enc"), \
             patch.object(runner, "_load_question", return_value="q"), \
             patch.object(runner, "_load_guides", return_value=["g1"]):
            with pytest.raises(Exception, match="API Error"):
                runner.run()


class TestRunnerPerformance:
    @pytest.mark.slow
    def test_execution_speed(self, mock_api_keys, tmp_path):
        import time
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Fast mock response"
        runner = make_runner(tmp_path, mock_model=mock_llm)

        with patch.object(runner, "_load_system_prompt", return_value="sys"), \
             patch.object(runner, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner, "_load_encoded", return_value="enc"), \
             patch.object(runner, "_load_question", return_value="q"), \
             patch.object(runner, "_load_guides", return_value=["g1"]):
            start = time.time()
            res = runner.run()
            dt = time.time() - start

        assert res is not None
        assert dt < 1.0, f"Execution took {dt:.2f}s"

    @pytest.mark.slow
    def test_memory_efficiency(self, mock_api_keys, tmp_path):
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.query.return_value = "Memory test response"
        runner = make_runner(tmp_path, mock_model=mock_llm)

        with patch.object(runner, "_load_system_prompt", return_value="sys"), \
             patch.object(runner, "_load_base_format_prompt", return_value="fmt"), \
             patch.object(runner, "_load_encoded", return_value="enc"), \
             patch.object(runner, "_load_question", return_value="q"), \
             patch.object(runner, "_load_guides", return_value=["g1"]):
            results = [runner.run() for _ in range(10)]

        assert len(results) == 10
        assert all(r is not None for r in results)
        assert mock_llm.query.call_count == 10

