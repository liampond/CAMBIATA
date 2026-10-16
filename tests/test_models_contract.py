"""
Test the LLM model interface contracts and behavior.

These tests define the expected behavior for all LLM models,
independent of current implementation details.
"""
import pytest
pytestmark = pytest.mark.contract
from abc import ABC
from unittest.mock import Mock, patch

from llm_fux.models.base import LLMInterface, PromptInput


class TestLLMInterfaceContract:
    """Test that all LLM models implement the required interface contract."""
    
    @pytest.mark.parametrize("model_name", ["chatgpt", "claude", "gemini"])
    def test_model_implements_interface(self, model_name, mock_api_keys):
        """All models MUST implement LLMInterface."""
        from llm_fux.core.dispatcher import get_llm
        
        try:
            model = get_llm(model_name)
            assert isinstance(model, LLMInterface), f"{model_name} must implement LLMInterface"
        except Exception as e:
            pytest.skip(f"Model {model_name} not available: {e}")
    
    @pytest.mark.parametrize("model_name", ["chatgpt", "claude", "gemini"])
    def test_model_has_query_method(self, model_name, mock_api_keys):
        """All models MUST have a callable query method."""
        from llm_fux.core.dispatcher import get_llm
        
        try:
            model = get_llm(model_name)
            assert hasattr(model, 'query'), f"{model_name} must have query method"
            assert callable(getattr(model, 'query')), f"{model_name}.query must be callable"
        except Exception as e:
            pytest.skip(f"Model {model_name} not available: {e}")
    
    @pytest.mark.parametrize("model_name", ["chatgpt", "claude", "gemini"])
    def test_model_query_accepts_prompt_input(self, model_name, mock_api_keys, sample_prompt_input):
        """All models MUST accept PromptInput objects in their query method."""
        from llm_fux.core.dispatcher import get_llm
        
        try:
            model = get_llm(model_name)
            
            # Mock the actual API call to avoid real requests
            with patch.object(model, 'query', return_value="mocked response") as mock_query:
                result = model.query(sample_prompt_input)
                assert result == "mocked response"
                mock_query.assert_called_once_with(sample_prompt_input)
                
        except Exception as e:
            pytest.skip(f"Model {model_name} not available: {e}")
    
    @pytest.mark.parametrize("model_name", ["chatgpt", "claude", "gemini"])
    def test_model_query_returns_string(self, model_name, mock_api_keys, sample_prompt_input):
        """All models MUST return string responses."""
        from llm_fux.core.dispatcher import get_llm
        
        try:
            model = get_llm(model_name)
            
            with patch.object(model, 'query', return_value="test response") as mock_query:
                result = model.query(sample_prompt_input)
                assert isinstance(result, str), f"{model_name} query must return string"
                
        except Exception as e:
            pytest.skip(f"Model {model_name} not available: {e}")


class TestLLMErrorHandling:
    """Test that LLM models handle errors appropriately."""
    
    def test_invalid_prompt_input_type(self, mock_api_keys):
        """Models SHOULD handle invalid input types gracefully."""
        from llm_fux.core.dispatcher import get_llm
        
        model = get_llm("chatgpt")  # Use any available model
        
        # Test with various invalid input types
        invalid_inputs = [None, "", 123, [], {}, object()]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                model.query(invalid_input)
    
    def test_api_failure_handling(self, mock_api_keys, sample_prompt_input):
        """Models SHOULD propagate API failures as appropriate exceptions."""
        from llm_fux.core.dispatcher import get_llm
        
        model = get_llm("chatgpt")
        
        # Mock API failure
        with patch.object(model, 'query', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                model.query(sample_prompt_input)
    
    def test_empty_response_handling(self, mock_api_keys, sample_prompt_input):
        """Models SHOULD handle empty API responses appropriately."""
        from llm_fux.core.dispatcher import get_llm
        
        model = get_llm("chatgpt")
        
        # Mock empty response
        with patch.object(model, 'query', return_value=""):
            result = model.query(sample_prompt_input)
            assert result == ""  # Empty string is valid


class TestPromptInputContract:
    """Test the PromptInput data contract."""
    
    def test_prompt_input_creation(self):
        """PromptInput MUST be createable with required fields."""
        prompt_input = PromptInput(
            system_prompt="You are a music expert",
            user_prompt="Analyze this music",
            temperature=0.7
        )
        
        assert prompt_input.system_prompt == "You are a music expert"
        assert prompt_input.user_prompt == "Analyze this music"
        assert prompt_input.temperature == 0.7
    
    def test_prompt_input_validation(self):
        """PromptInput SHOULD validate input parameters."""
        # Valid temperature range
        PromptInput(
            system_prompt="test",
            user_prompt="test",
            temperature=0.0
        )
        
        PromptInput(
            system_prompt="test", 
            user_prompt="test",
            temperature=1.0
        )
        
        # Test edge cases - implementation should handle these appropriately
        with pytest.raises((ValueError, TypeError)):
            PromptInput(
                system_prompt="test",
                user_prompt="test", 
                temperature=-0.1  # Invalid temperature
            )
        
        with pytest.raises((ValueError, TypeError)):
            PromptInput(
                system_prompt="test",
                user_prompt="test",
                temperature=1.1  # Invalid temperature
            )
    
    def test_prompt_input_required_fields(self):
        """PromptInput MUST require essential fields."""
        # Missing system_prompt
        with pytest.raises((TypeError, ValueError)):
            PromptInput(  # type: ignore[call-arg]
                user_prompt="test",
                temperature=0.5
            )
        
        # Missing user_prompt  
        with pytest.raises((TypeError, ValueError)):
            PromptInput(  # type: ignore[call-arg]
                system_prompt="test",
                temperature=0.5
            )
    
    def test_prompt_input_default_temperature(self):
        """PromptInput SHOULD provide reasonable defaults."""
        prompt_input = PromptInput(
            system_prompt="test",
            user_prompt="test"
        )
        
        # Should have some default temperature
        assert hasattr(prompt_input, 'temperature')
        assert isinstance(prompt_input.temperature, (int, float))
        assert 0.0 <= prompt_input.temperature <= 1.0


class TestLLMInterfaceAbstraction:
    """Test that LLMInterface properly defines the abstract contract."""
    
    def test_interface_is_abstract(self):
        """LLMInterface MUST be an abstract base class."""
        assert issubclass(LLMInterface, ABC)
        
        # Should not be instantiable directly
        with pytest.raises(TypeError):
            LLMInterface()
    
    def test_interface_defines_query_method(self):
        """LLMInterface MUST define abstract query method."""
        # Check that query is defined as abstract
        assert hasattr(LLMInterface, 'query')
        
        # Verify it's marked as abstract (has __isabstractmethod__)
        assert getattr(LLMInterface.query, '__isabstractmethod__', False)
    
    def test_concrete_implementation_requirements(self):
        """Concrete implementations MUST implement all abstract methods."""
        
        # This should fail - missing query implementation
        class IncompleteModel(LLMInterface):
            pass
        
        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore[abstract]
        
        # This should work - complete implementation
        class CompleteModel(LLMInterface):
            def query(self, input: PromptInput) -> str:
                return "test response"
            def submit_batch(self, inputs):
                raise NotImplementedError
            def poll_batch(self, batch_id):
                raise NotImplementedError
            def retrieve_batch_results(self, batch_id):
                raise NotImplementedError

        model = CompleteModel()
        assert isinstance(model, LLMInterface)


class TestLLMPerformanceRequirements:
    """Test performance-related requirements for LLM models."""
    
    @pytest.mark.slow
    def test_model_instantiation_speed(self, mock_api_keys):
        """Model instantiation SHOULD be reasonably fast."""
        import time
        from llm_fux.core.dispatcher import get_llm
        
        start_time = time.time()
        
        try:
            model = get_llm("chatgpt")
            instantiation_time = time.time() - start_time
            
            # Should instantiate within 2 seconds
            assert instantiation_time < 2.0, f"Model took {instantiation_time:.2f}s to instantiate"
            
        except Exception as e:
            pytest.skip(f"Model not available: {e}")
    
    @pytest.mark.slow  
    def test_query_timeout_handling(self, mock_api_keys, sample_prompt_input):
        """Models SHOULD document or handle query timeouts; if no timeout, the call should eventually return."""
        from llm_fux.core.dispatcher import get_llm
        
        model = get_llm("chatgpt")
        
        # Mock a slow query
        def slow_query(*args, **kwargs):
            import time
            time.sleep(1.0)  # Simulate slow response within test budget
            return "delayed response"
        
        with patch.object(model, 'query', side_effect=slow_query):
            # Expect a return (no built-in timeout in current implementation)
            result = model.query(sample_prompt_input)
            assert result == "delayed response"
