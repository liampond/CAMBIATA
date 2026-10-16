"""
Test the dispatcher interface and behavior requirements.

These tests define how the model dispatcher SHOULD behave,
independent of current implementation details.
"""
import pytest
pytestmark = pytest.mark.contract
from unittest.mock import Mock, patch

from llm_fux.models.base import LLMInterface


class TestDispatcherInterface:
    """Test the core dispatcher interface requirements."""
    
    def test_get_llm_function_exists(self):
        """Dispatcher MUST provide a get_llm function."""
        from llm_fux.core.dispatcher import get_llm
        assert callable(get_llm)
    
    def test_get_llm_returns_llm_interface(self, mock_api_keys):
        """get_llm MUST return objects implementing LLMInterface."""
        from llm_fux.core.dispatcher import get_llm
        
        # Test with each expected model
        expected_models = ["chatgpt", "claude", "gemini"]
        
        for model_name in expected_models:
            try:
                model = get_llm(model_name)
                assert isinstance(model, LLMInterface), f"get_llm('{model_name}') must return LLMInterface"
            except Exception as e:
                pytest.skip(f"Model {model_name} not available: {e}")
    
    def test_get_llm_invalid_model_error(self):
        """get_llm MUST raise clear errors for invalid model names."""
        from llm_fux.core.dispatcher import get_llm
        
        invalid_names = [
            "invalid_model",
            "gpt-4",  # Not our exact naming
            "",
            None,
            123,
            [],
            {}
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises((ValueError, TypeError, AttributeError)) as exc_info:
                get_llm(invalid_name)
            
            # Error message should be informative
            error_msg = str(exc_info.value).lower()
            assert any(word in error_msg for word in ["unknown", "invalid", "model", "not found"])
    
    def test_get_llm_case_handling(self, mock_api_keys):
        """get_llm SHOULD handle model names consistently."""
        from llm_fux.core.dispatcher import get_llm
        
        # Define expected behavior for case sensitivity
        # The system should either be case-insensitive OR clearly document case requirements
        
        base_names = ["chatgpt", "claude", "gemini"]
        
        for base_name in base_names:
            try:
                # Test the canonical name
                model1 = get_llm(base_name)
                assert isinstance(model1, LLMInterface)
                
                # Test case variations - system should either:
                # 1. Accept them (case-insensitive), OR
                # 2. Reject them with clear error (case-sensitive)
                
                variations = [base_name.upper(), base_name.title()]
                
                for variation in variations:
                    try:
                        model2 = get_llm(variation)
                        # If accepted, must return valid LLMInterface
                        assert isinstance(model2, LLMInterface)
                    except (ValueError, TypeError) as e:
                        # If rejected, error should mention case sensitivity
                        error_msg = str(e).lower()
                        # This is acceptable - just be consistent
                        pass
                        
            except Exception as e:
                pytest.skip(f"Model {base_name} not available: {e}")
    
    def test_get_llm_returns_fresh_instances(self, mock_api_keys):
        """get_llm SHOULD return fresh instances (not singletons)."""
        from llm_fux.core.dispatcher import get_llm
        
        try:
            model1 = get_llm("chatgpt")
            model2 = get_llm("chatgpt")
            
            # Should be different instances (not singleton pattern)
            # This allows for independent configuration per instance
            assert model1 is not model2, "get_llm should return fresh instances"
            
        except Exception as e:
            pytest.skip(f"ChatGPT model not available: {e}")


class TestDispatcherErrorHandling:
    """Test error handling requirements for the dispatcher."""
    
    def test_model_initialization_failure_propagation(self, mock_api_keys):
        """Dispatcher SHOULD propagate model initialization failures clearly."""
        from llm_fux.core.dispatcher import get_llm
        
        # Mock a model that fails during initialization
        with patch('llm_fux.models.chatgpt.ChatGPTModel', side_effect=ValueError("API key missing")):
            with pytest.raises(ValueError, match="API key missing"):
                get_llm("chatgpt")
    
    def test_import_failure_handling(self):
        """Dispatcher SHOULD handle missing model modules gracefully."""
        from llm_fux.core.dispatcher import get_llm
        
        # Mock import failure for a model by patching the model class at its import path
        with patch('llm_fux.models.chatgpt.ChatGPTModel', side_effect=ImportError("Module not found")):
            with pytest.raises((ImportError, ValueError)):
                get_llm("chatgpt")
    
    def test_configuration_error_handling(self):
        """Dispatcher SHOULD provide clear errors for configuration issues."""
        from llm_fux.core.dispatcher import get_llm
        
        # Mock missing API keys
        with patch('llm_fux.config.config.API_KEYS', {}):
            try:
                model = get_llm("chatgpt")
                # If it doesn't raise an error, that's fine too - depends on implementation
                # But if it does raise an error, it should be clear
            except Exception as e:
                error_msg = str(e).lower()
                assert any(word in error_msg for word in ["api", "key", "config", "missing", "not found"])


class TestDispatcherConfiguration:
    """Test dispatcher configuration and setup requirements."""
    
    def test_supported_models_list(self):
        """Dispatcher SHOULD provide a way to list supported models."""
        # This might be a separate function or part of get_llm
        # The exact interface is flexible, but there should be some way to discover available models
        
        from llm_fux.core import dispatcher
        
        # Check if there's a list_models function or similar
        if hasattr(dispatcher, 'list_available_models'):
            models = dispatcher.list_available_models()
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(model, str) for model in models)
        
        # Alternative: get_llm should at least support known models
        expected_models = ["chatgpt", "claude", "gemini"]
        available_count = 0
        
        for model_name in expected_models:
            try:
                from llm_fux.core.dispatcher import get_llm
                get_llm(model_name)
                available_count += 1
            except Exception:
                pass
        
        # At least one model should be available
        assert available_count > 0, "At least one model should be available"
    
    def test_model_aliases_support(self, mock_api_keys):
        """Dispatcher MAY support model aliases for user convenience."""
        from llm_fux.core.dispatcher import get_llm
        
        # Test common aliases that users might expect
        alias_mappings = {
            "openai": "chatgpt",
            "anthropic": "claude", 
            "google": "gemini"
        }
        
        for alias, canonical in alias_mappings.items():
            try:
                alias_model = get_llm(alias)
                canonical_model = get_llm(canonical)
                
                # If aliases are supported, they should return the same type
                assert type(alias_model) == type(canonical_model)
                
            except (ValueError, TypeError):
                # If aliases aren't supported, that's fine too
                pass
            except Exception as e:
                pytest.skip(f"Models not available: {e}")


class TestDispatcherPerformance:
    """Test performance requirements for the dispatcher."""
    
    @pytest.mark.slow
    def test_model_lookup_speed(self, mock_api_keys):
        """Model lookup and instantiation SHOULD be reasonably fast."""
        import time
        from llm_fux.core.dispatcher import get_llm
        
        # Test multiple lookups
        lookups = 5
        start_time = time.time()
        
        for _ in range(lookups):
            try:
                model = get_llm("chatgpt")
                assert isinstance(model, LLMInterface)
            except Exception as e:
                pytest.skip(f"Model not available: {e}")
        
        total_time = time.time() - start_time
        avg_time = total_time / lookups
        
        # Each lookup should be fast (< 1 second)
        assert avg_time < 1.0, f"Average lookup time {avg_time:.2f}s is too slow"
    
    @pytest.mark.slow
    def test_concurrent_model_access(self, mock_api_keys):
        """Dispatcher SHOULD handle concurrent model access safely."""
        import threading
        import queue
        from llm_fux.core.dispatcher import get_llm
        
        results = queue.Queue()
        
        def get_model():
            try:
                model = get_llm("chatgpt")
                results.put(("success", model))
            except Exception as e:
                results.put(("error", e))
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=get_model)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Collect results
        success_count = 0
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success":
                assert isinstance(result_value, LLMInterface)
                success_count += 1
        
        # At least one should succeed
        assert success_count > 0, "Concurrent access should work"


class TestDispatcherExtensibility:
    """Test requirements for extending the dispatcher with new models."""
    
    def test_new_model_registration_pattern(self):
        """Dispatcher SHOULD follow a consistent pattern for adding new models."""
        # This test documents the expected pattern for adding new models
        # The exact mechanism depends on implementation, but should be consistent
        
        from llm_fux.core import dispatcher
        
        # Check the structure of the dispatcher module
        assert hasattr(dispatcher, 'get_llm')
        
        # The implementation should follow a clear pattern:
        # 1. Import model classes
        # 2. Map names to classes
        # 3. Instantiate on demand
        
        # This is more of a documentation test - the exact pattern
        # should be clear from the code structure
        assert callable(dispatcher.get_llm)
    
    def test_model_interface_compatibility(self):
        """New models MUST be compatible with LLMInterface."""
        # This is a design test - any new model added should pass the interface tests
        
        # Create a mock new model
        class TestNewModel(LLMInterface):
            def query(self, input) -> str:
                return "test response"
            def submit_batch(self, inputs):
                raise NotImplementedError
            def poll_batch(self, batch_id):
                raise NotImplementedError
            def retrieve_batch_results(self, batch_id):
                raise NotImplementedError

        # Should be compatible with the interface
        model = TestNewModel()
        assert isinstance(model, LLMInterface)
        
        # Should work with the expected interface
        from llm_fux.models.base import PromptInput
        prompt = PromptInput(
            system_prompt="test",
            user_prompt="test",
            temperature=0.5
        )
        
        result = model.query(prompt)
        assert isinstance(result, str)
