# models/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Iterator


@dataclass
class PromptInput:
    """
    Encapsulates all parameters for a single LLM request.
    """
    system_prompt: str            # The system‐level instructions
    user_prompt: str              # The body: format intro + MusicXML data + guides + question
    temperature: float = 0.0      # Sampling temperature
    model_name: Optional[str] = None   # Override the default model if provided
    max_tokens: Optional[int] = None   # (Optional) token limit for the response

    def __post_init__(self):
        # Validate required string fields
        if not isinstance(self.system_prompt, str) or not isinstance(self.user_prompt, str):
            raise TypeError("system_prompt and user_prompt must be strings")

        # Validate temperature
        if not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number between 0.0 and 1.0")
        if not (0.0 <= float(self.temperature) <= 1.0):
            raise ValueError("temperature must be between 0.0 and 1.0")

        # Validate max_tokens if provided
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer if provided")


@dataclass
class TokenUsage:
    """Token counts returned by an LLM API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class LLMResponse:
    """Wraps the text response from an LLM together with token usage metadata."""
    text: str
    usage: Optional[TokenUsage] = None
    model_id: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BatchResult:
    """Represents a single result from a batch process."""
    def __init__(self, custom_id: str, response: Optional[LLMResponse] = None, error: Optional[str] = None):
        self.custom_id = custom_id
        self.response = response
        self.error = error

class LLMInterface(ABC):
    """
    Abstract base class for all LLM wrappers.
    Subclasses must implement the `query` method
    using their respective API format.
    """

    @abstractmethod
    def query(self, input: PromptInput) -> LLMResponse:
        """
        Send a prompt to the LLM and return the response with metadata.
        
        Parameters:
            input (PromptInput): Contains system/user prompt and parameters.
        
        Returns:
            LLMResponse: The LLM's generated response text and token usage.
        """
        pass

    @abstractmethod
    def submit_batch(self, inputs: Dict[str, PromptInput]) -> str:
        """
        Submit a batch of prompts for processing.

        Parameters:
            inputs: A dictionary mapping custom IDs to PromptInput instances.

        Returns:
            str: The batch ID.
        """
        pass

    @abstractmethod
    def poll_batch(self, batch_id: str) -> str:
        """
        Poll the status of a batch process.

        Parameters:
            batch_id: The ID of the batch to poll.

        Returns:
            str: The current status of the batch ('completed', 'failed', 'in_progress', etc.)
        """
        pass

    @abstractmethod
    def retrieve_batch_results(self, batch_id: str) -> 'Iterator[BatchResult]':
        """
        Retrieve the results of a completed batch.

        Parameters:
            batch_id: The ID of the completed batch.

        Returns:
            Iterator[BatchResult]: An iterator yielding the results.
        """
        pass

    def supports_batch_api(self) -> bool:
        """Check if provider has batch API"""
        return False
