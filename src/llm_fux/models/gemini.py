# models/gemini.py

import logging
import os
import time
import json
import tempfile
from typing import Optional, Dict, Iterator

from google import genai
from google.genai import types

from llm_fux.models.base import LLMInterface, LLMResponse, PromptInput, TokenUsage, BatchResult
from llm_fux.config.config import DEFAULT_MODELS, get_timeout, get_max_tokens
from llm_fux.utils.text_utils import clean_code_blocks

logger = logging.getLogger(__name__)


class GeminiModel(LLMInterface):
    """
    Wrapper for Google's Gemini models via the Google Gen AI SDK (google-genai).
    Reads GOOGLE_API_KEY from the environment, supports per-call model overrides,
    temperature control, and optional max_tokens.

    Uses the native ``system_instruction`` parameter so the system prompt receives
    elevated priority rather than being concatenated with user content.
    """

    def __init__(self, model_name: Optional[str] = None):
        # Load API key
        self.api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in the environment.")

        # Initialize the client explicitly with the API key (avoid relying on ambient config)
        self.client = genai.Client(api_key=self.api_key)
        
        # Choose default model from settings if not overridden
        self.model_name = model_name or DEFAULT_MODELS["google"]

    def query(self, input: PromptInput) -> LLMResponse:
        """
        Sends a prompt to Gemini via generate_content, using system_instruction
        for the system prompt and contents for the user prompt.

        Parameters:
            input (PromptInput): 
                - system_prompt (str): High-level instructions (sent via system_instruction).
                - user_prompt   (str): The combined prompt (format intro, MusicXML data, guides, question).
                - temperature   (float): Sampling temperature.
                - max_tokens    (Optional[int]): Maximum tokens for the response.
                - model_name    (Optional[str]): Override for the model to use.

        Returns:
            LLMResponse: Generated text and token usage metadata.
        """
        # Determine which model to call
        model_id = input.model_name or self.model_name
        
        # Fix model format - remove "models/" prefix if it exists
        if model_id.startswith("models/"):
            model_id = model_id[7:]  # Strip "models/" prefix

        # Get max_tokens from input, or fall back to config default
        max_tokens = getattr(input, "max_tokens", None) or get_max_tokens()

        # Build generation config with native system_instruction
        config = types.GenerateContentConfig(
            system_instruction=input.system_prompt,
            temperature=input.temperature,
            response_mime_type="text/plain",
            max_output_tokens=max_tokens,
        )
        
        logger.info(
            "Gemini API call — model: %s, max_tokens: %s, temperature: %s",
            model_id, max_tokens, input.temperature,
        )

        # Generate content using the client API
        response = self.client.models.generate_content(
            model=model_id,
            contents=input.user_prompt,
            config=config,
        )

        # Clean response: strip whitespace and remove any code block delimiters
        raw_response = response.text.strip()
        cleaned = clean_code_blocks(raw_response)

        # Extract token usage from response metadata
        usage = None
        meta = getattr(response, "usage_metadata", None)
        if meta:
            usage = TokenUsage(
                prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                total_tokens=getattr(meta, "total_token_count", 0) or 0,
                extra={
                    k: v
                    for k, v in {
                        "thoughts_token_count": getattr(meta, "thoughts_token_count", None),
                        "cached_content_token_count": getattr(meta, "cached_content_token_count", None),
                    }.items()
                    if v is not None
                },
            )
            logger.info(
                "Token usage — prompt: %d, completion: %d, total: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

        return LLMResponse(text=cleaned, usage=usage, model_id=model_id)

    def submit_batch(self, inputs: Dict[str, PromptInput]) -> str:
        requests_jsonl = []
        model = self.model_name
        for custom_id, prompt_input in inputs.items():
            model = prompt_input.model_name or self.model_name
            if model.startswith("models/"):
                model = model[7:]
            max_tokens = getattr(prompt_input, "max_tokens", None) or get_max_tokens()

            req = {
                "key": str(custom_id),
                "request": {
                    "contents": [{"parts": [{"text": prompt_input.user_prompt}], "role": "user"}],
                    "systemInstruction": {
                        "parts": [{"text": prompt_input.system_prompt}]
                    },
                    "generationConfig": {
                        "temperature": prompt_input.temperature,
                        "maxOutputTokens": max_tokens,
                    }
                }
            }
            requests_jsonl.append(json.dumps(req))

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as f:
            f.write("\n".join(requests_jsonl) + "\n")
            f.seek(0)

            uploaded_file = self.client.files.upload(
                file=f.name,
                config=types.UploadFileConfig(display_name=f"batch-{int(time.time())}", mime_type='application/jsonl')
            )

        os.unlink(f.name)

        batch_job = self.client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={'display_name': f"batch-job-{int(time.time())}"}
        )
        logger.info("Submitted Gemini batch job: %s", batch_job.name)
        return batch_job.name

    def poll_batch(self, batch_id: str) -> str:
        batch = self.client.batches.get(name=batch_id)
        if batch.state.name == "JOB_STATE_SUCCEEDED":
            return "completed"
        elif batch.state.name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            return "failed"
        return "in_progress"

    def retrieve_batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        batch = self.client.batches.get(name=batch_id)
        if hasattr(batch.dest, "file_name") and batch.dest.file_name:
            file_content_bytes = self.client.files.download(file=batch.dest.file_name)
            file_content = file_content_bytes.decode('utf-8')
            for line in file_content.splitlines():
                if not line.strip(): continue
                data = json.loads(line)
                custom_id = data.get("key", "Unknown")

                if "response" in data and data["response"]:
                    try:
                        text = ""
                        for part in data["response"]["candidates"][0]["content"]["parts"]:
                            if "text" in part:
                                text += part["text"]

                        cleaned = clean_code_blocks(text)

                        meta = data["response"].get("usageMetadata", {})
                        usage = TokenUsage(
                            prompt_tokens=meta.get("promptTokenCount", 0),
                            completion_tokens=meta.get("candidatesTokenCount", 0),
                            total_tokens=meta.get("totalTokenCount", 0)
                        )

                        yield BatchResult(
                            custom_id=custom_id,
                            response=LLMResponse(text=cleaned, usage=usage, model_id=self.model_name)
                        )
                    except Exception as e:
                        yield BatchResult(custom_id=custom_id, error=f"Parse error: {str(e)}")
                elif "error" in data:
                    yield BatchResult(custom_id=custom_id, error=str(data["error"]))
                else:
                    yield BatchResult(custom_id=custom_id, error="Unknown error in response")

    def supports_batch_api(self) -> bool:
        return True