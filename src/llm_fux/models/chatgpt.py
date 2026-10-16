import logging
import os
import time
import json
import tempfile
from typing import Optional, Dict, Iterator
from openai import OpenAI
from llm_fux.models.base import LLMInterface, LLMResponse, PromptInput, TokenUsage, BatchResult
from llm_fux.config.config import DEFAULT_MODELS, get_timeout, get_max_tokens
from llm_fux.utils.text_utils import clean_code_blocks

logger = logging.getLogger(__name__)


class ChatGPTModel(LLMInterface):
    """
    ChatGPTModel wraps OpenAI's API behind a uniform interface.

    GPT-5+ models use the Responses API with reasoning effort control.
    Older models use the Chat Completions API with temperature.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")
        timeout = get_timeout()
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model_name = model_name or DEFAULT_MODELS["openai"]

    def _is_reasoning_model(self, model: str) -> bool:
        return model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4")

    def query(self, input: PromptInput) -> LLMResponse:
        model = input.model_name or self.model_name
        max_tokens = getattr(input, "max_tokens", None) or get_max_tokens()

        if self._is_reasoning_model(model):
            return self._query_responses_api(input, model, max_tokens)
        else:
            return self._query_chat_completions(input, model, max_tokens)

    def _query_responses_api(self, input: PromptInput, model: str, max_tokens: int) -> LLMResponse:
        """Use the Responses API for GPT-5+ / reasoning models.

        Runs in background mode and polls for completion to avoid HTTP timeouts
        on long reasoning tasks.
        """
        reasoning_effort = "high"
        extra_params = {"reasoning_effort": reasoning_effort}

        logger.info(
            "ChatGPT Responses API call (background) — model: %s, max_output_tokens: %s, reasoning_effort: %s",
            model, max_tokens, reasoning_effort,
        )

        # Create in background mode to avoid HTTP timeout on long reasoning
        response = self.client.responses.create(
            model=model,
            instructions=input.system_prompt,
            input=input.user_prompt,
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_tokens,
            background=True,
        )

        # Poll until complete
        poll_interval = 5  # seconds
        max_wait = get_timeout() or 1800  # default 30 min if no timeout configured
        elapsed = 0
        while response.status in ("queued", "in_progress"):
            if elapsed >= max_wait:
                raise TimeoutError(
                    f"GPT response still {response.status} after {elapsed}s (id: {response.id})"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
            response = self.client.responses.retrieve(response.id)
            if elapsed % 30 == 0:
                logger.info("Waiting for GPT response... %ds elapsed (status: %s)", elapsed, response.status)

        if response.status != "completed":
            raise RuntimeError(f"GPT response failed with status: {response.status} (id: {response.id})")

        # Extract text from response output items
        raw_response = ""
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        raw_response += content.text
        raw_response = raw_response.strip()
        cleaned = clean_code_blocks(raw_response)

        # Extract token usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                total_tokens=(
                    (getattr(response.usage, "input_tokens", 0) or 0)
                    + (getattr(response.usage, "output_tokens", 0) or 0)
                ),
                extra={
                    k: v
                    for k, v in {
                        "reasoning_tokens": getattr(response.usage, "output_tokens_details", None)
                        and getattr(response.usage.output_tokens_details, "reasoning_tokens", None),
                    }.items()
                    if v is not None
                },
            )
            logger.info(
                "Token usage — input: %d, output: %d, total: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

        return LLMResponse(text=cleaned, usage=usage, model_id=model, extra_params=extra_params)

    def _query_chat_completions(self, input: PromptInput, model: str, max_tokens: int) -> LLMResponse:
        """Use the Chat Completions API for older models."""
        logger.info(
            "ChatGPT Chat Completions API call — model: %s, max_tokens: %s, temperature: %s",
            model, max_tokens, input.temperature,
        )

        messages = [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=input.temperature,
            max_completion_tokens=max_tokens,
        )

        raw_response = response.choices[0].message.content.strip()
        cleaned = clean_code_blocks(raw_response)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
            )
            logger.info(
                "Token usage — prompt: %d, completion: %d, total: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

        return LLMResponse(text=cleaned, usage=usage, model_id=model)

    def submit_batch(self, inputs: Dict[str, PromptInput]) -> str:
        requests_jsonl = []
        for custom_id, prompt_input in inputs.items():
            model = prompt_input.model_name or self.model_name
            max_tokens = getattr(prompt_input, "max_tokens", None) or get_max_tokens()

            req = {
                "custom_id": str(custom_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt_input.system_prompt},
                        {"role": "user", "content": prompt_input.user_prompt}
                    ],
                }
            }

            if self._is_reasoning_model(model):
                req["body"]["reasoning_effort"] = "high"
            else:
                req["body"]["temperature"] = prompt_input.temperature

            if max_tokens:
                req["body"]["max_completion_tokens"] = max_tokens

            requests_jsonl.append(json.dumps(req))

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as f:
            f.write("\n".join(requests_jsonl) + "\n")
            f.seek(0)

        with open(f.name, "rb") as file_obj:
            batch_input_file = self.client.files.create(
                file=file_obj,
                purpose="batch"
            )

        os.unlink(f.name)

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logger.info("Submitted OpenAI batch job: %s", batch.id)
        return batch.id

    def poll_batch(self, batch_id: str) -> str:
        batch = self.client.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch.status
        return "in_progress"

    def retrieve_batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        batch = self.client.batches.retrieve(batch_id)

        if batch.output_file_id:
            result_file_response = self.client.files.content(batch.output_file_id)
            for line in result_file_response.text.splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    custom_id = data["custom_id"]

                    response = data.get("response")
                    if response is None or response.get("status_code") != 200:
                        yield BatchResult(custom_id=custom_id, error=str(response or "null response"))
                        continue

                    resp_body = response["body"]
                    text = resp_body["choices"][0]["message"]["content"] or ""
                    cleaned = clean_code_blocks(text)

                    usage_data = resp_body.get("usage", {})
                    usage = TokenUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0)
                    )

                    resp = LLMResponse(text=cleaned, usage=usage, model_id=self.model_name)
                    yield BatchResult(custom_id=custom_id, response=resp)
                except Exception as e:
                    logger.error("Failed to parse batch result line: %s — %s", e, line[:200])
                    cid = "unknown"
                    try:
                        cid = json.loads(line).get("custom_id", "unknown")
                    except Exception:
                        pass
                    yield BatchResult(custom_id=cid, error=f"Parse error: {e}")

        if batch.error_file_id:
            error_file_response = self.client.files.content(batch.error_file_id)
            for line in error_file_response.text.splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    yield BatchResult(custom_id=data["custom_id"], error=str(data.get("error", "Unknown error")))
                except Exception as e:
                    logger.error("Failed to parse batch error line: %s", e)

    def supports_batch_api(self) -> bool:
        return True
