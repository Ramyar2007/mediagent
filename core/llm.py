# mediagent/core/llm.py
"""
Production-grade LLM client wrapper for MediAgent.
Handles text and multimodal (vision) completions against local Qwen model.
Implements retry logic, error handling, response parsing, and OpenAI-compatible API calls.
"""

import logging
import time
import re
from typing import Any, Dict, List, Optional, Union

import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Lightweight, framework-agnostic LLM client wrapping OpenAI Python SDK.
    Designed for local inference endpoints (vLLM, Ollama, TensorRT-LLM) 
    running at http://localhost:8000/v1 with model path "/model".
    """

    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MODEL = "/model"
    DEFAULT_API_KEY = "none"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout: float = 120.0,
        temperature: float = 0.0
    ):
        self.model = model
        self.max_retries = max_retries
        self.default_temperature = temperature
        self.timeout = timeout

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=self.DEFAULT_API_KEY,
            timeout=timeout
        )
        logger.info(f"LLMClient initialized | Model: {self.model} | Endpoint: {base_url}")

    # ─────────────────────────────────────────────────────────────────────────
    # CORE GENERATION METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        force_json: bool = False
    ) -> Dict[str, Any]:
        """
        Send a text-only completion request to the LLM.
        Returns standardized response dict with content, usage, success flag, and error.
        """
        messages = self._build_messages(system_prompt, prompt)
        response_format = {"type": "json_object"} if force_json else None
        return self._execute_with_retry(
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            call_type="TEXT"
        )

    def generate_vision(
        self,
        base64_image: str,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send a multimodal completion request with a base64 encoded medical image.
        Automatically detects image MIME type and formats per OpenAI vision spec.
        """
        img_url = self._format_image_url(base64_image)
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]
        messages = self._build_messages(system_prompt, user_content)
        return self._execute_with_retry(
            messages=messages,
            temperature=temperature,
            response_format=None,
            call_type="VISION",
            max_tokens=max_tokens
        )

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        system_prompt: str,
        user_content: Union[str, List[Dict]]
    ) -> List[Dict]:
        """Construct OpenAI-compatible message array."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(user_content, str):
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_content})
        return messages

    def _format_image_url(self, base64_data: str) -> str:
        """Normalize base64 image data into OpenAI vision-compatible URL format."""
        if base64_data.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/jpg;base64,")):
            return base64_data
        # Default to JPEG if no MIME prefix is present
        return f"data:image/jpeg;base64,{base64_data}"

    def _attempt_call(
        self,
        messages: List[Dict],
        temperature: Optional[float],
        response_format: Optional[Dict],
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a single API call with the OpenAI client."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        
        usage = response.usage
        return {
            "success": True,
            "content": content,
            "raw_response": response,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "model": response.model,
            "error": None
        }

    def _execute_with_retry(
        self,
        messages: List[Dict],
        temperature: Optional[float],
        response_format: Optional[Dict],
        call_type: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retry wrapper with exponential backoff for robust local inference."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._attempt_call(messages, temperature, response_format, max_tokens)
                if result["success"]:
                    logger.debug(f"{call_type} generation successful on attempt {attempt}")
                    return result
            except Exception as e:
                last_error = str(e)
                logger.warning(f"{call_type} generation failed on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    backoff = min(2 ** attempt, 16)
                    logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)

        logger.error(f"{call_type} generation failed permanently after {self.max_retries} attempts.")
        return {
            "success": False,
            "content": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": self.model,
            "error": last_error or f"{call_type} endpoint unreachable or max retries exceeded."
        }

    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE PARSING UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
        """
        Safely extract JSON from LLM output, stripping markdown formatting
        and handling partial/comma-separated JSON arrays if necessary.
        """
        if not content:
            return None
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
            # First try direct JSON decode
            return LLMClient._safe_json_decode(cleaned)
        except Exception:
            logger.debug("Direct JSON extraction failed. Attempting fallback parsing...")
            return LLMClient._fallback_json_parse(cleaned)

    @staticmethod
    def _safe_json_decode(text: str):
        """Import json lazily and decode, raising cleanly on failure."""
        import json
        return json.loads(text)

    @staticmethod
    def _fallback_json_parse(text: str) -> Optional[Dict[str, Any]]:
        """
        Fallback: scan for first valid JSON object or array in the text.
        Handles cases where the LLM adds conversational padding.
        """
        import json
        brace_depth = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == "{":
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0 and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
        # Try array fallback
        bracket_depth = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == "[":
                if bracket_depth == 0:
                    start_idx = i
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
                if bracket_depth == 0 and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
        return None
