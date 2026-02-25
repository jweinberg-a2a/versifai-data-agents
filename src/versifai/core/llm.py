"""
LLM client with multi-provider support via LiteLLM.

Provides a unified interface for tool-use conversations across Anthropic Claude,
OpenAI, and any other provider supported by LiteLLM. Handles retries, token
tracking, and provider-specific optimizations (e.g., Anthropic prompt caching).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import litellm

logger = logging.getLogger("versifai.llm")

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider.

    Wraps the raw provider response into a consistent shape so the rest
    of the framework doesn't need to care which provider was used.
    """

    content: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: Any = None

    @property
    def input_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


class LLMClient:
    """
    Multi-provider LLM client for tool-use conversations.

    Uses `LiteLLM <https://docs.litellm.ai/>`_ under the hood to support
    Anthropic Claude, OpenAI, Azure, Bedrock, Vertex, and 100+ other providers
    through a single unified API.

    Provider is inferred from the model string:

    - ``"claude-sonnet-4-6"`` → Anthropic
    - ``"gpt-4o"`` → OpenAI
    - ``"bedrock/anthropic.claude-3-5-sonnet"`` → AWS Bedrock

    API keys are resolved from environment variables automatically by LiteLLM
    (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, etc.) or can be passed explicitly.

    Example::

        from versifai.core import LLMClient

        # Anthropic Claude (default)
        llm = LLMClient()

        # OpenAI
        llm = LLMClient(model="gpt-4o")

        # Explicit key
        llm = LLMClient(model="claude-sonnet-4-6", api_key="sk-...")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        api_key: str | None = None,
        api_base: str | None = None,
        retry_attempts: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._api_base = api_base
        self._retry_attempts = retry_attempts
        self._retry_base_delay = retry_base_delay

        # Usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_creation_tokens = 0
        self._call_count = 0

    @property
    def _is_anthropic(self) -> bool:
        return "claude" in self._model.lower()

    # ------------------------------------------------------------------
    # Main send method
    # ------------------------------------------------------------------

    def send(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        """
        Send a message with tool definitions and return the response.

        For Anthropic models, applies prompt caching on the system prompt
        and tool definitions to reduce token costs.

        Parameters
        ----------
        messages : list[dict]
            Conversation history in OpenAI message format.
        system : str
            System prompt text.
        tools : list[dict]
            Tool definitions (Anthropic format — auto-converted for other
            providers by LiteLLM).

        Returns
        -------
        LLMResponse
            Normalized response with content blocks and usage stats.
        """
        # Build kwargs for litellm.completion
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "tools": tools or None,
        }

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        # Provider-specific optimizations
        if self._is_anthropic:
            # Anthropic prompt caching: mark system + last tool for caching
            kwargs["messages"] = [{"role": "system", "content": system}] + messages
            if tools:
                # LiteLLM passes cache_control through for Anthropic
                cached_tools = list(tools)
                last_tool = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}
                cached_tools[-1] = last_tool
                kwargs["tools"] = cached_tools
        else:
            # OpenAI and others: system goes in messages
            kwargs["messages"] = [{"role": "system", "content": system}] + messages

        # Retry loop with exponential backoff
        last_error: Optional[Exception] = None

        for attempt in range(self._retry_attempts):
            try:
                response = litellm.completion(**kwargs)
                self._call_count += 1

                # Track usage
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    self._total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self._total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
                    self._total_cache_read_tokens += getattr(
                        usage, "cache_read_input_tokens", 0
                    ) or 0
                    self._total_cache_creation_tokens += getattr(
                        usage, "cache_creation_input_tokens", 0
                    ) or 0

                return self._normalize_response(response)

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Retry on rate limits, connection errors, server errors
                if any(kw in err_str for kw in ("rate", "429", "500", "502", "503", "connection")):
                    delay = self._retry_base_delay * (2**attempt)
                    logger.warning(
                        f"LLM error (attempt {attempt + 1}/{self._retry_attempts}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"LLM call failed after {self._retry_attempts} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Response normalization
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> LLMResponse:
        """Convert a LiteLLM response to our normalized LLMResponse format."""
        content_blocks: list[dict] = []
        stop_reason = ""

        choice = response.choices[0] if response.choices else None
        if choice:
            msg = choice.message
            stop_reason = choice.finish_reason or ""

            # Text content
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})

            # Tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    import json

                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}

                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": args,
                    })

        usage_dict = {}
        if hasattr(response, "usage") and response.usage:
            usage_dict = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            }

        # Map stop reasons to a consistent vocabulary
        if stop_reason in ("tool_calls", "tool_use"):
            stop_reason = "tool_use"
        elif stop_reason in ("stop", "end_turn"):
            stop_reason = "end_turn"

        return LLMResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            usage=usage_dict,
            raw=response,
        )

    # ------------------------------------------------------------------
    # Action extraction — works on normalized LLMResponse
    # ------------------------------------------------------------------

    @staticmethod
    def extract_actions(response: LLMResponse | Any) -> list[dict]:
        """
        Parse a response into a list of actions.

        Each action is one of:
          - ``{"type": "text", "text": "..."}``
          - ``{"type": "tool_use", "id": "...", "name": "...", "input": {...}}``

        Accepts both our ``LLMResponse`` and raw Anthropic ``Message`` objects
        for backward compatibility.
        """
        # Handle our normalized LLMResponse
        if isinstance(response, LLMResponse):
            return list(response.content)

        # Backward compat: raw Anthropic Message objects
        actions = []
        for block in response.content:
            if block.type == "text":
                actions.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                actions.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return actions

    @staticmethod
    def build_tool_result_message(
        tool_use_id: str, result: str, is_error: bool = False,
    ) -> dict:
        """Build a tool_result message block for the conversation."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                    "is_error": is_error,
                }
            ],
        }

    @property
    def usage_summary(self) -> dict:
        return {
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "cache_read_tokens": self._total_cache_read_tokens,
            "cache_creation_tokens": self._total_cache_creation_tokens,
        }
