"""Tests for LLMClient provider detection and configuration."""

from __future__ import annotations

from versifai.core.llm import LLMClient

# ---------------------------------------------------------------------------
# Provider detection — _is_direct_anthropic
# ---------------------------------------------------------------------------


class TestIsDirectAnthropic:
    """Verify _is_direct_anthropic correctly identifies provider routes."""

    def test_direct_anthropic_default(self):
        llm = LLMClient(model="claude-sonnet-4-6")
        assert llm._is_direct_anthropic is True

    def test_direct_anthropic_haiku(self):
        llm = LLMClient(model="claude-haiku-4-5-20251001")
        assert llm._is_direct_anthropic is True

    def test_direct_anthropic_opus(self):
        llm = LLMClient(model="claude-opus-4-6")
        assert llm._is_direct_anthropic is True

    def test_databricks_claude_not_direct(self):
        """Databricks-hosted Claude should NOT trigger Anthropic-specific features."""
        llm = LLMClient(model="databricks/databricks-claude-sonnet-4-6")
        assert llm._is_direct_anthropic is False

    def test_bedrock_claude_not_direct(self):
        llm = LLMClient(model="bedrock/anthropic.claude-3-5-sonnet")
        assert llm._is_direct_anthropic is False

    def test_azure_not_anthropic(self):
        llm = LLMClient(model="azure/gpt-4o")
        assert llm._is_direct_anthropic is False

    def test_vertex_ai_claude_not_direct(self):
        llm = LLMClient(model="vertex_ai/claude-3-5-sonnet")
        assert llm._is_direct_anthropic is False

    def test_vertex_ai_beta_claude_not_direct(self):
        llm = LLMClient(model="vertex_ai_beta/claude-3-5-sonnet")
        assert llm._is_direct_anthropic is False

    def test_openai_model_not_anthropic(self):
        llm = LLMClient(model="gpt-4o")
        assert llm._is_direct_anthropic is False

    def test_gemini_model_not_anthropic(self):
        llm = LLMClient(model="gemini/gemini-1.5-pro")
        assert llm._is_direct_anthropic is False

    def test_ollama_claude_not_direct(self):
        llm = LLMClient(model="ollama/claude-something")
        assert llm._is_direct_anthropic is False

    def test_openrouter_claude_not_direct(self):
        llm = LLMClient(model="openrouter/anthropic/claude-3-opus")
        assert llm._is_direct_anthropic is False

    def test_sagemaker_claude_not_direct(self):
        llm = LLMClient(model="sagemaker/claude-v2")
        assert llm._is_direct_anthropic is False


# ---------------------------------------------------------------------------
# LLMClient construction
# ---------------------------------------------------------------------------


class TestLLMClientInit:
    """Verify LLMClient construction from various inputs."""

    def test_default_model(self):
        llm = LLMClient()
        assert llm._model == "claude-sonnet-4-6"
        assert llm._max_tokens == 8192

    def test_custom_databricks_model(self):
        llm = LLMClient(model="databricks/databricks-claude-sonnet-4-6")
        assert llm._model == "databricks/databricks-claude-sonnet-4-6"

    def test_api_base_stored(self):
        llm = LLMClient(
            model="databricks/databricks-claude-sonnet-4-6",
            api_base="https://my-workspace.databricks.com/serving-endpoints",
        )
        assert llm._api_base == "https://my-workspace.databricks.com/serving-endpoints"

    def test_extended_context_disabled(self):
        llm = LLMClient(extended_context=False)
        assert llm._extended_context is False


# ---------------------------------------------------------------------------
# Databricks api_base normalization
# ---------------------------------------------------------------------------


class TestNormalizeApiBase:
    """Verify _normalize_api_base appends /serving-endpoints for Databricks."""

    def test_databricks_bare_host_gets_suffix(self):
        llm = LLMClient(
            model="databricks/databricks-claude-sonnet-4-6",
            api_base="https://adb-123.14.azuredatabricks.net",
        )
        assert llm._api_base == "https://adb-123.14.azuredatabricks.net/serving-endpoints"

    def test_databricks_host_with_trailing_slash(self):
        llm = LLMClient(
            model="databricks/databricks-claude-sonnet-4-6",
            api_base="https://adb-123.14.azuredatabricks.net/",
        )
        assert llm._api_base == "https://adb-123.14.azuredatabricks.net/serving-endpoints"

    def test_databricks_already_has_suffix(self):
        """Don't double-append if api_base already includes /serving-endpoints."""
        llm = LLMClient(
            model="databricks/databricks-claude-sonnet-4-6",
            api_base="https://adb-123.14.azuredatabricks.net/serving-endpoints",
        )
        assert llm._api_base == "https://adb-123.14.azuredatabricks.net/serving-endpoints"

    def test_non_databricks_model_untouched(self):
        """Non-Databricks models should not get /serving-endpoints appended."""
        llm = LLMClient(
            model="claude-sonnet-4-6",
            api_base="https://my-proxy.example.com",
        )
        assert llm._api_base == "https://my-proxy.example.com"

    def test_none_api_base_stays_none(self):
        llm = LLMClient(model="databricks/databricks-claude-sonnet-4-6")
        assert llm._api_base is None
