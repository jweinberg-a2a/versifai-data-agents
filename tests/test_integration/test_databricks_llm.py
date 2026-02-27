"""Integration tests for Databricks-hosted LLM support.

Verifies that tool calling works end-to-end through a Databricks
serving endpoint with ``databricks-claude-sonnet-4-6``.

Credentials are resolved from either set of env vars:
  - ``DATABRICKS_TOKEN`` + ``DATABRICKS_HOST`` (Databricks SDK names, typical in CI)
  - ``DATABRICKS_API_KEY`` + ``DATABRICKS_API_BASE`` (LiteLLM names)

Run::

    DATABRICKS_TOKEN=... DATABRICKS_HOST=... \\
        pytest tests/test_integration/test_databricks_llm.py -v -m integration

Override model::

    VERSIFAI_TEST_MODEL=databricks/other-model \\
        pytest tests/test_integration/test_databricks_llm.py -v -m integration
"""

from __future__ import annotations

import os

import pytest

from versifai.core.agent import BaseAgent
from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.tools.registry import ToolRegistry
from versifai.core.tools.save_note import SaveNoteTool
from versifai.story_agents.storyteller.tools.evaluate_evidence import EvaluateEvidenceTool

from .conftest import requires_databricks_llm

# ---------------------------------------------------------------------------
# Credential resolution — accept either SDK or LiteLLM env var names
# ---------------------------------------------------------------------------


def _resolve_databricks_creds() -> tuple[str | None, str | None]:
    """Return (api_key, api_base) from whichever env vars are set.

    ``DATABRICKS_API_BASE`` (LiteLLM convention) is used as-is — callers are
    expected to include ``/serving-endpoints`` in the URL.

    ``DATABRICKS_HOST`` (SDK convention) is the bare workspace URL.  The
    ``/serving-endpoints`` suffix is appended by ``LLMClient._normalize_api_base``
    at construction time, so we pass the raw host here.
    """
    api_key = os.environ.get("DATABRICKS_API_KEY") or os.environ.get("DATABRICKS_TOKEN")
    api_base = os.environ.get("DATABRICKS_API_BASE") or os.environ.get("DATABRICKS_HOST")
    return api_key or None, api_base or None


# ---------------------------------------------------------------------------
# Minimal test agent for Databricks endpoints
# ---------------------------------------------------------------------------


class DatabricksReviewerAgent(BaseAgent):
    """Lightweight agent for testing Databricks-hosted LLM endpoints."""

    _SYSTEM_PROMPT = (
        "You are a helpful assistant. When asked to evaluate evidence, "
        "use the evaluate_evidence tool. When asked to record a note, "
        "use the save_note tool. Be concise."
    )

    def __init__(self, tools: list, model: str | None = None) -> None:
        display = AgentDisplay()
        memory = AgentMemory()
        model = model or os.environ.get(
            "VERSIFAI_TEST_MODEL",
            "databricks/databricks-claude-sonnet-4-6",
        )
        api_key, api_base = _resolve_databricks_creds()
        llm = LLMClient(model=model, api_key=api_key, api_base=api_base)
        registry = ToolRegistry()

        super().__init__(display=display, memory=memory, llm=llm, registry=registry)

        for tool in tools:
            registry.register(tool)

        self._system_prompt = self._SYSTEM_PROMPT

    def review(self, prompt: str, max_turns: int = 8) -> str:
        """Run a single review phase; return the agent's final text."""
        return self._run_phase(prompt, max_turns=max_turns)


# ---------------------------------------------------------------------------
# Unit-level sanity checks (no API key needed)
# ---------------------------------------------------------------------------


class TestDatabricksProviderDetection:
    """Verify Databricks models are not treated as direct Anthropic."""

    def test_no_anthropic_headers_for_databricks_claude(self):
        llm = LLMClient(model="databricks/databricks-claude-sonnet-4-6")
        assert llm._is_direct_anthropic is False

    def test_no_anthropic_headers_for_databricks_generic(self):
        llm = LLMClient(model="databricks/databricks-meta-llama-3-1-70b-instruct")
        assert llm._is_direct_anthropic is False


# ---------------------------------------------------------------------------
# Integration tests — require Databricks credentials (SDK or LiteLLM names)
# ---------------------------------------------------------------------------


@requires_databricks_llm
@pytest.mark.integration
class TestDatabricksToolCalling:
    """Verify tool calling works through Databricks serving endpoints."""

    def test_tool_call_round_trip(self, tmp_path):
        """Agent calls save_note through Databricks endpoint and returns."""
        notes_path = str(tmp_path / "notes")
        os.makedirs(notes_path, exist_ok=True)

        tools = [
            SaveNoteTool(notes_path=notes_path),
            EvaluateEvidenceTool(),
        ]

        agent = DatabricksReviewerAgent(tools=tools)

        response = agent.review(
            "Use the save_note tool to record a note with theme_id='test' "
            "and content='Databricks integration test successful'. "
            "Then confirm you saved it.",
            max_turns=6,
        )

        # Verify the agent completed and the note was actually saved
        assert response, "Agent returned empty response"
        note_file = os.path.join(notes_path, "test_notes.txt")
        assert os.path.isfile(note_file), "save_note tool should have created a note file"

    def test_evaluate_evidence_tool_call(self, tmp_path):
        """Agent calls evaluate_evidence through Databricks endpoint."""
        notes_path = str(tmp_path / "notes")
        os.makedirs(notes_path, exist_ok=True)

        tools = [
            SaveNoteTool(notes_path=notes_path),
            EvaluateEvidenceTool(),
        ]

        agent = DatabricksReviewerAgent(tools=tools)

        response = agent.review(
            "Evaluate the following evidence using the evaluate_evidence tool:\n"
            "- p_value: 0.001\n"
            "- effect_size: 0.8\n"
            "- sample_size: 500\n"
            "- finding_text: 'Strong correlation between X and Y'\n\n"
            "Report the evidence tier.",
            max_turns=6,
        )

        assert response, "Agent returned empty response"
        lower = response.lower()
        assert any(
            kw in lower for kw in ["strong", "definitive", "tier", "evidence", "significant"]
        ), f"Agent did not classify evidence. Response: {response[:500]}"
