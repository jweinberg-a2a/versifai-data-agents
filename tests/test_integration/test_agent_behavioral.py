"""Agent-level integration tests that invoke the actual LLM.

These tests verify that the agent ACTUALLY REASONS correctly about:
- Misattributed citations (CMS data attributed to wrong organization)
- Contradictory statistics (narrative claims vs. actual numbers)
- Confused sources (Medicare vs. Medicaid swapped)
- Statistical overclaiming (text says "GROUNDBREAKING" but p=0.73)

Unlike the tool-level unit tests, these tests run the full ReAct loop:
  LLM reasons → calls tools → reads results → reasons again → produces verdict

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.

Run integration tests only::

    pytest tests/test_integration/ -v -m integration

Skip integration tests::

    pytest tests/ -v -m "not integration"

Override model (default: claude-haiku-4-5-20251001)::

    VERSIFAI_TEST_MODEL=claude-sonnet-4-6 pytest tests/test_integration/ -v
"""

from __future__ import annotations

import os

import pytest

from versifai.core.agent import BaseAgent
from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.tools.registry import ToolRegistry

from .conftest import CMS_CITATION, requires_llm

# ---------------------------------------------------------------------------
# Minimal test agent — real LLM, real tools, focused prompt
# ---------------------------------------------------------------------------

class ReviewerAgent(BaseAgent):
    """Lightweight agent for integration testing.

    Uses a small set of real tools and a focused system prompt.
    The LLM makes real API calls and drives the ReAct loop.
    """

    _SYSTEM_PROMPT = """\
You are a rigorous fact-checker and evidence reviewer for data analysis reports.

Your job is to review findings, citations, and statistical claims for ACCURACY.

Rules:
1. Check that citations are attributed to the CORRECT source organization.
2. Verify that statistical claims MATCH the actual numbers provided.
3. Flag any misattributions, confused sources, or contradictory evidence.
4. Use the evaluate_evidence tool to classify evidence strength when p-values are provided.
5. Use the save_note tool to record any issues you find.

Critical domain knowledge:
- CMS (Centers for Medicare & Medicaid Services) publishes Medicare enrollment data.
  The CDC (Centers for Disease Control) is a DIFFERENT agency — it does NOT publish
  Medicare enrollment statistics.
- Medicare is federal health insurance for people 65+ (and some with disabilities).
  Approximately 65.7 million enrollees in 2023.
- Medicaid is a joint federal/state means-tested program for low-income individuals
  of ANY age. It is a DIFFERENT program from Medicare.
- These are common points of confusion — be alert for them.

When you find an error, clearly state what the error is, what the correct
information should be, and why it matters. Then stop — do NOT continue
searching for more issues after documenting the primary error."""

    def __init__(self, tools: list, model: str | None = None) -> None:
        display = AgentDisplay()
        memory = AgentMemory()
        model = model or os.environ.get(
            "VERSIFAI_TEST_MODEL", "claude-haiku-4-5-20251001"
        )
        llm = LLMClient(model=model)
        registry = ToolRegistry()

        super().__init__(
            display=display, memory=memory, llm=llm, registry=registry,
        )

        for tool in tools:
            registry.register(tool)

        self._system_prompt = self._SYSTEM_PROMPT

    def review(self, prompt: str, max_turns: int = 10) -> str:
        """Run a single review phase; return the agent's final text."""
        return self._run_phase(prompt, max_turns=max_turns)


# ---------------------------------------------------------------------------
# Helper — broad semantic check
# ---------------------------------------------------------------------------

def _response_contains_any(response: str, keywords: list[str]) -> bool:
    """Check if the response contains any of the given keywords (case-insensitive)."""
    lower = response.lower()
    return any(kw.lower() in lower for kw in keywords)


# ===================================================================
# Test class 1: Misattributed citation (CMS → CDC)
# ===================================================================

@requires_llm
@pytest.mark.integration
class TestAgentDetectsMisattribution:
    """The agent must catch when CMS Medicare data is attributed to the CDC."""

    def test_catches_cms_data_attributed_to_cdc(
        self, review_tools, findings_cms_misattributed_to_cdc,
    ):
        """AGENT BEHAVIORAL: CMS data wrongly cited as CDC → agent flags it.

        Real citation: CMS publishes Medicare Total Enrollment statistics.
        Error: The finding says "According to CDC Program Statistics..."

        The agent should identify that Medicare enrollment data comes from CMS,
        not the CDC, and flag the misattribution.
        """
        finding = findings_cms_misattributed_to_cdc[0]

        agent = ReviewerAgent(tools=review_tools)

        prompt = (
            "Review the following research finding for citation accuracy.\n\n"
            f"FINDING:\n"
            f"  Title: {finding['title']}\n"
            f"  Text: {finding['finding']}\n"
            f"  Attributed source: {finding['source']}\n"
            f"  Source URL: {finding['source_url']}\n\n"
            "Is the citation correctly attributed? The finding claims this "
            "data comes from the CDC. Evaluate whether that is accurate.\n\n"
            "Use save_note to document your assessment, then provide your verdict."
        )

        response = agent.review(prompt, max_turns=10)

        # The agent should identify CMS as the correct source
        assert _response_contains_any(response, [
            "CMS", "Centers for Medicare", "misattribut", "incorrect",
            "wrong source", "not CDC", "not the CDC", "should be CMS",
            "incorrectly attributed", "error", "inaccurate",
        ]), (
            f"Agent failed to detect CMS data misattributed to CDC.\n"
            f"Agent response:\n{response[:800]}"
        )


# ===================================================================
# Test class 2: Confused programs (Medicare vs. Medicaid)
# ===================================================================

@requires_llm
@pytest.mark.integration
class TestAgentDetectsProgramConfusion:
    """The agent must catch when Medicare and Medicaid are confused."""

    def test_catches_medicare_medicaid_swap(
        self, review_tools, findings_medicare_medicaid_confused,
    ):
        """AGENT BEHAVIORAL: 65.7M labeled as Medicaid seniors → agent flags it.

        Real fact: 65.7 million is the MEDICARE enrollment figure (all ages 65+
        and disabled). Medicaid is a different program for low-income individuals.

        The finding incorrectly calls this "Medicaid enrollment for seniors."
        """
        finding = findings_medicare_medicaid_confused[0]

        agent = ReviewerAgent(tools=review_tools)

        prompt = (
            "Review this finding for factual accuracy.\n\n"
            f"FINDING:\n"
            f"  Title: {finding['title']}\n"
            f"  Text: {finding['finding']}\n"
            f"  Source: {finding['source']}\n\n"
            "Check whether the program name and enrollment figure are consistent. "
            "Is 65.7 million the correct enrollment number for the program "
            "named in the finding?\n\n"
            "Use save_note to document any issues, then provide your verdict."
        )

        response = agent.review(prompt, max_turns=10)

        # The agent should flag the Medicare/Medicaid confusion
        assert _response_contains_any(response, [
            "Medicare", "not Medicaid", "confusion", "incorrect program",
            "mixed up", "wrong program", "should be Medicare",
            "65.7 million is Medicare", "Medicaid is", "means-tested",
            "different program", "error", "inaccurate",
        ]), (
            f"Agent failed to detect Medicare/Medicaid confusion.\n"
            f"Agent response:\n{response[:800]}"
        )


# ===================================================================
# Test class 3: Contradictory statistics (overclaiming weak results)
# ===================================================================

@requires_llm
@pytest.mark.integration
class TestAgentDetectsStatisticalOverclaiming:
    """The agent must catch when narrative text wildly overclaims weak stats."""

    def test_catches_inflated_narrative_with_p_073(
        self, review_tools, findings_inflated_text_weak_stats,
    ):
        """AGENT BEHAVIORAL: Text says "OVERWHELMING PROOF" but p=0.73 → agent flags it.

        The finding's narrative uses superlatives ("DEFINITIVE", "PROVES",
        "IRREFUTABLE") but the actual p-value is 0.73 (far above 0.05)
        and effect size is negligible (d=0.02).

        The agent should use evaluate_evidence to classify the finding as WEAK
        and flag the contradiction between the narrative and the statistics.
        """
        finding = findings_inflated_text_weak_stats[0]

        agent = ReviewerAgent(tools=review_tools)

        prompt = (
            "Evaluate this research finding for statistical validity.\n\n"
            f"FINDING:\n"
            f"  Title: {finding['title']}\n"
            f"  Text: {finding['finding']}\n"
            f"  P-value: {finding['p_value']}\n"
            f"  Effect size: {finding['effect_size']}\n"
            f"  Significance: {finding['significance']}\n\n"
            "Use the evaluate_evidence tool to classify this finding's evidence "
            "tier. Then assess whether the narrative text matches the actual "
            "statistical evidence.\n\n"
            "Document any discrepancies using save_note, then provide your verdict."
        )

        response = agent.review(prompt, max_turns=10)

        # The agent should flag the contradiction
        assert _response_contains_any(response, [
            "WEAK", "not significant", "0.73", "contradicts",
            "does not match", "doesn't match", "overstated", "inflated",
            "misleading", "overclaim", "negligible", "not supported",
            "no evidence", "exaggerat", "mismatch", "unsupported",
        ]), (
            f"Agent failed to detect narrative overclaiming with p=0.73.\n"
            f"Agent response:\n{response[:800]}"
        )


# ===================================================================
# Test class 4: Wrong enrollment number
# ===================================================================

@requires_llm
@pytest.mark.integration
class TestAgentDetectsWrongNumbers:
    """The agent must catch when a well-known statistic is reported incorrectly."""

    def test_catches_wrong_medicare_enrollment_count(
        self, review_tools, findings_wrong_enrollment_number,
    ):
        """AGENT BEHAVIORAL: 45.2M cited instead of ~65.7M → agent flags it.

        The actual CMS figure is ~65.7 million Medicare enrollees (2023).
        The finding claims 45.2 million — off by over 20 million.
        """
        finding = findings_wrong_enrollment_number[0]

        agent = ReviewerAgent(tools=review_tools)

        prompt = (
            "Review this finding for numerical accuracy.\n\n"
            f"FINDING:\n"
            f"  Title: {finding['title']}\n"
            f"  Text: {finding['finding']}\n"
            f"  Source: {finding['source']}\n"
            f"  Source URL: {finding['source_url']}\n\n"
            f"KNOWN REFERENCE DATA:\n"
            f"  CMS reports approximately 65.7 million Medicare enrollees "
            f"as of 2023 (source: {CMS_CITATION['url']})\n\n"
            "Does the enrollment number in the finding match the known "
            "reference data? Flag any discrepancies.\n\n"
            "Use save_note to document your assessment, then provide your verdict."
        )

        response = agent.review(prompt, max_turns=10)

        # The agent should catch the 45.2M vs 65.7M discrepancy
        assert _response_contains_any(response, [
            "65.7", "65,7", "incorrect", "discrepancy", "does not match",
            "doesn't match", "wrong", "inaccurate", "off by",
            "should be", "actual", "differs", "mismatch",
        ]), (
            f"Agent failed to detect wrong enrollment number (45.2M vs 65.7M).\n"
            f"Agent response:\n{response[:800]}"
        )


# ===================================================================
# Test class 5: Web search verification (network-dependent)
# ===================================================================

requires_network = pytest.mark.skipif(
    os.environ.get("VERSIFAI_SKIP_NETWORK_TESTS", "").lower() in ("1", "true", "yes"),
    reason="Network tests disabled via VERSIFAI_SKIP_NETWORK_TESTS",
)


@requires_llm
@requires_network
@pytest.mark.integration
class TestAgentWebVerification:
    """The agent uses web_search to verify a citation against the real source."""

    def test_verifies_cms_source_via_web_search(
        self, tmp_path, findings_cms_misattributed_to_cdc,
    ):
        """AGENT BEHAVIORAL: Agent fetches CMS URL to confirm source attribution.

        The agent is given a misattributed finding (CMS data → CDC) and a
        web_search tool configured with the real CMS documentation URL.
        It should fetch the URL, confirm the data comes from CMS, and flag
        the misattribution.
        """
        from versifai.core.tools.save_note import SaveNoteTool
        from versifai.core.tools.web_search import WebSearchTool
        from versifai.data_agents.engineer.config import ProjectConfig
        from versifai.story_agents.storyteller.tools.cite_source import CiteSourceTool
        from versifai.story_agents.storyteller.tools.evaluate_evidence import (
            EvaluateEvidenceTool,
        )

        cfg = ProjectConfig(
            name="Test",
            catalog="test",
            schema="test",
            volume_path=str(tmp_path),
            documentation_urls={
                "cms": [CMS_CITATION["url"]],
                "medicare": [CMS_CITATION["url"]],
            },
        )

        (tmp_path / "notes").mkdir(exist_ok=True)
        tools = [
            EvaluateEvidenceTool(),
            CiteSourceTool(),
            SaveNoteTool(notes_path=str(tmp_path / "notes")),
            WebSearchTool(cfg=cfg),
        ]

        agent = ReviewerAgent(tools=tools)
        # Extend system prompt to encourage web verification
        agent._system_prompt += (
            "\n\nYou have a web_search tool that can fetch documentation URLs. "
            "Use it to verify citation sources when the attribution seems wrong. "
            "Search for 'cms medicare enrollment' or use the URL parameter to "
            "fetch the actual source page."
        )

        finding = findings_cms_misattributed_to_cdc[0]

        prompt = (
            "Review this finding and VERIFY the citation source using web_search.\n\n"
            f"FINDING:\n"
            f"  Title: {finding['title']}\n"
            f"  Text: {finding['finding']}\n"
            f"  Attributed source: {finding['source']}\n"
            f"  Source URL: {finding['source_url']}\n\n"
            "The finding claims Medicare enrollment data comes from the CDC. "
            "Use the web_search tool to look up 'cms medicare enrollment' and "
            "verify which organization actually publishes this data.\n\n"
            "Document your findings with save_note, then provide your verdict."
        )

        response = agent.review(prompt, max_turns=12)

        # The agent should reference CMS as the correct source
        assert _response_contains_any(response, [
            "CMS", "Centers for Medicare",
            "misattribut", "incorrect", "wrong source",
        ]), (
            f"Agent failed to verify CMS as correct source via web search.\n"
            f"Agent response:\n{response[:800]}"
        )
