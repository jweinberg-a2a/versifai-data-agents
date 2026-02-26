"""Integration test: DataScientistAgent enforces save_finding for every theme.

Verifies that the agent's nudge mechanism works — when the LLM completes
a theme analysis without calling save_finding, the orchestrator detects
this and runs a follow-up phase to get the agent to record findings.

This test uses a minimal ResearchConfig with a single theme and a mock
SQL tool that returns canned data. The LLM is real (ReAct loop with
actual API calls), but no Databricks connection is needed.

Run::

    pytest tests/test_integration/test_save_finding_enforcement.py -v -m integration

Skip::

    pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import json
import os

import pytest

from versifai.core.agent import BaseAgent
from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.tools.base import BaseTool, ToolResult
from versifai.core.tools.registry import ToolRegistry
from versifai.science_agents.scientist.tools.save_finding import SaveFindingTool

from .conftest import requires_llm

# ---------------------------------------------------------------------------
# Mock SQL tool — returns canned data so the agent has something to analyze
# ---------------------------------------------------------------------------


class MockExecuteSQLTool(BaseTool):
    """Returns canned query results so the agent can 'analyze' data."""

    @property
    def name(self) -> str:
        return "execute_sql"

    @property
    def description(self) -> str:
        return (
            "Execute a SQL query against the data catalog. "
            "Returns query results as a list of row dicts."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute.",
                },
            },
            "required": ["query"],
        }

    def _execute(self, query: str = "", **kwargs) -> ToolResult:
        # Return plausible summary stats regardless of query
        data = [
            {"country": "USA", "gdp_per_capita": 63544, "life_expectancy": 77.3},
            {"country": "GBR", "gdp_per_capita": 46510, "life_expectancy": 81.0},
            {"country": "JPN", "gdp_per_capita": 39286, "life_expectancy": 84.6},
            {"country": "BRA", "gdp_per_capita": 8920, "life_expectancy": 75.9},
            {"country": "NGA", "gdp_per_capita": 2066, "life_expectancy": 54.7},
            {"country": "IND", "gdp_per_capita": 2257, "life_expectancy": 70.8},
        ]
        return ToolResult(
            success=True,
            data={"rows": data, "row_count": len(data)},
            summary=f"Query returned {len(data)} rows.",
        )


# ---------------------------------------------------------------------------
# Lightweight analysis agent — drives the ReAct loop with real LLM
# ---------------------------------------------------------------------------


class AnalysisAgent(BaseAgent):
    """Minimal agent for testing save_finding enforcement."""

    _SYSTEM_PROMPT = """\
You are a data analyst. You have access to execute_sql and save_finding tools.

When asked to analyze data:
1. Query the data using execute_sql
2. Describe the patterns you observe
3. Call save_finding to record your findings with the theme ID provided

IMPORTANT: You MUST call save_finding at least once per analysis theme.
Each finding needs: research_question_id, title, finding, evidence, significance."""

    def __init__(self, tools: list, model: str | None = None) -> None:
        display = AgentDisplay()
        memory = AgentMemory()
        model = model or os.environ.get("VERSIFAI_TEST_MODEL", "claude-haiku-4-5-20251001")
        llm = LLMClient(model=model)
        registry = ToolRegistry()

        super().__init__(display=display, memory=memory, llm=llm, registry=registry)

        for tool in tools:
            registry.register(tool)

        self._system_prompt = self._SYSTEM_PROMPT

    def analyze(self, prompt: str, max_turns: int = 15) -> str:
        """Run a single analysis phase; return the agent's final text."""
        return self._run_phase(prompt, max_turns=max_turns)


# ===================================================================
# Test: Agent calls save_finding when prompted
# ===================================================================


@requires_llm
@pytest.mark.integration
class TestSaveFindingEnforcement:
    """Verify that the agent calls save_finding to record structured findings."""

    def test_agent_calls_save_finding(self, tmp_path):
        """AGENT BEHAVIORAL: Agent must call save_finding when analyzing data.

        Given a dataset of GDP and life expectancy by country, the agent
        should analyze the data, observe the relationship, and record
        at least one structured finding via save_finding.
        """
        finding_tool = SaveFindingTool(results_path=str(tmp_path))
        tools = [MockExecuteSQLTool(), finding_tool]

        agent = AnalysisAgent(tools=tools)

        prompt = (
            "Analyze the relationship between GDP per capita and life expectancy.\n\n"
            "Theme ID: theme_1\n"
            "Question: Is there a relationship between wealth and health outcomes?\n\n"
            "Steps:\n"
            "1. Query the data using execute_sql\n"
            "2. Observe patterns in GDP vs life expectancy\n"
            "3. Record your findings using save_finding with theme ID 'theme_1'\n\n"
            "You MUST call save_finding at least once."
        )

        agent.analyze(prompt, max_turns=15)

        # The agent should have recorded at least one finding
        assert len(finding_tool.findings) >= 1, (
            f"Agent did not call save_finding. "
            f"Expected at least 1 finding, got {len(finding_tool.findings)}."
        )

        # Verify the finding has required fields
        finding = finding_tool.findings[0]
        assert finding["research_question_id"] == "theme_1", (
            f"Finding has wrong theme ID: {finding['research_question_id']}"
        )
        assert finding["title"], "Finding is missing a title"
        assert finding["finding"], "Finding is missing the finding text"
        assert finding["significance"] in ("high", "medium", "low"), (
            f"Invalid significance: {finding['significance']}"
        )

    def test_findings_export_to_json(self, tmp_path):
        """AGENT BEHAVIORAL: Findings are correctly exported to findings.json.

        After the agent records findings, export_findings_json should write
        a valid JSON file that can be read back by the StoryTeller.
        """
        finding_tool = SaveFindingTool(results_path=str(tmp_path))
        tools = [MockExecuteSQLTool(), finding_tool]

        agent = AnalysisAgent(tools=tools)

        prompt = (
            "Analyze GDP per capita across countries.\n\n"
            "Theme ID: theme_0\n"
            "Question: What does the distribution of GDP look like globally?\n\n"
            "Query the data, then call save_finding with:\n"
            "- research_question_id: 'theme_0'\n"
            "- A descriptive title\n"
            "- Your quantitative finding\n"
            "- The evidence (the SQL query or data summary)\n"
            "- significance: 'medium'\n"
        )

        agent.analyze(prompt, max_turns=15)

        # Export findings
        findings_path = finding_tool.export_findings_json()

        # Verify the file exists and is valid JSON
        assert os.path.isfile(findings_path), f"findings.json not created at {findings_path}"

        with open(findings_path) as f:
            data = json.load(f)

        assert isinstance(data, list), f"findings.json should be a list, got {type(data)}"
        assert len(data) >= 1, "findings.json is empty"

        # Verify structure matches what StoryTeller expects
        entry = data[0]
        assert "research_question_id" in entry
        assert "title" in entry
        assert "finding" in entry
        assert "evidence" in entry
        assert "significance" in entry

    def test_nudge_mechanism_records_findings(self, tmp_path):
        """AGENT BEHAVIORAL: Nudge prompt gets agent to record findings it missed.

        Simulates the orchestrator's nudge mechanism: first run a theme
        analysis, then if no findings were recorded, send a follow-up
        prompt explicitly asking the agent to call save_finding.

        This tests the exact flow from DataScientistAgent._nudge_save_finding().
        """
        finding_tool = SaveFindingTool(results_path=str(tmp_path))
        tools = [MockExecuteSQLTool(), finding_tool]

        agent = AnalysisAgent(tools=tools)

        # Phase 1: Run analysis WITHOUT mentioning save_finding
        # (simulating when the LLM "forgets" to call it)
        analysis_prompt = (
            "Query the data about GDP and life expectancy using execute_sql. "
            "Describe what you see in the data — which countries are richest, "
            "which have highest life expectancy, and any patterns you notice. "
            "Just describe your observations in text."
        )
        agent.analyze(analysis_prompt, max_turns=10)

        findings_before_nudge = len(finding_tool.findings)

        # Phase 2: Send the nudge (same as _nudge_save_finding)
        nudge_prompt = (
            "You just completed analysis for Theme 1: GDP and Life Expectancy, "
            "but you did NOT call `save_finding`. "
            "Every theme MUST produce at least one structured finding. "
            "Review the analysis you just performed and call `save_finding` "
            "now with the theme ID 'theme_1', a title, the key "
            "quantitative result, the supporting evidence (SQL or stats), "
            "and a significance rating. Do this for each major insight "
            "from the theme."
        )
        agent.analyze(nudge_prompt, max_turns=10)

        findings_after_nudge = len(finding_tool.findings)

        # The nudge should have produced at least one new finding
        new_findings = findings_after_nudge - findings_before_nudge
        assert new_findings >= 1, (
            f"Nudge failed to produce findings. "
            f"Before nudge: {findings_before_nudge}, after: {findings_after_nudge}."
        )
