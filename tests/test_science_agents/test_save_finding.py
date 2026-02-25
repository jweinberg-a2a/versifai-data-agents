"""Tests for SaveFindingTool."""

from __future__ import annotations

import json
import os

from versifai.science_agents.scientist.tools.save_finding import SaveFindingTool


class TestSaveFinding:
    def test_save_creates_finding(self):
        tool = SaveFindingTool()
        result = tool.execute(
            research_question_id="rq1",
            title="Test Finding",
            finding="We found X causes Y.",
            evidence="t-test p=0.001",
            significance="high",
        )
        assert result.success is True
        assert len(tool.findings) == 1
        assert tool.findings[0]["title"] == "Test Finding"

    def test_append_second_finding(self):
        tool = SaveFindingTool()
        tool.execute(
            research_question_id="rq1",
            title="First",
            finding="Finding one.",
            evidence="evidence 1",
        )
        tool.execute(
            research_question_id="rq2",
            title="Second",
            finding="Finding two.",
            evidence="evidence 2",
        )
        assert len(tool.findings) == 2

    def test_required_fields_validated(self):
        tool = SaveFindingTool()
        # Missing title
        result = tool.execute(
            research_question_id="rq1",
            title="",
            finding="A finding.",
            evidence="Some evidence",
        )
        assert result.success is False

    def test_get_findings_for_question(self):
        tool = SaveFindingTool()
        tool.execute(research_question_id="rq1", title="A", finding="a", evidence="e")
        tool.execute(research_question_id="rq2", title="B", finding="b", evidence="e")
        tool.execute(research_question_id="rq1", title="C", finding="c", evidence="e")

        rq1_findings = tool.get_findings_for_question("rq1")
        assert len(rq1_findings) == 2

    def test_get_high_significance(self):
        tool = SaveFindingTool()
        tool.execute(
            research_question_id="rq1", title="High", finding="h", evidence="e", significance="high"
        )
        tool.execute(
            research_question_id="rq1", title="Low", finding="l", evidence="e", significance="low"
        )

        high = tool.get_high_significance_findings()
        assert len(high) == 1
        assert high[0]["title"] == "High"

    def test_export_findings_json(self, tmp_path):
        tool = SaveFindingTool(results_path=str(tmp_path))
        tool.execute(research_question_id="rq1", title="Export Test", finding="f", evidence="e")
        tool.export_findings_json()

        json_path = tmp_path / "findings.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Export Test"

    def test_findings_summary_text(self):
        tool = SaveFindingTool()
        tool.execute(
            research_question_id="rq1",
            title="Summary Test",
            finding="f",
            evidence="e",
            significance="high",
        )
        summary = tool.findings_summary_text()
        assert "Summary Test" in summary
        assert "HIGH" in summary.upper() or "high" in summary
