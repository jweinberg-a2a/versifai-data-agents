"""Tests for ReadFindingsTool."""

from __future__ import annotations

import json

import pytest

from versifai.story_agents.storyteller.tools.read_findings import ReadFindingsTool


class TestReadFindings:
    def test_list_all_findings(self, sample_findings):
        tool = ReadFindingsTool(results_path=sample_findings)
        result = tool.execute(operation="list")
        assert result.success is True
        findings = result.data.get("findings", result.data.get("summaries", []))
        assert len(findings) == 4

    def test_get_by_index(self, sample_findings):
        tool = ReadFindingsTool(results_path=sample_findings)
        result = tool.execute(operation="get", index=0)
        assert result.success is True
        # result.data is the finding dict directly
        finding = result.data if isinstance(result.data, dict) else {}
        title = finding.get("title", str(result.data)).lower()
        assert "geographic" in title or "disparity" in title

    def test_filter_by_theme(self, sample_findings):
        tool = ReadFindingsTool(results_path=sample_findings)
        result = tool.execute(operation="by_theme", theme_id="theme0")
        assert result.success is True
        findings = result.data.get("findings", [])
        assert len(findings) == 2
        for f in findings:
            assert f.get("research_question_id", "") == "theme0"

    def test_high_significance_filter(self, sample_findings):
        tool = ReadFindingsTool(results_path=sample_findings)
        result = tool.execute(operation="high_significance")
        assert result.success is True
        findings = result.data.get("findings", [])
        # Only the first finding has significance="high"
        assert len(findings) >= 1
        for f in findings:
            assert f.get("significance", "").lower() in ("high", "critical")

    def test_search_by_keyword(self, sample_findings):
        tool = ReadFindingsTool(results_path=sample_findings)
        result = tool.execute(operation="search", query="geographic")
        assert result.success is True
        findings = result.data.get("findings", result.data.get("matches", []))
        assert len(findings) >= 1

    def test_empty_findings_graceful(self, tmp_path):
        empty_dir = tmp_path / "empty_results"
        empty_dir.mkdir()
        tool = ReadFindingsTool(results_path=str(empty_dir))
        result = tool.execute(operation="list")
        assert result.success is True
        findings = result.data.get("findings", result.data.get("summaries", []))
        assert len(findings) == 0

    def test_get_out_of_bounds(self, sample_findings):
        tool = ReadFindingsTool(results_path=sample_findings)
        result = tool.execute(operation="get", index=999)
        assert result.success is False or len(result.data.get("findings", [])) == 0
