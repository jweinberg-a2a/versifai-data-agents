"""Tests for CiteSourceTool."""

from __future__ import annotations

from versifai.story_agents.storyteller.tools.cite_source import CiteSourceTool


class TestCiteSource:
    def test_add_citation(self):
        tool = CiteSourceTool()
        result = tool.execute(
            operation="add",
            title="CMS Star Ratings Technical Notes",
            author="CMS",
            year="2024",
            url="https://cms.gov/stars",
        )
        assert result.success is True
        assert "cite_key" in result.data

    def test_list_citations(self):
        tool = CiteSourceTool()
        tool.execute(operation="add", title="Source One", author="Auth1")
        tool.execute(operation="add", title="Source Two", author="Auth2")
        result = tool.execute(operation="list")
        assert result.success is True
        citations = result.data.get("citations", {})
        assert len(citations) == 2

    def test_format_bibliography(self):
        tool = CiteSourceTool()
        tool.execute(
            operation="add",
            title="NOAA Weather Station Data",
            author="NOAA",
            year="2023",
            url="https://noaa.gov",
        )
        result = tool.execute(operation="format")
        assert result.success is True
        bib_text = result.data.get("bibliography", result.data.get("text", ""))
        assert "NOAA Weather Station Data" in bib_text
        assert "NOAA" in bib_text

    def test_search_by_keyword(self):
        tool = CiteSourceTool()
        tool.execute(operation="add", title="Stars Analysis Report", author="Research Team")
        tool.execute(operation="add", title="Enrollment Data Guide", author="Data Team")
        result = tool.execute(operation="search", query="stars")
        assert result.success is True
        matches = result.data.get("matches", result.data.get("citations", []))
        assert len(matches) >= 1

    def test_auto_generated_cite_key(self):
        tool = CiteSourceTool()
        result = tool.execute(
            operation="add",
            title="My Cool Research Paper",
        )
        assert result.success is True
        key = result.data.get("cite_key", "")
        assert len(key) > 0

    def test_custom_cite_key(self):
        tool = CiteSourceTool()
        result = tool.execute(
            operation="add",
            title="Custom Key Test",
            cite_key="custom_key_2024",
        )
        assert result.success is True
        assert result.data["cite_key"] == "custom_key_2024"

    def test_empty_bibliography(self):
        tool = CiteSourceTool()
        result = tool.execute(operation="format")
        assert result.success is True

    def test_search_no_match(self):
        tool = CiteSourceTool()
        tool.execute(operation="add", title="Something Unrelated")
        result = tool.execute(operation="search", query="xyznonexistent")
        assert result.success is True
        matches = result.data.get("matches", result.data.get("citations", []))
        assert len(matches) == 0

    def test_inline_ref_format(self):
        tool = CiteSourceTool()
        result = tool.execute(operation="add", title="Test Paper", cite_key="test_2024")
        inline = result.data.get("inline_ref", "")
        assert "test_2024" in inline
