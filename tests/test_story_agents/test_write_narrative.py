"""Tests for WriteNarrativeTool."""

from __future__ import annotations

import os

from versifai.story_agents.storyteller.tools.write_narrative import WriteNarrativeTool


class TestWriteNarrative:
    def test_write_section_creates_file(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir))
        result = tool.execute(
            operation="write_section",
            section_id="intro",
            title="Introduction",
            content="This report examines geographic disparities.",
            sequence=1,
        )
        assert result.success is True
        assert (output_dir / "intro.md").exists()

    def test_read_section_returns_content(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir))
        tool.execute(
            operation="write_section",
            section_id="methods",
            title="Methods",
            content="We used statistical analysis.",
            sequence=2,
        )
        result = tool.execute(operation="read_section", section_id="methods")
        assert result.success is True
        assert "statistical analysis" in str(result.data).lower()

    def test_list_sections(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir))
        tool.execute(operation="write_section", section_id="s1", content="Section 1", sequence=1)
        tool.execute(operation="write_section", section_id="s2", content="Section 2", sequence=2)

        result = tool.execute(operation="list_sections")
        assert result.success is True
        sections = result.data.get("sections", [])
        assert len(sections) == 2

    def test_assemble_combines_in_order(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir))
        tool.execute(
            operation="write_section",
            section_id="intro",
            content="Introduction content",
            sequence=1,
        )
        tool.execute(
            operation="write_section", section_id="results", content="Results content", sequence=2
        )
        tool.execute(
            operation="write_section",
            section_id="conclusion",
            content="Conclusion content",
            sequence=3,
        )

        result = tool.execute(operation="assemble")
        assert result.success is True

        # Read the assembled file
        output_file = output_dir / "narrative_report.md"
        assert output_file.exists()
        text = output_file.read_text()
        intro_pos = text.find("Introduction content")
        results_pos = text.find("Results content")
        conclusion_pos = text.find("Conclusion content")
        assert intro_pos < results_pos < conclusion_pos

    def test_assemble_includes_toc(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir), include_toc=True)
        tool.execute(
            operation="write_section",
            section_id="s1",
            title="First Section",
            content="Content of first section.",
            sequence=1,
        )
        result = tool.execute(operation="assemble")
        assert result.success is True

        text = (output_dir / "narrative_report.md").read_text()
        # TOC should reference the section title
        assert "First Section" in text

    def test_update_section(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir))
        tool.execute(operation="write_section", section_id="s1", content="Original", sequence=1)
        tool.execute(operation="update_section", section_id="s1", content="Updated")

        result = tool.execute(operation="read_section", section_id="s1")
        assert "Updated" in str(result.data)

    def test_read_nonexistent_section_error(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir))
        result = tool.execute(operation="read_section", section_id="missing")
        assert result.success is False

    def test_sections_written_counter(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        tool = WriteNarrativeTool(output_path=str(output_dir))
        tool.execute(operation="write_section", section_id="s1", content="C1", sequence=1)
        tool.execute(operation="write_section", section_id="s2", content="C2", sequence=2)
        assert tool.sections_written == 2
