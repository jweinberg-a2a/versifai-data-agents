"""Tests for ViewChartTool."""

from __future__ import annotations

import os

from versifai.core.tools.view_chart import ViewChartTool


class TestViewChart:
    def test_list_charts_and_tables(self, tmp_path):
        charts_dir = tmp_path / "charts"
        tables_dir = tmp_path / "tables"
        charts_dir.mkdir()
        tables_dir.mkdir()
        (charts_dir / "plot1.png").write_bytes(b"\x89PNG")
        (tables_dir / "data.csv").write_text("a,b\n1,2\n")

        tool = ViewChartTool(charts_path=str(charts_dir), tables_path=str(tables_dir))
        result = tool.execute(filename="list")
        assert result.success is True
        assert "plot1.png" in str(result.data)
        assert "data.csv" in str(result.data)

    def test_list_with_empty_filename(self, tmp_path):
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        (charts_dir / "chart.png").write_bytes(b"\x89PNG")

        tool = ViewChartTool(charts_path=str(charts_dir), tables_path="")
        result = tool.execute(filename="")
        assert result.success is True

    def test_view_png_returns_image_path(self, tmp_path):
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        (charts_dir / "scatter.png").write_bytes(b"\x89PNG fake image")

        tool = ViewChartTool(charts_path=str(charts_dir), tables_path="")
        result = tool.execute(filename="scatter.png")
        assert result.success is True
        assert result.image_path.endswith("scatter.png")

    def test_view_csv_returns_content(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        (tables_dir / "stats.csv").write_text("col_a,col_b\n1,2\n3,4\n")

        tool = ViewChartTool(charts_path="", tables_path=str(tables_dir))
        result = tool.execute(filename="stats.csv")
        assert result.success is True
        assert "col_a" in str(result.data)

    def test_view_missing_file_error(self, tmp_path):
        tool = ViewChartTool(charts_path=str(tmp_path), tables_path=str(tmp_path))
        result = tool.execute(filename="nonexistent.png")
        assert result.success is False
        assert "not found" in result.error.lower() or "available" in result.error.lower()

    def test_html_not_inline(self, tmp_path):
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        (charts_dir / "interactive.html").write_text("<html></html>")

        tool = ViewChartTool(charts_path=str(charts_dir), tables_path="")
        result = tool.execute(filename="interactive.html")
        assert result.success is True
