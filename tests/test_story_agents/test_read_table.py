"""Tests for ReadTableTool."""

from __future__ import annotations

import os

from versifai.story_agents.storyteller.tools.read_table import ReadTableTool


class TestReadTable:
    def test_list_tables(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        (tables_dir / "summary.csv").write_text("a,b\n1,2\n")
        (tables_dir / "detail.csv").write_text("x,y\n3,4\n")

        tool = ReadTableTool(tables_path=str(tables_dir))
        result = tool.execute(operation="list")
        assert result.success is True
        tables = result.data.get("tables", result.data.get("files", []))
        assert len(tables) == 2

    def test_read_csv_content(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        (tables_dir / "data.csv").write_text("col_a,col_b\n10,20\n30,40\n")

        tool = ReadTableTool(tables_path=str(tables_dir))
        result = tool.execute(operation="read", filename="data.csv")
        assert result.success is True
        rows = result.data.get("rows", result.data.get("data", []))
        assert len(rows) == 2

    def test_summary_row_count(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        (tables_dir / "big.csv").write_text(
            "id,value\n" + "".join(f"{i},{i * 10}\n" for i in range(50))
        )

        tool = ReadTableTool(tables_path=str(tables_dir))
        result = tool.execute(operation="summary", filename="big.csv")
        assert result.success is True
        assert result.data.get("total_rows", 0) == 50

    def test_missing_file_error(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()

        tool = ReadTableTool(tables_path=str(tables_dir))
        result = tool.execute(operation="read", filename="nonexistent.csv")
        assert result.success is False

    def test_read_max_rows_limit(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        (tables_dir / "large.csv").write_text(
            "id,value\n" + "".join(f"{i},{i}\n" for i in range(200))
        )

        tool = ReadTableTool(tables_path=str(tables_dir))
        result = tool.execute(operation="read", filename="large.csv", max_rows=10)
        assert result.success is True
        rows = result.data.get("rows", result.data.get("data", []))
        assert len(rows) <= 10

    def test_empty_dir_list(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()

        tool = ReadTableTool(tables_path=str(tables_dir))
        result = tool.execute(operation="list")
        assert result.success is True
        tables = result.data.get("tables", result.data.get("files", []))
        assert len(tables) == 0
