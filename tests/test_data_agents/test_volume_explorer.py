"""Tests for VolumeExplorerTool."""

from __future__ import annotations

import os

from versifai.data_agents.engineer.tools.volume_explorer import VolumeExplorerTool


class TestVolumeExplorer:
    def test_list_directory(self, tmp_path):
        (tmp_path / "file1.csv").write_text("a,b\n1,2\n")
        (tmp_path / "file2.txt").write_text("hello")

        tool = VolumeExplorerTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success is True
        entry_names = [e["name"] for e in result.data["entries"]]
        assert "file1.csv" in entry_names
        assert "file2.txt" in entry_names

    def test_recursive_listing(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.csv").write_text("x\n1\n")
        (tmp_path / "top.csv").write_text("y\n2\n")

        tool = VolumeExplorerTool()
        result = tool.execute(path=str(tmp_path), recursive=True)
        assert result.success is True
        # Should see both top-level and nested entries
        all_text = str(result.data)
        assert "nested.csv" in all_text
        assert "top.csv" in all_text

    def test_missing_path_error(self):
        tool = VolumeExplorerTool()
        result = tool.execute(path="/nonexistent/volume/path")
        assert result.success is False

    def test_file_metadata(self, tmp_path):
        (tmp_path / "test.csv").write_text("col\n1\n2\n3\n")
        tool = VolumeExplorerTool()
        result = tool.execute(path=str(tmp_path))
        file_entry = [e for e in result.data["entries"] if e["name"] == "test.csv"][0]
        assert file_entry["type"] == "file"
        assert "size_bytes" in file_entry
        assert file_entry["extension"] in (".csv", "csv")

    def test_directory_type(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        tool = VolumeExplorerTool()
        result = tool.execute(path=str(tmp_path))
        dir_entry = [e for e in result.data["entries"] if e["name"] == "mydir"][0]
        assert dir_entry["type"] == "directory"

    def test_empty_directory(self, tmp_path):
        tool = VolumeExplorerTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success is True
        assert len(result.data["entries"]) == 0
