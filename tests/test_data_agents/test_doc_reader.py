"""Tests for DocumentationReaderTool and ScanForDocumentationTool."""

from __future__ import annotations

import os

from versifai.data_agents.engineer.tools.doc_reader import (
    DocumentationReaderTool,
    ScanForDocumentationTool,
)


class TestDocumentationReader:
    def test_read_txt_doc(self, tmp_path):
        doc_file = tmp_path / "README.txt"
        doc_file.write_text("This is the project documentation.\nIt has multiple lines.")
        tool = DocumentationReaderTool()
        result = tool.execute(file_path=str(doc_file))
        assert result.success is True
        assert "documentation" in result.data["content"].lower()

    def test_read_markdown_doc(self, tmp_path):
        doc_file = tmp_path / "guide.md"
        doc_file.write_text("# Guide\n\nThis is a markdown guide.\n\n## Section 1\nDetails here.")
        tool = DocumentationReaderTool()
        result = tool.execute(file_path=str(doc_file))
        assert result.success is True
        assert "Guide" in result.data["content"]

    def test_read_csv_as_dict(self, tmp_path):
        dict_file = tmp_path / "data_dictionary.csv"
        dict_file.write_text("column_name,description,type\nfips_code,County FIPS,string\nvalue,Metric value,float\n")
        tool = DocumentationReaderTool()
        result = tool.execute(file_path=str(dict_file))
        assert result.success is True
        assert "fips_code" in str(result.data)

    def test_truncation_at_max_chars(self, tmp_path):
        long_doc = tmp_path / "long.txt"
        long_doc.write_text("x" * 50000)
        tool = DocumentationReaderTool()
        result = tool.execute(file_path=str(long_doc), max_chars=100)
        assert result.success is True
        assert result.data.get("truncated", False) is True

    def test_missing_file_error(self):
        tool = DocumentationReaderTool()
        result = tool.execute(file_path="/nonexistent/readme.txt")
        assert result.success is False

    def test_encoding_fallback(self, tmp_path):
        latin_doc = tmp_path / "latin.txt"
        latin_doc.write_bytes(b"Caf\xe9 documentation for \xe9l\xe8ves.")
        tool = DocumentationReaderTool()
        result = tool.execute(file_path=str(latin_doc))
        assert result.success is True


class TestScanForDocumentation:
    def test_scan_finds_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")

        tool = ScanForDocumentationTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success is True
        found_files = [f["file_name"] for f in result.data["documentation_files"]]
        assert "README.md" in found_files

    def test_scan_prioritizes_data_dict(self, tmp_path):
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "data_dictionary.csv").write_text("col,desc\nfips,FIPS code\n")
        (tmp_path / "notes.txt").write_text("Some notes.")

        tool = ScanForDocumentationTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success is True
        files = result.data["documentation_files"]
        file_names = [f["file_name"] for f in files]
        # Both data dictionary and README should be found
        assert "data_dictionary.csv" in file_names
        assert "README.md" in file_names
        # Data dictionary should have high priority (>= 9)
        dict_item = next(f for f in files if "dictionary" in f["file_name"].lower())
        assert dict_item["priority"] >= 9

    def test_scan_empty_dir(self, tmp_path):
        tool = ScanForDocumentationTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success is True
        assert result.data["count"] == 0

    def test_scan_skips_underscore_dirs(self, tmp_path):
        hidden = tmp_path / "_internal"
        hidden.mkdir()
        (hidden / "secret.md").write_text("Hidden doc")
        (tmp_path / "public.md").write_text("Visible doc")

        tool = ScanForDocumentationTool()
        result = tool.execute(path=str(tmp_path))
        found_files = [f["file_name"] for f in result.data["documentation_files"]]
        assert "secret.md" not in found_files
        assert "public.md" in found_files
