"""Tests for FileReaderTool."""

from __future__ import annotations

from versifai.data_agents.engineer.tools.file_reader import FileReaderTool


class TestFileReader:
    def test_read_csv_headers(self, sample_csv):
        tool = FileReaderTool()
        result = tool.execute(file_path=sample_csv)
        assert result.success is True
        assert result.data["columns"] == ["id", "name", "score", "grade", "fips_code"]

    def test_read_csv_sample_rows(self, sample_csv):
        tool = FileReaderTool()
        result = tool.execute(file_path=sample_csv, n_rows=3)
        assert result.success is True
        assert result.data["sample_row_count"] == 3

    def test_read_csv_column_count(self, sample_csv):
        tool = FileReaderTool()
        result = tool.execute(file_path=sample_csv)
        assert result.data["column_count"] == 5

    def test_read_tsv_auto_detect(self, sample_tsv):
        tool = FileReaderTool()
        result = tool.execute(file_path=sample_tsv)
        assert result.success is True
        assert "col_a" in result.data["columns"]
        assert "col_b" in result.data["columns"]

    def test_missing_file_error(self):
        tool = FileReaderTool()
        result = tool.execute(file_path="/nonexistent/path/data.csv")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_unsupported_extension(self, tmp_path):
        weird_file = tmp_path / "data.xyz"
        weird_file.write_text("some content")
        tool = FileReaderTool()
        result = tool.execute(file_path=str(weird_file))
        assert result.success is False
        assert "unsupported" in result.error.lower()

    def test_encoding_fallback(self, tmp_path):
        """CSV with latin-1 characters should be handled by encoding fallback."""
        latin_csv = tmp_path / "latin.csv"
        latin_csv.write_bytes(b"name,value\nCaf\xe9,42\n")
        tool = FileReaderTool()
        result = tool.execute(file_path=str(latin_csv))
        assert result.success is True
        assert result.data["column_count"] == 2

    def test_skip_rows(self, tmp_path):
        """skip_rows skips data rows after the header."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n2,20\n3,30\n4,40\n5,50\n")
        tool = FileReaderTool()
        result = tool.execute(file_path=str(csv_file), skip_rows=2, n_rows=10)
        assert result.success is True
        assert "id" in result.data["columns"]
        # Skipped 2 data rows, so should have 3 left
        assert result.data["sample_row_count"] == 3

    def test_pipe_separated(self, tmp_path):
        pipe_csv = tmp_path / "pipe.csv"
        pipe_csv.write_text("a|b|c\n1|2|3\n4|5|6\n")
        tool = FileReaderTool()
        result = tool.execute(file_path=str(pipe_csv))
        assert result.success is True
        assert result.data["column_count"] == 3

    def test_file_size_reported(self, sample_csv):
        tool = FileReaderTool()
        result = tool.execute(file_path=sample_csv)
        assert "file_size_mb" in result.data
        assert result.data["file_size_mb"] >= 0

    def test_estimated_rows(self, sample_csv):
        tool = FileReaderTool()
        result = tool.execute(file_path=sample_csv)
        assert "estimated_total_rows" in result.data
        assert result.data["estimated_total_rows"] > 0
