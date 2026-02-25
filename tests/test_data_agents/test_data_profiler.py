"""Tests for DataProfilerTool."""

from __future__ import annotations

from versifai.data_agents.engineer.tools.data_profiler import DataProfilerTool


class TestDataProfiler:
    def test_profile_column_count(self, sample_csv):
        tool = DataProfilerTool()
        result = tool.execute(file_path=sample_csv)
        assert result.success is True
        assert result.data["column_count"] == 5

    def test_profile_null_rates(self, sample_csv):
        """CSV has one null score (Eve's row) out of 10 rows → 10% null rate."""
        tool = DataProfilerTool()
        result = tool.execute(file_path=sample_csv)
        profiles = {p["column_name"]: p for p in result.data["column_profiles"]}
        score_profile = profiles["score"]
        assert score_profile["null_count"] == 1
        assert score_profile["null_rate"] == 0.1

    def test_profile_unique_counts(self, sample_csv):
        tool = DataProfilerTool()
        result = tool.execute(file_path=sample_csv)
        profiles = {p["column_name"]: p for p in result.data["column_profiles"]}
        # 'name' column has 10 unique values
        assert profiles["name"]["unique_count"] == 10
        # 'grade' has A, B, C, D, F = 5 unique values
        assert profiles["grade"]["unique_count"] == 5

    def test_detect_fips_codes(self, sample_csv):
        """The fips_code column should be detected as potential FIPS."""
        tool = DataProfilerTool()
        result = tool.execute(file_path=sample_csv)
        assert "fips_code" in result.data["potential_fips_columns"]

    def test_profile_numeric_stats(self, sample_csv):
        """Verify min/max/mean for the score column with known values."""
        tool = DataProfilerTool()
        result = tool.execute(file_path=sample_csv)
        profiles = {p["column_name"]: p for p in result.data["column_profiles"]}
        score = profiles["score"]
        # Known scores: 95.5, 82.3, 78.1, 91.0, (null), 88.2, 67.4, 73.9, 99.1, 55.0
        assert float(score["min"]) == 55.0
        assert float(score["max"]) == 99.1

    def test_missing_file_error(self):
        tool = DataProfilerTool()
        result = tool.execute(file_path="/nonexistent/data.csv")
        assert result.success is False

    def test_detect_geo_columns(self, tmp_path):
        """Column named 'county' should be flagged as geo."""
        geo_csv = tmp_path / "geo.csv"
        geo_csv.write_text("county,state,value\nDallas,TX,100\nKing,WA,200\n")
        tool = DataProfilerTool()
        result = tool.execute(file_path=str(geo_csv))
        assert "county" in result.data["potential_geo_columns"]
        assert "state" in result.data["potential_geo_columns"]

    def test_top_values(self, sample_csv):
        tool = DataProfilerTool()
        result = tool.execute(file_path=sample_csv)
        profiles = {p["column_name"]: p for p in result.data["column_profiles"]}
        grade_profile = profiles["grade"]
        assert "top_values" in grade_profile
        assert len(grade_profile["top_values"]) > 0

    def test_tsv_profiling(self, sample_tsv):
        tool = DataProfilerTool()
        result = tool.execute(file_path=sample_tsv)
        assert result.success is True
        assert result.data["column_count"] == 3
