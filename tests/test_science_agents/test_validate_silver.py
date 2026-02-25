"""Tests for ValidateSilverTool — data quality validation."""

from __future__ import annotations

import pytest

from versifai.science_agents.scientist.tools.validate_silver import ValidateSilverTool


@pytest.fixture
def tool():
    return ValidateSilverTool()


class TestGrainCheck:
    def test_duplicate_keys_flagged(self, tool):
        data = [
            {"fips": "06001", "year": 2020, "value": 100},
            {"fips": "06001", "year": 2020, "value": 105},  # duplicate
            {"fips": "06037", "year": 2020, "value": 200},
        ]
        result = tool.execute(
            check_type="grain",
            data=data,
            primary_key_columns=["fips", "year"],
        )
        assert result.success is True
        assert result.data.get("issues_found", False) or result.data.get("duplicate_count", 0) > 0

    def test_no_duplicates_clean(self, tool):
        """BEHAVIORAL: Clean data passes grain check."""
        data = [
            {"fips": "06001", "year": 2020, "value": 100},
            {"fips": "06037", "year": 2020, "value": 200},
            {"fips": "06001", "year": 2021, "value": 110},
        ]
        result = tool.execute(
            check_type="grain",
            data=data,
            primary_key_columns=["fips", "year"],
        )
        assert result.success is True
        issues = result.data.get("issues_found", result.data.get("duplicate_count", 0))
        if isinstance(issues, bool):
            assert not issues
        else:
            assert issues == 0


class TestValueRanges:
    def test_out_of_bounds_caught(self, tool):
        data = [
            {"pct_enrolled": 50},
            {"pct_enrolled": 150},  # >100% should be flagged
            {"pct_enrolled": -5},  # negative should be flagged
        ]
        result = tool.execute(
            check_type="value_ranges",
            data=data,
            column_ranges={"pct_enrolled": {"min": 0, "max": 100}},
        )
        assert result.success is True
        has_issues = (
            result.data.get("issues_found", False) or len(result.data.get("details", [])) > 0
        )
        assert has_issues, "Out-of-range values should be caught"

    def test_valid_ranges_pass(self, tool):
        """BEHAVIORAL: Clean data should pass range checks."""
        data = [
            {"pct_enrolled": 50},
            {"pct_enrolled": 75},
            {"pct_enrolled": 25},
        ]
        result = tool.execute(
            check_type="value_ranges",
            data=data,
            column_ranges={"pct_enrolled": {"min": 0, "max": 100}},
        )
        assert result.success is True


class TestZeroColumns:
    def test_all_zero_column_detected(self, tool):
        data = [
            {"fips": "06001", "value": 0, "enrollment": 100},
            {"fips": "06037", "value": 0, "enrollment": 200},
            {"fips": "36061", "value": 0, "enrollment": 150},
        ]
        result = tool.execute(check_type="zero_columns", data=data)
        assert result.success is True
        details = result.data.get("details", [])
        assert len(details) > 0, "All-zero column should be flagged"

    def test_normal_data_no_zero_alert(self, tool):
        """BEHAVIORAL: Normal data should not trigger zero column alert."""
        data = [
            {"fips": "06001", "value": 10, "enrollment": 100},
            {"fips": "06037", "value": 20, "enrollment": 200},
            {"fips": "36061", "value": 30, "enrollment": 150},
        ]
        result = tool.execute(check_type="zero_columns", data=data)
        assert result.success is True
        details = result.data.get("details", [])
        assert len(details) == 0, "Normal data should not trigger zero alerts"


class TestJoinCompleteness:
    def test_low_match_rate_flagged(self, tool):
        result = tool.execute(
            check_type="join_completeness",
            data=[{"x": 1}],  # data param is required
            left_count=1000,
            matched_count=500,  # 50% match rate
            unmatched_sample=[{"fips": "99999"}],
        )
        assert result.success is True
        has_issue = (
            result.data.get("issues_found", False) or result.data.get("match_rate_pct", 100) < 80
        )
        assert has_issue, "50% match rate should be flagged"

    def test_high_match_rate_ok(self, tool):
        result = tool.execute(
            check_type="join_completeness",
            data=[{"x": 1}],  # data param is required
            left_count=1000,
            matched_count=950,  # 95% match rate
        )
        assert result.success is True


class TestYearAlignment:
    def test_misaligned_years(self, tool):
        data = [
            {"star_year": 2020, "enrollment_year": 2020},  # expected offset -1
            {"star_year": 2021, "enrollment_year": 2021},  # off by 0, expected -1
        ]
        result = tool.execute(
            check_type="year_alignment",
            data=data,
            year_column_left="star_year",
            year_column_right="enrollment_year",
            expected_offset=-1,
        )
        assert result.success is True
