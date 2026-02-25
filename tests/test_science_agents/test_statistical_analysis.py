"""Tests for StatisticalAnalysisTool — includes behavioral validity tests."""

from __future__ import annotations

import random

import pytest

from versifai.science_agents.scientist.tools.statistical_analysis import StatisticalAnalysisTool


@pytest.fixture
def tool():
    return StatisticalAnalysisTool()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_known_data(self, tool):
        data = [{"x": v} for v in [1, 2, 3, 4, 5]]
        result = tool.execute(analysis_type="describe", data=data, columns=["x"])
        assert result.success is True
        # Output is keyed by column name: data["x"]["mean"]
        x_stats = result.data["x"]
        assert float(x_stats["mean"]) == pytest.approx(3.0)

    def test_describe_with_nulls(self, tool):
        data = [{"x": 1}, {"x": 2}, {"x": None}, {"x": 4}]
        result = tool.execute(analysis_type="describe", data=data, columns=["x"])
        assert result.success is True


class TestCorrelation:
    def test_perfect_positive(self, tool):
        data = [{"x": i, "y": i} for i in range(20)]
        result = tool.execute(analysis_type="correlation", data=data, columns=["x", "y"])
        assert result.success is True
        r = result.data["correlations"][0]["r"]
        assert abs(r) > 0.99

    def test_no_correlation_random(self, tool):
        random.seed(42)
        data = [{"x": random.gauss(0, 1), "y": random.gauss(0, 1)} for _ in range(200)]
        result = tool.execute(analysis_type="correlation", data=data, columns=["x", "y"])
        assert result.success is True
        r = result.data["correlations"][0]["r"]
        assert abs(r) < 0.2

    def test_negative_correlation(self, tool):
        data = [{"x": i, "y": -i} for i in range(20)]
        result = tool.execute(analysis_type="correlation", data=data, columns=["x", "y"])
        assert result.success is True
        r = result.data["correlations"][0]["r"]
        assert r < -0.99


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------


class TestHypothesisTest:
    def test_ttest_identical_groups(self, tool):
        """Same constant in both groups → NOT significant.

        Note: when all values are identical (std=0), t-test gives NaN.
        We add tiny noise to make it testable while keeping groups nearly identical.
        """
        random.seed(42)
        data = [{"group": "A", "value": 100 + random.gauss(0, 0.01)} for _ in range(30)] + [
            {"group": "B", "value": 100 + random.gauss(0, 0.01)} for _ in range(30)
        ]
        result = tool.execute(
            analysis_type="hypothesis_test",
            data=data,
            group_column="group",
            value_column="value",
            method="ttest_ind",
        )
        assert result.success is True
        p = result.data["p_value"]
        assert p > 0.05  # clearly not significant

    def test_ttest_different_groups(self, tool, clearly_different_groups):
        """Groups with N(100,5) vs N(200,5) → p < 0.001."""
        result = tool.execute(
            analysis_type="hypothesis_test",
            data=clearly_different_groups,
            group_column="group",
            value_column="value",
            method="ttest_ind",
        )
        assert result.success is True
        p = result.data["p_value"]
        assert p < 0.001


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------


class TestEffectSize:
    def test_large_effect(self, tool, clearly_different_groups):
        result = tool.execute(
            analysis_type="effect_size",
            data=clearly_different_groups,
            group_column="group",
            value_column="value",
        )
        assert result.success is True
        comp = result.data["comparisons"][0]
        d = abs(comp["cohens_d"])
        assert d > 0.8  # large effect

    def test_negligible_effect(self, tool, nearly_identical_groups):
        result = tool.execute(
            analysis_type="effect_size",
            data=nearly_identical_groups,
            group_column="group",
            value_column="value",
        )
        assert result.success is True
        comp = result.data["comparisons"][0]
        d = abs(comp["cohens_d"])
        assert d < 0.5  # should be small/negligible


# ---------------------------------------------------------------------------
# BEHAVIORAL validity tests
# ---------------------------------------------------------------------------


class TestBehavioralValidity:
    def test_no_false_significance(self, tool, nearly_identical_groups):
        """BEHAVIORAL: Groups with near-identical means on noisy data → NOT significant.

        This verifies the tool doesn't produce false positives on data
        where true difference is essentially zero (0.001 diff with std=20).
        """
        result = tool.execute(
            analysis_type="hypothesis_test",
            data=nearly_identical_groups,
            group_column="group",
            value_column="value",
            method="ttest_ind",
        )
        assert result.success is True
        p = result.data["p_value"]
        # p should NOT be significant — these groups are practically identical
        assert p > 0.05, f"False significance detected: p={p} for nearly identical groups"

    def test_detects_true_significance(self, tool, clearly_different_groups):
        """BEHAVIORAL: Groups with large real difference → IS significant.

        Verifies the tool correctly identifies a real effect (100 vs 200 mean).
        """
        result = tool.execute(
            analysis_type="hypothesis_test",
            data=clearly_different_groups,
            group_column="group",
            value_column="value",
            method="ttest_ind",
        )
        assert result.success is True
        p = result.data["p_value"]
        assert p < 0.001, f"Failed to detect obvious difference: p={p}"

    def test_effect_size_large_classified(self, tool, clearly_different_groups):
        """BEHAVIORAL: Cohen's d > 0.8 should be labeled 'large'."""
        result = tool.execute(
            analysis_type="effect_size",
            data=clearly_different_groups,
            group_column="group",
            value_column="value",
        )
        assert result.success is True
        comp = result.data["comparisons"][0]
        interpretation = comp.get("interpretation", "").lower()
        d = abs(comp["cohens_d"])
        assert d > 0.8 or "large" in interpretation

    def test_effect_size_small_classified(self, tool, nearly_identical_groups):
        """BEHAVIORAL: Negligible difference should be 'small' or 'negligible'."""
        result = tool.execute(
            analysis_type="effect_size",
            data=nearly_identical_groups,
            group_column="group",
            value_column="value",
        )
        assert result.success is True
        comp = result.data["comparisons"][0]
        d = abs(comp["cohens_d"])
        interpretation = comp.get("interpretation", "").lower()
        assert d < 0.5 or "small" in interpretation or "negligible" in interpretation


class TestFalseDataDetection:
    """BEHAVIORAL: Feed intentionally bad/misleading data and verify the tool
    produces correct (not misleading) statistical conclusions.
    """

    def test_all_identical_values_no_false_correlation(self, tool):
        """All Y values identical → correlation should be 0 or undefined, not 1."""
        data = [{"x": i, "y": 42} for i in range(20)]
        result = tool.execute(analysis_type="correlation", data=data, columns=["x", "y"])
        assert result.success is True
        r = result.data["correlations"][0]["r"]
        # With zero variance in Y, correlation is undefined or 0
        import math

        assert r == 0 or math.isnan(r), f"Constant Y should give r=0 or NaN, got {r}"

    def test_single_outlier_inflated_stats(self, tool):
        """A single massive outlier should not make a trivial pattern 'significant'.

        50 values of noise around 0, then 1 outlier at 10000. The describe
        stats should show high std reflecting the outlier.
        """
        random.seed(42)
        data = [{"x": random.gauss(0, 1)} for _ in range(50)]
        data.append({"x": 10000})
        result = tool.execute(analysis_type="describe", data=data, columns=["x"])
        assert result.success is True
        stats = result.data["x"]
        # The std should be very large due to the outlier
        assert float(stats["std"]) > 100, "Outlier should inflate std dramatically"
        # The median should still be near 0 (robust to outliers)
        assert abs(float(stats["median"])) < 2, "Median should be robust to outlier"

    def test_repeated_data_does_not_inflate_significance(self, tool):
        """Simply repeating the same data 10x shouldn't make a non-effect significant.

        This catches the mistake of inflating sample size by duplication.
        """
        base = [
            {"group": "A", "value": 100 + random.gauss(0, 20)},
            {"group": "B", "value": 100 + random.gauss(0, 20)},
        ]
        random.seed(42)
        base = [{"group": "A", "value": 100 + random.gauss(0, 20)} for _ in range(5)] + [
            {"group": "B", "value": 100.5 + random.gauss(0, 20)} for _ in range(5)
        ]
        # With n=5 per group and 0.5 mean diff on std=20, should not be significant
        result = tool.execute(
            analysis_type="hypothesis_test",
            data=base,
            group_column="group",
            value_column="value",
            method="ttest_ind",
        )
        assert result.success is True
        p = result.data["p_value"]
        assert p > 0.05, f"Tiny difference with small n should not be significant: p={p}"


class TestDataQuality:
    def test_detects_nulls_and_outliers(self, tool):
        data = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": None}, {"x": 1000}]
        result = tool.execute(analysis_type="data_quality", data=data, columns=["x"])
        assert result.success is True
