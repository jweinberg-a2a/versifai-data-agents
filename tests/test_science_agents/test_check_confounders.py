"""Tests for CheckConfoundersTool — Simpson's Paradox detection."""

from __future__ import annotations

import pytest

from versifai.science_agents.scientist.tools.check_confounders import CheckConfoundersTool


@pytest.fixture
def tool():
    return CheckConfoundersTool()


class TestSimpsonsParadox:
    def test_paradox_detected(self, tool, simpsons_paradox_data):
        """BEHAVIORAL: Aggregate positive trend but every subgroup negative → FLAGGED.

        Classic Simpson's Paradox: overall x-y correlation is positive
        (because group B has both higher x and higher y), but within each
        group the correlation is negative.
        """
        result = tool.execute(
            data=simpsons_paradox_data,
            outcome_column="y",
            predictor_column="x",
            grouping_columns=["group"],
        )
        assert result.success is True
        # Should flag paradox/reversal/confounding
        data = result.data
        has_paradox = (
            data.get("paradox_detected", False)
            or data.get("simpsons_paradox", False)
            or data.get("direction_reversal", False)
            or any(
                d.get("paradox_detected", False) or d.get("direction_reversal", False)
                for d in data.get("decompositions", data.get("grouping_results", []))
                if isinstance(d, dict)
            )
        )
        assert has_paradox, f"Simpson's Paradox should be detected. Result: {data}"

    def test_no_paradox_when_consistent(self, tool):
        """BEHAVIORAL: Aggregate and subgroups agree → NOT flagged.

        Both groups have positive x-y relationship and aggregate is positive.
        """
        data = [
            # Group A: positive slope
            {"group": "A", "x": 1, "y": 10},
            {"group": "A", "x": 2, "y": 20},
            {"group": "A", "x": 3, "y": 30},
            {"group": "A", "x": 4, "y": 40},
            {"group": "A", "x": 5, "y": 50},
            # Group B: also positive slope
            {"group": "B", "x": 6, "y": 60},
            {"group": "B", "x": 7, "y": 70},
            {"group": "B", "x": 8, "y": 80},
            {"group": "B", "x": 9, "y": 90},
            {"group": "B", "x": 10, "y": 100},
        ]
        result = tool.execute(
            data=data,
            outcome_column="y",
            predictor_column="x",
            grouping_columns=["group"],
        )
        assert result.success is True
        result_data = result.data
        has_paradox = result_data.get("paradox_detected", False) or result_data.get(
            "simpsons_paradox", False
        )
        assert not has_paradox, "No paradox should be detected when trends are consistent"

    def test_returns_decomposed_stats(self, tool, simpsons_paradox_data):
        """Result includes per-group statistics."""
        result = tool.execute(
            data=simpsons_paradox_data,
            outcome_column="y",
            predictor_column="x",
            grouping_columns=["group"],
        )
        assert result.success is True
        data = result.data
        # Should have aggregate stats and per-group breakdown
        has_aggregate = "aggregate" in data or "overall" in data
        has_groups = (
            "decompositions" in data
            or "grouping_results" in data
            or "groups" in data
            or "subgroups" in data
        )
        assert has_aggregate or has_groups, "Should include group-level decomposition"

    def test_missing_columns_error(self, tool):
        data = [{"x": 1, "y": 2}]
        result = tool.execute(
            data=data,
            outcome_column="",
            predictor_column="x",
            grouping_columns=["group"],
        )
        assert result.success is False
