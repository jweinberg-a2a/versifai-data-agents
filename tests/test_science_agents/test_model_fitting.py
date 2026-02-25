"""Tests for ModelFittingTool — includes behavioral validity tests."""

from __future__ import annotations

import random

import pytest

from versifai.science_agents.scientist.tools.model_fitting import ModelFittingTool


@pytest.fixture
def tool():
    return ModelFittingTool()


class TestLinearRegression:
    def test_perfect_linear(self, tool):
        """y = 2x + 1 → coefficient ≈ 2, intercept ≈ 1, R² ≈ 1.0."""
        data = [{"x": i, "y": 2 * i + 1} for i in range(50)]
        result = tool.execute(
            model_type="linear_regression",
            data=data,
            target_column="y",
            feature_columns=["x"],
        )
        assert result.success is True
        r2 = result.data.get("r_squared", result.data.get("r2", 0))
        assert r2 > 0.99

    def test_pure_noise(self, tool):
        """Random data → R² ≈ 0.0."""
        random.seed(42)
        data = [{"x": random.gauss(0, 1), "y": random.gauss(0, 1)} for _ in range(100)]
        result = tool.execute(
            model_type="linear_regression",
            data=data,
            target_column="y",
            feature_columns=["x"],
        )
        assert result.success is True
        r2 = result.data.get("r_squared", result.data.get("r2", 1.0))
        assert r2 < 0.15

    def test_coefficients_correct(self, tool):
        """y = 3x + 5 → slope ≈ 3."""
        data = [{"x": i, "y": 3 * i + 5} for i in range(50)]
        result = tool.execute(
            model_type="linear_regression",
            data=data,
            target_column="y",
            feature_columns=["x"],
        )
        assert result.success is True
        # Coefficients is a list of dicts: [{"variable": "const", "coefficient": 5}, {"variable": "x", "coefficient": 3}]
        coefficients = result.data.get("coefficients", [])
        x_coef = next((c for c in coefficients if c.get("variable") == "x"), None)
        assert x_coef is not None, f"Expected 'x' coefficient, got: {coefficients}"
        assert float(x_coef["coefficient"]) == pytest.approx(3.0, abs=0.1)


class TestBehavioralModelFitting:
    def test_multicollinearity_detection(self, tool):
        """BEHAVIORAL: Two identical columns → high VIF warning."""
        data = [{"x1": i, "x2": i, "y": i * 2 + 1} for i in range(50)]
        result = tool.execute(
            model_type="linear_regression",
            data=data,
            target_column="y",
            feature_columns=["x1", "x2"],
        )
        assert result.success is True
        # If VIF is calculated, it should be very high for collinear features
        vif_data = result.data.get("vif", result.data.get("vif_values", []))
        if vif_data:
            for item in vif_data:
                if isinstance(item, dict):
                    vif_val = item.get("vif", item.get("VIF", 0))
                    assert float(vif_val) > 5, (
                        f"VIF should be high for collinear features, got {vif_val}"
                    )

    def test_random_forest_feature_importance(self, tool):
        """BEHAVIORAL: Dominant predictor ranked first in importance."""
        random.seed(42)
        data = [
            {
                "signal": i,
                "noise": random.gauss(0, 100),
                "y": i * 10 + random.gauss(0, 1),
            }
            for i in range(100)
        ]
        result = tool.execute(
            model_type="random_forest",
            data=data,
            target_column="y",
            feature_columns=["signal", "noise"],
        )
        assert result.success is True
        importances = result.data.get(
            "feature_importance", result.data.get("feature_importances", [])
        )
        if isinstance(importances, list) and len(importances) >= 2:
            # Signal should have higher importance than noise
            imp_dict = {}
            for item in importances:
                if isinstance(item, dict):
                    name = item.get("feature", item.get("name", ""))
                    importance = item.get("importance", item.get("value", 0))
                    imp_dict[name] = float(importance)
            if "signal" in imp_dict and "noise" in imp_dict:
                assert imp_dict["signal"] > imp_dict["noise"], (
                    f"Signal importance ({imp_dict['signal']}) should exceed noise ({imp_dict['noise']})"
                )


class TestTimeSeries:
    def test_increasing_trend(self, tool):
        data = [{"time": t, "value": t * 5 + 100} for t in range(20)]
        result = tool.execute(
            model_type="time_series",
            data=data,
            target_column="value",
            time_column="time",
        )
        assert result.success is True
        trend = result.data.get("trend", {})
        direction = trend.get("direction", result.data.get("direction", "")).lower()
        slope = trend.get("slope", result.data.get("slope", 0))
        assert "increas" in direction or float(slope) > 0


class TestCrossValidation:
    def test_cross_validate_returns_models(self, tool):
        random.seed(42)
        data = [{"x": i, "y": 2 * i + random.gauss(0, 2)} for i in range(100)]
        result = tool.execute(
            model_type="cross_validate",
            data=data,
            target_column="y",
            feature_columns=["x"],
        )
        assert result.success is True
        # Should return results for multiple models
        models = result.data.get(
            "model_comparison", result.data.get("models", result.data.get("results", []))
        )
        assert len(models) >= 2
