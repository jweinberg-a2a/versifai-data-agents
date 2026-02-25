"""Shared test fixtures."""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def mock_dbutils():
    """Mock Databricks dbutils for testing outside Databricks."""

    class MockWidgets:
        def text(self, name, default, label=""):
            pass

        def get(self, name):
            return ""

    class MockFS:
        def ls(self, path):
            return []

        def cp(self, src, dst, recurse=False):
            pass

        def rm(self, path, recurse=False):
            pass

    class MockDBUtils:
        widgets = MockWidgets()
        fs = MockFS()

    return MockDBUtils()


@pytest.fixture
def mock_catalog_config():
    """Minimal CatalogConfig for testing."""
    from versifai.core.config import CatalogConfig

    return CatalogConfig(
        catalog="test_catalog",
        schema="test_schema",
        volume_path="/tmp/test_volume",
        staging_path="/tmp/test_staging",
    )


@pytest.fixture
def mock_project_config(tmp_path, mock_catalog_config):
    """Minimal ProjectConfig for testing."""
    from versifai.data_agents.engineer.config import ProjectConfig

    return ProjectConfig(
        name="Test Project",
        catalog=mock_catalog_config.catalog,
        schema=mock_catalog_config.schema,
        volume_path=str(tmp_path / "volume"),
        staging_path=str(tmp_path / "staging"),
    )


@pytest.fixture
def sample_csv(tmp_path):
    """Write a simple CSV with known values and return its path."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "id,name,score,grade,fips_code\n"
        "1,Alice,95.5,A,06001\n"
        "2,Bob,82.3,B,06037\n"
        "3,Charlie,78.1,C,36061\n"
        "4,Diana,91.0,A,17031\n"
        "5,Eve,,B,48201\n"  # null score
        "6,Frank,88.2,B,04013\n"
        "7,Grace,67.4,D,12086\n"
        "8,Hank,73.9,C,06059\n"
        "9,Ivy,99.1,A,36047\n"
        "10,Jack,55.0,F,48113\n"
    )
    return str(csv_path)


@pytest.fixture
def sample_tsv(tmp_path):
    """Write a tab-separated file and return its path."""
    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text(
        "col_a\tcol_b\tcol_c\n"
        "1\thello\t3.14\n"
        "2\tworld\t2.72\n"
    )
    return str(tsv_path)


@pytest.fixture
def sample_findings(tmp_path):
    """Write a realistic findings.json and return its directory path."""
    results_path = tmp_path / "results"
    results_path.mkdir()
    findings = [
        {
            "research_question_id": "theme0",
            "title": "Significant geographic disparity found",
            "finding": "Counties with higher SVI scores show 15% lower enrollment rates.",
            "evidence": "t-test p=0.0003, Cohen's d=0.82",
            "significance": "high",
            "p_value": 0.0003,
            "effect_size": 0.82,
            "visualization_path": "charts/theme0_svi_enrollment.png",
        },
        {
            "research_question_id": "theme0",
            "title": "Urban-rural gap persists",
            "finding": "Rural counties have 8% lower rates than urban.",
            "evidence": "Mann-Whitney U p=0.012, effect size=0.45",
            "significance": "medium",
            "p_value": 0.012,
            "effect_size": 0.45,
        },
        {
            "research_question_id": "theme1",
            "title": "No significant age effect",
            "finding": "Age distribution does not predict enrollment.",
            "evidence": "Regression p=0.78, R²=0.003",
            "significance": "low",
            "p_value": 0.78,
            "effect_size": 0.003,
        },
        {
            "research_question_id": "theme1",
            "title": "Weak seasonal pattern",
            "finding": "Q4 shows marginally higher enrollment.",
            "evidence": "ANOVA p=0.45, eta²=0.02",
            "significance": "low",
            "p_value": 0.45,
            "effect_size": 0.02,
        },
    ]
    (results_path / "findings.json").write_text(json.dumps(findings, indent=2))
    (results_path / "charts").mkdir()
    (results_path / "tables").mkdir()
    (results_path / "notes").mkdir()
    return str(results_path)


@pytest.fixture
def simpsons_paradox_data():
    """Data where aggregate trend is positive but every subgroup is negative.

    Classic Simpson's Paradox: overall correlation between X and Y is positive,
    but within each group (A, B), correlation is negative.
    """
    return [
        # Group A: low X, high Y baseline — negative slope within group
        {"group": "A", "x": 1, "y": 90},
        {"group": "A", "x": 2, "y": 88},
        {"group": "A", "x": 3, "y": 85},
        {"group": "A", "x": 4, "y": 82},
        {"group": "A", "x": 5, "y": 80},
        # Group B: high X, higher Y baseline — negative slope within group
        {"group": "B", "x": 6, "y": 110},
        {"group": "B", "x": 7, "y": 108},
        {"group": "B", "x": 8, "y": 105},
        {"group": "B", "x": 9, "y": 102},
        {"group": "B", "x": 10, "y": 100},
    ]


@pytest.fixture
def clearly_different_groups():
    """Two groups with large, obvious mean difference (for behavioral tests)."""
    import random

    random.seed(42)
    group_a = [{"group": "A", "value": random.gauss(100, 5)} for _ in range(50)]
    group_b = [{"group": "B", "value": random.gauss(200, 5)} for _ in range(50)]
    return group_a + group_b


@pytest.fixture
def nearly_identical_groups():
    """Two groups with nearly identical means (for false-significance tests)."""
    import random

    random.seed(42)
    group_a = [{"group": "A", "value": random.gauss(100, 20)} for _ in range(30)]
    group_b = [{"group": "B", "value": random.gauss(100.001, 20)} for _ in range(30)]
    return group_a + group_b
