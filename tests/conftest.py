"""Shared test fixtures."""

import pytest


@pytest.fixture
def mock_dbutils():
    """Mock Databricks dbutils for testing outside Databricks."""

    class MockWidgets:
        def text(self, name, default, label=""):
            pass

        def get(self, name):
            return ""

    class MockDBUtils:
        widgets = MockWidgets()

    return MockDBUtils()
