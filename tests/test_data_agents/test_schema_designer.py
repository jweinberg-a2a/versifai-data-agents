"""Tests for SchemaDesignerTool."""

from __future__ import annotations

import pytest

from versifai.data_agents.engineer.tools.schema_designer import SchemaDesignerTool


class TestSchemaDesigner:
    def test_create_schema_shorthand(self, mock_project_config):
        tool = SchemaDesignerTool(cfg=mock_project_config)
        result = tool.execute(
            source_name="test_source",
            table_name="test_table",
            column_names=["fips_code", "enrollment", "value_pct"],
        )
        assert result.success is True
        assert result.data["column_count"] >= 3
        assert "schema" in result.data
        assert "columns" in result.data["schema"]

    def test_returns_create_table_sql(self, mock_project_config):
        tool = SchemaDesignerTool(cfg=mock_project_config)
        result = tool.execute(
            source_name="src1",
            table_name="tbl1",
            column_names=["county_fips", "total_enrollment"],
        )
        assert result.success is True
        assert "sql" in result.data or "create_table_sql" in result.data

    def test_missing_source_name_error(self, mock_project_config):
        tool = SchemaDesignerTool(cfg=mock_project_config)
        result = tool.execute(
            source_name="",
            table_name="tbl",
            column_names=["col1"],
        )
        assert result.success is False

    def test_missing_table_name_error(self, mock_project_config):
        tool = SchemaDesignerTool(cfg=mock_project_config)
        result = tool.execute(
            source_name="src",
            table_name="",
            column_names=["col1"],
        )
        assert result.success is False

    def test_type_inference(self, mock_project_config):
        """Columns with year/fips/count patterns should get inferred types."""
        tool = SchemaDesignerTool(cfg=mock_project_config)
        result = tool.execute(
            source_name="src2",
            table_name="tbl2",
            column_names=["year", "county_fips", "total_count", "latitude"],
        )
        assert result.success is True

    def test_schema_registry(self, mock_project_config):
        tool = SchemaDesignerTool(cfg=mock_project_config)
        tool.execute(
            source_name="reg_test",
            table_name="reg_table",
            column_names=["col1"],
        )
        schema = tool.get_schema("reg_test")
        assert schema is not None

    def test_full_columns_mode(self, mock_project_config):
        tool = SchemaDesignerTool(cfg=mock_project_config)
        result = tool.execute(
            source_name="full_src",
            table_name="full_tbl",
            columns=[
                {
                    "source_name": "RAW_FIPS",
                    "target_name": "county_fips",
                    "data_type": "STRING",
                    "description": "County FIPS code",
                    "nullable": False,
                    "is_join_key": True,
                },
                {
                    "source_name": "ENR",
                    "target_name": "enrollment",
                    "data_type": "DOUBLE",
                    "description": "Total enrollment",
                },
            ],
        )
        assert result.success is True
        assert result.data["has_join_key"] is True

    def test_no_config_raises(self):
        with pytest.raises((ValueError, TypeError)):
            SchemaDesignerTool(cfg=None)
