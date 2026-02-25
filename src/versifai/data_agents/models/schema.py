"""
Data models for target schema definitions designed by the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ColumnDefinition:
    """Definition of a single column in a target schema."""

    target_name: str               # snake_case target column name
    data_type: str                 # spark SQL type: STRING, INT, DOUBLE, DATE, etc.
    source_name: str               # original column name in the source file
    description: str = ""
    nullable: bool = True
    is_fips: bool = False          # True if this is a FIPS code column
    is_metadata: bool = False      # True if this is an added metadata column
    transform_expression: str = "" # optional SQL/Python expression to apply

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "data_type": self.data_type,
            "source_name": self.source_name,
            "description": self.description,
            "nullable": self.nullable,
            "is_fips": self.is_fips,
            "is_metadata": self.is_metadata,
            "transform_expression": self.transform_expression,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ColumnDefinition:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TargetSchema:
    """
    Complete schema definition for a target Unity Catalog table.

    Designed by the agent after profiling a data source.
    """

    source_name: str                              # e.g. "cdc_svi"
    table_name: str                               # full table: catalog.schema.table
    columns: list[ColumnDefinition] = field(default_factory=list)
    description: str = ""
    partition_columns: list[str] = field(default_factory=list)

    # Mapping from historical file patterns to column name overrides
    # e.g. {"2018": {"old_col": "new_col"}, "2020": {}}
    historical_mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def fips_column(self) -> str | None:
        """Return the name of the FIPS column, if any."""
        for col in self.columns:
            if col.is_fips:
                return col.target_name
        return None

    @property
    def column_names(self) -> list[str]:
        return [c.target_name for c in self.columns]

    def get_column(self, target_name: str) -> ColumnDefinition | None:
        for col in self.columns:
            if col.target_name == target_name:
                return col
        return None

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "table_name": self.table_name,
            "columns": [c.to_dict() for c in self.columns],
            "description": self.description,
            "partition_columns": self.partition_columns,
            "historical_mappings": self.historical_mappings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TargetSchema:
        cols = [ColumnDefinition.from_dict(c) for c in d.get("columns", [])]
        return cls(
            source_name=d["source_name"],
            table_name=d["table_name"],
            columns=cols,
            description=d.get("description", ""),
            partition_columns=d.get("partition_columns", []),
            historical_mappings=d.get("historical_mappings", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_create_table_sql(self) -> str:
        """Generate a CREATE TABLE statement for this schema."""
        col_defs = []
        for col in self.columns:
            nullable = "" if col.nullable else " NOT NULL"
            comment = f' COMMENT "{col.description}"' if col.description else ""
            col_defs.append(f"  {col.target_name} {col.data_type}{nullable}{comment}")

        cols_sql = ",\n".join(col_defs)
        partition = ""
        if self.partition_columns:
            partition = f"\nPARTITIONED BY ({', '.join(self.partition_columns)})"

        comment = f'\nCOMMENT "{self.description}"' if self.description else ""

        return (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} (\n"
            f"{cols_sql}\n"
            f"){comment}{partition}\n"
            f"USING DELTA"
        )
