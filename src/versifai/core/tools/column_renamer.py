"""
Tool: rename_columns

Renames columns on existing Delta tables in Unity Catalog using
ALTER TABLE ... RENAME COLUMN. This is a metadata-only operation in
Databricks — no data rewrite occurs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from versifai._utils.naming import to_snake_case
from versifai.core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from versifai.data_agents.engineer.config import ProjectConfig


class RenameColumnsTool(BaseTool):
    """
    Rename columns on an existing Unity Catalog Delta table.

    Accepts a fully qualified table name and a dict mapping current
    column names to new descriptive names.  Validates that new names
    are snake_case, then executes ALTER TABLE RENAME COLUMN for each
    mapping.  Skips metadata columns and the join key to avoid breaking
    downstream consumers.
    """

    def __init__(self, cfg: ProjectConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "rename_columns"

    @property
    def description(self) -> str:
        return (
            "Rename columns on an existing Delta table in Unity Catalog. "
            "Provide the fully qualified table_name and a column_renames dict "
            "mapping current column names to new descriptive snake_case names. "
            "This is a metadata-only operation — no data is rewritten. "
            "Use this to fix cryptic/abbreviated column names on tables that "
            "have already been loaded."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": (
                        "Fully qualified table name (catalog.schema.table)."
                    ),
                },
                "column_renames": {
                    "type": "object",
                    "description": (
                        "Dict mapping current column names to new descriptive names. "
                        "Example: {'ep_pov150': 'pct_below_150_poverty', "
                        "'e_totpop': 'total_population_estimate'}"
                    ),
                },
            },
            "required": ["table_name", "column_renames"],
        }

    def _execute(
        self, table_name: str, column_renames: dict, **kwargs
    ) -> ToolResult:
        if not column_renames:
            return ToolResult(
                success=False,
                error="column_renames is empty. Provide at least one mapping.",
            )

        cfg = self._cfg
        # Protect metadata and join key columns from accidental rename
        protected = set(cfg.metadata_column_names + [cfg.join_key.column_name])

        # Validate new names
        warnings: list[str] = []
        validated: dict[str, str] = {}

        for old_name, new_name in column_renames.items():
            old_lower = old_name.lower()
            if old_lower in protected:
                warnings.append(
                    f"Skipped '{old_name}' — protected column (metadata/join key)."
                )
                continue

            # Ensure snake_case
            new_snake = to_snake_case(new_name)
            if new_snake != new_name:
                warnings.append(
                    f"'{new_name}' normalized to '{new_snake}' (snake_case)."
                )

            if old_lower == new_snake:
                warnings.append(
                    f"Skipped '{old_name}' — already named '{new_snake}'."
                )
                continue

            validated[old_lower] = new_snake

        if not validated:
            return ToolResult(
                success=True,
                data={"renamed": 0, "warnings": warnings},
                summary=(
                    f"No columns to rename on {table_name}. "
                    f"Warnings: {'; '.join(warnings) if warnings else 'none'}"
                ),
            )

        # Execute ALTER TABLE RENAME COLUMN for each mapping
        succeeded: list[dict] = []
        failed: list[dict] = []

        for old_col, new_col in validated.items():
            sql = (
                f"ALTER TABLE {table_name} "
                f"RENAME COLUMN `{old_col}` TO `{new_col}`"
            )
            try:
                self._execute_sql(sql)
                succeeded.append({"old": old_col, "new": new_col})
            except Exception as exc:
                failed.append({"old": old_col, "new": new_col, "error": str(exc)})

        if failed and not succeeded:
            return ToolResult(
                success=False,
                error=(
                    f"All {len(failed)} renames failed on {table_name}. "
                    f"First error: {failed[0]['error']}"
                ),
                data={"failed": failed, "warnings": warnings},
            )

        return ToolResult(
            success=True,
            data={
                "table_name": table_name,
                "renamed": len(succeeded),
                "succeeded": succeeded,
                "failed": failed,
                "warnings": warnings,
            },
            summary=(
                f"Renamed {len(succeeded)} column(s) on {table_name}. "
                f"Failed: {len(failed)}. Warnings: {len(warnings)}."
            ),
        )

    @staticmethod
    def _execute_sql(sql: str) -> None:
        """Execute a single SQL statement. Tries Spark, falls back to SDK."""
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark:
                spark.sql(sql)
                return
        except Exception:
            pass

        import os

        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient(
            host=os.environ.get("DATABRICKS_HOST", ""),
            token=os.environ.get("DATABRICKS_TOKEN", ""),
        )
        warehouses = list(client.warehouses.list())
        if not warehouses:
            raise RuntimeError("No SQL warehouse available")

        result = client.statement_execution.execute_statement(
            warehouse_id=warehouses[0].id,
            statement=sql,
            wait_timeout="30s",
        )
        # Check for error status from the SDK
        if result.status and result.status.error:
            raise RuntimeError(
                f"SQL error: {result.status.error.message}"
            )
