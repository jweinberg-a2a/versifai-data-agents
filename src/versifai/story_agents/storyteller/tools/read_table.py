"""
ReadTableTool — read CSV result tables produced by the DataScientist.

Provides listing, reading, and summary statistics for output tables.
"""

from __future__ import annotations

import csv
import os
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class ReadTableTool(BaseTool):
    """Read and summarize CSV result tables from the DataScientist."""

    def __init__(self, tables_path: str) -> None:
        self._tables_path = tables_path

    @property
    def name(self) -> str:
        return "read_table"

    @property
    def description(self) -> str:
        return (
            "Read CSV result tables from the DataScientist's output. "
            "Operations: 'list' (all tables), 'read' (full CSV content), "
            "'summary' (row count, columns, sample rows)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "read", "summary"],
                    "description": "Which operation to run.",
                },
                "filename": {
                    "type": "string",
                    "description": "CSV filename (for 'read' and 'summary').",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows to return for 'read' (default 100).",
                },
            },
            "required": ["operation"],
        }

    def _list_tables(self) -> list[str]:
        """List all CSV files in the tables directory."""
        if not os.path.isdir(self._tables_path):
            return []
        return sorted(
            f for f in os.listdir(self._tables_path)
            if f.lower().endswith(".csv")
        )

    def _execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs["operation"]

        if operation == "list":
            tables = self._list_tables()
            return ToolResult(
                success=True,
                data={"count": len(tables), "tables": tables},
                summary=f"{len(tables)} CSV tables available.",
            )

        elif operation == "read":
            filename = kwargs.get("filename", "")
            if not filename:
                return ToolResult(success=False, error="filename required.")

            path = os.path.join(self._tables_path, filename)
            if not os.path.isfile(path):
                return ToolResult(success=False, error=f"File not found: {filename}")

            max_rows = kwargs.get("max_rows", 100)
            rows = []
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(dict(row))

            return ToolResult(
                success=True,
                data={
                    "filename": filename,
                    "headers": headers,
                    "rows": rows,
                    "rows_returned": len(rows),
                    "truncated": len(rows) >= max_rows,
                },
                summary=f"Read {len(rows)} rows from {filename} ({len(headers)} columns).",
            )

        elif operation == "summary":
            filename = kwargs.get("filename", "")
            if not filename:
                return ToolResult(success=False, error="filename required.")

            path = os.path.join(self._tables_path, filename)
            if not os.path.isfile(path):
                return ToolResult(success=False, error=f"File not found: {filename}")

            total_rows = 0
            headers: list[str] = []
            sample_rows: list[dict] = []
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                for i, row in enumerate(reader):
                    total_rows += 1
                    if i < 5:
                        sample_rows.append(dict(row))

            return ToolResult(
                success=True,
                data={
                    "filename": filename,
                    "total_rows": total_rows,
                    "columns": headers,
                    "column_count": len(headers),
                    "sample_rows": sample_rows,
                },
                summary=f"{filename}: {total_rows} rows, {len(headers)} columns.",
            )

        return ToolResult(success=False, error=f"Unknown operation: {operation}")
