"""
ReadChartTool — list and inspect chart files produced by the DataScientist.

Provides an inventory of available charts and their metadata (SQL, interpretation)
from the notes files, so the storyteller can decide which charts to include.
"""

from __future__ import annotations

import json
import os
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class ReadChartTool(BaseTool):
    """List and inspect chart PNGs and their metadata."""

    def __init__(self, charts_path: str, notes_path: str) -> None:
        self._charts_path = charts_path
        self._notes_path = notes_path

    @property
    def name(self) -> str:
        return "read_chart"

    @property
    def description(self) -> str:
        return (
            "List and inspect chart files from the DataScientist's output. "
            "Operations: 'list' (all charts), 'by_theme' (charts for a theme), "
            "'metadata' (get SQL/interpretation notes for a specific chart)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "by_theme", "metadata"],
                    "description": "Which query to run.",
                },
                "theme_id": {
                    "type": "string",
                    "description": "Theme ID (e.g., 'theme_0'). For 'by_theme'.",
                },
                "chart_filename": {
                    "type": "string",
                    "description": "Chart filename for 'metadata' operation.",
                },
            },
            "required": ["operation"],
        }

    def _list_charts(self) -> list[str]:
        """List all chart files in the charts directory."""
        if not os.path.isdir(self._charts_path):
            return []
        return sorted(
            f for f in os.listdir(self._charts_path)
            if f.lower().endswith((".png", ".html", ".svg"))
        )

    def _load_notes(self, theme_id: str) -> dict:
        """Load notes JSON for a specific theme."""
        notes_file = os.path.join(self._notes_path, f"{theme_id}_notes.json")
        if not os.path.isfile(notes_file):
            return {}
        try:
            with open(notes_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs["operation"]

        if operation == "list":
            charts = self._list_charts()
            return ToolResult(
                success=True,
                data={"count": len(charts), "charts": charts},
                summary=f"{len(charts)} charts available.",
            )

        elif operation == "by_theme":
            theme_id = kwargs.get("theme_id", "")
            all_charts = self._list_charts()
            # Match by theme prefix patterns (theme0_, t0_, theme_0_)
            seq = theme_id.replace("theme_", "")
            prefixes = [f"theme{seq}_", f"t{seq}_", f"{theme_id}_"]
            matches = [
                c for c in all_charts
                if any(c.lower().startswith(p) for p in prefixes)
            ]
            return ToolResult(
                success=True,
                data={"theme": theme_id, "count": len(matches), "charts": matches},
                summary=f"{len(matches)} charts for {theme_id}.",
            )

        elif operation == "metadata":
            filename = kwargs.get("chart_filename", "")
            if not filename:
                return ToolResult(success=False, error="chart_filename required.")

            # Try to find notes by scanning all theme notes files
            chart_metadata: dict = {}
            if os.path.isdir(self._notes_path):
                for notes_file in os.listdir(self._notes_path):
                    if not notes_file.endswith("_notes.json"):
                        continue
                    notes = self._load_notes(
                        notes_file.replace("_notes.json", "")
                    )
                    # Notes may contain chart metadata keyed by filename
                    for key, value in notes.items():
                        if isinstance(value, dict) and filename in str(value):
                            chart_metadata[key] = value

            # Also return the file path for view_chart
            chart_path = os.path.join(self._charts_path, filename)
            exists = os.path.isfile(chart_path)

            return ToolResult(
                success=True,
                data={
                    "filename": filename,
                    "exists": exists,
                    "path": chart_path if exists else "",
                    "notes_metadata": chart_metadata,
                },
                summary=f"Metadata for {filename} (exists={exists}).",
            )

        return ToolResult(success=False, error=f"Unknown operation: {operation}")
