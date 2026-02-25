"""
Tool: view_chart

Reads an existing chart image (PNG) from the results directory and returns
it so the agent can visually inspect it.  The orchestrator's image_path
pipeline handles base64 encoding and injection into the conversation as a
Claude vision content block.

Also supports listing all available chart files when called without a
filename, so the agent can discover what exists before viewing.
"""

from __future__ import annotations

import os

from versifai.core.tools.base import BaseTool, ToolResult


class ViewChartTool(BaseTool):
    """
    View an existing chart image from the results volume.

    The agent calls this to visually inspect previously created charts —
    checking for formatting issues, label overlap, color scale problems,
    missing data, etc.  Returns the image so Claude can see it.
    """

    def __init__(self, charts_path: str = "", tables_path: str = "") -> None:
        super().__init__()
        self._charts_path = charts_path
        self._tables_path = tables_path

    @property
    def name(self) -> str:
        return "view_chart"

    @property
    def description(self) -> str:
        return (
            "View an existing chart image (PNG) or list available charts. "
            "Call with a filename to see the rendered chart — the image is "
            "returned so you can visually review it for formatting issues, "
            "overlapping labels, color scale problems, missing data, etc. "
            "Call without a filename (or with filename='list') to get a list "
            "of all available chart and table files."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "The chart filename to view (e.g., 'theme1_choropleth.png'). "
                        "Use 'list' or omit to list all available files."
                    ),
                },
            },
            "required": [],
        }

    def _execute(self, filename: str = "", **kwargs) -> ToolResult:
        # List mode — show what's available
        if not filename or filename.lower() == "list":
            return self._list_files()

        # View mode — read the chart image
        # Try charts directory first, then tables
        chart_path = os.path.join(self._charts_path, filename) if self._charts_path else ""
        table_path = os.path.join(self._tables_path, filename) if self._tables_path else ""

        if chart_path and os.path.isfile(chart_path):
            full_path = chart_path
        elif table_path and os.path.isfile(table_path):
            full_path = table_path
        else:
            # Try both directories for a helpful error
            available = self._get_available_files()
            return ToolResult(
                success=False,
                error=(
                    f"File not found: '{filename}'\n\n"
                    f"Available files ({len(available)}):\n"
                    + "\n".join(f"  - {f}" for f in available[:30])
                ),
            )

        # Only PNG images can be viewed visually
        if full_path.lower().endswith(".png"):
            return ToolResult(
                success=True,
                data={"filename": filename, "path": full_path},
                summary=f"Viewing chart: {filename}",
                image_path=full_path,
            )
        elif full_path.lower().endswith(".csv"):
            # For CSVs, read and return the content as text
            try:
                with open(full_path) as f:
                    content = f.read()
                # Truncate very large CSVs
                if len(content) > 10000:
                    content = content[:10000] + "\n\n[... truncated ...]"
                return ToolResult(
                    success=True,
                    data={"filename": filename, "content": content},
                    summary=f"Read table file: {filename} ({len(content)} chars)",
                )
            except OSError as e:
                return ToolResult(success=False, error=f"Cannot read {filename}: {e}")
        elif full_path.lower().endswith(".html"):
            return ToolResult(
                success=True,
                data={"filename": filename, "note": "HTML chart — cannot display inline"},
                summary=f"Chart '{filename}' is HTML (interactive plotly) — cannot view inline. Recreate as PNG to review visually.",
            )
        else:
            return ToolResult(
                success=False,
                error=f"Unsupported file type: {filename}. Only PNG images can be viewed.",
            )

    def _get_available_files(self) -> list[str]:
        """Gather all files from charts and tables directories."""
        files = []
        for path, _prefix in [(self._charts_path, ""), (self._tables_path, "")]:
            if path and os.path.isdir(path):
                files.extend(sorted(os.listdir(path)))
        return sorted(set(files))

    def _list_files(self) -> ToolResult:
        """List all available chart and table files."""
        charts = []
        tables = []

        if self._charts_path and os.path.isdir(self._charts_path):
            charts = sorted(os.listdir(self._charts_path))
        if self._tables_path and os.path.isdir(self._tables_path):
            tables = sorted(os.listdir(self._tables_path))

        data = {"charts": charts, "tables": tables}
        summary_parts = []
        if charts:
            summary_parts.append(f"Charts ({len(charts)}): {', '.join(charts[:10])}")
            if len(charts) > 10:
                summary_parts[-1] += f" ... +{len(charts) - 10} more"
        if tables:
            summary_parts.append(f"Tables ({len(tables)}): {', '.join(tables[:10])}")
            if len(tables) > 10:
                summary_parts[-1] += f" ... +{len(tables) - 10} more"

        return ToolResult(
            success=True,
            data=data,
            summary="; ".join(summary_parts) if summary_parts else "No files found.",
        )
