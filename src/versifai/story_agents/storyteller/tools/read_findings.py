"""
ReadFindingsTool — load and query the DataScientist's findings.json.

Provides filtering by theme, significance level, and free-text search
so the storyteller can efficiently find the evidence it needs.
"""

from __future__ import annotations

import json
import os
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class ReadFindingsTool(BaseTool):
    """Load and query structured findings from the DataScientist."""

    def __init__(self, results_path: str) -> None:
        self._results_path = results_path
        self._findings: list[dict] | None = None  # lazy-loaded cache

    @property
    def name(self) -> str:
        return "read_findings"

    @property
    def description(self) -> str:
        return (
            "Load and query the DataScientist's findings.json. "
            "Operations: 'list' (all findings), 'get' (by index), "
            "'by_theme' (filter by research_question_id), "
            "'high_significance' (only high/critical findings), "
            "'search' (free-text search across finding text)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "get", "by_theme", "high_significance", "search"],
                    "description": "Which query to run.",
                },
                "index": {
                    "type": "integer",
                    "description": "Finding index (for 'get' operation).",
                },
                "theme_id": {
                    "type": "string",
                    "description": "Theme ID to filter by (e.g., 'theme_0'). For 'by_theme'.",
                },
                "query": {
                    "type": "string",
                    "description": "Search text for 'search' operation.",
                },
            },
            "required": ["operation"],
        }

    def _load_findings(self) -> list[dict]:
        """Load findings from disk (cached after first call)."""
        if self._findings is not None:
            return self._findings

        path = os.path.join(self._results_path, "findings.json")
        if not os.path.isfile(path):
            self._findings = []
            return self._findings

        with open(path) as f:
            data = json.load(f)

        self._findings = data if isinstance(data, list) else []
        return self._findings

    def _execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs["operation"]
        findings = self._load_findings()

        if operation == "list":
            summaries = []
            for i, f in enumerate(findings):
                summaries.append(
                    {
                        "index": i,
                        "theme": f.get("research_question_id", ""),
                        "title": f.get("title", ""),
                        "significance": f.get("significance", ""),
                        "effect_size": f.get("effect_size", ""),
                    }
                )
            return ToolResult(
                success=True,
                data={"total": len(findings), "findings": summaries},
                summary=f"Listed {len(findings)} findings.",
            )

        elif operation == "get":
            idx = kwargs.get("index", 0)
            if idx < 0 or idx >= len(findings):
                return ToolResult(
                    success=False,
                    error=f"Index {idx} out of range (0-{len(findings) - 1}).",
                )
            return ToolResult(
                success=True,
                data=findings[idx],
                summary=f"Finding {idx}: {findings[idx].get('title', '')}",
            )

        elif operation == "by_theme":
            theme_id = kwargs.get("theme_id", "")
            matches = [f for f in findings if f.get("research_question_id", "") == theme_id]
            return ToolResult(
                success=True,
                data={"theme": theme_id, "count": len(matches), "findings": matches},
                summary=f"{len(matches)} findings for {theme_id}.",
            )

        elif operation == "high_significance":
            high = [
                f for f in findings if f.get("significance", "").lower() in ("high", "critical")
            ]
            return ToolResult(
                success=True,
                data={"count": len(high), "findings": high},
                summary=f"{len(high)} high/critical significance findings.",
            )

        elif operation == "search":
            query = kwargs.get("query", "").lower()
            if not query:
                return ToolResult(success=False, error="Search query required.")
            matches = []
            for f in findings:
                text = json.dumps(f).lower()
                if query in text:
                    matches.append(f)
            return ToolResult(
                success=True,
                data={"query": query, "count": len(matches), "findings": matches},
                summary=f"{len(matches)} findings matching '{query}'.",
            )

        return ToolResult(success=False, error=f"Unknown operation: {operation}")
