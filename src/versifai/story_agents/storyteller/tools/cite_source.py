"""
CiteSourceTool — manage external citations and bibliography.

Accumulates citations during the writing process and formats them
into a bibliography at the end.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class CiteSourceTool(BaseTool):
    """Manage external citations and format bibliography."""

    def __init__(self) -> None:
        self._citations: dict[str, dict] = {}  # cite_key -> citation data
        self._cite_counter = 0

    @property
    def name(self) -> str:
        return "cite_source"

    @property
    def description(self) -> str:
        return (
            "Manage external citations for the narrative report. "
            "Operations: 'add' (register a citation), 'list' (all citations), "
            "'format' (generate bibliography text), 'search' (find by keyword)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "list", "format", "search"],
                    "description": "Which operation to run.",
                },
                "title": {
                    "type": "string",
                    "description": "Source title (for 'add').",
                },
                "url": {
                    "type": "string",
                    "description": "Source URL (for 'add').",
                },
                "author": {
                    "type": "string",
                    "description": "Author or organization (for 'add').",
                },
                "year": {
                    "type": "string",
                    "description": "Publication year (for 'add').",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what this source provides.",
                },
                "cite_key": {
                    "type": "string",
                    "description": "Short citation key (e.g., 'cms_stars_2024'). Auto-generated if omitted.",
                },
                "query": {
                    "type": "string",
                    "description": "Search text for 'search' operation.",
                },
            },
            "required": ["operation"],
        }

    def _execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs["operation"]

        if operation == "add":
            title = kwargs.get("title", "")
            url = kwargs.get("url", "")
            if not title:
                return ToolResult(success=False, error="title required for add.")

            self._cite_counter += 1
            cite_key = kwargs.get("cite_key", "")
            if not cite_key:
                # Auto-generate key from title
                words = title.lower().split()[:3]
                cite_key = "_".join(w for w in words if w.isalnum())
                if not cite_key:
                    cite_key = f"source_{self._cite_counter}"

            self._citations[cite_key] = {
                "title": title,
                "url": url,
                "author": kwargs.get("author", ""),
                "year": kwargs.get("year", str(datetime.now().year)),
                "description": kwargs.get("description", ""),
                "number": self._cite_counter,
            }

            return ToolResult(
                success=True,
                data={"cite_key": cite_key, "inline_ref": f"[{cite_key}]"},
                summary=f"Added citation [{cite_key}]: {title}",
            )

        elif operation == "list":
            return ToolResult(
                success=True,
                data={
                    "count": len(self._citations),
                    "citations": self._citations,
                },
                summary=f"{len(self._citations)} citations registered.",
            )

        elif operation == "format":
            if not self._citations:
                return ToolResult(
                    success=True,
                    data={"bibliography": ""},
                    summary="No citations to format.",
                )

            lines = ["## References\n"]
            for key, cite in sorted(
                self._citations.items(),
                key=lambda kv: kv[1]["number"],
            ):
                author = cite["author"] or "Unknown"
                year = cite["year"]
                title = cite["title"]
                url = cite["url"]

                entry = f"**[{key}]** {author} ({year}). *{title}*."
                if url:
                    entry += f" [{url}]({url})"
                if cite["description"]:
                    entry += f" — {cite['description']}"

                lines.append(entry + "\n")

            bibliography = "\n".join(lines)
            return ToolResult(
                success=True,
                data={"bibliography": bibliography},
                summary=f"Formatted {len(self._citations)} citations.",
            )

        elif operation == "search":
            query = kwargs.get("query", "").lower()
            if not query:
                return ToolResult(success=False, error="query required for search.")

            matches = {}
            for key, cite in self._citations.items():
                searchable = f"{key} {cite['title']} {cite['author']} {cite['description']}".lower()
                if query in searchable:
                    matches[key] = cite

            return ToolResult(
                success=True,
                data={"query": query, "count": len(matches), "matches": matches},
                summary=f"{len(matches)} citations matching '{query}'.",
            )

        return ToolResult(success=False, error=f"Unknown operation: {operation}")
