"""
WriteNarrativeTool — accumulate and assemble narrative sections.

Manages the document as a collection of sections that can be written,
read back, updated, and finally assembled into the complete report.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class WriteNarrativeTool(BaseTool):
    """Write, read, update, and assemble narrative sections."""

    def __init__(
        self,
        output_path: str,
        output_filename: str = "narrative_report.md",
        include_toc: bool = True,
        report_title: str = "Research Analysis Report",
    ) -> None:
        self._output_path = output_path
        self._output_filename = output_filename
        self._include_toc = include_toc
        self._report_title = report_title
        self._sections: dict[str, str] = {}  # section_id -> markdown content
        self._section_order: list[str] = []  # ordered section IDs
        self._section_sequence: dict[str, int] = {}  # section_id -> sequence number
        self.sections_written: int = 0

    @property
    def name(self) -> str:
        return "write_narrative"

    @property
    def description(self) -> str:
        return (
            "Accumulate and assemble narrative sections into the final report. "
            "Operations: 'write_section' (create/overwrite a section), "
            "'read_section' (read back a section), 'update_section' (replace), "
            "'list_sections' (show all written sections), "
            "'assemble' (combine all sections into the final document)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "write_section",
                        "read_section",
                        "update_section",
                        "list_sections",
                        "assemble",
                    ],
                    "description": "Which operation to run.",
                },
                "section_id": {
                    "type": "string",
                    "description": "Section ID (e.g., 'section_stakes'). Required for write/read/update.",
                },
                "title": {
                    "type": "string",
                    "description": "Section title for write_section.",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown content for write_section / update_section.",
                },
                "sequence": {
                    "type": "integer",
                    "description": "Section order (0-based). For write_section.",
                },
            },
            "required": ["operation"],
        }

    def _execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs["operation"]

        if operation == "write_section":
            section_id = kwargs.get("section_id", "")
            title = kwargs.get("title", "")
            content = kwargs.get("content", "")
            sequence = kwargs.get("sequence", len(self._section_order))

            if not section_id or not content:
                return ToolResult(
                    success=False,
                    error="section_id and content are required for write_section.",
                )

            # Add title header if provided
            if title and not content.startswith("#"):
                content = f"## {title}\n\n{content}"

            self._sections[section_id] = content

            # Maintain order
            if section_id not in self._section_order:
                self._section_order.append(section_id)
                # Sort by sequence if we can infer it
                self._section_order.sort(key=lambda sid: self._section_sequence.get(sid, 999))

            self._section_sequence[section_id] = sequence
            self.sections_written += 1

            # Persist section to disk immediately
            self._persist_section(section_id, content)

            word_count = len(content.split())
            return ToolResult(
                success=True,
                data={
                    "section_id": section_id,
                    "word_count": word_count,
                    "total_sections": len(self._sections),
                },
                summary=f"Wrote section '{section_id}' ({word_count} words). "
                f"Total: {len(self._sections)} sections.",
            )

        elif operation == "read_section":
            section_id = kwargs.get("section_id", "")
            if section_id not in self._sections:
                return ToolResult(
                    success=False,
                    error=f"Section '{section_id}' not written yet.",
                )
            content = self._sections[section_id]
            return ToolResult(
                success=True,
                data={"section_id": section_id, "content": content},
                summary=f"Section '{section_id}': {len(content.split())} words.",
            )

        elif operation == "update_section":
            section_id = kwargs.get("section_id", "")
            content = kwargs.get("content", "")
            if not section_id or not content:
                return ToolResult(
                    success=False,
                    error="section_id and content required for update_section.",
                )
            if section_id not in self._sections:
                return ToolResult(
                    success=False,
                    error=f"Section '{section_id}' doesn't exist. Use write_section first.",
                )
            self._sections[section_id] = content
            self._persist_section(section_id, content)
            return ToolResult(
                success=True,
                data={"section_id": section_id, "word_count": len(content.split())},
                summary=f"Updated section '{section_id}'.",
            )

        elif operation == "list_sections":
            section_list = []
            for sid in self._section_order:
                content = self._sections.get(sid, "")
                section_list.append(
                    {
                        "section_id": sid,
                        "word_count": len(content.split()),
                        "sequence": self._section_sequence.get(sid, 0),
                    }
                )
            return ToolResult(
                success=True,
                data={"sections": section_list, "total": len(section_list)},
                summary=f"{len(section_list)} sections written.",
            )

        elif operation == "assemble":
            return self._assemble_document()

        return ToolResult(success=False, error=f"Unknown operation: {operation}")

    def _persist_section(self, section_id: str, content: str) -> None:
        """Save individual section to disk for durability."""
        os.makedirs(self._output_path, exist_ok=True)
        path = os.path.join(self._output_path, f"{section_id}.md")
        with open(path, "w") as f:
            f.write(content)

    def _assemble_document(self) -> ToolResult:
        """Combine all sections into the final document."""
        if not self._sections:
            return ToolResult(
                success=False,
                error="No sections written yet. Write sections first.",
            )

        os.makedirs(self._output_path, exist_ok=True)

        # Sort sections by sequence
        ordered_ids = sorted(
            self._section_order,
            key=lambda sid: self._section_sequence.get(sid, 999),
        )

        parts = []

        # Title
        parts.append(f"# {self._report_title}\n")
        parts.append(f"*Generated {datetime.now().strftime('%Y-%m-%d')}*\n")

        # Table of contents
        if self._include_toc:
            parts.append("## Table of Contents\n")
            for i, sid in enumerate(ordered_ids):
                content = self._sections[sid]
                # Extract first heading
                title = sid.replace("section_", "").replace("_", " ").title()
                for line in content.split("\n"):
                    if line.startswith("## "):
                        title = line.replace("## ", "").strip()
                        break
                parts.append(f"{i + 1}. [{title}](#{sid})")
            parts.append("")

        # Sections
        parts.append("---\n")
        for sid in ordered_ids:
            content = self._sections[sid]
            parts.append(f'<a id="{sid}"></a>\n')
            parts.append(content)
            parts.append("\n---\n")

        document = "\n".join(parts)

        # Write final document
        output_file = os.path.join(self._output_path, self._output_filename)
        with open(output_file, "w") as f:
            f.write(document)

        word_count = len(document.split())
        return ToolResult(
            success=True,
            data={
                "output_file": output_file,
                "word_count": word_count,
                "sections_assembled": len(ordered_ids),
            },
            summary=(
                f"Assembled {len(ordered_ids)} sections into {output_file} ({word_count} words)."
            ),
        )
