"""
Tool: save_note

Appends free-form research notes to per-theme text files.  These capture
softer observations that don't fit neatly into a chart or table —
data-quality quirks, validation checks, join-key gotchas, behavioral
observations about distributions, etc.

One file per theme, append-only.  On resume the orchestrator reads these
back so the agent can pick up where it left off.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class SaveNoteTool(BaseTool):
    """
    Append a research note to the per-theme notes file.

    The agent calls this throughout analysis to record observations
    that are important for continuity but don't belong in a structured
    finding, chart, or table.
    """

    def __init__(self, notes_path: str = "") -> None:
        super().__init__()
        self._notes_path = notes_path
        # In-memory accumulator for prompt injection
        self._notes: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "save_note"

    @property
    def description(self) -> str:
        return (
            "Append a research note to the per-theme notes file. Use this to "
            "record observations that don't fit into a chart or table — data "
            "quality quirks, validation results, join-key issues, distribution "
            "anomalies, methodology decisions, things to watch out for on "
            "resume, etc. Notes are appended (never overwritten) and read back "
            "on resume so context is preserved across sessions."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "theme_id": {
                    "type": "string",
                    "description": (
                        "ID of the theme this note belongs to "
                        "(e.g. 'theme_0', 'theme_5', 'general', 'silver')."
                    ),
                },
                "note": {
                    "type": "string",
                    "description": (
                        "The observation or note to record. Be specific — "
                        "include column names, table names, numeric values, "
                        "and any context that will help future sessions."
                    ),
                },
            },
            "required": ["theme_id", "note"],
        }

    def _execute(
        self,
        theme_id: str = "",
        note: str = "",
        **kwargs,
    ) -> ToolResult:
        if not theme_id:
            return ToolResult(
                success=False,
                error="Missing required parameter 'theme_id'.",
            )
        if not note:
            return ToolResult(
                success=False,
                error="Missing required parameter 'note'.",
            )

        # Accumulate in memory
        self._notes.setdefault(theme_id, []).append(note)

        # Append to file (read-then-write for Databricks Volumes compatibility
        # since FUSE-mounted volumes don't support open("a") append mode)
        if self._notes_path:
            os.makedirs(self._notes_path, exist_ok=True)
            filepath = os.path.join(self._notes_path, f"{theme_id}_notes.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing = ""
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r") as f:
                        existing = f.read()
                except OSError:
                    pass
            with open(filepath, "w") as f:
                f.write(existing + f"\n--- {timestamp} ---\n{note}\n")

            return ToolResult(
                success=True,
                data={"theme_id": theme_id, "file": filepath},
                summary=f"Note appended to {filepath}",
            )

        return ToolResult(
            success=True,
            data={"theme_id": theme_id},
            summary=f"Note recorded in memory for {theme_id} (no file path configured).",
        )

    # ------------------------------------------------------------------
    # Public helpers for the orchestrator
    # ------------------------------------------------------------------

    @property
    def notes(self) -> dict[str, list[str]]:
        """All notes accumulated in memory, keyed by theme_id."""
        return self._notes

    def get_notes_for_theme(self, theme_id: str) -> list[str]:
        """Return in-memory notes for a specific theme."""
        return self._notes.get(theme_id, [])

    def load_notes_from_disk(self) -> dict[str, str]:
        """Read all existing notes files from disk.

        Returns a dict of theme_id -> full file contents.
        """
        result: dict[str, str] = {}
        if not self._notes_path or not os.path.isdir(self._notes_path):
            return result
        for filename in sorted(os.listdir(self._notes_path)):
            if filename.endswith("_notes.txt"):
                theme_id = filename.replace("_notes.txt", "")
                filepath = os.path.join(self._notes_path, filename)
                try:
                    with open(filepath) as f:
                        result[theme_id] = f.read()
                except OSError:
                    pass
        return result

    def get_theme_notes_from_disk(self, theme_id: str) -> str:
        """Read notes for a single theme from disk. Returns empty string if none."""
        if not self._notes_path:
            return ""
        filepath = os.path.join(self._notes_path, f"{theme_id}_notes.txt")
        if os.path.isfile(filepath):
            try:
                with open(filepath) as f:
                    return f.read()
            except OSError:
                pass
        return ""
