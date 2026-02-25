"""
Tool: explore_volume

Crawls a Databricks Unity Volume directory, returning file and subdirectory
metadata (names, sizes, modification dates, types).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class VolumeExplorerTool(BaseTool):
    @property
    def name(self) -> str:
        return "explore_volume"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory inside a Databricks Unity Volume. "
            "Returns file names, sizes, modification dates, and whether each entry "
            "is a file or subdirectory. Use this to discover what data files are "
            "available in a given path."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Volume directory to explore.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, recurse into subdirectories (max 3 levels deep). Default false.",
                    "default": False,
                },
            },
            "required": ["path"],
        }

    def _execute(self, path: str, recursive: bool = False, **kwargs) -> ToolResult:
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error=f"Path does not exist: {path}",
            )
        if not os.path.isdir(path):
            return ToolResult(
                success=False,
                error=f"Path is not a directory: {path}",
            )

        entries = self._scan_directory(path, recursive=recursive, max_depth=3)
        dir_count = sum(1 for e in entries if e["type"] == "directory")
        file_count = sum(1 for e in entries if e["type"] == "file")
        total_size = sum(e.get("size_bytes", 0) for e in entries if e["type"] == "file")

        summary = (
            f"Found {file_count} files and {dir_count} subdirectories "
            f"({round(total_size / (1024*1024), 2)} MB total) in {path}"
        )

        # Include a compact file list in the summary so the agent sees ALL
        # filenames even if the detailed entries get truncated at 15K chars
        all_files = sorted(e["name"] for e in entries if e["type"] == "file")
        all_dirs = sorted(e["name"] for e in entries if e["type"] == "directory")
        if all_files:
            summary += f"\n\nAll files ({len(all_files)}):\n"
            summary += "\n".join(f"  - {f}" for f in all_files)
        if all_dirs:
            summary += f"\n\nSubdirectories ({len(all_dirs)}):\n"
            summary += "\n".join(f"  - {d}/" for d in all_dirs)

        return ToolResult(
            success=True,
            data={"path": path, "entry_count": len(entries), "entries": entries},
            summary=summary,
        )

    def _scan_directory(
        self, path: str, recursive: bool, max_depth: int, current_depth: int = 0
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return [{"name": os.path.basename(path), "type": "error", "error": "Permission denied"}]

        for item_name in items:
            full_path = os.path.join(path, item_name)
            try:
                stat = os.stat(full_path)
            except OSError:
                continue

            if os.path.isdir(full_path):
                entry = {
                    "name": item_name,
                    "path": full_path,
                    "type": "directory",
                }
                entries.append(entry)
                if recursive and current_depth < max_depth:
                    children = self._scan_directory(
                        full_path, recursive=True, max_depth=max_depth,
                        current_depth=current_depth + 1,
                    )
                    entry["children"] = children
            else:
                _, ext = os.path.splitext(item_name)
                entries.append({
                    "name": item_name,
                    "path": full_path,
                    "type": "file",
                    "extension": ext.lower().lstrip("."),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        return entries
