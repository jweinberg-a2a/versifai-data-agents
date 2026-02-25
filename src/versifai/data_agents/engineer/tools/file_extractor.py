"""
Tool: extract_archive

Extracts ZIP, GZ, TAR, and other compressed archives to a staging directory.
"""

from __future__ import annotations

import gzip
import os
import shutil
import tarfile
import zipfile
from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult


class FileExtractorTool(BaseTool):
    @property
    def name(self) -> str:
        return "extract_archive"

    @property
    def description(self) -> str:
        return (
            "Extract a compressed archive (ZIP, GZ, TAR, TGZ) to a destination directory. "
            "Returns the list of extracted files with their paths, sizes, and types. "
            "Use this when you discover archive files that need to be unpacked before reading."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the archive file to extract.",
                },
                "dest_path": {
                    "type": "string",
                    "description": (
                        "Directory to extract into. Will be created if it doesn't exist. "
                        "If not provided, extracts to a subfolder next to the archive."
                    ),
                },
            },
            "required": ["file_path"],
        }

    def _execute(self, file_path: str, dest_path: str = "", **kwargs) -> ToolResult:  # type: ignore[override]
        if not os.path.isfile(file_path):
            return ToolResult(success=False, error=f"File not found: {file_path}")

        # Default destination: same folder as the archive, in a subfolder named after the file
        if not dest_path:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            dest_path = os.path.join(os.path.dirname(file_path), f"_extracted_{base_name}")

        os.makedirs(dest_path, exist_ok=True)

        ext = os.path.splitext(file_path)[1].lower()
        extracted_files: list[dict[str, Any]] = []

        if ext == ".zip":
            extracted_files = self._extract_zip(file_path, dest_path)
        elif ext in (".gz", ".gzip"):
            extracted_files = self._extract_gzip(file_path, dest_path)
        elif ext in (".tar", ".tgz"):
            extracted_files = self._extract_tar(file_path, dest_path)
        else:
            return ToolResult(
                success=False,
                error=f"Unsupported archive format: {ext}. Supported: .zip, .gz, .tar, .tgz",
            )

        return ToolResult(
            success=True,
            data={
                "source_archive": file_path,
                "destination": dest_path,
                "file_count": len(extracted_files),
                "files": extracted_files,
            },
            summary=f"Extracted {len(extracted_files)} files from {os.path.basename(file_path)} to {dest_path}",
        )

    def _extract_zip(self, file_path: str, dest_path: str) -> list[dict]:
        results = []
        with zipfile.ZipFile(file_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                zf.extract(info, dest_path)
                extracted_path = os.path.join(dest_path, info.filename)
                _, ext = os.path.splitext(info.filename)
                results.append(
                    {
                        "name": info.filename,
                        "path": extracted_path,
                        "size_bytes": info.file_size,
                        "extension": ext.lower().lstrip("."),
                    }
                )
        return results

    def _extract_gzip(self, file_path: str, dest_path: str) -> list[dict]:
        # Gzip contains a single file — strip the .gz suffix for the output name
        base_name = os.path.basename(file_path)
        if base_name.endswith(".gz"):
            out_name = base_name[:-3]
        else:
            out_name = base_name + ".out"
        out_path = os.path.join(dest_path, out_name)

        with gzip.open(file_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        stat = os.stat(out_path)
        _, ext = os.path.splitext(out_name)
        return [
            {
                "name": out_name,
                "path": out_path,
                "size_bytes": stat.st_size,
                "extension": ext.lower().lstrip("."),
            }
        ]

    def _extract_tar(self, file_path: str, dest_path: str) -> list[dict]:
        results = []
        with tarfile.open(file_path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            tf.extractall(dest_path, members=members, filter="data")
            for m in members:
                extracted_path = os.path.join(dest_path, m.name)
                _, ext = os.path.splitext(m.name)
                results.append(
                    {
                        "name": m.name,
                        "path": extracted_path,
                        "size_bytes": m.size,
                        "extension": ext.lower().lstrip("."),
                    }
                )
        return results
