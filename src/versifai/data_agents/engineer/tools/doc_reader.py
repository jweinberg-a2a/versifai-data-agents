"""
Tool: read_documentation

Reads and extracts content from documentation files found alongside data files.
Handles plain text, HTML, PDF (text extraction), and common data dictionary formats.

The agent uses this to understand field definitions, data dictionaries,
and methodology before designing schemas.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from versifai.core.tools.base import BaseTool, ToolResult


# File extensions considered documentation
DOC_EXTENSIONS = {
    "txt", "text", "readme", "md", "rst",
    "html", "htm",
    "pdf",
    "doc", "docx", "rtf",
    "csv",  # Some data dictionaries are CSVs
    "xlsx", "xls",  # Layout files sometimes in Excel
}

# Filename patterns that suggest documentation rather than data
DOC_FILENAME_PATTERNS = [
    r"readme",
    r"data.?dict",
    r"layout",
    r"codebook",
    r"metadata",
    r"documentation",
    r"description",
    r"field.?def",
    r"variable.?list",
    r"record.?layout",
    r"user.?guide",
    r"methodology",
    r"about",
    r"notes",
    r"changelog",
    r"schema",
]


class DocumentationReaderTool(BaseTool):
    @property
    def name(self) -> str:
        return "read_documentation"

    @property
    def description(self) -> str:
        return (
            "Read and extract the text content from a documentation file. "
            "Supports TXT, MD, HTML, PDF (text layer), CSV/Excel data dictionaries, "
            "and other common formats found alongside data files. "
            "Use this to understand field definitions, data dictionaries, codebooks, "
            "and methodology documents before designing a target schema."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the documentation file.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return. Default 15000. Long docs will be truncated.",
                    "default": 15000,
                },
            },
            "required": ["file_path"],
        }

    def _execute(self, file_path: str, max_chars: int = 15000, **kwargs) -> ToolResult:
        if not os.path.isfile(file_path):
            return ToolResult(success=False, error=f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        content = ""
        doc_type = "unknown"

        if ext in ("txt", "text", "md", "rst", "readme", ""):
            content, doc_type = self._read_text(file_path)
        elif ext in ("html", "htm"):
            content, doc_type = self._read_html(file_path)
        elif ext == "pdf":
            content, doc_type = self._read_pdf(file_path)
        elif ext in ("csv", "tsv"):
            content, doc_type = self._read_csv_dict(file_path)
        elif ext in ("xlsx", "xls"):
            content, doc_type = self._read_excel_dict(file_path)
        else:
            # Try reading as plain text
            content, doc_type = self._read_text(file_path)

        # Classify what kind of documentation this is
        classification = self._classify_document(file_name, content)

        # Truncate if necessary
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[... TRUNCATED ...]"
            truncated = True

        return ToolResult(
            success=True,
            data={
                "file_path": file_path,
                "file_name": file_name,
                "file_size_bytes": file_size,
                "doc_type": doc_type,
                "classification": classification,
                "content": content,
                "char_count": len(content),
                "truncated": truncated,
            },
            summary=(
                f"Read {doc_type} documentation: {file_name} "
                f"({len(content)} chars, classified as: {classification})"
            ),
        )

    # ------------------------------------------------------------------
    # Format readers
    # ------------------------------------------------------------------

    def _read_text(self, file_path: str) -> tuple[str, str]:
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read(), "text"
            except UnicodeDecodeError:
                continue
        return "[Could not decode file with any encoding]", "text"

    def _read_html(self, file_path: str) -> tuple[str, str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback: strip tags manually
            raw, _ = self._read_text(file_path)
            clean = re.sub(r"<[^>]+>", " ", raw)
            return re.sub(r"\s+", " ", clean).strip(), "html"

        raw, _ = self._read_text(file_path)
        soup = BeautifulSoup(raw, "html.parser")

        # Extract tables (common for data dictionaries)
        tables_text = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                rows.append(" | ".join(cells))
            tables_text.append("\n".join(rows))

        # Get body text
        body_text = soup.get_text(separator="\n", strip=True)

        if tables_text:
            combined = body_text + "\n\n=== TABLES ===\n\n" + "\n\n---\n\n".join(tables_text)
            return combined, "html"
        return body_text, "html"

    def _read_pdf(self, file_path: str) -> tuple[str, str]:
        # Try pypdf (current maintained package) first, then PyPDF2 (legacy name)
        for pdf_module in ("pypdf", "PyPDF2"):
            try:
                mod = __import__(pdf_module)
                with open(file_path, "rb") as f:
                    reader = mod.PdfReader(f)
                    pages = []
                    for page in reader.pages[:50]:  # Limit to 50 pages
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                    if pages:
                        return "\n\n--- PAGE BREAK ---\n\n".join(pages), "pdf"
            except ImportError:
                continue
            except Exception:
                continue

        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                pages = []
                for page in pdf.pages[:50]:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                    # Also try to extract tables
                    for table in page.extract_tables():
                        rows = [" | ".join(str(c or "") for c in row) for row in table]
                        pages.append("\n".join(rows))
                return "\n\n--- PAGE BREAK ---\n\n".join(pages), "pdf"
        except ImportError:
            pass

        return "[PDF reading requires pypdf or pdfplumber — neither is installed. Install with: pip install pypdf]", "pdf"

    def _read_csv_dict(self, file_path: str) -> tuple[str, str]:
        """Read a CSV that might be a data dictionary / layout file."""
        import pandas as pd
        try:
            df = pd.read_csv(file_path, nrows=500, on_bad_lines="skip")
            header = f"Columns: {list(df.columns)}\n\n"
            return header + df.to_string(max_rows=200, max_cols=20), "csv_data_dictionary"
        except Exception as e:
            return f"[Failed to parse CSV data dictionary: {e}]", "csv_data_dictionary"

    def _read_excel_dict(self, file_path: str) -> tuple[str, str]:
        """Read an Excel file that might be a data dictionary / layout file."""
        import pandas as pd
        try:
            xls = pd.ExcelFile(file_path)
            parts = [f"Sheets: {xls.sheet_names}\n"]
            for sheet in xls.sheet_names[:5]:  # Limit sheets
                df = pd.read_excel(file_path, sheet_name=sheet, nrows=200)
                parts.append(f"\n=== Sheet: {sheet} ===\n")
                parts.append(f"Columns: {list(df.columns)}\n")
                parts.append(df.to_string(max_rows=100, max_cols=20))
            return "\n".join(parts), "excel_data_dictionary"
        except Exception as e:
            return f"[Failed to parse Excel data dictionary: {e}]", "excel_data_dictionary"

    # ------------------------------------------------------------------
    # Document classification
    # ------------------------------------------------------------------

    def _classify_document(self, filename: str, content: str) -> str:
        """Classify what kind of documentation this is."""
        name_lower = filename.lower()
        content_lower = content[:3000].lower()

        for pattern in DOC_FILENAME_PATTERNS:
            if re.search(pattern, name_lower):
                if "dict" in pattern or "codebook" in pattern:
                    return "data_dictionary"
                if "layout" in pattern or "record" in pattern:
                    return "record_layout"
                if "readme" in pattern:
                    return "readme"
                if "method" in pattern:
                    return "methodology"
                return "documentation"

        # Content-based classification
        if any(kw in content_lower for kw in ["field name", "column name", "variable name", "field description"]):
            return "data_dictionary"
        if any(kw in content_lower for kw in ["record layout", "file layout", "field position"]):
            return "record_layout"
        if "methodology" in content_lower or "how the data" in content_lower:
            return "methodology"

        return "general_documentation"


class ScanForDocumentationTool(BaseTool):
    """
    Scans a directory tree for documentation files and returns a prioritized list.
    Useful for the agent to discover what documentation is available before profiling data.
    """

    @property
    def name(self) -> str:
        return "scan_for_documentation"

    @property
    def description(self) -> str:
        return (
            "Scan a directory (and subdirectories) for documentation files such as "
            "READMEs, data dictionaries, codebooks, layout files, and methodology docs. "
            "Returns a prioritized list of documentation files found, classified by type. "
            "Use this FIRST when exploring a new data source to find supporting documentation "
            "before reading the actual data files."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to scan for documentation files.",
                },
            },
            "required": ["path"],
        }

    def _execute(self, path: str, **kwargs) -> ToolResult:
        if not os.path.isdir(path):
            return ToolResult(success=False, error=f"Directory not found: {path}")

        found_docs: list[dict] = []

        for root, dirs, files in os.walk(path):
            # Don't recurse into extracted staging dirs
            dirs[:] = [d for d in dirs if not d.startswith("_")]

            for fname in files:
                full_path = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower().lstrip(".")
                name_lower = fname.lower()

                is_doc_ext = ext in DOC_EXTENSIONS
                is_doc_name = any(re.search(p, name_lower) for p in DOC_FILENAME_PATTERNS)

                if is_doc_ext or is_doc_name:
                    priority = self._prioritize(fname, ext)
                    found_docs.append({
                        "file_path": full_path,
                        "file_name": fname,
                        "extension": ext,
                        "size_bytes": os.path.getsize(full_path),
                        "priority": priority,
                        "reason": self._reason(fname, ext),
                    })

        # Sort by priority (highest first)
        found_docs.sort(key=lambda d: d["priority"], reverse=True)

        return ToolResult(
            success=True,
            data={
                "path": path,
                "documentation_files": found_docs,
                "count": len(found_docs),
            },
            summary=f"Found {len(found_docs)} documentation files in {path}",
        )

    def _prioritize(self, fname: str, ext: str) -> int:
        """Higher number = read this first."""
        name_lower = fname.lower()
        score = 0
        if "readme" in name_lower:
            score += 10
        if "data" in name_lower and "dict" in name_lower:
            score += 9
        if "codebook" in name_lower:
            score += 9
        if "layout" in name_lower:
            score += 8
        if "metadata" in name_lower:
            score += 7
        if "variable" in name_lower:
            score += 7
        if ext == "pdf":
            score += 2
        if ext in ("txt", "md"):
            score += 3
        return score

    def _reason(self, fname: str, ext: str) -> str:
        name_lower = fname.lower()
        if "readme" in name_lower:
            return "README file — likely contains overview and field descriptions"
        if "dict" in name_lower:
            return "Data dictionary — field definitions and descriptions"
        if "codebook" in name_lower:
            return "Codebook — variable coding and value labels"
        if "layout" in name_lower:
            return "Record layout — field positions and data types"
        if "metadata" in name_lower:
            return "Metadata — data source details"
        return f"Documentation file ({ext})"
