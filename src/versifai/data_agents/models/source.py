"""
Data models representing discovered files and data sources in the Volume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class FileInfo:
    """Metadata about a single file discovered in the Volume."""

    path: str
    name: str
    extension: str
    size_bytes: int
    modified_at: Optional[datetime] = None

    # Populated after inspection
    is_archive: bool = False
    extracted_to: Optional[str] = None
    encoding: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: str) -> "FileInfo":
        """Build a FileInfo from a filesystem path."""
        stat = os.stat(path)
        name = os.path.basename(path)
        _, ext = os.path.splitext(name)
        return cls(
            path=path,
            name=name,
            extension=ext.lower().lstrip("."),
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            is_archive=ext.lower() in (".zip", ".gz", ".tar", ".tgz", ".7z"),
        )

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)

    def __repr__(self) -> str:
        return f"FileInfo({self.name}, {self.size_mb}MB, ext={self.extension})"


@dataclass
class FileGroup:
    """
    A logical grouping of related files within a Volume subfolder.

    For example, all Medicare Advantage enrollment CSVs across multiple years
    might be grouped together under a single FileGroup.
    """

    name: str                          # e.g. "ma_enrollment"
    folder_path: str                   # Volume subfolder path
    files: list[FileInfo] = field(default_factory=list)
    documentation_files: list[FileInfo] = field(default_factory=list)
    data_files: list[FileInfo] = field(default_factory=list)
    archive_files: list[FileInfo] = field(default_factory=list)

    def classify_files(self) -> None:
        """Sort files into documentation, data, and archive buckets."""
        doc_extensions = {"txt", "pdf", "doc", "docx", "md", "html", "htm", "rtf"}
        data_extensions = {"csv", "xlsx", "xls", "tsv", "dat", "sas7bdat", "sav", "dta", "parquet"}
        archive_extensions = {"zip", "gz", "tar", "tgz", "7z", "bz2"}

        self.documentation_files.clear()
        self.data_files.clear()
        self.archive_files.clear()

        for f in self.files:
            if f.extension in archive_extensions:
                self.archive_files.append(f)
            elif f.extension in doc_extensions:
                self.documentation_files.append(f)
            elif f.extension in data_extensions:
                self.data_files.append(f)
            else:
                # Unknown extension — still keep as data for the agent to inspect
                self.data_files.append(f)

    @property
    def total_size_mb(self) -> float:
        return round(sum(f.size_bytes for f in self.files) / (1024 * 1024), 2)

    def most_recent_data_file(self) -> Optional[FileInfo]:
        """Return the data file with the latest modification date."""
        candidates = self.data_files or self.files
        if not candidates:
            return None
        return max(candidates, key=lambda f: f.modified_at or datetime.min)


@dataclass
class DataSource:
    """
    Top-level representation of a data source that the agent will process.

    Combines the file group with profiling results, schema decisions, and
    processing metadata.
    """

    name: str
    file_group: FileGroup
    description: str = ""

    # Populated during processing
    documentation_summary: str = ""
    profile_summary: str = ""
    target_table_name: str = ""
    year_range: str = ""
    county_fips_column: str = ""
    notes: list[str] = field(default_factory=list)
