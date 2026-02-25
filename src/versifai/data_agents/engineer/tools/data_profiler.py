"""
Tool: profile_data

Runs column-level statistical profiling on a data file: types, null rates,
unique counts, min/max, distributions, and sample values.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from versifai.core.tools.base import BaseTool, ToolResult


class DataProfilerTool(BaseTool):
    @property
    def name(self) -> str:
        return "profile_data"

    @property
    def description(self) -> str:
        return (
            "Run detailed column-level profiling on a tabular data file. "
            "Returns for each column: data type, null count/rate, unique count, "
            "min/max values, most frequent values, and sample values. "
            "Also detects potential FIPS code columns and geographic identifiers. "
            "Use this after reading headers to deeply understand a data file's content."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the data file to profile.",
                },
                "sample_size": {
                    "type": "integer",
                    "description": "Number of rows to sample for profiling. Default 500. Increase only if needed for statistical significance.",
                    "default": 500,
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding. Auto-detected if omitted.",
                },
                "separator": {
                    "type": "string",
                    "description": "Column separator for CSV. Auto-detected if omitted.",
                },
            },
            "required": ["file_path"],
        }

    def _execute(
        self,
        file_path: str,
        sample_size: int = 500,
        encoding: Optional[str] = None,
        separator: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        if not os.path.isfile(file_path):
            return ToolResult(success=False, error=f"File not found: {file_path}")

        df = self._load_sample(file_path, sample_size, encoding, separator)
        if df is None:
            return ToolResult(success=False, error="Failed to load file for profiling.")

        total_rows = len(df)
        total_cols = len(df.columns)

        column_profiles = []
        potential_fips_columns = []
        potential_geo_columns = []

        for col in df.columns:
            profile = self._profile_column(df[col], col)
            column_profiles.append(profile)

            # Detect FIPS-like columns
            if profile.get("is_potential_fips"):
                potential_fips_columns.append(col)
            if profile.get("is_potential_geo"):
                potential_geo_columns.append(col)

        return ToolResult(
            success=True,
            data={
                "file_path": file_path,
                "rows_sampled": total_rows,
                "column_count": total_cols,
                "columns": [str(c) for c in df.columns],
                "column_profiles": column_profiles,
                "potential_fips_columns": potential_fips_columns,
                "potential_geo_columns": potential_geo_columns,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            },
            summary=(
                f"Profiled {total_cols} columns across {total_rows} sampled rows. "
                f"Potential FIPS columns: {potential_fips_columns or 'none detected'}. "
                f"Potential geo columns: {potential_geo_columns or 'none detected'}."
            ),
        )

    def _load_sample(
        self, file_path: str, sample_size: int, encoding: Optional[str], separator: Optional[str]
    ) -> Optional[pd.DataFrame]:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")

        if ext in ("xlsx", "xls"):
            return pd.read_excel(file_path, nrows=sample_size)
        if ext == "parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(file_path)
            # Read only the first row group (usually enough for profiling)
            table = pf.read_row_groups([0]) if pf.metadata.num_row_groups > 0 else pf.read()
            return table.to_pandas().head(sample_size)
        if ext == "sas7bdat":
            try:
                reader = pd.read_sas(file_path, encoding="latin-1", chunksize=sample_size)
                df = next(reader)
                reader.close()
                return df
            except Exception:
                return pd.read_sas(file_path, encoding="latin-1").head(sample_size)

        # CSV-like
        encodings = [encoding] if encoding else ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                sep = separator
                if not sep:
                    with open(file_path, "r", encoding=enc) as f:
                        line = f.readline()
                    if "\t" in line:
                        sep = "\t"
                    elif "|" in line:
                        sep = "|"
                    else:
                        sep = ","
                return pd.read_csv(
                    file_path, sep=sep, encoding=enc, nrows=sample_size,
                    on_bad_lines="skip", low_memory=False,
                )
            except (UnicodeDecodeError, UnicodeError):
                continue
        return None

    def _profile_column(self, series: pd.Series, col_name: str) -> dict:
        """Generate a profile dict for a single column."""
        total = len(series)
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        unique_count = int(non_null.nunique())

        profile = {
            "column_name": str(col_name),
            "dtype": str(series.dtype),
            "total_count": total,
            "null_count": null_count,
            "null_rate": round(null_count / total, 4) if total > 0 else 0,
            "unique_count": unique_count,
            "unique_rate": round(unique_count / total, 4) if total > 0 else 0,
            "sample_values": [str(v) for v in non_null.head(3).tolist()],
            "is_potential_fips": False,
            "is_potential_geo": False,
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            profile["min"] = str(non_null.min()) if len(non_null) > 0 else None
            profile["max"] = str(non_null.max()) if len(non_null) > 0 else None
            profile["mean"] = str(round(non_null.mean(), 4)) if len(non_null) > 0 else None
            profile["median"] = str(round(non_null.median(), 4)) if len(non_null) > 0 else None
            profile["std"] = str(round(non_null.std(), 4)) if len(non_null) > 0 else None
        else:
            # String stats
            str_series = non_null.astype(str)
            if len(str_series) > 0:
                lengths = str_series.str.len()
                profile["min_length"] = int(lengths.min())
                profile["max_length"] = int(lengths.max())
                profile["avg_length"] = round(float(lengths.mean()), 1)

        # Top frequent values (keep compact to save tokens)
        if unique_count > 0 and unique_count <= 50:
            value_counts = non_null.value_counts().head(5)
            profile["top_values"] = [
                {"value": str(v), "count": int(c)}
                for v, c in value_counts.items()
            ]
        elif unique_count > 50:
            value_counts = non_null.value_counts().head(3)
            profile["top_values"] = [
                {"value": str(v), "count": int(c)}
                for v, c in value_counts.items()
            ]

        # FIPS detection heuristics
        col_lower = str(col_name).lower()
        profile["is_potential_fips"] = self._is_fips_like(col_lower, non_null)
        profile["is_potential_geo"] = self._is_geo_like(col_lower)

        return profile

    def _is_fips_like(self, col_lower: str, series: pd.Series) -> bool:
        """Heuristic: does this column look like a FIPS code?"""
        fips_keywords = ["fips", "fip", "county_code", "cnty_cd", "geoid", "geo_id"]
        if any(kw in col_lower for kw in fips_keywords):
            return True

        # Check if values look like FIPS codes (2-5 digit numbers)
        if len(series) > 0:
            sample = series.head(100).astype(str).str.strip()
            digit_match = sample.str.match(r"^\d{2,5}$")
            if digit_match.mean() > 0.9:
                unique_lens = sample.str.len().unique()
                if set(unique_lens).issubset({2, 3, 4, 5}):
                    return True
        return False

    def _is_geo_like(self, col_lower: str) -> bool:
        """Heuristic: does this column name suggest geographic data?"""
        geo_keywords = [
            "county", "state", "zip", "city", "region", "tract",
            "latitude", "longitude", "lat", "lon", "lng",
            "address", "cbsa", "msa", "metropolitan",
        ]
        return any(kw in col_lower for kw in geo_keywords)
