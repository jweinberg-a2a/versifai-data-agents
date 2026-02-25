"""
Tool: transform_and_load

Reads a source data file, applies column mappings and transformations
defined in a registered schema, and writes the result to a staging
DataFrame that can be written to Unity Catalog.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from versifai._utils.fips import pad_fips_series
from versifai._utils.naming import to_snake_case
from versifai.core.tools.base import BaseTool, ToolResult
from versifai.data_agents.models.schema import TargetSchema


class DataTransformerTool(BaseTool):
    """
    Transforms raw data files into the target schema format.

    Reads the source file, applies column mappings and type conversions,
    adds metadata columns, and stages the result.

    Auto-flush: when staged data exceeds a memory threshold, the tool
    automatically writes to Unity Catalog (append mode) and clears the
    staging buffer to prevent OOM. Requires a catalog writer reference
    and a registered flush target (table name).
    """

    # Default auto-flush threshold: 30 million rows
    DEFAULT_FLUSH_THRESHOLD_ROWS = 30_000_000

    def __init__(self, schema_designer_tool=None, cfg=None):
        super().__init__()
        self._schema_tool = schema_designer_tool
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg
        # Staging area: holds transformed DataFrames keyed by source_name
        self._staged: dict[str, pd.DataFrame] = {}

        # Auto-flush support
        self._catalog_writer = None  # set via set_catalog_writer()
        self._staging_threshold_rows = getattr(
            cfg, "staging_flush_threshold_rows", self.DEFAULT_FLUSH_THRESHOLD_ROWS
        )
        # Tracks per-source flush state: {table_name, rows_flushed, flush_count}
        self._flush_state: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "transform_and_load"

    @property
    def description(self) -> str:
        return (
            "Read a source data file and transform it to match a previously designed "
            "target schema. Applies column name mappings, type conversions, FIPS "
            "zero-padding, and adds metadata columns. ALL source columns are preserved — "
            "columns in the schema get renamed/typed, and any extra source columns not in "
            "the schema are carried forward with snake_case names (bronze layer: keep everything). "
            "The transformed data is staged in memory, ready to be written to Unity Catalog "
            "via the write_to_catalog tool. If the source has schema drift (different columns "
            "than expected), provide column_overrides to map source columns to target columns."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the source data file.",
                },
                "source_name": {
                    "type": "string",
                    "description": "Name of the registered schema to transform against.",
                },
                "source_year": {
                    "type": "integer",
                    "description": "The data year for this file (added as metadata).",
                },
                "source_period_start": {
                    "type": "string",
                    "description": (
                        "The start date of the period this file covers, as YYYY-MM-DD string. "
                        "Derive from the filename: annual file '2023' -> '2023-01-01', "
                        "monthly file '2023-03' -> '2023-03-01', "
                        "quarterly file '2023_Q2' -> '2023-04-01'. "
                        "This creates proper date alignment when concatenating multi-period files."
                    ),
                },
                "column_overrides": {
                    "type": "object",
                    "description": (
                        "Optional dict mapping source column names to target column names "
                        "for this specific file. Used when historical files have different "
                        "column names than the reference schema. "
                        "Format: {'source_col_in_this_file': 'target_col_name'}"
                    ),
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding. Auto-detected if omitted.",
                },
                "separator": {
                    "type": "string",
                    "description": "Column separator for CSV. Auto-detected if omitted.",
                },
                "skip_rows": {
                    "type": "integer",
                    "description": "Rows to skip at top of file. Default 0.",
                    "default": 0,
                },
                "sheet_name": {
                    "type": "string",
                    "description": "Sheet name for Excel files.",
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to existing staged data. If false, replace. Default true.",
                    "default": True,
                },
                "files": {
                    "type": "array",
                    "description": (
                        "BATCH MODE: Process multiple files in a single call. "
                        "Each element is an object with file_path (required), "
                        "source_year, and source_period_start. Much more efficient "
                        "than calling transform_and_load once per file. The same "
                        "schema, encoding, separator, and column_overrides are "
                        "applied to every file in the batch. "
                        "SCHEMA NORMALIZATION: In batch mode, every file is "
                        "automatically normalized to the target schema columns. "
                        "Files with extra columns (e.g., older files with trailing "
                        "nulls) have those extras dropped. Files missing columns "
                        "get NULLs. This ensures all files produce identical "
                        "DataFrames for clean concatenation — no custom tools needed."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "source_year": {"type": "integer"},
                            "source_period_start": {"type": "string"},
                        },
                        "required": ["file_path"],
                    },
                },
            },
            "required": ["source_name"],
        }

    def _execute(
        self,
        source_name: str,
        file_path: str | None = None,
        source_year: int | None = None,
        source_period_start: str | None = None,
        column_overrides: dict[str, str] | None = None,
        encoding: str | None = None,
        separator: str | None = None,
        skip_rows: int = 0,
        sheet_name: str | None = None,
        append: bool = True,
        files: list[dict] | None = None,
        **kwargs,
    ) -> ToolResult:
        # Get the target schema
        schema = None
        if self._schema_tool:
            schema = self._schema_tool.get_schema(source_name)
        if schema is None:
            return ToolResult(
                success=False,
                error=f"No schema registered for source '{source_name}'. Call design_schema first.",
            )

        # ── Batch mode: process multiple files in one call ────────────
        if files:
            return self._execute_batch(
                files=files,
                source_name=source_name,
                schema=schema,
                column_overrides=column_overrides,
                encoding=encoding,
                separator=separator,
                skip_rows=skip_rows,
                sheet_name=sheet_name,
            )

        if not file_path:
            return ToolResult(
                success=False,
                error="Either 'file_path' or 'files' must be provided.",
            )

        # Load the source data
        df = self._load_file(file_path, encoding, separator, skip_rows, sheet_name)
        if df is None:
            return ToolResult(success=False, error=f"Failed to load file: {file_path}")

        original_row_count = len(df)
        list(df.columns)

        # Build the source→target column mapping
        col_map = self._build_column_mapping(schema, df.columns.tolist(), column_overrides)

        # Transform
        transformed = pd.DataFrame()
        unmapped_target = []

        for col_def in schema.columns:
            if col_def.is_metadata:
                continue  # metadata columns added separately

            source_col = col_map.get(col_def.target_name)
            if source_col and source_col in df.columns:
                series = df[source_col].copy()

                # Apply transform expression if provided
                if col_def.transform_expression:
                    series = self._apply_transform(series, col_def.transform_expression)
                elif col_def.is_fips:
                    # Vectorized FIPS zero-padding
                    join_key_width = self._cfg.join_key.width
                    non_null = series.dropna()
                    if col_def.target_name == self._cfg.join_key.column_name or (
                        len(non_null) > 0 and len(str(non_null.iloc[0])) >= (join_key_width - 1)
                    ):
                        series = pad_fips_series(series, width=join_key_width)
                    else:
                        series = pad_fips_series(series, width=2)

                # Type conversion
                series = self._cast_type(series, col_def.data_type)
                transformed[col_def.target_name] = series
            else:
                unmapped_target.append(col_def.target_name)
                # Add null column for missing source columns
                transformed[col_def.target_name] = None

        # Carry forward ALL unmapped source columns (bronze layer: keep everything)
        mapped_sources = set(col_map.values())
        extra_columns_added = []
        for src_col in df.columns:
            if src_col not in mapped_sources:
                # Rename to snake_case and add to output
                clean_name = to_snake_case(str(src_col))
                # Avoid collision with existing target columns
                if clean_name in transformed.columns:
                    clean_name = f"_source_{clean_name}"
                transformed[clean_name] = df[src_col]
                extra_columns_added.append(f"{src_col} -> {clean_name}")

        # Add metadata columns (names from config)
        file_name = os.path.basename(file_path)

        # Parse source_period_start into a date if provided
        period_date = None
        if source_period_start:
            try:
                period_date = pd.to_datetime(source_period_start).date()
            except Exception:
                period_date = None
        elif source_year:
            # Fallback: if only year given, default to Jan 1 of that year
            try:
                period_date = pd.to_datetime(f"{source_year}-01-01").date()
            except Exception:
                period_date = None

        meta_map = {col.name: col for col in self._cfg.metadata_columns}
        for meta_name, _meta_col in meta_map.items():
            if "file" in meta_name and "name" in meta_name:
                transformed[meta_name] = file_name
            elif "period" in meta_name and "start" in meta_name:
                transformed[meta_name] = period_date
            elif "year" in meta_name:
                transformed[meta_name] = source_year
            elif "timestamp" in meta_name:
                transformed[meta_name] = datetime.now()
            else:
                transformed[meta_name] = None

        # Stage the result
        if append and source_name in self._staged:
            self._staged[source_name] = pd.concat(
                [self._staged[source_name], transformed],
                ignore_index=True,
                copy=False,
            )
        else:
            self._staged[source_name] = transformed

        # Check if auto-flush is needed (staging buffer too large)
        flush_msg = self._check_auto_flush(source_name)

        total_staged = len(self._staged[source_name])
        flush_state = self._flush_state.get(source_name, {})
        rows_already_flushed = flush_state.get("rows_flushed", 0)

        summary = (
            f"Transformed {original_row_count} rows from {file_name} "
            f"(year={source_year}) into schema '{source_name}'. "
            f"Total staged: {total_staged} rows, {len(transformed.columns)} columns. "
            f"Extra source cols kept: {len(extra_columns_added)}, "
            f"unmapped target cols: {len(unmapped_target)}."
        )

        if rows_already_flushed > 0:
            summary += (
                f" Previously flushed: {rows_already_flushed:,} rows "
                f"({flush_state.get('flush_count', 0)} batches)."
            )

        if flush_msg:
            summary += f"\n\n{flush_msg}"

        return ToolResult(
            success=True,
            data={
                "file_path": file_path,
                "source_name": source_name,
                "source_year": source_year,
                "original_rows": original_row_count,
                "transformed_rows": len(transformed),
                "total_staged_rows": total_staged,
                "rows_previously_flushed": rows_already_flushed,
                "columns_mapped": len(transformed.columns),
                "extra_source_columns_kept": extra_columns_added,
                "unmapped_target_columns": unmapped_target,
                "auto_flush": flush_msg,
                "sample_data": self._safe_sample(transformed),
            },
            summary=summary,
        )

    def _execute_batch(
        self,
        files: list[dict],
        source_name: str,
        schema: TargetSchema,
        column_overrides: dict[str, str] | None = None,
        encoding: str | None = None,
        separator: str | None = None,
        skip_rows: int = 0,
        sheet_name: str | None = None,
    ) -> ToolResult:
        """Process multiple files in a single tool call.

        Each file is transformed using the same schema and settings,
        then appended to the staging buffer. Auto-flush triggers
        normally if the buffer grows too large.

        Performance: collects transformed DataFrames in a list and
        concatenates in bulk (once per flush boundary) instead of
        growing via pd.concat after each file (O(n) vs O(n^2)).
        """
        results = []
        total_rows = 0
        errors = []
        # Collect chunks between flushes — concat once at flush or at end
        pending_chunks: list[pd.DataFrame] = []
        pending_rows = 0

        # ── Canonical column set: ensures schema consistency across files ──
        # In batch mode, all DataFrames MUST have identical columns for clean
        # concatenation. Extra unmapped source columns (e.g., trailing null
        # columns in older files) are dropped to prevent schema mismatch
        # that causes silent data loss during Spark writes.
        canonical_cols: list[str] = []
        canonical_set: set[str] = set()
        for col_def in schema.columns:
            if not col_def.is_metadata:
                canonical_cols.append(col_def.target_name)
                canonical_set.add(col_def.target_name)
        for meta_col in self._cfg.metadata_columns:
            canonical_cols.append(meta_col.name)
            canonical_set.add(meta_col.name)
        drift_notes: list[str] = []

        # ── Pre-scan: detect schema drift across files ──────────────
        # Quick header scan (reads only first row) to detect column count
        # differences before processing. Reports drift upfront so the
        # agent understands what's happening if some files have extra columns.
        header_info = self._scan_batch_headers(
            files,
            encoding,
            separator,
            skip_rows,
            sheet_name,
        )
        col_counts = set(h["column_count"] for h in header_info if h["column_count"] > 0)
        if len(col_counts) > 1:
            count_summary = ", ".join(
                f"{h['file_name']}={h['column_count']}"
                for h in header_info
                if h["column_count"] > 0
            )
            drift_notes.append(
                f"Schema drift detected: files have different column counts "
                f"({sorted(col_counts)}). All files will be normalized to the "
                f"target schema ({len(canonical_cols)} columns). "
                f"Per-file counts: {count_summary}"
            )

        for i, file_entry in enumerate(files):
            fp = file_entry.get("file_path", "")
            yr = file_entry.get("source_year")
            period = file_entry.get("source_period_start")

            if not fp:
                errors.append(f"File {i}: missing file_path")
                continue

            # Load
            df = self._load_file(fp, encoding, separator, skip_rows, sheet_name)
            if df is None:
                errors.append(f"{os.path.basename(fp)}: failed to load")
                continue

            row_count = len(df)

            # Build column mapping
            col_map = self._build_column_mapping(
                schema,
                df.columns.tolist(),
                column_overrides,
            )

            # Transform — build dict of columns, construct DataFrame at end
            col_data = {}
            for col_def in schema.columns:
                if col_def.is_metadata:
                    continue
                source_col = col_map.get(col_def.target_name)
                if source_col and source_col in df.columns:
                    series = df[source_col].copy()
                    if col_def.transform_expression:
                        series = self._apply_transform(
                            series,
                            col_def.transform_expression,
                        )
                    elif col_def.is_fips:
                        join_key_width = self._cfg.join_key.width
                        non_null = series.dropna()
                        if col_def.target_name == self._cfg.join_key.column_name or (
                            len(non_null) > 0 and len(str(non_null.iloc[0])) >= (join_key_width - 1)
                        ):
                            series = pad_fips_series(series, width=join_key_width)
                        else:
                            series = pad_fips_series(series, width=2)
                    series = self._cast_type(series, col_def.data_type)
                    col_data[col_def.target_name] = series
                else:
                    col_data[col_def.target_name] = None

            transformed = pd.DataFrame(col_data)

            # Carry forward unmapped source columns
            mapped_sources = set(col_map.values())
            for src_col in df.columns:
                if src_col not in mapped_sources:
                    clean_name = to_snake_case(str(src_col))
                    if clean_name in transformed.columns:
                        clean_name = f"_source_{clean_name}"
                    transformed[clean_name] = df[src_col]

            # Add metadata
            file_name = os.path.basename(fp)
            period_date = None
            if period:
                try:
                    period_date = pd.to_datetime(period).date()
                except Exception:
                    pass
            elif yr:
                try:
                    period_date = pd.to_datetime(f"{yr}-01-01").date()
                except Exception:
                    pass

            meta_map = {col.name: col for col in self._cfg.metadata_columns}
            for meta_name in meta_map:
                if "file" in meta_name and "name" in meta_name:
                    transformed[meta_name] = file_name
                elif "period" in meta_name and "start" in meta_name:
                    transformed[meta_name] = period_date
                elif "year" in meta_name:
                    transformed[meta_name] = yr
                elif "timestamp" in meta_name:
                    transformed[meta_name] = datetime.now()
                else:
                    transformed[meta_name] = None

            # Free the raw dataframe immediately
            del df

            # ── Normalize to canonical schema (batch consistency) ──
            # Ensure every DataFrame has the exact same columns in the same
            # order. This prevents schema mismatch when files across years
            # have different column counts (e.g., older files with trailing
            # null columns that newer files don't have).

            # Add any missing canonical columns as None
            for col in canonical_cols:
                if col not in transformed.columns:
                    transformed[col] = None

            # Track extra unmapped columns being dropped
            extra = [c for c in transformed.columns if c not in canonical_set]
            if extra:
                drift_notes.append(
                    f"{file_name}: dropped {len(extra)} extra source column(s) "
                    f"not in target schema ({', '.join(extra[:5])}"
                    f"{'...' if len(extra) > 5 else ''})"
                )

            # Reorder to canonical column order (all files get identical shape)
            transformed = transformed[canonical_cols]

            # Collect chunk (no concat yet)
            pending_chunks.append(transformed)
            pending_rows += row_count
            total_rows += row_count
            results.append(f"{file_name}: {row_count:,} rows (year={yr})")

            # Check if we need to flush (bulk concat + auto-flush check)
            existing_staged = len(self._staged.get(source_name, pd.DataFrame()))
            if existing_staged + pending_rows > self._staging_threshold_rows:
                # Bulk concat pending chunks into staging, then defragment
                if pending_chunks:
                    combined = pd.concat(pending_chunks, ignore_index=True, copy=False)
                    if source_name in self._staged and len(self._staged[source_name]) > 0:
                        self._staged[source_name] = pd.concat(
                            [self._staged[source_name], combined],
                            ignore_index=True,
                            copy=False,
                        )
                    else:
                        self._staged[source_name] = combined
                    pending_chunks.clear()
                    pending_rows = 0

                flush_msg = self._check_auto_flush(source_name)
                if flush_msg:
                    results.append(f"  >> {flush_msg}")

        # Final bulk concat of any remaining pending chunks
        if pending_chunks:
            combined = pd.concat(pending_chunks, ignore_index=True, copy=False)
            if source_name in self._staged and len(self._staged[source_name]) > 0:
                self._staged[source_name] = pd.concat(
                    [self._staged[source_name], combined],
                    ignore_index=True,
                    copy=False,
                )
            else:
                self._staged[source_name] = combined
            pending_chunks.clear()
            # Defragment: consolidate memory blocks for fast parquet writes
            self._staged[source_name] = self._staged[source_name].copy()

        # Final auto-flush check
        flush_msg = self._check_auto_flush(source_name)
        if flush_msg:
            results.append(f"  >> {flush_msg}")

        # Build summary
        staged_now = len(self._staged.get(source_name, pd.DataFrame()))
        flush_state = self._flush_state.get(source_name, {})
        flushed_total = flush_state.get("rows_flushed", 0)

        file_list = "\n".join(results)
        summary = (
            f"BATCH: Processed {len(files)} files, {total_rows:,} total rows.\n"
            f"Currently staged in memory: {staged_now:,} rows.\n"
            f"Canonical schema: {len(canonical_cols)} columns "
            f"(all files normalized to this shape).\n"
        )
        if flushed_total > 0:
            summary += (
                f"Flushed to parquet: {flushed_total:,} rows "
                f"({flush_state.get('flush_count', 0)} batches).\n"
            )
        if drift_notes:
            summary += (
                f"\nSchema drift detected in {len(drift_notes)} file(s) "
                f"(extra columns dropped for batch consistency):\n"
                + "\n".join(f"  - {n}" for n in drift_notes)
                + "\n"
            )
        if errors:
            summary += f"Errors ({len(errors)}): {'; '.join(errors)}\n"
        summary += f"\nPer-file breakdown:\n{file_list}"

        return ToolResult(
            success=len(errors) < len(files),  # success if at least one file worked
            data={
                "source_name": source_name,
                "files_processed": len(files) - len(errors),
                "files_failed": len(errors),
                "total_rows": total_rows,
                "staged_rows": staged_now,
                "flushed_rows": flushed_total,
                "errors": errors,
                "schema_drift": drift_notes,
                "canonical_columns": len(canonical_cols),
            },
            summary=summary,
        )

    def _scan_batch_headers(
        self,
        files: list[dict],
        encoding: str | None,
        separator: str | None,
        skip_rows: int,
        sheet_name: str | None,
    ) -> list[dict]:
        """Quick-scan file headers to detect schema differences before batch processing.

        Returns a list of {file_name, column_count, columns} dicts. Only reads
        the first row of each file (nrows=0 for CSV, header-only for Excel).
        """
        header_info: list[dict] = []
        for file_entry in files:
            fp = file_entry.get("file_path", "")
            if not fp:
                continue
            fname = os.path.basename(fp)
            ext = os.path.splitext(fp)[1].lower().lstrip(".")
            try:
                if ext in ("xlsx", "xls"):
                    df_header = pd.read_excel(
                        fp,
                        sheet_name=sheet_name or 0,
                        skiprows=skip_rows or None,
                        nrows=0,
                    )
                elif ext == "parquet":
                    import pyarrow.parquet as pq

                    pf = pq.ParquetFile(fp)
                    cols = pf.schema.names
                    header_info.append(
                        {
                            "file_name": fname,
                            "column_count": len(cols),
                            "columns": cols,
                        }
                    )
                    continue
                else:
                    encodings = [encoding] if encoding else ["utf-8", "latin-1"]
                    df_header = None
                    for enc in encodings:
                        try:
                            sep = separator
                            if not sep:
                                with open(fp, encoding=enc) as f:
                                    line = f.readline()
                                sep = "\t" if "\t" in line else ("|" if "|" in line else ",")
                            df_header = pd.read_csv(
                                fp,
                                sep=sep,
                                encoding=enc,
                                nrows=0,
                                skiprows=range(1, skip_rows + 1) if skip_rows else None,
                            )
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                if df_header is not None:
                    header_info.append(
                        {
                            "file_name": fname,
                            "column_count": len(df_header.columns),
                            "columns": list(df_header.columns),
                        }
                    )
            except Exception:
                header_info.append(
                    {
                        "file_name": fname,
                        "column_count": -1,
                        "columns": [],
                    }
                )
        return header_info

    def get_staged(self, source_name: str) -> pd.DataFrame | None:
        """Retrieve staged DataFrame for a source."""
        return self._staged.get(source_name)

    def stage_dataframe(
        self,
        source_name: str,
        df: pd.DataFrame,
        append: bool = False,
    ) -> dict:
        """
        Stage an external DataFrame (e.g. from a custom tool) for catalog writing.

        This bridges the gap between custom in-memory processing and
        write_to_catalog. After calling this, use write_to_catalog with
        the same source_name.

        Returns a summary dict.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        if append and source_name in self._staged:
            self._staged[source_name] = pd.concat(
                [self._staged[source_name], df],
                ignore_index=True,
            )
        else:
            self._staged[source_name] = df

        total = len(self._staged[source_name])
        return {
            "source_name": source_name,
            "rows_staged": len(df),
            "total_staged": total,
            "columns": list(df.columns),
        }

    # ------------------------------------------------------------------
    # Auto-flush — write to Unity Catalog when staging gets too large
    # ------------------------------------------------------------------

    def set_catalog_writer(self, writer) -> None:
        """Set the catalog writer reference for auto-flush."""
        self._catalog_writer = writer

    def register_flush_target(self, source_name: str, table_name: str) -> None:
        """
        Register a Unity Catalog table as the auto-flush target for a source.

        Call this before processing files for a source. When staged data
        exceeds the memory threshold, it will be flushed to parquet on the
        staging volume. The final write_to_catalog call then creates the
        Delta table from all accumulated parquet batches.
        """
        staging_dir = f"{self._cfg.staging_path}/{source_name}"
        os.makedirs(staging_dir, exist_ok=True)

        self._flush_state[source_name] = {
            "table_name": table_name,
            "staging_dir": staging_dir,
            "rows_flushed": 0,
            "flush_count": 0,
        }

    def get_flush_state(self, source_name: str) -> dict | None:
        """Get the flush state for a source (None if no flushes occurred)."""
        state = self._flush_state.get(source_name)
        if state and state["flush_count"] > 0:
            return state
        return None

    def _check_auto_flush(self, source_name: str) -> str | None:
        """
        Check if staged data exceeds the row threshold and flush to parquet.

        Flushes write to parquet files on the staging volume — NOT to Delta.
        The final write_to_catalog call creates the Delta table from all
        accumulated parquet batches.

        Returns a status message if a flush occurred, or None if not needed.
        """
        if source_name not in self._staged:
            return None

        staged_rows = len(self._staged[source_name])
        if staged_rows <= self._staging_threshold_rows:
            return None

        # Auto-register flush target if not already registered.
        if source_name not in self._flush_state:
            table_name = f"{self._cfg.full_schema}.{source_name}"
            self.register_flush_target(source_name, table_name)

        return self._auto_flush(source_name, staged_rows)

    @staticmethod
    def _clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Clean object columns for parquet compatibility, vectorized."""
        obj_cols = df.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            for col in obj_cols:
                s = df[col].astype(str)
                mask = s.isin(["nan", "None", "<NA>", "NaT", ""])
                df[col] = s.where(~mask, other=None)
        return df

    def _auto_flush(self, source_name: str, staged_rows: int) -> str:
        """Flush staged data to parquet on the staging volume and clear memory.

        Does NOT write to Delta — just dumps to parquet. The final
        write_to_catalog call reads all parquet batches and creates the table.
        """
        state = self._flush_state[source_name]
        staging_dir = state["staging_dir"]
        batch_num = state["flush_count"]
        parquet_path = f"{staging_dir}/batch_{batch_num}.parquet"

        # Defragment before parquet write — consolidates scattered memory
        # blocks into contiguous arrays for much faster serialization
        df = self._staged[source_name].copy()

        # Clean object columns before writing parquet (vectorized)
        df = self._clean_object_columns(df)

        try:
            df.to_parquet(
                parquet_path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True
            )
        except Exception as e:
            return (
                f"AUTO-FLUSH FAILED: Could not write parquet to {parquet_path}: "
                f"{e}. Data remains in memory ({staged_rows:,} rows)."
            )

        state["rows_flushed"] += staged_rows
        state["flush_count"] += 1
        # Clear the staging buffer
        self._staged[source_name] = pd.DataFrame()

        return (
            f"AUTO-FLUSH: Wrote {staged_rows:,} rows to {parquet_path}. "
            f"Total flushed: {state['rows_flushed']:,} rows in "
            f"{state['flush_count']} batch(es). Memory cleared — "
            f"continue staging remaining files. "
            f"Call write_to_catalog when ALL files are processed."
        )

    def _load_file(
        self,
        file_path: str,
        encoding: str | None,
        separator: str | None,
        skip_rows: int,
        sheet_name: str | None,
    ) -> pd.DataFrame | None:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")

        if ext in ("xlsx", "xls"):
            return pd.read_excel(
                file_path,
                sheet_name=sheet_name or 0,
                skiprows=skip_rows or None,
            )
        if ext == "parquet":
            return pd.read_parquet(file_path)

        # CSV-like
        encodings = [encoding] if encoding else ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                sep = separator
                if not sep:
                    with open(file_path, encoding=enc) as f:
                        line = f.readline()
                    if "\t" in line:
                        sep = "\t"
                    elif "|" in line:
                        sep = "|"
                    else:
                        sep = ","
                return pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding=enc,
                    low_memory=False,
                    on_bad_lines="skip",
                    skiprows=range(1, skip_rows + 1) if skip_rows else None,
                )
            except (UnicodeDecodeError, UnicodeError):
                continue
        return None

    def _build_column_mapping(
        self,
        schema: TargetSchema,
        source_columns: list[str],
        overrides: dict[str, str] | None,
    ) -> dict[str, str]:
        """
        Build a mapping from target_name → source_column_name.

        Priority:
        1. column_overrides (explicit mapping for this file)
        2. schema.columns[].source_name (designed mapping)
        3. Exact name match (source col == target col)
        """
        target_to_source: dict[str, str] = {}

        # Inverted overrides: {source_col: target_col} → {target_col: source_col}
        inverted_overrides: dict[str, str] = {}
        if overrides:
            for src, tgt in overrides.items():
                inverted_overrides[tgt] = src

        source_set = set(source_columns)
        source_lower_map = {c.lower().strip(): c for c in source_columns}

        for col_def in schema.columns:
            if col_def.is_metadata:
                continue

            target = col_def.target_name

            # 1. Check overrides
            if target in inverted_overrides:
                override_src = inverted_overrides[target]
                if override_src in source_set:
                    target_to_source[target] = override_src
                    continue

            # 2. Check schema-defined source_name
            if col_def.source_name in source_set:
                target_to_source[target] = col_def.source_name
                continue

            # 3. Case-insensitive match
            if col_def.source_name.lower().strip() in source_lower_map:
                target_to_source[target] = source_lower_map[col_def.source_name.lower().strip()]
                continue

            # 4. Target name match
            if target in source_set:
                target_to_source[target] = target
                continue
            if target in source_lower_map:
                target_to_source[target] = source_lower_map[target]

        return target_to_source

    def _apply_transform(self, series: pd.Series, expression: str) -> pd.Series:
        """Apply a Python expression to each value in the series."""

        def safe_eval(value):
            try:
                return eval(
                    expression,
                    {"__builtins__": {}},
                    {"value": value, "str": str, "int": int, "float": float, "pd": pd},
                )
            except Exception:
                return value

        return series.apply(safe_eval)

    @staticmethod
    def _safe_sample(df: pd.DataFrame, n: int = 5) -> list[dict]:
        """Serialize sample rows to dicts, handling nullable types safely."""
        try:
            sample = df.head(n).copy()
            for col in sample.columns:
                sample[col] = sample[col].astype(object).fillna("").astype(str)
                sample[col] = sample[col].replace({"nan": "", "None": "", "<NA>": ""})
            return sample.to_dict(orient="records")
        except Exception:
            return [{"_note": f"Could not serialize sample ({len(df)} rows transformed)"}]

    def _cast_type(self, series: pd.Series, data_type: str) -> pd.Series:
        """Best-effort type casting."""
        try:
            dt = data_type.upper()
            if dt == "STRING":
                return series.astype(str).replace("nan", None).replace("None", None)
            elif dt in ("INT", "BIGINT"):
                return pd.to_numeric(series, errors="coerce").astype("Int64")
            elif dt in ("DOUBLE", "FLOAT", "DECIMAL"):
                return pd.to_numeric(series, errors="coerce")
            elif dt == "DATE":
                return pd.to_datetime(series, errors="coerce").dt.date
            elif dt == "TIMESTAMP":
                return pd.to_datetime(series, errors="coerce")
            elif dt == "BOOLEAN":
                return series.astype(bool)
        except Exception:
            pass
        return series
