"""
Tool: write_to_catalog

Writes a staged DataFrame to a Delta table in Databricks Unity Catalog.
Also provides a tool to list existing tables and execute SQL.
"""

from __future__ import annotations

import os

import pandas as pd

from versifai.core.tools.base import BaseTool, ToolResult


class CatalogWriterTool(BaseTool):
    """
    Write staged data to a Unity Catalog Delta table.

    Uses either PySpark (when running in Databricks) or the Databricks SDK
    (when running externally) to write data.
    """

    def __init__(self, transformer_tool=None):
        super().__init__()
        self._transformer = transformer_tool

    @property
    def name(self) -> str:
        return "write_to_catalog"

    @property
    def description(self) -> str:
        return (
            "Write staged, transformed data to a Delta table in Unity Catalog. "
            "The data must first be staged via the transform_and_load tool. "
            "Specify the source_name (to find staged data) and table_name (fully qualified). "
            "ALWAYS use mode='overwrite' so the pipeline is idempotent and re-runnable. "
            "The tool creates the table if it doesn't exist and writes as a Delta table."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source_name": {
                    "type": "string",
                    "description": "Source name used when staging data via transform_and_load.",
                },
                "table_name": {
                    "type": "string",
                    "description": ("Fully qualified table name (catalog.schema.table)."),
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": "Write mode. 'overwrite' replaces the table, 'append' adds to it. Default: overwrite.",
                    "default": "overwrite",
                },
            },
            "required": ["source_name", "table_name"],
        }

    def _execute(  # type: ignore[override]
        self, source_name: str, table_name: str, mode: str = "overwrite", **kwargs
    ) -> ToolResult:
        # Check for auto-flush state (parquet batches on staging volume)
        flush_state = None
        flushed_rows = 0
        if self._transformer and hasattr(self._transformer, "get_flush_state"):
            flush_state = self._transformer.get_flush_state(source_name)
            if flush_state:
                flushed_rows = flush_state.get("rows_flushed", 0)

        # Get any remaining staged data in memory
        df = None
        if self._transformer:
            df = self._transformer.get_staged(source_name)
        has_remaining = df is not None and len(df) > 0

        if not has_remaining and flushed_rows == 0:
            return ToolResult(
                success=False,
                error=(
                    f"No staged data found for source '{source_name}'. "
                    f"Run transform_and_load first."
                ),
            )

        # ── Large data path: parquet batches exist from auto-flush ────
        if flushed_rows > 0:
            return self._write_from_flushed_parquet(
                flush_state,  # type: ignore[arg-type]
                df,
                table_name,
                mode,
                source_name,
            )

        # ── Small data path: everything fits in memory, no flushes ────
        return self._write_direct(df, table_name, mode, source_name)

    # spark.createDataFrame() fails on Spark Connect with gRPC protobuf
    # errors for large DataFrames (~10M+ rows). Above this threshold we
    # write to a temp parquet file first and use Spark SQL to create the
    # Delta table, bypassing the gRPC serialisation path entirely.
    DIRECT_WRITE_MAX_ROWS = 2_000_000

    def _write_direct(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str,
        source_name: str,
    ) -> ToolResult:
        """Write datasets that had no auto-flush during processing.

        Small datasets (≤ DIRECT_WRITE_MAX_ROWS) go through
        spark.createDataFrame(). Larger ones are written to a temp
        parquet file and loaded via Spark SQL to avoid the Spark Connect
        gRPC protobuf size limit.
        """
        row_count = len(df)

        if row_count > self.DIRECT_WRITE_MAX_ROWS:
            return self._write_via_temp_parquet(df, table_name, mode, source_name)

        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active Spark session found")

        spark_df = spark.createDataFrame(df)

        writer = spark_df.write.format("delta").mode(mode).option("overwriteSchema", "true")
        writer.saveAsTable(table_name)

        result_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table_name}").collect()[0]["cnt"]

        return ToolResult(
            success=True,
            data={
                "table_name": table_name,
                "mode": mode,
                "rows_written": row_count,
                "verified_row_count": result_count,
                "source_name": source_name,
                "method": "spark_direct",
            },
            summary=(
                f"Wrote {row_count:,} rows to {table_name} (mode={mode}, "
                f"verified={result_count:,} rows)."
            ),
        )

    def _write_via_temp_parquet(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str,
        source_name: str,
    ) -> ToolResult:
        """Write a large in-memory DataFrame via a temp parquet file.

        Used when the DataFrame is too large for spark.createDataFrame()
        but no auto-flush batches exist (the data accumulated in memory
        without hitting the flush threshold).
        """
        staging_dir = f"{self._transformer._cfg.staging_path}/{source_name}_direct"
        os.makedirs(staging_dir, exist_ok=True)

        row_count = len(df)
        parquet_path = f"{staging_dir}/data.parquet"

        # Clean object columns for parquet compatibility (vectorized)
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            s = df[col].astype(str)
            mask = s.isin(["nan", "None", "<NA>", "NaT", ""])
            df[col] = s.where(~mask, other=None)

        try:
            df.to_parquet(
                parquet_path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True
            )
        except Exception as e:
            import traceback

            return ToolResult(
                success=False,
                error=(
                    f"Failed to write temp parquet for {source_name}: {e}\n"
                    f"Traceback:\n{traceback.format_exc()[-500:]}"
                ),
            )

        try:
            result = self._write_from_parquet(
                staging_dir,
                table_name,
                mode,
                source_name,
                row_count,
            )
        except Exception as e:
            import traceback

            return ToolResult(
                success=False,
                error=(
                    f"Failed to create Delta table from temp parquet.\n"
                    f"Staging dir: {staging_dir}\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{traceback.format_exc()[-500:]}"
                ),
            )

        if result.success:
            self._cleanup_staging(staging_dir)

        return result

    def _write_from_flushed_parquet(
        self,
        flush_state: dict,
        remaining_df: pd.DataFrame | None,
        table_name: str,
        mode: str,
        source_name: str,
    ) -> ToolResult:
        """Write large datasets from auto-flushed parquet batches."""
        staging_dir = flush_state["staging_dir"]
        flushed_rows = flush_state["rows_flushed"]

        # Write any remaining in-memory data as a final parquet batch
        remaining_rows = 0
        if remaining_df is not None and len(remaining_df) > 0:
            remaining_rows = len(remaining_df)
            batch_num = flush_state["flush_count"]
            parquet_path = f"{staging_dir}/batch_{batch_num}.parquet"

            obj_cols = remaining_df.select_dtypes(include=["object"]).columns
            for col in obj_cols:
                s = remaining_df[col].astype(str)
                mask = s.isin(["nan", "None", "<NA>", "NaT", ""])
                remaining_df[col] = s.where(~mask, other=None)

            remaining_df.to_parquet(
                parquet_path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True
            )

        total_rows = flushed_rows + remaining_rows

        # Create Delta table from ALL parquet files via Spark SQL
        try:
            result = self._write_from_parquet(
                staging_dir,
                table_name,
                mode,
                source_name,
                total_rows,
            )
        except Exception as e:
            import traceback

            return ToolResult(
                success=False,
                error=(
                    f"Failed to create Delta table from parquet.\n"
                    f"Staging dir: {staging_dir}\n"
                    f"Error: {e}\n"
                    f"Traceback:\n{traceback.format_exc()[-500:]}"
                ),
            )

        # Clean up staging parquet files on success
        if result.success:
            self._cleanup_staging(staging_dir)

        return result

    def _cleanup_staging(self, staging_dir: str) -> None:
        """Remove all parquet batch files from the staging directory."""
        try:
            for f in os.listdir(staging_dir):
                if f.endswith(".parquet"):
                    os.remove(os.path.join(staging_dir, f))
            os.rmdir(staging_dir)
        except OSError:
            pass  # directory not empty or already removed

    def _write_from_parquet(
        self,
        staging_dir: str,
        table_name: str,
        mode: str,
        source_name: str,
        total_rows: int,
    ) -> ToolResult:
        """Create a Delta table from all parquet files in a staging directory.

        Uses Spark SQL to read the parquet directory and create/append to a
        Delta table. All data has already been written as parquet batch files
        by auto-flush and/or the final write_to_catalog call.
        """
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active Spark session found")

        # Spark SQL reads all parquet files in the directory
        parquet_glob = f"{staging_dir}/"
        if mode == "overwrite":
            spark.sql(f"DROP TABLE IF EXISTS {table_name}")
            spark.sql(
                f"CREATE TABLE {table_name} USING DELTA AS SELECT * FROM parquet.`{parquet_glob}`"
            )
        else:
            # append mode
            spark.sql(f"INSERT INTO {table_name} SELECT * FROM parquet.`{parquet_glob}`")

        # Verify
        result_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table_name}").collect()[0]["cnt"]

        return ToolResult(
            success=True,
            data={
                "table_name": table_name,
                "mode": mode,
                "rows_written": total_rows,
                "verified_row_count": result_count,
                "source_name": source_name,
                "method": "spark_sql_from_parquet",
                "staging_dir": staging_dir,
            },
            summary=(
                f"Created {table_name} from parquet staging ({mode}). "
                f"Expected {total_rows:,} rows, verified {result_count:,} rows."
            ),
        )


class ExecuteSQLTool(BaseTool):
    """Execute a SQL statement against the Databricks warehouse."""

    @property
    def name(self) -> str:
        return "execute_sql"

    @property
    def description(self) -> str:
        return (
            "Execute a SQL statement against the Databricks SQL warehouse. "
            "Use for DDL (CREATE TABLE, ALTER TABLE), DML (INSERT, UPDATE), "
            "or queries (SELECT). Returns up to 100 rows of results for queries."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL statement to execute.",
                },
            },
            "required": ["sql"],
        }

    # Maximum wall-clock seconds for any single SQL query (Spark path).
    # Queries exceeding this are killed and the agent gets an actionable
    # error message telling it to simplify.
    QUERY_TIMEOUT_SECONDS = 600  # 10 minutes

    def _execute(self, sql: str, **kwargs) -> ToolResult:  # type: ignore[override]
        import time

        t0 = time.time()
        stripped_upper = sql.strip().upper()
        is_ddl = stripped_upper.startswith(
            ("CREATE", "DROP", "ALTER", "INSERT", "MERGE", "TRUNCATE")
        )

        # Try Spark first (preferred in Databricks notebooks)
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark:
                # Safety: disable broadcast joins for DDL/CTAS to prevent
                # driver OOM when Spark misjudges table sizes
                if is_ddl:
                    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

                try:
                    result = spark.sql(sql)
                except Exception as e:
                    err_msg = str(e)
                    elapsed = time.time() - t0
                    # Detect OOM / driver crash patterns
                    oom_keywords = [
                        "OutOfMemoryError",
                        "Java heap space",
                        "GC overhead limit",
                        "Container killed",
                        "ExecutorLostFailure",
                        "SparkException",
                    ]
                    if any(kw.lower() in err_msg.lower() for kw in oom_keywords):
                        return ToolResult(
                            success=False,
                            error=(
                                f"OUT OF MEMORY after {elapsed:.0f}s. The query is "
                                f"too large to materialize at once. To fix this:\n"
                                f"1. Check for accidental Cartesian products — ensure "
                                f"every JOIN has an explicit ON clause.\n"
                                f"2. Estimate output size first: run SELECT COUNT(*) "
                                f"on the same query to see how many rows it produces.\n"
                                f"3. Build incrementally: join 2 tables at a time into "
                                f"intermediate silver tables, then combine.\n"
                                f"4. Add WHERE filters to reduce data volume.\n"
                                f"Error: {err_msg[:500]}"
                            ),
                        )
                    return ToolResult(
                        success=False,
                        error=f"SQL execution failed after {elapsed:.0f}s: {err_msg[:500]}",
                    )

                # Run .collect() in a thread with a timeout so a slow query
                # doesn't block the agent forever.
                #
                # IMPORTANT: We must NOT use `with ThreadPoolExecutor(...) as pool:`
                # because pool.__exit__ calls shutdown(wait=True), which blocks
                # until the thread finishes — defeating the timeout entirely.
                # Instead, create the pool manually and shut down with wait=False
                # on timeout so the agent can continue immediately.
                import concurrent.futures

                timeout = self.QUERY_TIMEOUT_SECONDS

                def _collect_ddl():
                    # DDL: just trigger execution, no rows to collect
                    result.collect()

                def _collect_query():
                    return [row.asDict() for row in result.limit(100).collect()]

                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    if is_ddl:
                        future = pool.submit(_collect_ddl)
                        future.result(timeout=timeout)
                        pool.shutdown(wait=False)
                        elapsed = time.time() - t0
                        return ToolResult(
                            success=True,
                            data={"rows": [], "row_count": 0, "method": "spark"},
                            summary=f"DDL/DML executed via Spark in {elapsed:.1f}s.",
                        )
                    else:
                        future = pool.submit(_collect_query)
                        rows = future.result(timeout=timeout)
                        pool.shutdown(wait=False)
                        elapsed = time.time() - t0
                        return ToolResult(
                            success=True,
                            data={"rows": rows, "row_count": len(rows), "method": "spark"},
                            summary=f"SQL executed via Spark in {elapsed:.1f}s, returned {len(rows)} rows.",
                        )
                except concurrent.futures.TimeoutError:
                    elapsed = time.time() - t0
                    # Don't wait for the thread — let it die in the background
                    pool.shutdown(wait=False)
                    # Cancel the Spark jobs so the cluster stops working on it
                    try:
                        spark.sparkContext.cancelAllJobs()
                    except Exception:
                        pass
                    return ToolResult(
                        success=False,
                        error=(
                            f"QUERY TIMEOUT after {elapsed:.0f}s (limit: {timeout}s). "
                            f"The query is too slow. To fix this:\n"
                            f"1. Add WHERE filters to reduce data volume (e.g., "
                            f"filter to a single year or state first).\n"
                            f"2. Check for accidental Cartesian products — ensure "
                            f"every JOIN has an explicit ON clause.\n"
                            f"3. Break the query into smaller steps: build "
                            f"intermediate silver tables with 2-table joins, then "
                            f"combine.\n"
                            f"4. Run SELECT COUNT(*) first to estimate output size "
                            f"before collecting results.\n"
                            f"5. Use SELECT DISTINCT or GROUP BY to reduce row count "
                            f"if the query returns many near-duplicate rows."
                        ),
                    )

        except ImportError:
            pass

        # Fallback to SDK — use long timeout + async polling for DDL/CTAS
        import os

        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient(
            host=os.environ.get("DATABRICKS_HOST", ""),
            token=os.environ.get("DATABRICKS_TOKEN", ""),
        )
        warehouses = list(client.warehouses.list())
        warehouse_id = warehouses[0].id if warehouses else None
        if not warehouse_id:
            return ToolResult(success=False, error="No SQL warehouse available.")

        # Detect DDL/CTAS which may need longer execution time
        stripped = sql.strip().upper()
        is_ddl = stripped.startswith(("CREATE", "DROP", "ALTER", "INSERT", "MERGE"))
        initial_timeout = "300s" if is_ddl else "120s"

        result = client.statement_execution.execute_statement(  # type: ignore[assignment]
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout=initial_timeout,
        )

        # If still running after initial timeout, poll until complete (up to 10 min)
        max_poll_seconds = 600
        poll_interval = 5
        poll_elapsed = 0
        while (
            result.status
            and result.status.state
            and result.status.state.value in ("PENDING", "RUNNING")
            and poll_elapsed < max_poll_seconds
        ):
            time.sleep(poll_interval)
            poll_elapsed += poll_interval
            result = client.statement_execution.get_statement(result.statement_id)

        # Check for failure or still running
        if result.status and result.status.state:
            state = result.status.state.value
            if state in ("PENDING", "RUNNING"):
                elapsed = time.time() - t0
                return ToolResult(
                    success=False,
                    error=(
                        f"SQL statement still running after {elapsed:.0f}s. "
                        f"Statement ID: {result.statement_id}. "
                        f"The query may be too complex — consider breaking it "
                        f"into smaller steps."
                    ),
                )
            if state in ("FAILED", "CANCELED", "CLOSED"):
                error_msg = ""
                if result.status.error:
                    error_msg = result.status.error.message or str(result.status.error)
                elapsed = time.time() - t0
                return ToolResult(
                    success=False,
                    error=f"SQL {state.lower()} after {elapsed:.0f}s: {error_msg}",
                )

        rows = []
        if result.result and result.result.data_array:
            col_names = [c.name for c in result.manifest.schema.columns] if result.manifest else []
            for row in result.result.data_array[:100]:
                if col_names:
                    rows.append(dict(zip(col_names, row)))
                else:
                    rows.append(row)

        elapsed = time.time() - t0
        return ToolResult(
            success=True,
            data={"rows": rows, "row_count": len(rows), "method": "sdk"},
            summary=f"SQL executed via SDK in {elapsed:.1f}s, returned {len(rows)} rows.",
        )


class SilverOnlyExecuteSQLTool(ExecuteSQLTool):
    """Execute SQL with write protection for bronze tables.

    Allows SELECT on any table. DDL/DML (CREATE, DROP, ALTER, INSERT,
    UPDATE, DELETE) is only allowed on tables with a ``silver_`` prefix
    or the ``data_catalog`` table.
    """

    # SQL keywords that modify data/schema
    _WRITE_KEYWORDS = {"CREATE", "DROP", "ALTER", "INSERT", "UPDATE", "DELETE", "MERGE", "TRUNCATE"}

    @property
    def description(self) -> str:
        return (
            "Execute a SQL statement against the Databricks SQL warehouse. "
            "SELECT is allowed on any table. DDL/DML (CREATE, DROP, ALTER, INSERT, "
            "UPDATE, DELETE) is only allowed on silver_ tables. Bronze tables are read-only."
        )

    def _execute(self, sql: str, **kwargs) -> ToolResult:  # type: ignore[override]
        import re

        # Check if this is a write operation
        stripped = sql.strip().upper()
        first_keyword = stripped.split()[0] if stripped else ""

        if first_keyword in self._WRITE_KEYWORDS:
            # Extract the table name from the SQL
            # Match patterns like: CREATE TABLE x, DROP TABLE x, ALTER TABLE x,
            # INSERT INTO x, UPDATE x, DELETE FROM x, MERGE INTO x
            table_match = re.search(
                r"(?:TABLE|INTO|UPDATE|FROM|MERGE\s+INTO)\s+"
                r"(?:IF\s+(?:NOT\s+)?EXISTS\s+)?"
                r"(`?[\w.]+`?)",
                stripped,
            )
            if table_match:
                table_ref = table_match.group(1).strip("`").lower()
                # Get the simple table name (last part of dotted name)
                simple_name = table_ref.split(".")[-1]

                if not simple_name.startswith("silver_") and simple_name != "data_catalog":
                    return ToolResult(
                        success=False,
                        error=(
                            f"BLOCKED: Cannot run {first_keyword} on '{simple_name}'. "
                            f"Bronze tables are read-only. You may only modify tables "
                            f"with the 'silver_' prefix. Use SELECT to read bronze data, "
                            f"then CREATE TABLE silver_<name> to store your results."
                        ),
                    )

        return super()._execute(sql=sql, **kwargs)


class ListCatalogTablesTool(BaseTool):
    """List existing tables in the target Unity Catalog schema."""

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "list_catalog_tables"

    @property
    def description(self) -> str:
        return (
            f"List all existing tables in the target Unity Catalog schema "
            f"({self._cfg.full_schema}). Returns table names, "
            f"types, and row counts. Use this to check what has already been "
            f"loaded or to validate your work."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }

    def _execute(self, **kwargs) -> ToolResult:
        catalog = self._cfg.catalog
        schema = self._cfg.schema

        sql = f"SHOW TABLES IN {catalog}.{schema}"

        # Delegate to ExecuteSQLTool logic
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark:
                result = spark.sql(sql)
                tables = [row.asDict() for row in result.collect()]
                return ToolResult(
                    success=True,
                    data={"tables": tables, "count": len(tables)},
                    summary=f"Found {len(tables)} tables in {catalog}.{schema}.",
                )
        except Exception:
            pass

        import os

        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient(
            host=os.environ.get("DATABRICKS_HOST", ""),
            token=os.environ.get("DATABRICKS_TOKEN", ""),
        )
        warehouses = list(client.warehouses.list())
        if not warehouses:
            return ToolResult(success=False, error="No SQL warehouse found.")

        result = client.statement_execution.execute_statement(  # type: ignore[assignment]
            warehouse_id=warehouses[0].id, statement=sql, wait_timeout="30s"  # type: ignore[arg-type]
        )

        tables = []
        if result.result and result.result.data_array:
            col_names = [c.name for c in result.manifest.schema.columns] if result.manifest else []
            for row in result.result.data_array:
                tables.append(dict(zip(col_names, row)) if col_names else row)

        return ToolResult(
            success=True,
            data={"tables": tables, "count": len(tables)},
            summary=f"Found {len(tables)} tables in {catalog}.{schema}.",
        )
