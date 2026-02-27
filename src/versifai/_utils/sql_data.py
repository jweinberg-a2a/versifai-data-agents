"""
Shared SQL data fetcher for tools that need to execute queries directly.

Tries Spark first (native in Databricks notebooks), then falls back to the
Databricks SDK. Returns all rows as a pandas DataFrame — no row cap.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger("versifai.sql_data")


def fetch_sql_data(sql: str) -> pd.DataFrame | None:
    """Execute a SQL query and return ALL rows as a DataFrame.

    Tries Spark first (native in Databricks), then falls back to the
    Databricks SDK. Unlike ``execute_sql``, there is no 100-row cap —
    this fetches the complete result set.

    Returns ``None`` if both Spark and SDK fail.
    """
    # Try Spark first
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark:
            sdf = spark.sql(sql)
            return sdf.toPandas()
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Spark SQL failed: %s", e)

    # Fallback to Databricks SDK
    try:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient(
            host=os.environ.get("DATABRICKS_HOST", ""),
            token=os.environ.get("DATABRICKS_TOKEN", ""),
        )
        warehouses = list(client.warehouses.list())
        warehouse_id = warehouses[0].id if warehouses else None
        if not warehouse_id:
            return None

        result = client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="120s",
        )

        if result.result and result.result.data_array:
            col_names = (
                [c.name for c in result.manifest.schema.columns]  # type: ignore[union-attr]
                if result.manifest
                else []
            )
            rows = result.result.data_array
            if col_names:
                return pd.DataFrame(rows, columns=col_names)
            return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("SDK SQL failed: %s", e)

    return None
