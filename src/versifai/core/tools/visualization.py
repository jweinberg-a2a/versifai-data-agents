"""
Tool: create_visualization

Generates matplotlib/seaborn charts from data provided by the agent,
saves them to the results volume, and displays them inline in the
Databricks notebook.
"""

from __future__ import annotations

import base64
import io
import os
from typing import TYPE_CHECKING, Any

import pandas as pd

from versifai.core.display import AgentDisplay
from versifai.core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from versifai.science_agents.scientist.config import ResearchConfig


class CreateVisualizationTool(BaseTool):
    """
    Create publication-quality charts and result tables.

    The agent sends data (as a list of dicts, typically from SQL results)
    along with chart configuration, and this tool renders the chart,
    saves the PNG to the results volume, and displays it in the notebook.

    For choropleths and large datasets, the agent can pass a ``sql_query``
    instead of ``data``. The tool executes the SQL directly against Spark /
    Databricks SDK and fetches **all** rows — no 100-row limit.
    """

    def __init__(
        self,
        cfg: ResearchConfig | None = None,
        display: AgentDisplay | None = None,
        notes_path: str = "",
        results_path: str = "",
    ) -> None:
        super().__init__()
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg
        self._display = display or AgentDisplay()
        self._charts_created: list[str] = []
        self._notes_path = notes_path
        # When results_path is set (e.g. run-isolated directory), use it
        # for charts/ and tables/ instead of cfg.results_volume_path.
        self._results_path = results_path or cfg.results_volume_path

    def _fetch_sql_data(self, sql: str) -> pd.DataFrame | None:
        """Execute a SQL query and return ALL rows as a DataFrame.

        Tries Spark first (native in Databricks), then falls back to the
        Databricks SDK. Unlike execute_sql, there is no 100-row cap — this
        fetches the complete result set for visualization.
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
            self._display.warning(f"Spark SQL failed: {e}")

        # Fallback to SDK
        try:
            import os

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
                    [c.name for c in result.manifest.schema.columns] if result.manifest else []  # type: ignore[union-attr]
                )
                rows = result.result.data_array
                if col_names:
                    return pd.DataFrame(rows, columns=col_names)
                return pd.DataFrame(rows)
        except Exception as e:
            self._display.warning(f"SDK SQL failed: {e}")

        return None

    def _append_chart_metadata(
        self,
        theme_id: str,
        filename: str,
        chart_type: str,
        title: str,
        save_path: str,
        df: pd.DataFrame,
        sql_query: str = "",
        datasets: dict[str, str] | None = None,
        render_code: str = "",
        interpretation: str = "",
    ) -> None:
        """Append chart reproducibility metadata to the theme's notes file.

        Records the interpretation, SQL queries, dataset queries, render code
        (for custom), and a sample of the input data so every chart can be
        reproduced or edited later. Uses read-then-write for FUSE compatibility.
        """
        if not theme_id or not self._notes_path:
            return

        from datetime import datetime

        os.makedirs(self._notes_path, exist_ok=True)
        filepath = os.path.join(self._notes_path, f"{theme_id}_notes.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [f"\n--- {timestamp} | CHART: {filename} ---"]
        lines.append(f"Type: {chart_type}")
        lines.append(f"Title: {title}")
        lines.append(f"Path: {save_path}")

        if interpretation:
            lines.append(f"\nInterpretation: {interpretation.strip()}")

        if sql_query:
            lines.append(f"\nSQL Query:\n  {sql_query.strip()}")

        if datasets:
            lines.append("\nDataset Queries:")
            for name, query in datasets.items():
                lines.append(f"  {name}: {query.strip()}")

        if render_code:
            lines.append(f"\nRender Code:\n  {render_code.strip()}")

        # Input data sample
        if df is not None and not df.empty:
            sample_n = min(20, len(df))
            csv_sample = df.head(sample_n).to_csv(index=False)
            lines.append(
                f"\nInput Data ({sample_n} of {len(df)} rows):\n  "
                + csv_sample.strip().replace("\n", "\n  ")
            )

        block = "\n".join(lines) + "\n"

        # Read-then-write for FUSE compatibility (no append mode)
        existing = ""
        if os.path.isfile(filepath):
            try:
                with open(filepath) as f:
                    existing = f.read()
            except OSError:
                pass
        try:
            with open(filepath, "w") as f:
                f.write(existing + block)
        except OSError:
            pass

    @property
    def name(self) -> str:
        return "create_visualization"

    @property
    def description(self) -> str:
        return (
            "Create a chart or result table from data and save it to the results volume. "
            "Supported chart types: bar, scatter, heatmap, line, box, histogram, "
            "waterfall, dumbbell, lollipop, violin, "
            "choropleth, dual_choropleth, sankey, table, **custom**. "
            "For 'choropleth': renders a US county-level map colored by a value column. "
            "Data MUST include a FIPS column (5-digit county FIPS code) and "
            "a value column specified via color_column. Uses plotly for interactive maps. "
            "For 'dual_choropleth': renders two side-by-side US county maps for comparison. "
            "Data MUST include FIPS plus TWO value columns specified as "
            "color_column (left map) and y_column (right map). Use x_label/y_label for "
            "the left/right map subtitles. "
            "For 'table' type, renders an HTML table and saves as CSV. "
            "For all other types, generates a matplotlib/seaborn chart saved as PNG. "
            "\n\n"
            "**custom**: Full Python environment for signature visualizations. "
            "Write a complete program in `render_code` — define functions, "
            "transform data with pandas/numpy/scipy, compute derived metrics, "
            "then render the final chart. Available libraries: pd, np, math, "
            "json, re, Counter, defaultdict, plt, matplotlib (mcolors, mpatches, "
            "mticker), sns, go (plotly), px, make_subplots, scipy_stats, "
            "scipy_hierarchy, StandardScaler, MinMaxScaler, KMeans. "
            "Use `datasets` (dict of name→SQL) to load MULTIPLE data sources — "
            "each becomes a named DataFrame. You can use sql_query + datasets "
            "together, or datasets alone (no sql_query needed). "
            "Your code MUST set `fig` (matplotlib Figure) or `plotly_fig` "
            "(plotly Figure). The tool handles saving and display. "
            "This is the PREFERRED type for signature visualizations, multi-panel "
            "layouts, bivariate choropleths, and any chart requiring custom "
            "data processing or composition.\n\n"
            "DATA SOURCE: For standard types, always use 'sql_query'. "
            "For custom types, use sql_query and/or datasets. "
            "Do NOT call execute_sql first then pass results as 'data' — "
            "that truncates to 100 rows.\n\n"
            "The chart is displayed inline in the notebook and saved to the results volume. "
            "Returns the file path of the saved chart/table."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "description": (
                        "Type of visualization: 'bar', 'scatter', 'heatmap', "
                        "'line', 'box', 'histogram', 'waterfall', 'dumbbell', "
                        "'lollipop', 'violin', 'choropleth', "
                        "'dual_choropleth', 'sankey', 'table', or 'custom'. "
                        "waterfall: decomposition chart showing positive/negative contributions "
                        "to a total (x_column=categories, y_column=values). "
                        "dumbbell: paired comparison showing two values per category "
                        "(y_column=categories, x_column=start value, color_column=end value). "
                        "lollipop: ranked horizontal stems with dots (x_column=values, "
                        "y_column=categories). "
                        "violin: distribution plot richer than box (same params as box). "
                        "sankey: flow diagram (data needs 'source', 'target', 'value' columns). "
                        "custom: freeform Python rendering via render_code — USE THIS for "
                        "signature visualizations, multi-panel layouts, bivariate choropleths, "
                        "and any chart requiring custom styling."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Chart title.",
                },
                "sql_query": {
                    "type": "string",
                    "description": (
                        "A SELECT SQL query to fetch chart data. The tool executes it "
                        "directly and retrieves ALL rows — no row limit. This is the "
                        "PREFERRED way to provide data. Do NOT call execute_sql first. "
                        "Example: 'SELECT county_fips_code, ma_penetration "
                        "FROM catalog.schema.silver_county_master'"
                    ),
                },
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Fallback: pre-computed data as a list of row dicts. "
                        "Only use if you have already transformed data in Python "
                        "and cannot express it as SQL. Prefer sql_query."
                    ),
                },
                "x_column": {
                    "type": "string",
                    "description": "Column name for the x-axis.",
                },
                "y_column": {
                    "type": "string",
                    "description": "Column name for the y-axis.",
                },
                "color_column": {
                    "type": "string",
                    "description": (
                        "Column for color/grouping. For standard charts: hue grouping. "
                        "For choropleth: the value column to color counties by. "
                        "For dual_choropleth: the LEFT map value column."
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Filename to save as (e.g. 'rq1_svi_vs_penetration.png'). "
                        "PNG extension is added automatically for charts."
                    ),
                },
                "theme_id": {
                    "type": "string",
                    "description": (
                        "Theme ID for this chart (e.g. 'theme_1'). Used to log "
                        "chart metadata (SQL queries, input data, render code) "
                        "to the theme's notes file for reproducibility."
                    ),
                },
                "interpretation": {
                    "type": "string",
                    "description": (
                        "2-3 sentence interpretation of what the chart shows and "
                        "how to read it. Logged to the theme notes alongside the "
                        "chart metadata. Example: 'This choropleth maps MA "
                        "penetration by county. Darker red counties have lower "
                        "penetration. The Southeast cluster shows a clear disparity "
                        "corridor overlapping high-SVI areas.'"
                    ),
                },
                "x_label": {
                    "type": "string",
                    "description": "Optional custom x-axis label.",
                },
                "y_label": {
                    "type": "string",
                    "description": "Optional custom y-axis label.",
                },
                "annotations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional text annotations to add to the chart.",
                },
                "fips_column": {
                    "type": "string",
                    "description": (
                        "Column containing 5-digit county FIPS codes (for choropleth "
                        "and dual_choropleth only). Defaults to 'fips' if not specified."
                    ),
                },
                "color_scale": {
                    "type": "string",
                    "description": (
                        "Color scale for choropleth maps. Options: 'Viridis', 'RdYlGn', "
                        "'RdBu', 'Blues', 'Reds', 'YlOrRd', 'Plasma', 'Inferno'. "
                        "Defaults to 'Viridis'. Use 'RdBu' or 'RdYlGn' for diverging data."
                    ),
                },
                "render_code": {
                    "type": "string",
                    "description": (
                        "Full Python program for chart_type='custom'. Write complete "
                        "data transformation AND visualization code — define functions, "
                        "munge DataFrames, compute derived metrics, then render. "
                        "Available: df (primary DataFrame), title (string), pd, np, "
                        "math, json, re, Counter, defaultdict, plt, matplotlib, "
                        "mcolors, mpatches, mticker, sns, go (plotly), px, "
                        "make_subplots, scipy_stats, scipy_hierarchy, StandardScaler, "
                        "MinMaxScaler, KMeans, plus named DataFrames from `datasets`. "
                        "MUST set `fig` (matplotlib) or `plotly_fig` (plotly). "
                        "Example with data processing:\n"
                        "def compute_quartile_stats(data, value_col, group_col):\n"
                        "    data['quartile'] = pd.qcut(data[group_col], 4, labels=['Q1','Q2','Q3','Q4'])\n"
                        "    return data.groupby('quartile')[value_col].agg(['mean','std','count'])\n\n"
                        "stats = compute_quartile_stats(df, 'ma_penetration', 'svi_score')\n"
                        "merged = df.merge(other_df, on='fips')  # other_df from datasets\n"
                        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
                        "# ... custom rendering ..."
                    ),
                },
                "datasets": {
                    "type": "object",
                    "description": (
                        "Additional named data sources for chart_type='custom'. "
                        "A dict mapping variable names to SQL queries. Each query is "
                        "executed and the result is available as a DataFrame in render_code. "
                        'Example: {"ref_data": "SELECT * FROM ref_table", '
                        '"cutpoints": "SELECT * FROM cutpoint_table"} — then '
                        "render_code can use `ref_data` and `cutpoints` as DataFrames "
                        "alongside the primary `df` from sql_query."
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["chart_type", "title", "filename"],
        }

    def _execute(
        self,
        chart_type: str = "",
        title: str = "",
        data: list[dict] | None = None,
        sql_query: str = "",
        filename: str = "",
        theme_id: str = "",
        interpretation: str = "",
        x_column: str = "",
        y_column: str = "",
        color_column: str = "",
        x_label: str = "",
        y_label: str = "",
        annotations: list[str] | None = None,
        fips_column: str = "",
        color_scale: str = "",
        render_code: str = "",
        datasets: dict[str, str] | None = None,
        **kwargs,
    ) -> ToolResult:
        if not chart_type:
            return ToolResult(success=False, error="Missing 'chart_type'.")
        if not filename:
            return ToolResult(success=False, error="Missing 'filename'.")

        # Resolve data: prefer sql_query (fetches ALL rows), fall back to data.
        # For chart_type='custom', data sources are optional (can use datasets only).
        if sql_query:
            df = self._fetch_sql_data(sql_query)
            if df is None:
                return ToolResult(
                    success=False,
                    error="sql_query execution failed — check the SQL syntax.",
                )
        elif data:
            df = pd.DataFrame(data)
        elif chart_type == "custom" and datasets:
            # Custom charts can get all data from datasets — df is empty placeholder
            df = pd.DataFrame()
        else:
            return ToolResult(
                success=False,
                error="Provide either 'data' (list of dicts) or 'sql_query' (SELECT statement).",
            )

        if df.empty and chart_type != "custom":
            return ToolResult(success=False, error="Data is empty — nothing to plot.")

        # Route to the right renderer
        result: ToolResult | None = None
        if chart_type == "table":
            result = self._render_table(df, title, filename)
        elif chart_type == "choropleth":
            result = self._render_choropleth(
                df=df,
                title=title,
                filename=filename,
                fips_column=fips_column or "fips",
                value_column=color_column,
                color_scale=color_scale or "Viridis",
                label=y_label or color_column,
            )
        elif chart_type == "dual_choropleth":
            result = self._render_dual_choropleth(
                df=df,
                title=title,
                filename=filename,
                fips_column=fips_column or "fips",
                left_column=color_column,
                right_column=y_column,
                left_label=x_label or color_column,
                right_label=y_label or y_column,
                color_scale=color_scale or "Viridis",
            )
        elif chart_type == "sankey":
            result = self._render_sankey(
                df=df,
                title=title,
                filename=filename,
            )
        elif chart_type == "custom":
            if not render_code:
                return ToolResult(
                    success=False,
                    error="chart_type='custom' requires 'render_code' — Python code that "
                    "creates either a matplotlib `fig` or a plotly `plotly_fig`.",
                )
            result = self._render_custom(
                df=df,
                title=title,
                filename=filename,
                render_code=render_code,
                datasets=datasets,
                extra_vars=kwargs,
            )
        else:
            result = self._render_chart(
                df=df,
                chart_type=chart_type,
                title=title,
                filename=filename,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                x_label=x_label,
                y_label=y_label,
                annotations=annotations or [],
            )

        # Post-processing for successful results
        if result and result.success:
            save_path = (result.data or {}).get("chart_path") or (result.data or {}).get(
                "table_path", ""
            )

            # Return the chart image so Claude can visually review it
            if save_path and save_path.endswith(".png"):
                result.image_path = save_path

            # Log chart metadata to theme notes for reproducibility
            if theme_id:
                self._append_chart_metadata(
                    theme_id=theme_id,
                    filename=filename,
                    chart_type=chart_type,
                    title=title,
                    save_path=save_path,
                    df=df,
                    sql_query=sql_query,
                    datasets=datasets,
                    render_code=render_code,
                    interpretation=interpretation,
                )

        return result

    # ------------------------------------------------------------------
    # Chart rendering
    # ------------------------------------------------------------------

    def _render_chart(
        self,
        df: pd.DataFrame,
        chart_type: str,
        title: str,
        filename: str,
        x_column: str,
        y_column: str,
        color_column: str,
        x_label: str,
        y_label: str,
        annotations: list[str],
    ) -> ToolResult:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        cfg = self._cfg
        try:
            plt.style.use(cfg.chart_style)
        except OSError:
            pass  # Style not available, use default

        sns.set_palette(cfg.color_palette)
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            if chart_type == "bar":
                if color_column and color_column in df.columns:
                    sns.barplot(data=df, x=x_column, y=y_column, hue=color_column, ax=ax)
                else:
                    sns.barplot(data=df, x=x_column, y=y_column, ax=ax)
                plt.xticks(rotation=45, ha="right")

            elif chart_type == "scatter":
                if color_column and color_column in df.columns:
                    sns.scatterplot(
                        data=df, x=x_column, y=y_column, hue=color_column, ax=ax, alpha=0.7
                    )
                else:
                    sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax, alpha=0.7)

            elif chart_type == "line":
                if color_column and color_column in df.columns:
                    sns.lineplot(
                        data=df, x=x_column, y=y_column, hue=color_column, ax=ax, marker="o"
                    )
                else:
                    sns.lineplot(data=df, x=x_column, y=y_column, ax=ax, marker="o")

            elif chart_type == "box":
                if color_column and color_column in df.columns:
                    sns.boxplot(data=df, x=x_column, y=y_column, hue=color_column, ax=ax)
                else:
                    sns.boxplot(data=df, x=x_column, y=y_column, ax=ax)

            elif chart_type == "histogram":
                col = x_column or y_column or df.columns[0]
                if color_column and color_column in df.columns:
                    sns.histplot(data=df, x=col, hue=color_column, ax=ax, kde=True)
                else:
                    sns.histplot(data=df, x=col, ax=ax, kde=True)

            elif chart_type == "heatmap":
                # Pivot the data for heatmap — requires x, y, and a value column
                if x_column and y_column and color_column:
                    pivot = df.pivot_table(
                        index=y_column,
                        columns=x_column,
                        values=color_column,
                        aggfunc="mean",
                    )
                    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap=cfg.color_palette)
                else:
                    # Correlation matrix fallback
                    numeric = df.select_dtypes(include="number")
                    if not numeric.empty:
                        sns.heatmap(
                            numeric.corr(), ax=ax, annot=True, fmt=".2f", cmap=cfg.color_palette
                        )
                    else:
                        plt.close(fig)
                        return ToolResult(
                            success=False,
                            error="Heatmap needs numeric data or x_column/y_column/color_column for a pivot.",
                        )
            elif chart_type == "waterfall":
                # Waterfall decomposition chart — positive/negative contributions
                categories = (
                    df[x_column].astype(str).tolist()
                    if x_column
                    else df.iloc[:, 0].astype(str).tolist()
                )
                values = (
                    df[y_column].astype(float).tolist()
                    if y_column
                    else df.iloc[:, 1].astype(float).tolist()
                )

                cumulative = []
                running = 0.0
                for v in values:
                    cumulative.append(running)
                    running += v

                colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]
                # Last bar is the total
                if len(categories) > 0:
                    categories.append("Total")
                    values.append(running)
                    cumulative.append(0)
                    colors.append("#3b82f6")

                y_pos = range(len(categories))
                bottoms = [c if v >= 0 else c + v for c, v in zip(cumulative, values)]
                bar_heights = [abs(v) for v in values]

                ax.barh(
                    y_pos, bar_heights, left=bottoms, color=colors, edgecolor="white", linewidth=0.5
                )
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(categories)
                ax.invert_yaxis()

                # Connector lines
                for i in range(len(values) - 2):  # skip last (total)
                    end_x = cumulative[i] + values[i]
                    ax.plot(
                        [end_x, end_x],
                        [i - 0.4, i + 1.4],
                        color="#64748b",
                        linewidth=0.8,
                        linestyle="--",
                    )

                # Value labels
                for i, (b, h, v) in enumerate(zip(bottoms, bar_heights, values)):
                    label_x = b + h / 2
                    ax.text(
                        label_x,
                        i,
                        f"{v:+,.1f}" if i < len(values) - 1 else f"{v:,.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )

            elif chart_type == "dumbbell":
                # Dumbbell chart — paired comparison per category
                # y_column = categories, x_column = start value, color_column = end value
                cats = (
                    df[y_column].astype(str).tolist()
                    if y_column
                    else df.iloc[:, 0].astype(str).tolist()
                )
                starts = (
                    pd.to_numeric(df[x_column], errors="coerce").tolist()
                    if x_column
                    else df.iloc[:, 1].astype(float).tolist()
                )
                ends = (
                    pd.to_numeric(df[color_column], errors="coerce").tolist()
                    if color_column and color_column in df.columns
                    else starts
                )

                y_pos = range(len(cats))
                ax.hlines(y=y_pos, xmin=starts, xmax=ends, color="#64748b", linewidth=2, zorder=1)
                ax.scatter(
                    starts,
                    y_pos,
                    color="#3b82f6",
                    s=80,
                    zorder=2,
                    label=x_label or x_column or "Start",
                )
                ax.scatter(
                    ends,
                    y_pos,
                    color="#ef4444",
                    s=80,
                    zorder=2,
                    label=y_label or color_column or "End",
                )
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(cats)
                ax.invert_yaxis()
                ax.legend(loc="lower right")

            elif chart_type == "lollipop":
                # Lollipop chart — ranked horizontal stems with dots
                if y_column and y_column in df.columns:
                    cats = df[y_column].astype(str).tolist()
                else:
                    cats = df.iloc[:, 0].astype(str).tolist()
                if x_column and x_column in df.columns:
                    vals = pd.to_numeric(df[x_column], errors="coerce").tolist()
                else:
                    vals = df.iloc[:, 1].astype(float).tolist()

                # Sort by value
                paired = sorted(zip(vals, cats), key=lambda x: x[0])
                vals, cats = zip(*paired) if paired else ([], [])

                y_pos = range(len(cats))
                dot_color = "#7c3aed"
                if color_column and color_column in df.columns:
                    # Use color_column for coloring by category
                    unique_colors = df[color_column].unique()
                    cmap = sns.color_palette(cfg.color_palette, len(unique_colors))
                    color_map = dict(zip(unique_colors, cmap))
                    # Re-sort to match
                    sorted_df = df.sort_values(
                        by=x_column if x_column in df.columns else df.columns[1]
                    )
                    dot_colors = [color_map.get(c, dot_color) for c in sorted_df[color_column]]
                else:
                    dot_colors = [dot_color] * len(cats)

                ax.hlines(y=y_pos, xmin=0, xmax=vals, color="#94a3b8", linewidth=1.5)
                ax.scatter(vals, y_pos, color=dot_colors, s=60, zorder=2)
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(cats)

            elif chart_type == "violin":
                # Violin plot — richer distribution view than box
                if color_column and color_column in df.columns:
                    sns.violinplot(
                        data=df, x=x_column, y=y_column, hue=color_column, ax=ax, inner="box", cut=0
                    )
                else:
                    sns.violinplot(data=df, x=x_column, y=y_column, ax=ax, inner="box", cut=0)

            else:
                plt.close(fig)
                return ToolResult(
                    success=False,
                    error=f"Unknown chart_type '{chart_type}'. Use: bar, scatter, heatmap, line, box, histogram, waterfall, dumbbell, lollipop, violin, choropleth, dual_choropleth, sankey, table.",
                )

        except Exception as e:
            plt.close(fig)
            return ToolResult(success=False, error=f"Chart rendering failed: {e}")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        # Add annotations
        for i, ann in enumerate(annotations[:5]):
            ax.annotate(
                ann,
                xy=(0.02, 0.98 - i * 0.05),
                xycoords="axes fraction",
                fontsize=9,
                color="gray",
                va="top",
            )

        plt.tight_layout()

        # Save to results volume
        if not filename.endswith(".png"):
            filename += ".png"
        charts_dir = os.path.join(self._results_path, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        save_path = os.path.join(charts_dir, filename)
        fig.savefig(save_path, dpi=cfg.chart_dpi, bbox_inches="tight")

        # Display inline in notebook
        self._display_chart_inline(fig)
        plt.close(fig)

        self._charts_created.append(save_path)

        return ToolResult(
            success=True,
            data={
                "chart_path": save_path,
                "chart_type": chart_type,
                "title": title,
                "rows_plotted": len(df),
            },
            summary=f"Chart saved: {save_path} ({chart_type}, {len(df)} rows).",
        )

    # ------------------------------------------------------------------
    # Choropleth rendering (plotly — US county-level geo maps)
    # ------------------------------------------------------------------

    _counties_geojson: dict | None = None  # Class-level cache

    @classmethod
    def _get_counties_geojson(cls) -> dict:
        """Fetch and cache US county GeoJSON boundaries from plotly's dataset.

        The GeoJSON is ~7 MB and contains boundaries for all ~3,200 US counties,
        keyed by 5-digit FIPS code. Cached on the class so it's only fetched
        once per process.
        """
        if cls._counties_geojson is not None:
            return cls._counties_geojson

        import json
        from urllib.request import urlopen

        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urlopen(url, timeout=30) as response:
            cls._counties_geojson = json.load(response)
        return cls._counties_geojson

    def _render_choropleth(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
        fips_column: str,
        value_column: str,
        color_scale: str,
        label: str,
    ) -> ToolResult:
        """Render a US county-level choropleth map using plotly."""
        import plotly.express as px

        if fips_column not in df.columns:
            return ToolResult(
                success=False,
                error=f"FIPS column '{fips_column}' not found in data. Columns: {list(df.columns)}",
            )
        if not value_column or value_column not in df.columns:
            return ToolResult(
                success=False,
                error=f"Value column '{value_column}' not found. Specify via "
                f"color_column. Columns: {list(df.columns)}",
            )

        # Ensure FIPS codes are 5-digit zero-padded strings
        df = df.copy()
        df[fips_column] = df[fips_column].astype(str).str.zfill(5)

        try:
            counties = self._get_counties_geojson()
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to fetch county GeoJSON: {e}",
            )

        try:
            fig = px.choropleth(
                df,
                geojson=counties,
                locations=fips_column,
                color=value_column,
                color_continuous_scale=color_scale,
                scope="usa",
                labels={value_column: label},
                title=title,
            )
            fig.update_layout(
                geo=dict(
                    lakecolor="rgb(255,255,255)",
                    landcolor="rgb(245,245,245)",
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                title_font_size=16,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Choropleth rendering failed: {e}")

        return self._save_plotly_figure(fig, title, filename, "choropleth", len(df))

    def _render_dual_choropleth(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
        fips_column: str,
        left_column: str,
        right_column: str,
        left_label: str,
        right_label: str,
        color_scale: str,
    ) -> ToolResult:
        """Render two side-by-side US county choropleth maps for comparison."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if fips_column not in df.columns:
            return ToolResult(
                success=False,
                error=f"FIPS column '{fips_column}' not found. Columns: {list(df.columns)}",
            )
        for col, name in [(left_column, "color_column"), (right_column, "y_column")]:
            if not col or col not in df.columns:
                return ToolResult(
                    success=False,
                    error=f"Column '{col}' ({name}) not found. Columns: {list(df.columns)}",
                )

        df = df.copy()
        df[fips_column] = df[fips_column].astype(str).str.zfill(5)

        try:
            counties = self._get_counties_geojson()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to fetch county GeoJSON: {e}")

        try:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(left_label, right_label),
                specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
            )

            fig.add_trace(
                go.Choropleth(
                    geojson=counties,
                    locations=df[fips_column],
                    z=df[left_column],
                    colorscale=color_scale,
                    colorbar=dict(title=left_label, x=0.45, len=0.75),
                    marker_line_width=0.1,
                    marker_line_color="white",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Choropleth(
                    geojson=counties,
                    locations=df[fips_column],
                    z=df[right_column],
                    colorscale=color_scale,
                    colorbar=dict(title=right_label, x=1.0, len=0.75),
                    marker_line_width=0.1,
                    marker_line_color="white",
                ),
                row=1,
                col=2,
            )

            fig.update_geos(
                scope="usa",
                lakecolor="rgb(255,255,255)",
                landcolor="rgb(245,245,245)",
            )
            fig.update_layout(
                title_text=title,
                title_font_size=16,
                margin=dict(l=0, r=0, t=60, b=0),
                width=1400,
                height=500,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Dual choropleth rendering failed: {e}")

        return self._save_plotly_figure(fig, title, filename, "dual_choropleth", len(df))

    def _render_custom(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
        render_code: str,
        datasets: dict[str, str] | None = None,
        extra_vars: dict | None = None,
    ) -> ToolResult:
        """Execute a custom Python program to transform data and produce a visualization.

        The code runs with the full Python data-science stack pre-imported.
        The agent can define functions, transform DataFrames, compute derived
        metrics, and then render the final visualization.  The code must
        assign either ``fig`` (matplotlib Figure) or ``plotly_fig`` (plotly
        Figure) into the namespace.

        Args:
            df: Primary DataFrame (from sql_query or data).
            title: Chart title.
            filename: Output filename.
            render_code: Full Python program — may include function definitions,
                data transformation pipelines, and visualization rendering.
            datasets: Dict of name → SQL query. Each query is executed and
                the resulting DataFrame is injected into the namespace.
            extra_vars: Any additional kwargs passed by the agent — injected
                directly into the namespace.
        """
        import json as json_mod
        import math
        import re as re_mod
        from collections import Counter, defaultdict
        from itertools import groupby

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
        import seaborn as sns

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            go = px = make_subplots = None

        try:
            import scipy.cluster.hierarchy as scipy_hierarchy
            import scipy.stats as scipy_stats
        except ImportError:
            scipy_stats = scipy_hierarchy = None

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
        except ImportError:
            StandardScaler = MinMaxScaler = KMeans = None

        cfg = self._cfg

        # Execute additional dataset queries
        extra_dfs: dict[str, pd.DataFrame] = {}
        if datasets:
            for var_name, sql in datasets.items():
                result_df = self._fetch_sql_data(sql)
                if result_df is None:
                    return ToolResult(
                        success=False,
                        error=f"datasets['{var_name}'] query failed: {sql[:200]}",
                    )
                extra_dfs[var_name] = result_df

        # Build the execution namespace — full data-science stack
        exec_globals: dict[str, Any] = {
            "__builtins__": __builtins__,
            # Data
            "df": df,
            "title": title,
            # Core libraries
            "pd": pd,
            "np": np,
            "math": math,
            "json": json_mod,
            "re": re_mod,
            # Collections / itertools
            "Counter": Counter,
            "defaultdict": defaultdict,
            "groupby": groupby,
            # Visualization — matplotlib
            "plt": plt,
            "matplotlib": matplotlib,
            "mcolors": mcolors,
            "mpatches": mpatches,
            "mticker": mticker,
            # Visualization — seaborn
            "sns": sns,
            # Visualization — plotly
            "go": go,
            "px": px,
            "make_subplots": make_subplots,
            # Statistics
            "scipy_stats": scipy_stats,
            "scipy_hierarchy": scipy_hierarchy,
            # Sklearn
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "KMeans": KMeans,
        }

        # Inject additional datasets as named variables
        exec_globals.update(extra_dfs)

        # Inject any extra kwargs from the agent
        if extra_vars:
            exec_globals.update(extra_vars)

        try:
            exec(render_code, exec_globals)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Custom render_code execution failed: {type(e).__name__}: {e}",
            )

        # Check what the code produced
        plotly_fig = exec_globals.get("plotly_fig")
        mpl_fig = exec_globals.get("fig")

        if plotly_fig is not None:
            # Plotly path — save via existing helper
            return self._save_plotly_figure(
                plotly_fig,
                title,
                filename,
                "custom",
                len(df),
            )
        elif mpl_fig is not None:
            # Matplotlib path — save PNG
            if not filename.endswith(".png"):
                filename += ".png"
            charts_dir = os.path.join(self._results_path, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            save_path = os.path.join(charts_dir, filename)
            mpl_fig.savefig(save_path, dpi=cfg.chart_dpi, bbox_inches="tight")
            self._display_chart_inline(mpl_fig)
            plt.close(mpl_fig)

            self._charts_created.append(save_path)
            return ToolResult(
                success=True,
                data={
                    "chart_path": save_path,
                    "chart_type": "custom",
                    "title": title,
                    "rows_plotted": len(df),
                },
                summary=f"Custom chart saved: {save_path} ({len(df)} rows).",
            )
        else:
            return ToolResult(
                success=False,
                error=(
                    "render_code must assign either `fig` (matplotlib Figure) "
                    "or `plotly_fig` (plotly Figure). Neither was found after "
                    "executing the code."
                ),
            )

    def _render_sankey(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
    ) -> ToolResult:
        """Render a Sankey flow diagram using plotly.

        Expects columns: 'source', 'target', 'value', and optionally 'label'.
        """
        import plotly.graph_objects as go

        for col in ("source", "target", "value"):
            if col not in df.columns:
                return ToolResult(
                    success=False,
                    error=f"Sankey requires a '{col}' column. Columns: {list(df.columns)}",
                )

        # Build unique node labels
        all_labels = pd.concat([df["source"], df["target"]]).unique().tolist()
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        source_indices = [label_to_idx[s] for s in df["source"]]
        target_indices = [label_to_idx[t] for t in df["target"]]
        values = pd.to_numeric(df["value"], errors="coerce").fillna(0).tolist()

        # Optional custom node labels
        node_labels = all_labels
        if "label" in df.columns:
            label_map = {}
            for _, row in df.iterrows():
                label_map[row["source"]] = row.get("label", row["source"])
            node_labels = [label_map.get(n, n) for n in all_labels]

        # Color nodes by position (sources vs targets)
        source_set = set(df["source"].unique())
        target_set = set(df["target"].unique())
        node_colors = []
        for n in all_labels:
            if n in source_set and n not in target_set:
                node_colors.append("rgba(59, 130, 246, 0.8)")  # blue — pure source
            elif n in target_set and n not in source_set:
                node_colors.append("rgba(16, 185, 129, 0.8)")  # green — pure target
            else:
                node_colors.append("rgba(124, 58, 237, 0.8)")  # purple — both

        try:
            fig = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=node_labels,
                            color=node_colors,
                        ),
                        link=dict(
                            source=source_indices,
                            target=target_indices,
                            value=values,
                            color="rgba(148, 163, 184, 0.4)",
                        ),
                    )
                ]
            )
            fig.update_layout(
                title_text=title,
                title_font_size=16,
                font_size=12,
                margin=dict(l=20, r=20, t=60, b=20),
                width=1000,
                height=600,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Sankey rendering failed: {e}")

        return self._save_plotly_figure(fig, title, filename, "sankey", len(df))

    def _save_plotly_figure(
        self,
        fig,
        title: str,
        filename: str,
        chart_type: str,
        row_count: int,
    ) -> ToolResult:
        """Save a plotly figure as PNG (or HTML fallback) and display inline."""
        charts_dir = os.path.join(self._results_path, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Try PNG export via kaleido, fall back to HTML
        saved_as = "png"
        if not filename.endswith((".png", ".html")):
            filename += ".png"

        save_path = os.path.join(charts_dir, filename)

        try:
            fig.write_image(save_path, scale=2)
        except Exception:
            # kaleido not available — save as interactive HTML instead
            saved_as = "html"
            save_path = save_path.rsplit(".", 1)[0] + ".html"
            fig.write_html(save_path, include_plotlyjs="cdn")

        # Display inline in notebook — plotly HTML works natively in Databricks
        try:
            html_str = fig.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={"responsive": True},
            )
            self._display._raw_html(html_str)
        except Exception:
            # Fall back to PNG display if plotly HTML fails
            if saved_as == "png":
                self._display_chart_inline_from_path(save_path)

        self._charts_created.append(save_path)

        return ToolResult(
            success=True,
            data={
                "chart_path": save_path,
                "chart_type": chart_type,
                "format": saved_as,
                "title": title,
                "rows_plotted": row_count,
            },
            summary=f"Map saved: {save_path} ({chart_type}, {row_count} counties, {saved_as}).",
        )

    def _display_chart_inline_from_path(self, path: str) -> None:
        """Display a saved PNG file inline in the notebook."""
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            html_content = (
                f'<div style="margin:8px 0;text-align:center;">'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;border-radius:8px;border:1px solid #334155;" />'
                f"</div>"
            )
            self._display._display_html(html_content)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    def _render_table(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
    ) -> ToolResult:
        import html as html_mod

        # Save as CSV
        if not filename.endswith(".csv"):
            filename += ".csv"
        tables_dir = os.path.join(self._results_path, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        save_path = os.path.join(tables_dir, filename)
        df.to_csv(save_path, index=False)

        # Display as HTML table in notebook
        header_row = "".join(f"<th>{html_mod.escape(str(c))}</th>" for c in df.columns)
        body_rows = ""
        for _, row in df.head(50).iterrows():
            cells = "".join(f"<td>{html_mod.escape(str(v))}</td>" for v in row)
            body_rows += f"<tr>{cells}</tr>"

        truncation = ""
        if len(df) > 50:
            truncation = f'<div style="color:#64748b;font-size:11px;margin-top:4px;">Showing 50 of {len(df)} rows</div>'

        html_content = (
            f'<div style="margin:8px 0;">'
            f'<div style="font-weight:700;font-size:14px;color:#e2e8f0;margin-bottom:8px;">'
            f"{html_mod.escape(title)}</div>"
            f'<table style="border-collapse:collapse;width:100%;font-size:12px;color:#e2e8f0;">'
            f'<thead><tr style="background:#1e293b;border-bottom:2px solid #334155;">'
            f"{header_row}</tr></thead>"
            f"<tbody>{body_rows}</tbody></table>"
            f"{truncation}</div>"
        )
        self._display._display_html(html_content)

        return ToolResult(
            success=True,
            data={
                "table_path": save_path,
                "rows": len(df),
                "columns": list(df.columns),
                "title": title,
            },
            summary=f"Table saved: {save_path} ({len(df)} rows, {len(df.columns)} cols).",
        )

    # ------------------------------------------------------------------
    # Inline display helper
    # ------------------------------------------------------------------

    def _display_chart_inline(self, fig) -> None:
        """Display a matplotlib figure inline in the notebook."""
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            html_content = (
                f'<div style="margin:8px 0;text-align:center;">'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;border-radius:8px;border:1px solid #334155;" />'
                f"</div>"
            )
            self._display._display_html(html_content)
        except Exception:
            pass  # Silently fail if display isn't available

    @property
    def charts_created(self) -> list[str]:
        return self._charts_created
