# Versifai Tool Inventory

Complete reference for every tool available to each agent. All tools extend `BaseTool` and return `ToolResult`.

**ToolResult fields:** `success` (bool), `data` (Any), `error` (str), `summary` (str), `image_path` (str)

---

## Tool Cross-Reference Matrix

Which tools are available to which agents:

| Tool Name | Data Engineer | Data Scientist | StoryTeller |
|-----------|:---:|:---:|:---:|
| `execute_sql` | Full | Silver-only | Silver-only |
| `list_catalog_tables` | Y | Y | Y |
| `web_search` | Y | Y | Y |
| `scrape_web` | - | Y | Y |
| `create_visualization` | - | Y | Y |
| `view_chart` | - | Y | Y |
| `save_note` | - | Y | Y |
| `create_custom_tool` | Y | Y | Y |
| `explore_volume` | Y | - | - |
| `extract_archive` | Y | - | - |
| `read_file_header` | Y | - | - |
| `read_documentation` | Y | - | - |
| `scan_for_documentation` | Y | - | - |
| `profile_data` | Y | - | - |
| `design_schema` | Y | - | - |
| `transform_and_load` | Y | - | - |
| `write_to_catalog` | Y | - | - |
| `rename_columns` | Y | - | - |
| `statistical_analysis` | - | Y | - |
| `fit_model` | - | Y | - |
| `check_confounders` | - | Y | - |
| `validate_silver` | - | Y | - |
| `validate_statistics` | - | Y | - |
| `review_literature` | - | Y | - |
| `save_finding` | - | Y | - |
| `log_model` | - | Y* | - |
| `read_findings` | - | - | Y |
| `read_chart` | - | - | Y |
| `read_table` | - | - | Y |
| `write_narrative` | - | - | Y |
| `evaluate_evidence` | - | - | Y |
| `cite_source` | - | - | Y |

\* `log_model` is conditionally registered only when `cfg.mlflow_experiment` is set.

---

## Shared Tools

These tools are registered by two or more agents.

---

### `execute_sql`

**Class:** `ExecuteSQLTool` / `SilverOnlyExecuteSQLTool` | **File:** `core/tools/catalog_writer.py`

Execute SQL queries against Unity Catalog. Data Engineer gets full write access; Data Scientist and StoryTeller get write-protected access (DDL/DML restricted to `silver_*` tables).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `sql` | string | Y | SQL query (DDL, DML, or SELECT) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `rows` | list[dict] | Query result rows (SELECT only, capped at 100) |
| `row_count` | int | Number of rows returned |
| `method` | string | Execution method (`spark` or `sdk`) |

**Notes:**
- SELECT results capped at 100 rows for agent display; use WHERE/LIMIT for large sets
- OOM detection with actionable error messages suggesting LIMIT, WHERE, or GROUP BY
- Spark execution with 10-minute timeout, SDK fallback with async polling
- SilverOnly variant returns `success=False` if DDL/DML targets non-silver tables

---

### `list_catalog_tables`

**Class:** `ListCatalogTablesTool` | **File:** `core/tools/catalog_writer.py`

List all tables in the configured Unity Catalog schema.

**Parameters:** None required.

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `tables` | list[str] | Table names in the schema |
| `count` | int | Number of tables |

---

### `web_search`

**Class:** `WebSearchTool` | **File:** `core/tools/web_search.py`

Search the web or fetch a specific URL for data documentation and metadata.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `query` | string | Y | Search query, matched against `ProjectConfig.documentation_urls` |
| `url` | string | - | Specific URL to fetch directly (bypasses query matching) |
| `max_chars` | int | - | Max characters to return (default 10000) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Fetched URL (direct mode) |
| `content` | string | Extracted text content (direct mode) |
| `results` | list[dict] | Search results with url/content (search mode) |
| `urls_checked` | list[str] | URLs that were fetched (search mode) |

**Notes:**
- Matches queries against `ProjectConfig.documentation_urls` for known data portals
- Falls back to DuckDuckGo search if no documentation URL match
- HTML text extraction via BeautifulSoup (falls back to regex strip)

---

### `scrape_web`

**Class:** `WebScraperTool` | **File:** `core/tools/web_scraper.py`

Advanced web scraping with JavaScript rendering. Three operations.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `discover_site`, `fetch_page`, or `extract_tables` |
| `url` | string | Y | Target URL |
| `section_filter` | string | - | Keyword filter for `discover_site` |

**Returns (by operation):**

| Operation | Key Fields |
|-----------|------------|
| `discover_site` | `base_url`, `total_pages`, `sections` (list of name/url/page_count) |
| `fetch_page` | `url`, `text`, `content_length`, `source` (direct/google_cache/playwright) |
| `extract_tables` | `url`, `tables` (list of headers/rows), `count` |

**Notes:**
- Playwright headless browser for JavaScript-rendered pages
- Falls back to Google cache when direct fetch fails
- Supports PDF extraction

---

### `create_visualization`

**Class:** `CreateVisualizationTool` | **File:** `core/tools/visualization.py`

Create publication-quality charts, maps, and result tables. 15 chart types including choropleth maps. Logs all metadata (SQL, data, render code) to theme notes for reproducibility.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `chart_type` | string | Y | `bar`, `scatter`, `heatmap`, `line`, `box`, `histogram`, `waterfall`, `dumbbell`, `lollipop`, `violin`, `choropleth`, `dual_choropleth`, `sankey`, `table`, `custom` |
| `title` | string | Y | Chart title |
| `filename` | string | Y | Output filename (e.g., `rq1_chart.png`) |
| `sql_query` | string | - | SQL to fetch data (preferred over `data` for large datasets) |
| `data` | list[dict] | - | Pre-computed data rows (fallback if SQL not possible) |
| `x_column` | string | - | X-axis column name |
| `y_column` | string | - | Y-axis column name |
| `color_column` | string | - | Color/grouping column |
| `theme_id` | string | - | Theme ID for notes logging |
| `interpretation` | string | - | 2-3 sentence chart interpretation (logged to notes) |
| `render_code` | string | - | Full Python program for `custom` chart type |
| `datasets` | dict | - | Additional named SQL data sources for `custom` charts |
| `fips_column` | string | - | FIPS code column (choropleth only, default `fips`) |
| `color_scale` | string | - | Color scale: `Viridis`, `RdYlGn`, `RdBu`, `Blues`, `Reds`, `YlOrRd`, `Plasma`, `Inferno` |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `chart_path` | string | Path to saved PNG/CSV |
| `chart_type` | string | Chart type used |
| `row_count` | int | Number of data rows rendered |

**Notes:**
- `sql_query` executes via Spark with no row cap (unlike execute_sql's 100-row display limit)
- Chart metadata (SQL, data, render code, interpretation) logged to theme notes file
- `custom` chart type: agent writes full Python; available variables include `df`, `pd`, `np`, `plt`, `sns`, `go`, `px`
- Saves to configured results volume path

---

### `view_chart`

**Class:** `ViewChartTool` | **File:** `core/tools/view_chart.py`

View a previously created chart or list all available charts and tables.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `filename` | string | - | Specific file to view. Omit to list all. |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `charts` | list[str] | Available PNG files (list mode) |
| `tables` | list[str] | Available CSV files (list mode) |
| `filename` | string | Viewed file name (view mode) |
| `path` | string | Full file path (view mode) |

**Notes:**
- PNG files displayed inline via base64 in Databricks notebooks
- CSV files returned as formatted text
- Sets `image_path` on ToolResult for automatic rendering

---

### `save_note`

**Class:** `SaveNoteTool` | **File:** `core/tools/save_note.py`

Save a research note to a per-theme notes file for reproducibility and audit.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `theme_id` | string | Y | Theme identifier (e.g., `theme_1`, `rq_1`) |
| `note` | string | Y | Note content (supports markdown) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `theme_id` | string | Theme ID |
| `file` | string | Path to notes file |
| `timestamp` | string | ISO timestamp |
| `char_count` | int | Note length |

**Notes:**
- Uses read-then-write pattern for Databricks FUSE compatibility (no append mode)
- One notes file per theme
- Timestamps included for audit trail

---

### `create_custom_tool`

**Class:** `DynamicToolBuilderTool` | **File:** `core/tools/dynamic_tool_builder.py`

Create a custom tool at runtime from agent-provided Python code. Registered immediately for use.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `tool_name` | string | Y | Name for the new tool |
| `tool_description` | string | Y | Description visible to agent |
| `parameters` | dict | Y | JSON Schema for the tool's parameters |
| `code` | string | Y | Python implementation (receives **kwargs, must return dict) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | string | Created tool name |
| `registered` | bool | Whether registration succeeded |
| `total_custom_tools` | int | Count of custom tools in session |

**Security guardrails -blocked operations:**
- `subprocess`, `os.system` -no shell commands
- `eval`, `exec` -no dynamic code execution
- `open`, file I/O -no direct file access
- `socket` -no network operations
- `spark`, `dbutils` -no direct Databricks access

**Allowed:** pandas, numpy, dict/list/string operations, math, `stage_dataframe()` bridge function.

---

## Data Engineer Agent Tools

**Agent:** `DataEngineerAgent` | **File:** `data_agents/engineer/agent.py`

**Total tools:** 14 + 1 pseudo-tool (`ask_human`) | **SQL access:** Full (read/write)

### Summary

| Tool | Purpose |
|------|---------|
| `explore_volume` | Browse Databricks Volume directories |
| `extract_archive` | Unpack ZIP/GZ/TAR archives |
| `read_file_header` | Preview file headers and sample rows |
| `read_documentation` | Read and classify documentation files |
| `scan_for_documentation` | Find documentation in a directory |
| `profile_data` | Profile column types, distributions, nulls |
| `design_schema` | Design Delta table schema from source columns |
| `transform_and_load` | Transform and stage data for catalog write |
| `write_to_catalog` | Write staged DataFrames to Unity Catalog |
| `rename_columns` | Rename columns in Delta tables |
| + 4 shared tools | `execute_sql`, `list_catalog_tables`, `web_search`, `create_custom_tool` |

---

### `explore_volume`

**Class:** `VolumeExplorerTool` | **File:** `data_agents/engineer/tools/volume_explorer.py`

List files and subdirectories in a Databricks Volume with metadata.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `path` | string | Y | Volume path (e.g., `/Volumes/catalog/schema/volume/data`) |
| `recursive` | bool | - | Recurse into subdirectories (default false) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Explored path |
| `entry_count` | int | Number of entries |
| `entries` | list[dict] | File/directory entries with `name`, `type`, `size_bytes`, `size_mb` |

---

### `extract_archive`

**Class:** `FileExtractorTool` | **File:** `data_agents/engineer/tools/file_extractor.py`

Extract compressed archive files.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `file_path` | string | Y | Path to archive file |
| `dest_path` | string | - | Destination directory (default: same as archive) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `source_archive` | string | Archive path |
| `destination` | string | Extraction directory |
| `file_count` | int | Number of extracted files |
| `files` | list[str] | Paths to extracted files |

**Supported formats:** ZIP (.zip), GZIP (.gz), TAR (.tar), TAR.GZ (.tgz, .tar.gz)

---

### `read_file_header`

**Class:** `FileReaderTool` | **File:** `data_agents/engineer/tools/file_reader.py`

Read file headers and sample rows from data files.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `file_path` | string | Y | Path to data file |
| `n_rows` | int | - | Sample rows to read (default 10) |
| `encoding` | string | - | Encoding override (auto-detected) |
| `separator` | string | - | CSV separator override (auto-detected) |
| `sheet_name` | string | - | Excel sheet name (default: first) |
| `skip_rows` | int | - | Rows to skip before reading (default 0) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `columns` | list[str] | Column names |
| `column_count` | int | Number of columns |
| `columns_detail` | list[dict] | Per-column dtype and sample values |
| `sample_data` | list[dict] | Sample rows |
| `estimated_total_rows` | int | Estimated row count |
| `file_size_mb` | float | File size |

**Supported formats:** CSV, TSV, Excel (.xls/.xlsx), Parquet, SAS (.sas7bdat), Stata (.dta)

---

### `read_documentation`

**Class:** `DocumentationReaderTool` | **File:** `data_agents/engineer/tools/doc_reader.py`

Read and classify a documentation file.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `file_path` | string | Y | Path to documentation file |
| `max_chars` | int | - | Max characters to return (default 15000) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | string | File path |
| `doc_type` | string | Detected type (e.g., `markdown`, `pdf`) |
| `classification` | string | `data_dictionary`, `readme`, `schema_documentation`, `api_reference`, `general_documentation` |
| `content` | string | Extracted text content |
| `truncated` | bool | Whether content was truncated |

**Supported formats:** TXT, MD, HTML, PDF, CSV, Excel

---

### `scan_for_documentation`

**Class:** `ScanForDocumentationTool` | **File:** `data_agents/engineer/tools/doc_reader.py`

Scan a directory for documentation files, prioritized by type.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `path` | string | Y | Directory path to scan |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `documentation_files` | list[dict] | Found files with `filename`, `path`, `priority`, `doc_type` |
| `count` | int | Number of files found |

**Priority order:** README > DICTIONARY/DATA_DICTIONARY > SCHEMA > other docs

---

### `profile_data`

**Class:** `DataProfilerTool` | **File:** `data_agents/engineer/tools/data_profiler.py`

Profile a data file's structure -column types, distributions, missing values, outliers.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `file_path` | string | Y | Path to data file |
| `sample_size` | int | - | Rows to sample (default 500) |
| `encoding` | string | - | Encoding override |
| `separator` | string | - | Separator override |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `rows_sampled` | int | Actual rows sampled |
| `column_count` | int | Number of columns |
| `columns` | list[str] | Column names |
| `column_profiles` | list[dict] | Per-column: dtype, null_count, unique_count, min/max/mean/median/std, sample_values |
| `potential_fips_columns` | list[str] | Columns detected as FIPS codes |
| `potential_geo_columns` | list[str] | Columns detected as geographic |
| `memory_usage_mb` | float | Estimated memory usage |

**Heuristics:**
- FIPS: columns with "fips" in name or 5-digit zero-padded patterns
- Geographic: columns named "state", "county", "city", "zip", etc.

---

### `design_schema`

**Class:** `SchemaDesignerTool` | **File:** `data_agents/engineer/tools/schema_designer.py`

Design a Delta table schema from source file columns. Auto-infers types, detects cryptic names, generates CREATE TABLE SQL.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `source_name` | string | Y | Source name key (must match a profiled file) |
| `table_name` | string | Y | Target table name (e.g., `silver_county_demographics`) |
| `description` | string | - | Table description |
| `column_names` | list[str] | - | Subset of columns to include (default: all) |
| `join_key_source_column` | string | - | Column to use as primary/join key |
| `type_overrides` | dict | - | Manual type overrides: `{col: data_type}` |
| `name_overrides` | dict | - | Manual name overrides: `{source: target}` |
| `columns` | list[dict] | - | Full column definitions with transform expressions |
| `partition_columns` | list[str] | - | Partition columns (e.g., `["year", "state"]`) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `schema` | dict | Full schema with column definitions |
| `create_table_sql` | string | Executable CREATE TABLE SQL |
| `warnings` | list[str] | Cryptic names, missing join keys, etc. |
| `column_count` | int | Number of columns in schema |
| `has_join_key` | bool | Whether a join key was identified |

**Auto-detection:** `pct_`/`rate_` -> DOUBLE; `id`/`key` -> STRING; date patterns -> DATE/TIMESTAMP

---

### `transform_and_load`

**Class:** `DataTransformerTool` | **File:** `data_agents/engineer/tools/data_transformer.py`

Transform source data per designed schema and stage for catalog write. Supports single-file and batch modes.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `source_name` | string | Y | Source key (must match a designed schema) |
| `file_path` | string | - | Source file (single-file mode) |
| `files` | list[dict] | - | Batch mode: list of `{file_path, encoding, separator}` |
| `source_year` | string | - | Year label (e.g., `2024`) |
| `column_overrides` | dict | - | Per-column transform overrides |
| `encoding` | string | - | Encoding override |
| `separator` | string | - | Separator override |
| `skip_rows` | int | - | Rows to skip |
| `sheet_name` | string | - | Excel sheet |
| `append` | bool | - | Append to staged data (default true) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `original_rows` | int | Source row count |
| `transformed_rows` | int | Rows after transform |
| `total_staged_rows` | int | Cumulative staged rows |
| `columns_mapped` | int | Columns successfully mapped |
| `auto_flush` | bool | Whether auto-flush triggered |
| `sample_data` | list[dict] | Sample transformed rows |

**Notes:**
- Auto-flush to parquet at 30M row threshold to prevent OOM
- Column mapping, type casting, null handling per schema
- Batch mode normalizes multiple files to single schema

---

### `write_to_catalog`

**Class:** `CatalogWriterTool` | **File:** `core/tools/catalog_writer.py`

Write staged DataFrames to Unity Catalog as managed Delta tables.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `source_name` | string | Y | Staged data key |
| `table_name` | string | Y | Target table (e.g., `silver_county_demographics`) |
| `mode` | string | - | `overwrite` (default) or `append` |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `table_name` | string | Full catalog.schema.table name |
| `rows_written` | int | Rows written |
| `verified_row_count` | int | Post-write verification count |
| `method` | string | `spark` or `sdk` |

**Notes:**
- Direct write up to 2M rows; stages to parquet above that threshold
- Post-write verification via COUNT(*) query
- Spark first, Databricks SDK fallback

---

### `rename_columns`

**Class:** `RenameColumnsTool` | **File:** `core/tools/column_renamer.py`

Rename columns in a Delta table via ALTER TABLE (metadata-only, no data rewrite).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `table_name` | string | Y | Table name (without catalog.schema prefix) |
| `column_renames` | dict | Y | `{old_name: new_name}` mapping |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `table_name` | string | Full table name |
| `succeeded` | int | Columns renamed |
| `failed` | list | Columns that failed |

**Notes:** Metadata-only operation via ALTER TABLE RENAME COLUMN -instant, no data rewrite.

---

## Data Scientist Agent Tools

**Agent:** `DataScientistAgent` | **File:** `science_agents/scientist/agent.py`

**Total tools:** 14-15 | **SQL access:** Silver-only (write-protected)

### Summary

| Tool | Purpose |
|------|---------|
| `statistical_analysis` | 8 analysis types: describe, distribution, hypothesis test, correlation, effect size, data quality, assumptions, Bayesian |
| `fit_model` | 9 model types: regression, classification, clustering, time series, counterfactual, Bayesian |
| `check_confounders` | Detect Simpson's Paradox and confounding variables |
| `validate_silver` | 6 data quality checks for silver-layer tables |
| `validate_statistics` | 4 checks: multiple comparisons, multicollinearity, ecological fallacy, robustness |
| `review_literature` | Search, fetch, and compare published research |
| `save_finding` | Save structured findings to JSON for StoryTeller |
| `log_model` | Log trained models to MLflow (conditional) |
| + 8 shared tools | `execute_sql`, `list_catalog_tables`, `web_search`, `scrape_web`, `create_visualization`, `view_chart`, `save_note`, `create_custom_tool` |

---

### `statistical_analysis`

**Class:** `StatisticalAnalysisTool` | **File:** `science_agents/scientist/tools/statistical_analysis.py`

Perform statistical analysis with 8 analysis types.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `analysis_type` | string | Y | `describe`, `distribution`, `hypothesis_test`, `correlation`, `effect_size`, `data_quality`, `assumption_check`, `bayesian_test` |
| `data` | list[dict] | Y | Data rows from SQL results |
| `columns` | list[str] | - | Columns to analyze (default: all numeric) |
| `group_column` | string | - | Grouping column (hypothesis_test, effect_size) |
| `value_column` | string | - | Value column (hypothesis_test, effect_size) |
| `method` | string | - | Specific method (see below) |
| `confidence_level` | float | - | Confidence level (default 0.95) |
| `prior` | dict | - | Bayesian prior: `{mean, std}`, `{alpha, beta}` |
| `rope` | list[float] | - | Region of Practical Equivalence `[low, high]` |

**Methods by analysis type:**

| Analysis Type | Available Methods |
|---------------|-------------------|
| `hypothesis_test` | `ttest_ind`, `ttest_rel`, `mannwhitney`, `chi_square`, `anova`, `kruskal` |
| `correlation` | `pearson`, `spearman` |
| `assumption_check` | `regression`, `ttest`, `anova`, `chi_square` |
| `bayesian_test` | `bayesian_ttest`, `bayesian_proportion`, `bayesian_correlation` |

**Returns (key fields by type):**

| Analysis Type | Key Return Fields |
|---------------|-------------------|
| `describe` | Per-column: `count`, `mean`, `median`, `std`, `min`, `max`, `q25`, `q75`, `skew`, `kurtosis` |
| `distribution` | Normality tests (Shapiro-Wilk, D'Agostino), distribution fitting |
| `hypothesis_test` | `test_statistic`, `p_value`, `result` (REJECT_NULL / FAIL_TO_REJECT) |
| `correlation` | Correlation matrix with p-values |
| `effect_size` | Cohen's d, rank-biserial, magnitude classification |
| `data_quality` | Missing rates, outlier detection, cardinality |
| `assumption_check` | Normality, homoscedasticity, independence checks |
| `bayesian_test` | `posterior_mean`, `credible_interval`, `bayes_factor`, `rope_analysis` |

---

### `fit_model`

**Class:** `ModelFittingTool` | **File:** `science_agents/scientist/tools/model_fitting.py`

Fit statistical and ML models with 9 model types.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `model_type` | string | Y | `linear_regression`, `logistic_regression`, `random_forest`, `gradient_boosting`, `kmeans`, `time_series`, `counterfactual`, `cross_validate`, `bayesian_regression` |
| `data` | list[dict] | Y | Data rows |
| `target_column` | string | - | Target variable |
| `feature_columns` | list[str] | - | Feature variables |
| `time_column` | string | - | Time column (time_series only) |
| `parameters` | dict | - | Model-specific params (e.g., `{n_clusters: 4}`, `{scenarios: [...]}`) |

**Returns (key fields by model type):**

| Model Type | Key Return Fields |
|------------|-------------------|
| `linear_regression` | `r_squared`, `coefficients`, `p_values`, `vif_scores`, `f_statistic` |
| `logistic_regression` | `accuracy`, `auc`, `precision`, `recall`, confusion matrix |
| `random_forest` | `accuracy`, `auc`, `feature_importance` |
| `gradient_boosting` | `accuracy`, `feature_importance` |
| `kmeans` | Silhouette scores, cluster profiles |
| `time_series` | Trend decomposition, change points, forecasts |
| `counterfactual` | `scenario`, `predicted_outcome`, `interpretation` |
| `cross_validate` | Multi-model comparison metrics |
| `bayesian_regression` | `coefficients` (mean/std), `credible_intervals`, `probability_direction` |

**Notes:**
- VIF (Variance Inflation Factor) computed automatically for linear regression to detect multicollinearity
- `counterfactual` enables what-if scenario analysis
- `bayesian_regression` supports informative priors from published research

---

### `check_confounders`

**Class:** `CheckConfoundersTool` | **File:** `science_agents/scientist/tools/check_confounders.py`

Detect Simpson's Paradox by decomposing aggregate relationships into subgroups.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `data` | list[dict] | Y | Data rows |
| `outcome_column` | string | Y | Outcome/dependent variable |
| `predictor_column` | string | Y | Predictor/independent variable |
| `grouping_columns` | list[str] | Y | Potential confounder columns to stratify by |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `aggregate` | dict | Overall `correlation`, `direction`, `strength` |
| `decompositions` | dict | Per-group correlations for each grouping column |
| `paradox_detected` | bool | Whether Simpson's Paradox found |
| `paradox_type` | string | `direction_reversal`, `strength_masking`, or `strength_reversal` |
| `strongest_confounder` | string | Most impactful grouping column |
| `recommendation` | string | Interpretation and guidance |

---

### `validate_silver`

**Class:** `ValidateSilverTool` | **File:** `science_agents/scientist/tools/validate_silver.py`

Validate silver-layer data quality with 6 check types.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `check_type` | string | Y | `grain`, `enrollment_sanity`, `year_alignment`, `join_completeness`, `value_ranges`, `zero_columns` |
| `data` | list[dict] | Y | Data rows |
| `primary_key_columns` | list[str] | - | PK columns (for `grain`) |
| `enrollment_columns` | list[str] | - | Enrollment columns (for `enrollment_sanity`) |
| `max_enrollment_per_row` | int | - | Max reasonable value (default 1M) |
| `year_column_left` | string | - | Left year column (for `year_alignment`) |
| `year_column_right` | string | - | Right year column |
| `expected_offset` | int | - | Expected year offset (default -1) |
| `left_count` | int | - | Left table rows (for `join_completeness`) |
| `matched_count` | int | - | Matched rows |
| `column_ranges` | dict | - | Expected ranges: `{col: {min, max}}` |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `issues_found` | bool | Whether any issues detected |
| `issue_count` | int | Number of issues |
| `details` | list[str] | Issue descriptions |
| `suggested_fix` | string | Recommended fix |

---

### `validate_statistics`

**Class:** `ValidateStatisticsTool` | **File:** `science_agents/scientist/tools/validate_statistics.py`

Validate statistical analyses for common pitfalls.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `check_type` | string | Y | `multiple_comparisons`, `multicollinearity`, `ecological_fallacy`, `robustness` |
| `data` | list[dict] | - | Data (for multicollinearity/robustness) |
| `p_values` | list[float] | - | P-values (for multiple_comparisons) |
| `alpha` | float | - | Significance threshold (default 0.05) |
| `feature_columns` | list[str] | - | Features (for multicollinearity) |
| `outcome_column` | string | - | Outcome (for robustness) |
| `predictor_column` | string | - | Predictor (for ecological_fallacy) |
| `data_fine_grain` | list[dict] | - | Individual-level data (for ecological_fallacy) |

**Returns (by check type):**

| Check Type | Key Return Fields |
|------------|-------------------|
| `multiple_comparisons` | `bonferroni_corrected`, `bh_corrected`, per-p-value significance |
| `multicollinearity` | `vif_scores`, `high_vif` (columns with VIF > 5) |
| `ecological_fallacy` | `aggregate_correlation`, `individual_correlation`, `risk` level |
| `robustness` | Sensitivity to outlier removal, stability assessment |

---

### `review_literature`

**Class:** `LiteratureReviewTool` | **File:** `science_agents/scientist/tools/literature_review.py`

Search for and compare published research.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `search`, `fetch_article`, `compare_findings` |
| `query` | string | - | Search query (for `search`) |
| `url` | string | - | Article URL (for `fetch_article`) |
| `own_finding` | string | - | Your finding (for `compare_findings`) |
| `published_finding` | string | - | Published finding to compare against |
| `source_title` | string | - | Published source title |
| `max_results` | int | - | Max search results (default 10) |

**Returns (by operation):**

| Operation | Key Return Fields |
|-----------|-------------------|
| `search` | `results` list with title, authors, year, url, abstract |
| `fetch_article` | `title`, `authors`, `abstract`, `full_text`, `key_findings` |
| `compare_findings` | `alignment` (ALIGNED/CONTRADICTS/NOVEL), `similarity_score`, `differences` |

---

### `save_finding`

**Class:** `SaveFindingTool` | **File:** `science_agents/scientist/tools/save_finding.py`

Save a structured research finding to findings JSON for the StoryTeller agent.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `research_question_id` | string | Y | RQ identifier (e.g., `rq_1`) |
| `title` | string | Y | Short finding title |
| `finding` | string | Y | Finding statement |
| `evidence` | string | Y | Supporting evidence (stats, p-values, N) |
| `significance` | string | Y | `high`, `medium`, or `low` |
| `visualization_path` | string | - | Path to associated chart |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `research_question_id` | string | RQ ID |
| `title` | string | Finding title |
| `timestamp` | string | ISO timestamp |
| `index` | int | Finding index in JSON file |

---

### `log_model`

**Class:** `LogModelTool` | **File:** `science_agents/scientist/tools/log_model.py`

Log a trained model to MLflow with metrics and optional Unity Catalog registration.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `model_type` | string | Y | Model type (e.g., `linear_regression`) |
| `data` | list[dict] | Y | Training data |
| `target_column` | string | Y | Target column |
| `feature_columns` | list[str] | Y | Feature columns |
| `metrics` | dict | Y | Metrics to log (e.g., `{r_squared: 0.85}`) |
| `model_name` | string | Y | Registered model name |
| `parameters` | dict | - | Hyperparameters |
| `tags` | dict | - | Custom MLflow tags |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `mlflow_run_id` | string | MLflow run ID |
| `model_uri` | string | Model URI for loading |
| `registered_model` | string | Registered model name |
| `metrics_logged` | dict | Logged metrics |
| `top_features` | list[str] | Top features by importance |

**Notes:**
- Only registered when `cfg.mlflow_experiment` is configured
- Optional Unity Catalog model registration for production deployment

---

## StoryTeller Agent Tools

**Agent:** `StoryTellerAgent` | **File:** `story_agents/storyteller/agent.py`

**Total tools:** 14 | **SQL access:** Silver-only (SELECT in practice)

### Summary

| Tool | Purpose |
|------|---------|
| `read_findings` | Read findings saved by Data Scientist |
| `read_chart` | Read chart metadata from results directory |
| `read_table` | Read CSV result tables |
| `write_narrative` | Write, read, and assemble narrative sections |
| `evaluate_evidence` | Score evidence strength, curate findings |
| `cite_source` | Manage citations and references |
| + 8 shared tools | `execute_sql`, `list_catalog_tables`, `web_search`, `scrape_web`, `create_visualization`, `view_chart`, `save_note`, `create_custom_tool` |

---

### `read_findings`

**Class:** `ReadFindingsTool` | **File:** `story_agents/storyteller/tools/read_findings.py`

Read research findings saved by the Data Scientist agent. 5 operations.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `list`, `get`, `by_theme`, `high_significance`, `search` |
| `index` | int | - | Finding index (for `get`) |
| `theme_id` | string | - | Theme/RQ ID (for `by_theme`) |
| `query` | string | - | Search query (for `search`) |

**Returns (by operation):**

| Operation | Key Return Fields |
|-----------|-------------------|
| `list` | `findings` list (index, title, significance), `total_count` |
| `get` | Full finding: title, finding, evidence, significance, visualization_path |
| `by_theme` | Filtered findings for theme |
| `high_significance` | Only high/medium significance findings |
| `search` | Keyword-matched findings with relevance scores |

---

### `read_chart`

**Class:** `ReadChartTool` | **File:** `story_agents/storyteller/tools/read_chart.py`

Read chart metadata from the results directory.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `list`, `by_theme`, `metadata` |
| `theme_id` | string | - | Theme ID (for `by_theme`) |
| `chart_filename` | string | - | Chart filename (for `metadata`) |

**Returns (by operation):**

| Operation | Key Return Fields |
|-----------|-------------------|
| `list` | `charts` list (filename, theme_id, title), `total_count` |
| `by_theme` | Charts filtered by theme |
| `metadata` | Full metadata: chart_type, x/y columns, row_count, interpretation |

---

### `read_table`

**Class:** `ReadTableTool` | **File:** `story_agents/storyteller/tools/read_table.py`

Read CSV result tables from the results directory.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `list`, `read`, `summary` |
| `filename` | string | - | Table filename (for `read`/`summary`) |
| `max_rows` | int | - | Max rows to return (default 100) |

**Returns (by operation):**

| Operation | Key Return Fields |
|-----------|-------------------|
| `list` | `tables` list (filename, row_count, column_count), `total_count` |
| `read` | `columns`, `rows` (as list[dict]), `row_count` |
| `summary` | Per-column statistics: mean, std, min, max |

---

### `write_narrative`

**Class:** `WriteNarrativeTool` | **File:** `story_agents/storyteller/tools/write_narrative.py`

Write, read, update, and assemble narrative sections for the final report. 5 operations.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `write_section`, `read_section`, `update_section`, `list_sections`, `assemble` |
| `section_id` | string | - | Section identifier (e.g., `introduction`, `rq_1_findings`) |
| `title` | string | - | Section title (for `write_section`) |
| `content` | string | - | Markdown content (for `write_section`/`update_section`) |
| `sequence` | int | - | Order in final document (for `write_section`) |

**Returns (by operation):**

| Operation | Key Return Fields |
|-----------|-------------------|
| `write_section` | `section_id`, `sequence`, `created` |
| `read_section` | `title`, `content`, `sequence` |
| `list_sections` | `sections` list (id, title, sequence), `total_count` |
| `assemble` | `document_path`, `table_of_contents`, `section_count`, `word_count` |

**Notes:**
- `assemble` generates the final markdown report with auto-generated table of contents
- Sections ordered by `sequence` number in final document

---

### `evaluate_evidence`

**Class:** `EvaluateEvidenceTool` | **File:** `story_agents/storyteller/tools/evaluate_evidence.py`

Score evidence strength and curate findings for narrative sections.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `evaluate` or `curate` |
| `finding` | dict | - | Single finding (for `evaluate`) |
| `findings` | list[dict] | - | Multiple findings (for `curate`) |
| `purpose` | string | - | Section purpose for curation (e.g., `lead finding for introduction`) |
| `max_findings` | int | - | Max curated findings (default 5) |

**Returns (by operation):**

| Operation | Key Return Fields |
|-----------|-------------------|
| `evaluate` | `tier`, `tier_description`, `usable_as_lead`, `usable_as_support`, `effect_size`, `p_value` |
| `curate` | `curated` list (ranked by tier), `lead_candidates`, `support_candidates` |

**Evidence Tier Classification:**

| Tier | Criteria | Suitable For |
|------|----------|--------------|
| **DEFINITIVE** | p < 0.001, large effect size | Primary conclusions |
| **STRONG** | p < 0.01, medium+ effect | Leading paragraphs |
| **SUGGESTIVE** | p < 0.05 | Supporting evidence |
| **CONTEXTUAL** | Descriptive, no test | Background context |
| **WEAK** | p >= 0.05, negligible effect | Limitations |

---

### `cite_source`

**Class:** `CiteSourceTool` | **File:** `story_agents/storyteller/tools/cite_source.py`

Manage citations and references for the narrative report.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `operation` | string | Y | `add`, `list`, `format`, `search` |
| `title` | string | - | Source title (for `add`) |
| `url` | string | - | Source URL (for `add`) |
| `author` | string | - | Author (for `add`) |
| `year` | string | - | Publication year (for `add`) |
| `description` | string | - | Brief description (for `add`) |
| `cite_key` | string | - | Citation key (for `format`) |
| `query` | string | - | Search query (for `search`) |

**Returns (by operation):**

| Operation | Key Return Fields |
|-----------|-------------------|
| `add` | `cite_key` (auto-generated), `added` |
| `list` | `sources` list (cite_key, title, author, year), `total_count` |
| `format` | `citation_formats` with apa, chicago, inline |
| `search` | `results` with relevance scores |
