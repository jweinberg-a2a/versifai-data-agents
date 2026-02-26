# Data Engineer Agent

The `DataEngineerAgent` discovers raw files, profiles data, designs schemas, and transforms and loads data into Databricks Unity Catalog Delta tables. It handles archives, schema drift, encoding issues, and multi-year historical data autonomously.

The `DataAnalystAgent` works in tandem -it validates every table the engineer builds, checking schema quality, join key integrity, and cross-table joinability.

## Pipeline Phases

### Phase 1: Discovery

Explores the volume, maps directories, identifies file types and archives, creates a processing plan.

### Phase 1.5: Incremental Detection

Queries Unity Catalog for existing tables. Compares loaded files (via `source_file_name` metadata) against current directory contents. Routes each source to:

- **Full processing** -new source, no existing table
- **Incremental append** -existing table, new files detected
- **Skip** -existing table, no new files

Pass `force_full=True` to `run()` to bypass detection and reprocess everything.

### Phase 2: Source Processing (per source)

For each source: read documentation, extract archives, profile a reference file, design the target schema, batch-transform all files, write to catalog.

### Phase 3: Acceptance Loop

The Data Analyst agent reviews all tables via SQL queries. Issues get sent back to the engineer for fixes. Repeats up to 3 iterations until all tables are accepted.

## Batch Transform & Auto-Flush

For sources with many files (e.g., 45 monthly CSVs), `transform_and_load` supports **batch mode** -pass a `files` array to process all files in a single tool call instead of one-at-a-time.

When staged data exceeds **30M rows**, the tool **auto-flushes** to parquet on the staging volume and clears memory. The final `write_to_catalog` call creates the Delta table from all accumulated parquet batches.

## Write Pipeline

`write_to_catalog` uses a three-tier strategy based on data size:

| Data Size | Method | Path |
|-----------|--------|------|
| ≤ 2M rows | `spark.createDataFrame()` | Direct in-memory |
| > 2M rows (no flush) | Pandas → temp parquet → Spark SQL | Bypasses gRPC protobuf limits |
| Auto-flushed batches | Spark SQL reads parquet directory | `CREATE TABLE ... AS SELECT * FROM parquet.path` |

## Available Tools

| Tool | Description |
|------|-------------|
| `explore_volume` | Browse Unity Volume directories (recursive, max 3 levels) |
| `extract_archive` | Extract ZIP, GZ, TAR, TGZ archives |
| `scan_for_documentation` | Find docs (READMEs, data dictionaries, codebooks) |
| `read_documentation` | Extract text from TXT, MD, HTML, PDF, CSV/Excel data dictionaries |
| `read_file_header` | Column names, dtypes, sample rows (CSV, Excel, Parquet, SAS, Stata) |
| `profile_data` | Column-level stats: nulls, cardinality, ranges, FIPS detection |
| `design_schema` | Validate and register target schema |
| `transform_and_load` | Transform files against schema with batch mode and auto-flush |
| `write_to_catalog` | Write staged data to Delta table |
| `execute_sql` | Execute SQL against Databricks |
| `list_catalog_tables` | List all tables in the target schema |
| `web_search` | Fetch external documentation |
| `create_custom_tool` | Create tools at runtime with pandas |
| `ask_human` | Pause and ask the operator a question |

See the [Tool Reference](tool-inventory.md) for detailed parameter and return value tables.

## Usage

```python
from versifai.data_agents import DataEngineerAgent, ProjectConfig

cfg = ProjectConfig(
    name="Sales Pipeline",
    catalog="analytics",
    schema="sales",
    volume_path="/Volumes/analytics/sales/raw_data",
    # Domain-specific guidance (optional, but recommended)
    grain_detection_guidance=(
        "Account-level: Look for account_id columns\n"
        "Transaction-level: Look for transaction_id or order_id"
    ),
    column_naming_examples=(
        "`AcctID` → `account_id`\n"
        "`TxnAmt` → `transaction_amount_usd`"
    ),
)

agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)

# Full run
result = agent.run()

# Force reprocess everything
result = agent.run(force_full=True)
```

### Domain Guidance Fields

The agent prompts are **domain-agnostic**. Use these config fields to inject domain knowledge:

| Field | Purpose | Default |
|-------|---------|---------|
| `grain_detection_guidance` | How to detect data grain (account? county? device?) | `""` (generic fallback) |
| `column_naming_examples` | Domain-specific column rename patterns | `""` (generic fallback) |

When these fields are empty, the agent uses generic grain detection and naming heuristics.

## Data Analyst (Acceptance Testing)

The `DataAnalystAgent` runs after the engineer finishes loading tables. It performs acceptance testing via SQL queries.

**Checks:** schema quality, join key integrity (exists, correct type, zero-padded, no nulls), volume and completeness, data quality (null rates, cardinality, ranges), cross-table joinability, metadata columns.

**Returns:** Structured verdict per table (`ACCEPTED`, `NEEDS_FIX`, `REJECTED`) with specific issues and fixes.

**Available Tools:** `execute_sql`, `list_catalog_tables`, `create_custom_tool`, `ask_human`
