# Agentic Data Analysis Platform

An autonomous, LLM-powered framework with three specialized AI agents that discover, engineer, validate, and analyze data in Databricks Unity Catalog. Fully configurable via `ProjectConfig` — swap the config to point at any data source, catalog, or domain.

---

## How It Works

Three agents collaborate across the data lifecycle:

1. **Data Engineer** — Discovers raw files, profiles them, designs schemas, transforms and loads into Delta tables. Handles archives, schema drift, encoding issues, and multi-year historical data autonomously.
2. **Data Analyst** — Validates every table the engineer builds. Checks schema quality, join key integrity, data completeness, cross-table joinability. Produces structured fix requests that drive an engineer-analyst feedback loop.
3. **Data Scientist** — Runs statistical analysis, fits models, reviews literature, and produces visualizations from the validated tables.

The system follows a **ReAct** pattern: Reason → Act (tool call) → Observe (result) → Reflect → next action.

---

## Project Structure

```
analysis/
├── config.py                               # ProjectConfig + all settings
├── README.md
│
├── core/                                   # Shared infrastructure
│   ├── llm_client.py                       # Claude API wrapper (prompt caching, retries)
│   ├── memory.py                           # Conversation history + context management
│   └── tools/
│       ├── base.py                         # BaseTool ABC + ToolResult
│       ├── registry.py                     # ToolRegistry dispatcher
│       ├── catalog_writer.py               # Delta table writer (3-tier: direct/parquet/flushed)
│       ├── dynamic_tool_builder.py         # Runtime tool creation with perf guardrails
│       ├── web_scraper.py                  # Web content extraction
│       └── column_renamer.py               # Bulk column renaming
│
├── engineer/                               # Data Engineer agent
│   ├── orchestrator.py                     # Main pipeline (discovery → processing → acceptance)
│   ├── analyst_orchestrator.py             # Data Analyst agent (acceptance testing)
│   ├── planner.py                          # Source discovery & prioritization
│   ├── prompts.py                          # Engineer system/task prompts
│   ├── analyst_prompts.py                  # Analyst system/task prompts
│   └── tools/
│       ├── volume_explorer.py              # Directory listing in Unity Volumes
│       ├── file_extractor.py               # Archive extraction (ZIP/GZ/TAR)
│       ├── file_reader.py                  # Tabular file header/sample reader
│       ├── doc_reader.py                   # Documentation text extraction
│       ├── data_profiler.py                # Column-level statistical profiling
│       ├── data_transformer.py             # Schema transform + batch mode + auto-flush
│       ├── schema_designer.py              # Schema validation & registration
│       └── web_search.py                   # External documentation fetcher
│
├── scientist/                              # Data Scientist agent
│   ├── orchestrator.py                     # Research analysis pipeline
│   ├── prompts.py                          # Scientist system/task prompts
│   └── tools/
│       ├── statistical_analysis.py         # Descriptive & inferential statistics
│       ├── model_fitting.py                # Regression, classification, clustering
│       ├── literature_review.py            # Research context from web sources
│       ├── create_visualization.py         # Charts and plots
│       └── save_finding.py                 # Persist research findings
│
├── models/                                 # Data classes
│   ├── source.py                           # FileInfo, FileGroup, DataSource
│   ├── schema.py                           # ColumnDefinition, TargetSchema
│   └── state.py                            # SourceStatus, SourceState, AgentState
│
├── utils/                                  # Shared utilities
│   ├── display.py                          # Notebook HTML display + human input
│   ├── fips.py                             # FIPS normalization (scalar + vectorized)
│   └── naming.py                           # snake_case column naming
│
└── notebooks/
    ├── run_data_profiler.py                # Engineer pipeline entry point
    └── run_research_analysis.py            # Scientist pipeline entry point
```

---

## Configuration

All project-specific settings live in `ProjectConfig`. The agents and tools are generic — swap the config for different data.

| Setting | Purpose | Default |
|---------|---------|---------|
| `catalog` / `schema` | Unity Catalog target | `versifai_intelhub_dev_v2.stars_analysis` |
| `volume_path` | Raw data file location | `/Volumes/.../external_data` |
| `join_key` | Column all tables must share | `county_fips_code` (STRING, 5-digit) |
| `geographic_grain` | Level of geographic analysis | `county` |
| `metadata_columns` | Auto-added to every table | `source_file_name`, `source_year`, `source_period_start`, `load_timestamp` |
| `known_sources` | Hints for source identification | CMS, CDC SVI, CDC PLACES, HRSA, USDA, Census |
| `documentation_urls` | Known doc URLs for web search | CMS, CDC, HRSA, USDA, Census links |

```python
from analysis.config import ProjectConfig, JoinKeyConfig

cfg = ProjectConfig(
    name="My Custom Analysis",
    catalog="my_catalog",
    schema="my_schema",
    volume_path="/Volumes/my_catalog/my_schema/raw_data",
    join_key=JoinKeyConfig(
        column_name="zip_code",
        data_type="STRING",
        width=5,
        description="5-digit ZIP code",
        expected_entity_count=42000,
    ),
    geographic_grain="zip code",
)
```

---

## Data Engineer Agent

**File:** `engineer/orchestrator.py` — Class: `DataEngineerAgent`

### Pipeline Phases

**Phase 1: Discovery** — Explores the volume, maps directories, identifies file types and archives, creates a processing plan.

**Phase 1.5: Incremental Detection** — Queries Unity Catalog for existing tables. Compares loaded files (via `source_file_name` metadata) against current directory contents. Routes each source to:
- **Full processing** — new source, no existing table
- **Incremental append** — existing table, new files detected
- **Skip** — existing table, no new files

Pass `force_full=True` to `run()` to bypass detection and reprocess everything.

**Phase 2: Source Processing** (per source) — For each source: read documentation, extract archives, profile a reference file, design the target schema, batch-transform all files, write to catalog.

**Phase 3: Acceptance Loop** — The Data Analyst agent reviews all tables via SQL queries. Issues get sent back to the engineer for fixes. Repeats up to 3 iterations until all tables are accepted.

### Batch Transform & Auto-Flush

For sources with many files (e.g., 45 monthly CSVs), `transform_and_load` supports **batch mode** — pass a `files` array to process all files in a single tool call instead of one-at-a-time.

When staged data exceeds **30M rows**, the tool **auto-flushes** to parquet on the staging volume and clears memory. The final `write_to_catalog` call creates the Delta table from all accumulated parquet batches.

### Write Pipeline

`write_to_catalog` uses a three-tier strategy based on data size:

| Data Size | Method | Path |
|-----------|--------|------|
| ≤ 2M rows | `spark.createDataFrame()` | Direct in-memory |
| > 2M rows (no flush) | Pandas → temp parquet → Spark SQL | Bypasses gRPC protobuf limits |
| Auto-flushed batches | Spark SQL reads parquet directory | `CREATE TABLE ... AS SELECT * FROM parquet.\`path\`` |

### Available Tools

`explore_volume`, `extract_archive`, `read_file_header`, `read_documentation`, `scan_for_documentation`, `profile_data`, `design_schema`, `transform_and_load`, `write_to_catalog`, `execute_sql`, `list_catalog_tables`, `web_search`, `create_custom_tool`, `ask_human`

---

## Data Analyst Agent

**File:** `engineer/analyst_orchestrator.py` — Class: `DataAnalystAgent`

Runs after the engineer finishes loading tables. Performs acceptance testing via SQL queries.

**Acceptance checks:** schema quality, join key integrity (exists, correct type, zero-padded, no nulls), volume and completeness, data quality (null rates, cardinality, ranges), cross-table joinability, metadata columns.

**Returns:** Structured verdict per table (`ACCEPTED`, `NEEDS_FIX`, `REJECTED`) with specific issues and fixes.

**Available Tools:** `execute_sql`, `list_catalog_tables`, `create_custom_tool`, `ask_human`

---

## Data Scientist Agent

**File:** `scientist/orchestrator.py` — Class: `DataScientistAgent`

Runs research analysis against validated bronze tables. Reads from catalog (bronze tables are read-only), creates silver tables for derived data.

**Available Tools:**
- `statistical_analysis` — Descriptive stats, correlations, distributions, hypothesis testing
- `model_fitting` — Regression, classification, clustering with automated feature selection
- `literature_review` — Web-based research context gathering
- `create_visualization` — Charts, plots, geographic maps
- `save_finding` — Persist research findings with evidence
- `execute_sql` — Read-only on bronze tables, read-write on `silver_*` tables
- `create_custom_tool` — Runtime tool creation for custom analysis

---

## Core Infrastructure

### LLM Client (`core/llm_client.py`)

Wrapper around the Anthropic Python SDK with:
- **Prompt caching** — system prompt and tool definitions marked with `cache_control` to avoid re-billing static tokens each turn
- **Retry logic** — exponential backoff on rate limits, connection errors, 5xx errors
- **Usage tracking** — input/output/cache-read/cache-creation token counts

### Memory & Context (`core/memory.py`)

Manages conversation history and prevents context overflow:
- **Auto-summarization** — at 30 messages, older messages are compressed. Tool-use/tool-result pairs are never split.
- **Tool result compression** — results older than 10 messages and larger than 500 chars are trimmed to 300 chars + summary section.
- **Per-source reset** — conversation cleared between sources, but decisions, summaries, and context notes carry forward.

### Dynamic Tool Builder (`core/tools/dynamic_tool_builder.py`)

Allows agents to create custom tools at runtime. Security guardrails block file I/O, shell commands, network access, and direct Spark/dbutils access. Includes enforced performance best practices:
- Vectorized pandas operations required (no `.apply()`, `.iterrows()`)
- `pad_fips_series()` available for FIPS padding
- `stage_dataframe()` for bridging to `write_to_catalog`

---

## Tool Reference

### Discovery & Profiling

| Tool | Description |
|------|-------------|
| `explore_volume` | Browse Unity Volume directories (recursive, max 3 levels) |
| `extract_archive` | Extract ZIP, GZ, TAR, TGZ archives |
| `scan_for_documentation` | Find docs (READMEs, data dictionaries, codebooks) sorted by relevance |
| `read_documentation` | Extract text from TXT, MD, HTML, PDF, CSV/Excel data dictionaries |
| `read_file_header` | Column names, dtypes, sample rows. Supports CSV, Excel, Parquet, SAS, Stata |
| `profile_data` | Column-level stats: nulls, cardinality, ranges, FIPS detection, top values |

### Schema & Transform

| Tool | Description |
|------|-------------|
| `design_schema` | Validate and register target schema. Enforces naming, types, join key, auto-adds metadata |
| `transform_and_load` | Transform files against schema. Supports single file or batch mode (`files` array). Auto-flushes at 30M rows |

### Catalog & SQL

| Tool | Description |
|------|-------------|
| `write_to_catalog` | Write staged data to Delta table. Three-tier write strategy by size. Supports `overwrite` and `append` |
| `execute_sql` | Execute SQL against Databricks (Spark or SDK fallback). Scientist agent gets silver-only write access |
| `list_catalog_tables` | List all tables in the target schema |

### Utility

| Tool | Description |
|------|-------------|
| `web_search` | Fetch external documentation from known URLs |
| `create_custom_tool` | Create tools at runtime with pandas. Performance guardrails enforced |
| `ask_human` | Pause and ask the operator a question for ambiguous decisions |

---

## Running the Pipeline

### Data Engineer (in Databricks)

```python
from analysis.config import ProjectConfig
from analysis.engineer.orchestrator import DataEngineerAgent

cfg = ProjectConfig()
agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)
results = agent.run()                    # Full run
results = agent.run(force_full=True)     # Force reprocess everything
```

### Data Scientist (in Databricks)

```python
from analysis.config import ProjectConfig
from analysis.scientist.orchestrator import DataScientistAgent

cfg = ProjectConfig()
agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)
results = agent.run()
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CLAUDE_TOKEN` | yes | Anthropic API key |
| `DATABRICKS_HOST` | no | Workspace URL (auto-detected in notebooks) |
| `DATABRICKS_TOKEN` | no | PAT token (auto-detected in notebooks) |

### Dependencies

```
anthropic>=0.40.0
python-dotenv>=1.0.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
pandas
pyarrow
openpyxl
```

PySpark is provided by the Databricks runtime.

### Safety Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Max agent turns (global) | 200 | Prevent infinite loops |
| Max turns per source | 120 | Allow batch processing of large file sets |
| Max acceptance iterations | 3 | Engineer-analyst feedback cycles |
| Max consecutive tool errors | 5 | Trigger error escalation |
| Memory summarization trigger | 30 messages | Keep context window manageable |
| Auto-flush threshold | 30M rows | Prevent OOM during staging |
| Direct write threshold | 2M rows | Above this, route through parquet to avoid gRPC limits |
| LLM retry attempts | 3 | Exponential backoff for API resilience |
