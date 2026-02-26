# Architecture

Versifai is an autonomous, LLM-powered framework with three specialized AI agents that discover, engineer, validate, and analyze data on Databricks Unity Catalog — then write a narrative report from the results.

This page walks through the architecture from the top down: the overall pipeline, the core abstractions, how tools work, and finally a complete tool-level walkthrough of a real run.

---

## The Pipeline

Three agents run in sequence. Each one reads from what the previous agent produced.

```mermaid
flowchart LR
    subgraph inputs["Your Data"]
        RAW[/"CSVs, Excel, ZIP,<br>Parquet, SAS"/]
    end

    subgraph engineer["Stage 1"]
        DE["**Data Engineer**<br><br>Discovers files,<br>profiles columns,<br>designs schemas,<br>transforms & loads"]
    end

    subgraph catalog["Unity Catalog"]
        DT[("Delta Tables<br><br>silver_weather<br>silver_quacks<br>silver_ice_cream")]
    end

    subgraph scientist["Stage 2"]
        DS["**Data Scientist**<br><br>Builds silver datasets,<br>runs statistics,<br>fits models,<br>saves findings"]
    end

    subgraph outputs["Research Outputs"]
        FO[/"findings.json<br>charts/ (PNG)<br>tables/ (CSV)<br>notes/ (MD)"/]
    end

    subgraph storyteller["Stage 3"]
        ST["**StoryTeller**<br><br>Reads findings,<br>evaluates evidence,<br>writes narrative,<br>cites sources"]
    end

    subgraph report["Final Report"]
        RPT[/"report.md<br><br>Markdown with TOC,<br>citations, chart refs,<br>methodology appendix"/]
    end

    RAW --> DE
    DE --> DT
    DT --> DS
    DS --> FO
    FO --> ST
    ST --> RPT
```

Each stage has exactly three moving parts:

| Part | What It Is | What Changes Between Projects |
|------|-----------|-------------------------------|
| **Config** | A Python dataclass holding all domain knowledge | Everything — this is where your project lives |
| **Agent** | A generic Python class that reads the config and does work | Nothing — agents are reusable across projects |
| **Notebook** | A Databricks notebook that creates the agent and runs it | Just the import path to your config |

The agents are generic. **All domain-specific knowledge lives in configs.** You never modify agent code — you write new configs.

---

## Core Abstractions

Before diving into how each agent works, here are the building blocks everything is built on.

### The ReAct Loop

Every agent runs using the same execution pattern: **Reason → Act → Observe → Repeat.**

```mermaid
flowchart TD
    START([Agent receives prompt]) --> REASON["**Reason**<br>LLM reads context and<br>decides what to do next"]
    REASON --> TOOL{"Tool call?"}
    TOOL -->|"Yes"| ACT["**Act**<br>ToolRegistry executes<br>the requested tool"]
    ACT --> OBSERVE["**Observe**<br>Tool returns ToolResult<br>stored in AgentMemory"]
    OBSERVE --> REASON

    TOOL -->|"No — done"| END([Agent returns final answer])

    style REASON fill:#e8f0fe,stroke:#4a6f93
    style ACT fill:#e8f4e8,stroke:#4a8a4a
    style OBSERVE fill:#fef3e0,stroke:#b38600
```

The agent never executes arbitrary code. It reasons about what to do, calls a tool, observes the result, and reasons again. This loop continues until the agent decides it's done or hits a turn limit.

### Tools — The Unit of Capability

Tools are how agents interact with the world. Every tool follows the same contract:

```mermaid
flowchart LR
    LLM["LLM Agent<br><br>*I need to profile<br>this CSV file*"] -->|"tool_use block<br>name + parameters"| REG["ToolRegistry<br><br>Looks up tool by name,<br>validates params,<br>calls _execute()"]
    REG -->|"calls"| TOOL["BaseTool<br><br>profile_data._execute(<br>  file_path='weather.csv',<br>  sample_size=500<br>)"]
    TOOL -->|"returns"| TR["ToolResult<br><br>success: true<br>data: {columns, stats}<br>summary: '12 cols profiled'"]
    TR -->|"stored in"| MEM["AgentMemory<br><br>Conversation history<br>for next reasoning step"]
```

Every tool:

- Has a **name**, **description**, and **parameter schema** (JSON Schema)
- Implements `_execute()` which does the work and returns a `ToolResult`
- Is registered in a `ToolRegistry` at agent construction time
- Can be tested in isolation — no LLM needed for unit tests

### How Tools Are Registered

At construction time, each agent builds a `ToolRegistry` with the tools it needs. The registry handles dispatch and generates the tool definitions the LLM sees.

```python
from versifai.core.tools.registry import ToolRegistry

# Each agent builds its own registry
registry = ToolRegistry()
registry.register(VolumeExplorerTool(cfg=cfg))
registry.register(DataProfilerTool(cfg=cfg))
registry.register(SchemaDesignerTool(cfg=cfg))
# ... more tools

# The registry generates the schema the LLM sees
tool_definitions = registry.to_claude_tools()

# When the LLM calls a tool, the registry dispatches it
result = registry.execute(tool_name="profile_data", tool_input={...})
```

### ToolResult — The Standard Return Type

Every tool returns a `ToolResult`. This is the only way tools communicate back to the agent:

| Field | Type | Purpose |
|-------|------|---------|
| `success` | `bool` | Did the operation complete? |
| `data` | `Any` | Result payload (dict, list, string) |
| `error` | `str` | Error message if `success=False` |
| `summary` | `str` | Human-readable summary for the agent |
| `image_path` | `str` | Path to PNG for inline display |

Tools never raise exceptions. They always return a `ToolResult`, even on failure. This keeps the ReAct loop stable — the agent sees the error and can reason about what to do next.

### AgentMemory — Context Management

The `AgentMemory` class manages conversation history and prevents context overflow:

- **Auto-summarization** — at 30 messages, older messages are compressed
- **Tool result trimming** — large results older than 10 messages are truncated to 300 chars
- **Per-source reset** — history clears between sources, but decisions carry forward

---

## Tool Inventory by Agent

Each agent has a specialized toolkit. Some tools are shared across agents.

```mermaid
flowchart TD
    subgraph shared["Shared Tools"]
        direction LR
        SQL["execute_sql"]
        LIST["list_catalog_tables"]
        WEB["web_search"]
        CUSTOM["create_custom_tool"]
    end

    subgraph shared2["Shared — Scientist & StoryTeller"]
        direction LR
        VIZ["create_visualization"]
        VIEW["view_chart"]
        NOTE["save_note"]
        SCRAPE["scrape_web"]
    end

    subgraph eng_tools["Data Engineer Tools"]
        direction LR
        EV["explore_volume"]
        EA["extract_archive"]
        RFH["read_file_header"]
        RD["read_documentation"]
        SFD["scan_for_documentation"]
        PD["profile_data"]
        DS["design_schema"]
        TL["transform_and_load"]
        WTC["write_to_catalog"]
        RC["rename_columns"]
    end

    subgraph sci_tools["Data Scientist Tools"]
        direction LR
        STAT["statistical_analysis"]
        FIT["fit_model"]
        CONF["check_confounders"]
        VS["validate_silver"]
        VSTAT["validate_statistics"]
        LIT["review_literature"]
        SF["save_finding"]
    end

    subgraph story_tools["StoryTeller Tools"]
        direction LR
        RF["read_findings"]
        RCHART["read_chart"]
        RT["read_table"]
        WN["write_narrative"]
        EE["evaluate_evidence"]
        CS["cite_source"]
    end

    ENG["**Data Engineer**<br>14 tools"] --- eng_tools
    ENG --- shared
    SCI["**Data Scientist**<br>15 tools"] --- sci_tools
    SCI --- shared
    SCI --- shared2
    STORY["**StoryTeller**<br>14 tools"] --- story_tools
    STORY --- shared
    STORY --- shared2

    style shared fill:#f0f0f0,stroke:#999
    style shared2 fill:#f0f0f0,stroke:#999
    style eng_tools fill:#e8f0fe,stroke:#4a6f93
    style sci_tools fill:#e8f4e8,stroke:#4a8a4a
    style story_tools fill:#fef3e0,stroke:#b38600
```

!!! note "SQL Write Protection"
    The Data Engineer gets full SQL access (`ExecuteSQLTool`). The Data Scientist and StoryTeller get a write-protected variant (`SilverOnlyExecuteSQLTool`) that blocks DDL/DML on anything except `silver_*` tables. SELECT queries are unrestricted for everyone.

For the complete parameter and return type reference for every tool, see the [Tool Inventory](tool-inventory.md).

---

## Data Engineer Agent — Deep Dive

The Data Engineer is the first agent in the pipeline. It takes a directory of raw files and turns them into clean, validated Delta tables in Unity Catalog.

### Phases

```mermaid
flowchart LR
    P1["**Phase 1**<br>Discovery<br><br>Explore volume,<br>map directories,<br>find documentation"]
    P2["**Phase 2**<br>Processing<br><br>Per source: profile,<br>design schema,<br>transform, load"]
    P3["**Phase 3**<br>Acceptance<br><br>Analyst validates<br>via SQL queries,<br>engineer fixes issues"]

    P1 --> P2
    P2 --> P3

    style P1 fill:#e8f0fe,stroke:#4a6f93
    style P2 fill:#dce8f5,stroke:#4a6f93
    style P3 fill:#d0e0f0,stroke:#4a6f93
```

### Tool-Level Walkthrough

Imagine a Volume containing weather CSVs, a zipped duck observation archive, and an Excel file of ice cream sales. Here's exactly what the agent does:

```mermaid
flowchart TD
    subgraph phase1["Phase 1: Discovery"]
        EV["**explore_volume**<br><br>Scans /Volumes/.../raw_data/<br>Finds 4 subdirectories,<br>12 files across formats"]
        EV --> SFD["**scan_for_documentation**<br><br>Finds README.md,<br>data_dictionary.csv<br>in noaa_weather/"]
        SFD --> RD["**read_documentation**<br><br>Extracts field definitions:<br>TMAX = max temp (tenths °C)<br>PRCP = precipitation (tenths mm)"]
    end

    subgraph phase2_src1["Phase 2: Source 1 — NOAA Weather (3 CSVs)"]
        RFH1["**read_file_header**<br><br>Reads weather_2020.csv<br>→ 14 columns, 365K rows<br>→ STATION, DATE, TMAX, PRCP..."]
        RFH1 --> PD1["**profile_data**<br><br>Null rates, value ranges,<br>detects STATION as join key,<br>DATE as date column"]
        PD1 --> DS1["**design_schema**<br><br>Generates CREATE TABLE SQL:<br>silver_daily_weather<br>→ station_id STRING,<br>→ observation_date DATE,<br>→ temp_max_c DOUBLE..."]
        DS1 --> TL1["**transform_and_load**<br><br>Batch mode: processes all 3 CSVs<br>Renames columns, casts types,<br>converts tenths→degrees,<br>adds metadata columns"]
        TL1 --> WTC1["**write_to_catalog**<br><br>→ catalog.schema.silver_daily_weather<br>→ 1.1M rows written<br>→ Verified via COUNT(*)"]
    end

    subgraph phase2_src2["Phase 2: Source 2 — Duck Observations (ZIP)"]
        EA["**extract_archive**<br><br>Unpacks duck_obs.zip<br>→ quack_frequency.csv<br>→ feather_index.csv"]
        EA --> RFH2["**read_file_header**<br><br>Peeks at each extracted CSV"]
        RFH2 --> PD2["**profile_data**<br><br>Profiles both files independently"]
        PD2 --> DS2["**design_schema** (×2)<br><br>Designs separate schemas:<br>silver_quack_frequency<br>silver_feather_fluffing"]
        DS2 --> TL2["**transform_and_load** (×2)"]
        TL2 --> WTC2["**write_to_catalog** (×2)<br><br>→ 2 Delta tables created"]
    end

    subgraph phase2_src3["Phase 2: Source 3 — Ice Cream (Excel)"]
        RFH3["**read_file_header**<br><br>Reads ice_cream_sales.xlsx<br>sheet_name='Monthly Sales'"]
        RFH3 --> PD3["**profile_data**"]
        PD3 --> DS3["**design_schema**<br><br>→ silver_ice_cream_sales"]
        DS3 --> TL3["**transform_and_load**"]
        TL3 --> WTC3["**write_to_catalog**"]
    end

    subgraph phase3["Phase 3: Acceptance Loop"]
        ANALYST["**Data Analyst Agent**<br><br>execute_sql: checks each table<br>→ Schema quality<br>→ Join key integrity<br>→ Null rates & ranges<br>→ Cross-table joinability"]
        ANALYST --> V{All accepted?}
        V -->|"Yes"| DONE["All 4 tables validated"]
        V -->|"NEEDS_FIX"| FIX["Engineer receives fix list<br>e.g. 'station_id has nulls<br>in silver_quack_frequency'"]
        FIX --> FIXACT["Engineer calls<br>execute_sql or<br>transform_and_load<br>to fix issues"]
        FIXACT --> ANALYST
    end

    phase1 --> phase2_src1
    phase1 --> phase2_src2
    phase1 --> phase2_src3
    phase2_src1 --> phase3
    phase2_src2 --> phase3
    phase2_src3 --> phase3

    style phase1 fill:#f7f9fc,stroke:#4a6f93
    style phase2_src1 fill:#f7f9fc,stroke:#4a6f93
    style phase2_src2 fill:#f7f9fc,stroke:#4a6f93
    style phase2_src3 fill:#f7f9fc,stroke:#4a6f93
    style phase3 fill:#f7f9fc,stroke:#4a6f93
```

### Key Behaviors

**Smart resume** — If the notebook crashes after loading 2 of 4 sources, re-running skips the completed sources. The agent queries Unity Catalog for existing tables and compares loaded files (via `source_file_name` metadata) against current directory contents. See [Run Management & Reproducibility](run-management.md) for the full resume system.

**Batch transform** — For sources with many files (e.g., 45 monthly CSVs), `transform_and_load` supports batch mode — pass a `files` array to process everything in one tool call.

**Auto-flush** — When staged data exceeds 30M rows, the tool auto-flushes to parquet on the staging volume and clears memory. The final `write_to_catalog` creates the Delta table from all accumulated parquet batches.

**Three-tier write strategy:**

| Data Size | Method | Why |
|-----------|--------|-----|
| ≤ 2M rows | `spark.createDataFrame()` | Fast, in-memory |
| > 2M rows | Pandas → temp parquet → Spark SQL | Avoids gRPC protobuf limits |
| Auto-flushed | Spark reads parquet directory | Already on disk |

---

## Data Scientist Agent — Deep Dive

The Data Scientist reads from the Delta tables the engineer created, builds analytical datasets, runs statistics, fits models, and saves structured findings.

### Phases

```mermaid
flowchart LR
    P1["**Phase 1**<br>Orientation<br><br>Inventory tables,<br>assess data quality,<br>plan analysis"]
    P2["**Phase 2**<br>Silver Construction<br><br>Join source tables<br>into analytical<br>datasets"]
    P3["**Phase 3**<br>Theme Analysis<br><br>Run each research<br>theme: stats, models,<br>charts, findings"]
    P4["**Phase 4**<br>Synthesis<br><br>Cross-validate,<br>compile summary,<br>flag gaps"]

    P1 --> P2
    P2 --> P3
    P3 --> P4

    style P1 fill:#e8f4e8,stroke:#4a8a4a
    style P2 fill:#dcefd8,stroke:#4a8a4a
    style P3 fill:#d0ead0,stroke:#4a8a4a
    style P4 fill:#c4e5c4,stroke:#4a8a4a
```

### Tool-Level Walkthrough

Here's what the agent does for a single research theme — "Does quack frequency correlate with next-day rain?"

```mermaid
flowchart TD
    subgraph orient["Phase 1: Orientation"]
        LIST["**list_catalog_tables**<br><br>Finds 4 tables:<br>silver_daily_weather<br>silver_quack_frequency<br>silver_feather_fluffing<br>silver_ice_cream_sales"]
        LIST --> SQL1["**execute_sql**<br><br>SELECT COUNT(*), MIN(date),<br>MAX(date) FROM each table<br>→ Assess completeness"]
    end

    subgraph silver["Phase 2: Silver Construction"]
        SQL2["**execute_sql**<br><br>CREATE TABLE silver_weather_duck_daily AS<br>SELECT w.*, q.quack_count,<br>  f.fluff_intensity<br>FROM silver_daily_weather w<br>JOIN silver_quack_frequency q<br>  ON w.station_id = q.station_id<br>  AND w.observation_date = q.obs_date<br>JOIN silver_feather_fluffing f ..."]
        SQL2 --> VS["**validate_silver**<br><br>Grain check: is station_id +<br>observation_date unique?<br>Join completeness: 85% match rate<br>Null check: 15% null duck obs"]
    end

    subgraph theme["Phase 3: Theme Analysis — 'Quack Before the Storm'"]
        SQL3["**execute_sql**<br><br>SELECT quack_count,<br>  LEAD(precip_mm, 1) OVER<br>    (PARTITION BY station_id<br>     ORDER BY observation_date)<br>    AS next_day_precip<br>FROM silver_weather_duck_daily"]
        SQL3 --> STAT["**statistical_analysis**<br><br>type: correlation<br>method: spearman<br>→ r = 0.42, p < 0.001"]
        STAT --> STAT2["**statistical_analysis**<br><br>type: hypothesis_test<br>method: mannwhitney<br>→ Quacks higher before rain<br>→ p = 0.003"]
        STAT2 --> CONF["**check_confounders**<br><br>predictor: quack_count<br>outcome: next_day_precip<br>grouping: [season, temp_bin]<br>→ No Simpson's Paradox"]
        CONF --> VSTAT["**validate_statistics**<br><br>type: multiple_comparisons<br>p_values: [0.001, 0.003, ...]<br>→ Bonferroni-corrected results"]
        VSTAT --> VIZ["**create_visualization**<br><br>chart_type: scatter<br>x: quack_count<br>y: next_day_precip<br>color: season<br>→ Saves quack_rain_scatter.png"]
        VIZ --> NOTE["**save_note**<br><br>Logs methodology, SQL queries,<br>statistical reasoning to<br>notes/theme_1.md"]
        NOTE --> SF["**save_finding**<br><br>title: 'Quack-Rain Correlation'<br>finding: 'r=0.42, p<0.001'<br>evidence: 'Spearman rho...'<br>significance: high<br>→ Appended to findings.json"]
    end

    orient --> silver
    silver --> theme

    style orient fill:#f7faf7,stroke:#4a8a4a
    style silver fill:#f7faf7,stroke:#4a8a4a
    style theme fill:#f7faf7,stroke:#4a8a4a
```

### Key Behaviors

**Theme-driven analysis** — Each theme in the `ResearchConfig` is a self-contained research question with methodology steps, required tables, expected outputs, and a signature visualization. The agent executes themes in sequence order.

**Evidence tiers** — Every finding is classified by statistical strength:

| Tier | Criteria | Used For |
|------|----------|----------|
| DEFINITIVE | p < 0.001, large effect size | Primary conclusions |
| STRONG | p < 0.01, medium+ effect | Leading findings |
| SUGGESTIVE | p < 0.05 | Supporting evidence |
| CONTEXTUAL | Descriptive, no hypothesis test | Background context |
| WEAK | p ≥ 0.05, negligible effect | Limitations, caveats |

**Confounder detection** — `check_confounders` decomposes aggregate relationships into subgroups to detect Simpson's Paradox — where the overall trend reverses within every subgroup.

**Reproducibility** — Every SQL query, statistical test, and chart is logged to per-theme notes files via `save_note`. A human can follow the exact reasoning path without the AI. See [Run Management & Reproducibility](run-management.md) for the full artifact and notes system.

---

## StoryTeller Agent — Deep Dive

The StoryTeller reads the scientist's outputs (findings, charts, tables, notes) and produces a narrative report grounded in statistical evidence.

### Phases

```mermaid
flowchart LR
    P1["**Phase 1**<br>Inventory<br><br>Scan findings,<br>charts, tables,<br>notes files"]
    P2["**Phase 2**<br>Evidence Eval<br><br>Score finding strength,<br>build bill of<br>materials per section"]
    P3["**Phase 3**<br>Section Writing<br><br>Write each narrative<br>section using<br>curated evidence"]
    P4["**Phase 4**<br>Coherence<br><br>Fix transitions,<br>consistency across<br>all sections"]
    P5["**Phase 5**<br>Finalize<br><br>Assemble document,<br>TOC, bibliography,<br>appendices"]

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5

    style P1 fill:#fef9f0,stroke:#b38600
    style P2 fill:#fef3e0,stroke:#b38600
    style P3 fill:#fdecd0,stroke:#b38600
    style P4 fill:#fce6c0,stroke:#b38600
    style P5 fill:#fbe0b0,stroke:#b38600
```

### Tool-Level Walkthrough

Here's how the StoryTeller writes one section of the report — the "Duck vs. Doppler" showdown:

```mermaid
flowchart TD
    subgraph inventory["Phase 1: Inventory"]
        RF["**read_findings**<br><br>operation: list<br>→ 14 findings across 7 themes<br>→ 5 high, 6 medium, 3 low"]
        RF --> RC["**read_chart**<br><br>operation: list<br>→ 9 charts available"]
        RC --> RT["**read_table**<br><br>operation: list<br>→ 6 CSV result tables"]
    end

    subgraph evidence["Phase 2: Evidence Evaluation"]
        RF2["**read_findings**<br><br>operation: by_theme<br>theme_id: theme_5<br>→ 3 findings about<br>  duck vs meteorologist"]
        RF2 --> EE["**evaluate_evidence**<br><br>operation: curate<br>purpose: 'head-to-head comparison'<br>→ Lead: Duck F1=0.71 vs NWS F1=0.83<br>→ Support: McNemar p=0.02"]
    end

    subgraph writing["Phase 3: Write Section"]
        RC2["**read_chart**<br><br>operation: metadata<br>chart: precision_recall_comparison.png<br>→ Gets interpretation,<br>  axis labels, data source"]
        RC2 --> WN["**write_narrative**<br><br>operation: write_section<br>section_id: section_showdown<br>title: 'Duck vs. Doppler'<br>content: 'The precision-recall<br>curves tell a humbling story...<br>[Fig 5: precision_recall_comparison.png]'"]
        WN --> CS["**cite_source**<br><br>operation: add<br>title: 'NWS Forecast Verification'<br>url: weather.gov/verification<br>→ cite_key: nws_verification_2024"]
    end

    subgraph coherence["Phase 4: Coherence Pass"]
        LIST_SEC["**write_narrative**<br><br>operation: list_sections<br>→ 8 sections in order"]
        LIST_SEC --> READ_SEC["**write_narrative**<br><br>operation: read_section<br>→ Reads adjacent sections<br>→ Checks transitions flow"]
        READ_SEC --> UPDATE["**write_narrative**<br><br>operation: update_section<br>→ Smooths transition:<br>'The duck signal is real.<br>But can it compete?'"]
    end

    subgraph finalize["Phase 5: Finalize"]
        ASSEMBLE["**write_narrative**<br><br>operation: assemble<br>→ Concatenates all sections<br>→ Generates table of contents<br>→ Adds bibliography from cite_source<br>→ Writes duck_weather_report.md"]
    end

    inventory --> evidence
    evidence --> writing
    writing --> coherence
    coherence --> finalize

    style inventory fill:#fefaf2,stroke:#b38600
    style evidence fill:#fefaf2,stroke:#b38600
    style writing fill:#fefaf2,stroke:#b38600
    style coherence fill:#fefaf2,stroke:#b38600
    style finalize fill:#fefaf2,stroke:#b38600
```

### Key Behaviors

**Evidence-grounded writing** — The StoryTeller cannot make claims that aren't backed by findings. `evaluate_evidence` scores each finding's statistical strength and `curate` ranks them for each section's purpose.

**Narrative text must match statistics** — If a finding has p=0.73, it's classified as WEAK evidence regardless of how the text describes it. The evidence threshold config controls what's allowed as a lead finding vs. supporting evidence.

**Citation management** — `cite_source` maintains a bibliography. The `assemble` operation generates formatted references at the end of the report.

**Editorial review** — After the initial write, `run_editor()` enables a human-in-the-loop pass where the operator can give specific rewrite instructions (e.g., "simplify the methodology for a policymaker audience").

---

## Config-Driven Design

The agents are generic — the intelligence about *your* data lives in config dataclasses.

```mermaid
flowchart TD
    subgraph configs["Your Configs"]
        PC["**ProjectConfig**<br><br>catalog, schema, volume_path,<br>join_key, known_sources,<br>processing_hints, metadata_columns"]
        RC["**ResearchConfig**<br><br>thesis, analysis_themes,<br>silver_datasets,<br>research_references"]
        SC["**StorytellerConfig**<br><br>narrative_sections,<br>style_guide,<br>evidence_thresholds,<br>output_format"]
    end

    subgraph agents["Generic Agents (unchanged)"]
        DEA["DataEngineerAgent"]
        DSA["DataScientistAgent"]
        STA["StoryTellerAgent"]
    end

    PC --> DEA
    RC --> DSA
    SC --> STA

    style configs fill:#fff8e1,stroke:#b38600
    style agents fill:#e8f0fe,stroke:#4a6f93
```

| Config | Controls | Key Fields |
|--------|----------|------------|
| `ProjectConfig` | What data to ingest and how | `catalog`, `schema`, `volume_path`, `join_key`, `known_sources`, `source_processing_hints` |
| `ResearchConfig` | What questions to investigate | `thesis`, `analysis_themes`, `silver_datasets`, `research_references` |
| `StorytellerConfig` | How to write the report | `narrative_sections`, `style_guide`, `evidence_thresholds`, `output_format` |

To start a new project, copy an example config, change the domain-specific fields, and run the same agents. See the [Tutorial](tutorial-silly-weather.md) for a complete walkthrough.

---

## LLM Client

The `LLMClient` wraps [LiteLLM](https://docs.litellm.ai/) for multi-provider support:

```python
from versifai.core.llm import LLMClient

# Any LiteLLM-supported provider
llm = LLMClient(model="claude-sonnet-4-6")      # Anthropic
llm = LLMClient(model="gpt-4o")                  # OpenAI
llm = LLMClient(model="azure/gpt-4o")            # Azure
llm = LLMClient(model="gemini/gemini-1.5-pro")   # Google
```

Key features:

- **Prompt caching** — system prompt and tool definitions use `cache_control` to avoid re-billing static tokens each turn
- **Retry logic** — exponential backoff on rate limits, connection errors, 5xx errors
- **Usage tracking** — input/output/cache-read/cache-creation token counts per call

---

## Databricks Integration

### Unity Catalog

All tables live in a three-level namespace: `catalog.schema.table`.

```
my_catalog.silly_weather.silver_daily_weather
│          │              │
│          │              └── Table name (silver_ prefix = processed)
│          └── Schema (one per project)
└── Catalog (org-level grouping)
```

### Volumes (FUSE Mount)

Raw data files are accessed via Databricks Volumes at `/Volumes/catalog/schema/volume/`.

!!! warning "No file append on FUSE"
    Databricks FUSE mounts don't support file append mode. Versifai uses a read-then-write pattern everywhere:
    ```python
    existing = path.read_text() if path.exists() else ""
    path.write_text(existing + new_content)
    ```

### SQL Execution

Tools that run SQL follow a two-tier pattern:

1. **Try Spark first** — faster, native in Databricks notebooks
2. **Fall back to Databricks SDK** — works outside notebooks, uses async polling

---

## Safety Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Max agent turns (global) | 200 | Prevent infinite loops |
| Max turns per source | 120 | Allow batch processing of large file sets |
| Max acceptance iterations | 3 | Engineer-analyst feedback cycles |
| Max consecutive tool errors | 5 | Trigger error escalation |
| Memory summarization trigger | 30 messages | Keep context window manageable |
| Auto-flush threshold | 30M rows | Prevent OOM during staging |
| Direct write threshold | 2M rows | Above this, route through parquet |
| LLM retry attempts | 3 | Exponential backoff for API resilience |
