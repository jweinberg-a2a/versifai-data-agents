# Tutorial: Silly Weather Example

This walkthrough builds a complete Versifai project from scratch using the
**Silly Weather** example — a tongue-in-cheek investigation into whether duck
behavior can predict rain better than professional meteorologists.

The topic is intentionally absurd, but the framework patterns are exactly what
you'd use for a real analysis. By the end, you'll understand how to:

1. Write a **ProjectConfig** for the Data Engineer
2. Write a **ResearchConfig** for the Data Scientist
3. Write a **StorytellerConfig** for the StoryTeller
4. Wire everything together in **Databricks notebook entrypoints**

All example files live in
[`examples/silly_weather/`](https://github.com/jweinberg-a2a/versifai-data-agents/tree/main/examples/silly_weather).

---

## What You're About to Build

Here's the full plan. Three agents run in sequence — each one picks up where
the last one left off.

```mermaid
flowchart TD
    subgraph stage1["Stage 1 — Data Engineer"]
        direction TB
        RAW[/"**Raw Data Volume**\n\nnoaa_weather/ → 3 CSVs\nduck_observations/ → ZIP archive\nice_cream/ → Excel\nforecast_accuracy/ → CSV"/]
        RAW --> DISCOVER["**Discover & Profile**\n\nexplore_volume → map files\nread_file_header → peek at columns\nprofile_data → stats, nulls, types"]
        DISCOVER --> DESIGN["**Design & Load**\n\ndesign_schema → CREATE TABLE SQL\ntransform_and_load → map columns, cast types\nwrite_to_catalog → Delta tables"]
        DESIGN --> VALIDATE["**Validate**\n\nAnalyst agent runs SQL checks\n→ join key integrity\n→ null rates, ranges\n→ cross-table joinability"]
    end

    subgraph tables["Unity Catalog"]
        DT[("**4 Delta Tables**\n\nsilver_daily_weather\nsilver_quack_frequency\nsilver_feather_fluffing\nsilver_ice_cream_sales")]
    end

    subgraph stage2["Stage 2 — Data Scientist"]
        direction TB
        JOIN["**Build Silver Datasets**\n\nexecute_sql → JOIN weather +\nduck + ice cream tables\nvalidate_silver → quality checks"]
        JOIN --> ANALYZE["**7 Research Themes**\n\n0. The Quack Census\n1. Quack Before the Storm\n2. The Fluff Factor\n3. The Ice Cream Confounder\n4. V-Formation Tornado Warning\n5. Duck vs Doppler\n6. Grand Unified Duck Theory"]
        ANALYZE --> TOOLS["**Per Theme**\n\nstatistical_analysis → correlations, tests\nfit_model → regression, classification\ncheck_confounders → Simpson's Paradox\ncreate_visualization → charts\nsave_finding → structured evidence"]
    end

    subgraph outputs["Research Outputs"]
        FO[/"**findings.json** — 14 findings\n**charts/** — 9 visualizations\n**tables/** — 6 CSV summaries\n**notes/** — per-theme reasoning"/]
    end

    subgraph stage3["Stage 3 — StoryTeller"]
        direction TB
        READ["**Inventory & Evaluate**\n\nread_findings → load all findings\nevaluate_evidence → score strength\ncurate → rank for each section"]
        READ --> WRITE["**Write 8 Sections**\n\n1. Abstract\n2. The Quack Census\n3. Quack Before the Storm\n4. The Fluff Factor\n5. Duck vs Doppler Showdown\n6. Grand Unified Duck Theory\n7. Limitations\n8. Conclusions"]
        WRITE --> FINISH["**Finalize**\n\nCoherence pass → fix transitions\ncite_source → bibliography\nassemble → table of contents\n→ duck_weather_report.md"]
    end

    VALIDATE --> DT
    DT --> JOIN
    TOOLS --> FO
    FO --> READ

    style stage1 fill:#f7f9fc,stroke:#4a6f93
    style tables fill:#f0f0f0,stroke:#666
    style stage2 fill:#f7faf7,stroke:#4a8a4a
    style outputs fill:#f0f0f0,stroke:#666
    style stage3 fill:#fefaf2,stroke:#b38600
```

Each stage has exactly three parts:

| Part | What It Is | What Changes Between Projects |
|------|-----------|-------------------------------|
| **Config** | A Python dataclass holding all domain knowledge | Everything — this is where your project lives |
| **Agent** | A generic Python class that reads the config and does work | Nothing — agents are reusable across projects |
| **Notebook** | A Databricks notebook that creates the agent and runs it | Just the import path to your config |

The agents are generic. All the domain-specific knowledge lives in the configs.
This means you never modify agent code — you just write new configs.

For a deeper look at the ReAct loop, tool system, and agent internals, see the
[Architecture](architecture.md) page.

---

## Step 1: The Engineer Config

**File:** `examples/silly_weather/engineer_config.py`

The `ProjectConfig` tells the Data Engineer Agent everything it needs to ingest
raw data files from a Databricks Volume into Delta tables.

### The Essentials

```python
from versifai.data_agents.engineer.config import ProjectConfig, JoinKeyConfig

SILLY_WEATHER = ProjectConfig(
    name="Silly Weather: Do Ducks Predict Rain Better Than Meteorologists?",
    catalog="my_catalog",       # Your Unity Catalog name
    schema="silly_weather",     # Schema where tables will be created
    volume_path="/Volumes/my_catalog/silly_weather/raw_data",  # Where CSVs live
)
```

These four fields are the minimum. The agent will scan the Volume, profile
files, and figure out what to do. But you can help it work faster and smarter
by adding hints.

### Join Key — How Tables Connect

The `join_key` defines the primary column used to join all tables together.
Every table the engineer creates should include this column:

```python
from versifai.data_agents.engineer.config import JoinKeyConfig

join_key = JoinKeyConfig(
    column_name="station_id",
    data_type="STRING",
    description="NOAA weather station identifier.",
    validation_rule="Must match pattern 'USW\\d{8}' or 'USC\\d{8}'",
    expected_entity_count=500,
    related_columns=[
        {"name": "state_code", "description": "Two-letter US state code", "required": True},
        {"name": "latitude", "description": "Decimal degrees", "required": False},
    ],
)
```

!!! tip "Why declare a join key?"
    The agent uses this to validate that loaded tables can actually be joined.
    If a table is missing the join key column, the agent flags it during
    quality checks.

### Known Sources — Help the Agent Recognize Files

If you know what files are in your Volume, tell the agent:

```python
from versifai.data_agents.engineer.config import DataSourceHint

known_sources = [
    DataSourceHint(
        name="NOAA Daily Weather Summaries",
        description="Daily temp, precip, snow, wind for US stations.",
        keywords=["GHCND", "daily_summary", "NOAA"],
    ),
    DataSourceHint(
        name="Duck Pond Observation Logs",
        description="Daily quack frequency, feather-fluffing intensity.",
        keywords=["duck", "pond", "quack"],
    ),
]
```

These are just hints — the agent will still explore and profile everything.
But hints help it match files to sources faster.

### Source Processing Hints — Multi-File Instructions

For data sources that contain multiple files (like a ZIP with several CSVs),
use `SourceProcessingHint` to tell the agent exactly what to do:

```python
from versifai.data_agents.engineer.config import SourceProcessingHint, SourceFileHint

source_processing_hints = [
    SourceProcessingHint(
        source_pattern="duck_observations",
        description="Duck pond observation logs",
        multi_table=True,    # Create SEPARATE tables for each file type
        files=[
            SourceFileHint(
                file_pattern="quack_frequency",
                target_table="silver_quack_frequency",
                description="Daily quack counts per pond per hour",
            ),
            SourceFileHint(
                file_pattern="feather_index",
                target_table="silver_feather_fluffing",
                description="Daily feather fluffing intensity (0-10)",
            ),
        ],
    ),
]
```

!!! note "multi_table=True"
    When `multi_table=True`, the agent creates a separate Delta table for each
    file type. When `False` (default), all files in the source are combined
    into a single table.

### Metadata Columns — Added to Every Table

Define columns that should appear in every table automatically:

```python
from versifai.data_agents.engineer.config import MetadataColumnConfig

metadata_columns = [
    MetadataColumnConfig(
        name="source_file_name", data_type="STRING",
        description="Original filename the row was loaded from",
    ),
    MetadataColumnConfig(
        name="source_year", data_type="INT",
        description="The observation year extracted from the data",
    ),
    MetadataColumnConfig(
        name="load_timestamp", data_type="TIMESTAMP",
        description="When this row was loaded into the catalog",
    ),
]
```

### Putting It All Together

See the complete config in
[`engineer_config.py`](https://github.com/jweinberg-a2a/versifai-data-agents/blob/main/examples/silly_weather/engineer_config.py).

---

## Step 2: The Research Config

**File:** `examples/silly_weather/research_configs/duck_rain_prediction.py`

The `ResearchConfig` tells the Data Scientist Agent what to investigate. It
defines your thesis, analysis themes, silver datasets, and references.

### The Thesis

Every research config starts with a thesis — the core argument to investigate:

```python
from versifai.science_agents.scientist.config import ResearchConfig

DUCK_RAIN = ResearchConfig(
    name="Do Ducks Predict Rain Better Than Meteorologists?",
    thesis=(
        "Duck behavioral signals (quack frequency, feather-fluffing intensity, "
        "and V-formation flight patterns) contain genuine meteorological "
        "information that provides statistically significant 24-hour "
        "precipitation forecasts."
    ),
)
```

The thesis drives the entire analysis. Every theme either supports, refutes,
or adds nuance to this claim.

### Analysis Themes — The Research Arc

Themes are the heart of the config. Each theme is one unit of analysis with
a specific research question, methodology, and expected outputs:

```python
from versifai.science_agents.scientist.config import AnalysisTheme

theme_1 = AnalysisTheme(
    id="theme_1",
    title="Quack Before the Storm",
    question="Is there a significant correlation between quack frequency and next-day rain?",
    analysis_type="correlation",       # descriptive | comparative | correlation | trend
    sequence=1,                        # Execution order
    required_tables=[                  # Tables the agent needs
        "silver_weather_duck_daily",
    ],
    analysis_steps=[                   # Step-by-step methodology
        "Compute Pearson and Spearman correlation: quacks(t) vs precip(t+1)",
        "Run time-lagged cross-correlation for lags 0-3 days",
        "Test significance with permutation test (10,000 shuffles)",
        "Stratify by season",
        "Control for temperature (partial correlation)",
    ],
    tables_to_produce=[                # Expected output tables
        "quack_precip_correlation_matrix",
        "lagged_cross_correlation_results",
        "seasonal_correlation_breakdown",
    ],
    signature_visualization=(          # The ONE key chart for this theme
        "A lag-correlation plot (x=lag days, y=correlation coefficient) "
        "with confidence bands. One line per season."
    ),
    punchline=(                        # One-line summary of expected finding
        "Quack frequency at lag-1 shows r=X.XX with next-day rain (p=X.XXX)."
    ),
    data_notes=(                       # Domain-specific hints for the agent
        "Use silver_weather_duck_daily. Create lag columns. "
        "Rain threshold: precip_mm > 2.54 (0.1 inch)."
    ),
)
```

!!! info "How many themes?"
    Most projects have 5-10 themes. Each theme should answer one focused
    research question. Themes build on each other — early themes establish
    baselines, middle themes test hypotheses, and later themes synthesize.

The Silly Weather example has 7 themes:

| # | Title | Question |
|---|-------|----------|
| 0 | The Quack Census | What does our data look like? |
| 1 | Quack Before the Storm | Do quacks correlate with next-day rain? |
| 2 | The Fluff Factor | Does feather-fluffing predict storm severity? |
| 3 | The Ice Cream Confounder | Is the duck signal just a temperature proxy? |
| 4 | V-Formation Tornado Warning | Do formation flights predict severe weather? |
| 5 | Duck vs Doppler | Who predicts better: ducks or meteorologists? |
| 6 | The Grand Unified Duck Theory | Can we build a combined prediction model? |

### Silver Datasets — Pre-Joined Analytical Tables

Silver datasets are intermediate tables the agent builds by joining multiple
source tables. Declare them upfront so the agent knows what to construct:

```python
from versifai.science_agents.scientist.config import SilverDatasetSpec

silver_weather_duck = SilverDatasetSpec(
    name="silver_weather_duck_daily",
    description="Daily weather joined with duck behavioral metrics.",
    source_tables=["silver_daily_weather", "silver_quack_frequency", "silver_feather_fluffing"],
    join_key="station_id",
    time_column="observation_date",
    data_notes="Join via station_id + observation_date. ~15% NULL duck obs on weekdays.",
)
```

The agent builds these in its "Silver Construction" phase before running
any theme analysis.

### Research References

Point the agent to relevant published work:

```python
from versifai.science_agents.scientist.config import ResearchReference

refs = [
    ResearchReference(
        title="Animal Behavior as Weather Predictors: A Meta-Analysis",
        url="https://en.wikipedia.org/wiki/Weather_lore",
        description="Survey of folklore and scientific evidence",
    ),
]
```

The agent uses these during its literature review tool to compare its
findings against published results.

---

## Step 3: The Storyteller Config

**File:** `examples/silly_weather/storyteller_config.py`

The `StorytellerConfig` tells the StoryTeller Agent how to write the
narrative report from the scientist's findings.

### Style Guide — Voice and Tone

```python
from versifai.story_agents.storyteller.config import StyleGuide

style = StyleGuide(
    voice="third-person with dry humor",
    audience="Data scientists and curious duck enthusiasts",
    document_type="Research white paper (tongue-in-cheek)",
    tone_guidance=(
        "Deadpan scientific. Write as if this were a real Nature paper "
        "that happens to be about ducks."
    ),
    anti_patterns=(
        "- NO: Exclamation marks in scientific claims\n"
        "- NO: 'Interestingly' or 'Surprisingly'\n"
        "- NO: Hedging on clearly significant results\n"
    ),
)
```

### Narrative Sections — The Story Arc

Each section maps to one or more research themes and has specific writing
instructions:

```python
from versifai.story_agents.storyteller.config import NarrativeSection

section_showdown = NarrativeSection(
    id="section_showdown",
    title="Duck vs. Doppler: The Showdown",
    purpose="Head-to-head comparison with professional meteorologists",
    source_theme_ids=["theme_5"],    # Maps to research theme(s)
    tone="analytical",
    max_words=1500,
    key_evidence="Precision-recall curves, F1 scores, McNemar's test",
    narrative_guidance=(
        "This is the climax. Present paired PR curves side by side. "
        "Be ruthlessly honest about where ducks fail."
    ),
    transition_from="The duck signal is real. But can it compete?",
    transition_to="What if we combined forces?",
    sequence=4,
)
```

!!! tip "Transitions"
    The `transition_from` and `transition_to` fields help the agent write
    smooth connective tissue between sections. The coherence pass uses these
    to verify the narrative flows naturally.

### Evidence Thresholds

Control how strong evidence must be before the agent cites it:

```python
from versifai.story_agents.storyteller.config import EvidenceThreshold

evidence = EvidenceThreshold(
    min_significance_for_lead="high",      # Lead findings need p < 0.01
    min_significance_for_support="medium", # Supporting evidence needs p < 0.05
    require_effect_size=True,              # Must report effect sizes
    max_unsupported_claims=0,              # Zero tolerance for ungrounded claims
)
```

### Output Format

```python
from versifai.story_agents.storyteller.config import OutputFormat

output = OutputFormat(
    format="markdown",
    filename="duck_weather_report.md",
    include_toc=True,
    include_methodology_appendix=True,
    include_data_sources_appendix=True,
)
```

---

## Step 4: Notebook Entrypoints

The notebooks are where you actually run the agents. Each notebook follows
the same pattern:

1. Install versifai
2. Load the config
3. Create the agent
4. Run it

### Running the Engineer

**File:** `examples/silly_weather/notebooks/run_engineer.py`

```python
from examples.silly_weather.engineer_config import SILLY_WEATHER
from versifai.data_agents.engineer.agent import DataEngineerAgent

cfg = SILLY_WEATHER
agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)

# Stage 1: Discover, profile, design schemas, transform & load
results = agent.run(source_path=cfg.volume_path)

# Stage 2: Standardize column names
rename_results = agent.run_rename()

# Stage 3: Build data catalog table
catalog_results = agent.run_catalog()

# Stage 4: Validate all tables
quality_results = agent.run_quality_check()
```

Each stage is a separate method call. If the notebook crashes, re-run it —
the agent has **smart resume** and will skip completed work.

### Running the Scientist

**File:** `examples/silly_weather/notebooks/run_scientist.py`

```python
from examples.silly_weather.research_configs.duck_rain_prediction import DUCK_RAIN
from versifai.science_agents.scientist.agent import DataScientistAgent

cfg = DUCK_RAIN
agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)

# Full pipeline: Orientation → Silver Construction → Analysis → Synthesis
results = agent.run()
```

The scientist runs through 4 phases automatically:

1. **Orientation** — Inventory tables, assess data quality
2. **Silver Construction** — Build the pre-joined analytical datasets
3. **Theme Analysis** — Execute each theme sequentially
4. **Synthesis** — Cross-validate findings, compile summary

You can also run specific themes:

```python
# Skip themes 0-2, run only 3-6
agent.run_themes(start_theme=3)

# Run only specific themes
agent.run_themes(themes=[1, 5])
```

### Running the StoryTeller

**File:** `examples/silly_weather/notebooks/run_storyteller.py`

```python
from examples.silly_weather.storyteller_config import DUCK_STORY
from versifai.story_agents.storyteller.agent import StoryTellerAgent

cfg = DUCK_STORY
agent = StoryTellerAgent(cfg=cfg, dbutils=dbutils)

# Full pipeline: Inventory → Evidence → Write → Coherence → Finalize
results = agent.run()
```

The storyteller reads from the scientist's outputs (`findings.json`,
`charts/`, `tables/`) and produces a narrative Markdown report.

You can also rewrite specific sections or run an editorial pass:

```python
# Rewrite just the showdown section
agent.run_sections(sections=[4])

# Editor review with instructions
agent.run_editor(instructions="Tighten the transition between sections 2 and 3.")
```

---

## How Data Flows: Raw Files → Tables → Analysis → Report

This section shows what actually happens to your data at each stage — from
raw files on disk to a finished narrative report.

### Stage 1: Raw Files → Delta Tables

You start by uploading raw data files to a Databricks Volume. The Data Engineer
agent discovers them and turns each source into a clean Delta table.

```mermaid
flowchart LR
    subgraph volume["Databricks Volume: /Volumes/.../raw_data/"]
        direction TB
        F1["noaa_weather/\n├── weather_2020.csv\n├── weather_2021.csv\n└── weather_2022.csv"]
        F2["duck_observations/\n└── duck_obs.zip\n    ├── quack_frequency.csv\n    └── feather_index.csv"]
        F3["ice_cream/\n└── ice_cream_sales.xlsx"]
        F4["forecast_accuracy/\n└── nws_forecasts.csv"]
    end

    subgraph engineer["Data Engineer Agent"]
        direction TB
        PROFILE["Profile & Design\n\nReads headers, detects types,\ndesigns CREATE TABLE SQL\nfor each source"]
        TRANSFORM["Transform & Load\n\nRenames columns to snake_case,\ncasts types, adds metadata,\nbatch-processes multi-file sources"]
    end

    subgraph catalog["Unity Catalog: my_catalog.silly_weather"]
        direction TB
        T1["**silver_daily_weather**\n\nstation_id · observation_date\ntemp_max_c · temp_min_c\nprecip_mm · snow_depth_mm\nwind_speed_ms\nsource_file_name · load_timestamp\n\n*1.1M rows · 3 years*"]
        T2["**silver_quack_frequency**\n\nstation_id · observation_date\nhour_of_day · quack_count\nambient_noise_db\nsource_file_name · load_timestamp\n\n*890K rows*"]
        T3["**silver_feather_fluffing**\n\nstation_id · observation_date\nfluff_intensity · humidity_pct\nsource_file_name · load_timestamp\n\n*365K rows*"]
        T4["**silver_ice_cream_sales**\n\nstation_id · month · year\nunits_sold · revenue_usd\navg_temp_c\nsource_file_name · load_timestamp\n\n*18K rows*"]
    end

    F1 --> PROFILE
    F2 --> PROFILE
    F3 --> PROFILE
    F4 --> PROFILE
    PROFILE --> TRANSFORM
    TRANSFORM --> T1
    TRANSFORM --> T2
    TRANSFORM --> T3
    TRANSFORM --> T4

    style volume fill:#fff8e1,stroke:#b38600
    style engineer fill:#e8f0fe,stroke:#4a6f93
    style catalog fill:#e8f4e8,stroke:#4a8a4a
```

Every table gets:

- **snake_case column names** — the agent renames `TMAX` → `temp_max_c`, `PRCP` → `precip_mm`
- **Proper types** — dates become `DATE`, measurements become `DOUBLE`, IDs stay `STRING`
- **Metadata columns** — `source_file_name` and `load_timestamp` for audit
- **Join key** — `station_id` in every table so they can be joined

### Stage 2: Silver Datasets → Statistical Analysis

The Data Scientist joins the source tables into analytical datasets, then runs
each research theme against them.

```mermaid
flowchart TD
    subgraph source_tables["Source Tables (from Engineer)"]
        T1["silver_daily_weather"]
        T2["silver_quack_frequency"]
        T3["silver_feather_fluffing"]
        T4["silver_ice_cream_sales"]
    end

    subgraph silver_construction["Silver Dataset Construction (SQL JOINs)"]
        J1["**silver_weather_duck_daily**\n\nJOIN weather + quacks + fluffing\nON station_id + observation_date\n\n→ 365K rows, 15 columns\n→ Weather + duck behavior per day"]
        J2["**silver_duck_forecast_comparison**\n\nJOIN duck signals + NWS forecasts\nwith next-day actual weather\n\n→ 365K rows\n→ Who predicted better?"]
        J3["**silver_ice_cream_weather**\n\nJOIN ice cream sales + weather\n+ duck activity (monthly agg)\n\n→ 18K rows\n→ Confounder analysis"]
    end

    subgraph analysis["Theme Analysis (7 themes)"]
        direction TB
        TH0["Theme 0: **Quack Census**\n*describe* → summary stats\n*distribution* → normality tests\n→ 2 findings, 1 chart"]
        TH1["Theme 1: **Quack Before the Storm**\n*correlation* → r=0.42, p<0.001\n*hypothesis_test* → Mann-Whitney\n→ 3 findings, 2 charts"]
        TH5["Theme 5: **Duck vs Doppler**\n*fit_model* → logistic regression\n*cross_validate* → F1 comparison\n→ 2 findings, 2 charts"]
        TH6["Theme 6: **Grand Unified Duck Theory**\n*fit_model* → gradient boosting\n*check_confounders* → Simpson's check\n→ 2 findings, 1 chart"]
    end

    T1 --> J1
    T2 --> J1
    T3 --> J1
    T2 --> J2
    T1 --> J3
    T4 --> J3
    J1 --> TH0
    J1 --> TH1
    J2 --> TH5
    J1 --> TH6
    J3 --> TH6

    style source_tables fill:#e8f4e8,stroke:#4a8a4a
    style silver_construction fill:#dcefd8,stroke:#4a8a4a
    style analysis fill:#d0ead0,stroke:#4a8a4a
```

### Stage 3: Findings → Narrative Report

The StoryTeller reads the structured findings, evaluates evidence strength,
and writes each section of the report grounded in statistical evidence.

### What the Final Report Looks Like

The StoryTeller produces a complete Markdown report. Here's a mockup of what
the silly weather report looks like:

---

<div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 24px; margin: 16px 0; font-family: Georgia, serif;">

<h2 style="text-align: center; margin-bottom: 4px;">Do Ducks Predict Rain Better Than Meteorologists?</h2>
<p style="text-align: center; color: #666; font-style: italic; margin-top: 0;">A Rigorous Statistical Investigation</p>

**Table of Contents**

1. Abstract
2. The Quack Census: Baseline Characterization
3. Quack Before the Storm: The Core Signal
4. The Fluff Factor: Feather-Based Forecasting
5. The Ice Cream Confounder
6. Duck vs. Doppler: The Showdown
7. The Grand Unified Duck Theory
8. Limitations & Future Work
9. Conclusions
10. References

---

**1. Abstract**

This investigation examines whether observable duck behaviors — specifically
quack frequency, feather-fluffing intensity, and V-formation flight patterns —
contain genuine meteorological information. Analysis of 1.1 million daily
observations across 47 weather stations reveals a statistically significant
correlation between quack frequency and next-day precipitation
(Spearman ρ = 0.42, p < 0.001). However, the National Weather Service
maintains a decisive advantage in 24-hour precipitation forecasting
(F1 = 0.83 vs. duck-based F1 = 0.71, McNemar's p = 0.02).

---

**3. Quack Before the Storm: The Core Signal**

The central question of this investigation — whether ducks quack more before
rain — yields an unambiguous answer: they do.

A Spearman rank correlation between daily quack frequency and next-day
precipitation reveals ρ = 0.42 (p < 0.001, n = 312,847). The relationship
holds across all four seasons, though summer exhibits the strongest signal
(ρ = 0.51) while winter shows the weakest (ρ = 0.28).

*[Figure 2: Lag-correlation plot showing quack frequency vs. precipitation at
lags 0–3 days, stratified by season. The lag-1 peak is consistent across all
seasons, with 95% confidence bands shown.]*

A Mann-Whitney U test confirms that pre-rain quack counts (median = 847
quacks/hr) significantly exceed fair-weather counts (median = 612 quacks/hr,
U = 2.34 × 10⁹, p = 0.003). The effect is not subtle — Cohen's d = 0.38
places it firmly in the "small to medium" range.

---

**6. Duck vs. Doppler: The Showdown**

The precision-recall curves tell a humbling story. While duck-based forecasting
achieves respectable precision (0.74) at the 50% recall threshold, the National
Weather Service's Doppler-based system maintains precision of 0.89 at the same
recall level [1].

The McNemar test on paired 2×2 contingency tables confirms this is not a
statistical artifact (χ² = 5.41, p = 0.02). The NWS correctly predicts 83% of
rain events that ducks miss entirely — primarily light precipitation events
below 5mm.

However, ducks show a surprising advantage for heavy precipitation events
(> 25mm): duck-based recall = 0.91 vs. NWS recall = 0.84. The biological
signal appears strongest precisely when it matters most.

---

**9. Conclusions**

Ducks do not predict rain better than meteorologists. But they predict it
better than random chance, and in certain extreme weather scenarios, they
outperform professional forecasting systems. The combined duck-Doppler model
(gradient boosting, F1 = 0.87) suggests the optimal meteorological strategy is
to check both the radar and the nearest pond.

---

**References**

[1] National Weather Service. "Forecast Verification Statistics 2020-2022."
weather.gov/verification.

[2] Lorenz, K. "Animal Behavior as Environmental Indicators." *Journal of
Ethology*, 1973.

</div>

---

### Output File Structure

After running all three agents, your results Volume will contain everything
needed to verify the work without the AI. For a deep dive into run isolation,
smart resume, and the reproducibility contract, see
[Run Management & Reproducibility](run-management.md).

```
/Volumes/my_catalog/silly_weather/
├── raw_data/                          # Input (you uploaded these)
│   ├── noaa_weather/
│   ├── duck_observations/
│   ├── ice_cream/
│   └── forecast_accuracy/
│
├── results/                           # Data Scientist outputs
│   ├── findings.json                  # 14 structured findings with p-values
│   ├── charts/                        # PNG visualizations
│   │   ├── quack_census_distribution.png
│   │   ├── quack_rain_scatter.png
│   │   ├── lag_correlation_by_season.png
│   │   ├── fluff_storm_severity.png
│   │   ├── ice_cream_confounder_partial.png
│   │   ├── precision_recall_comparison.png
│   │   ├── feature_importance_unified.png
│   │   ├── v_formation_tornado.png
│   │   └── duck_doppler_confusion_matrix.png
│   ├── tables/                        # CSV summary tables
│   │   ├── correlation_matrix.csv
│   │   ├── model_comparison.csv
│   │   ├── seasonal_breakdown.csv
│   │   ├── confounder_decomposition.csv
│   │   ├── forecast_accuracy_paired.csv
│   │   └── unified_model_coefficients.csv
│   └── notes/                         # Per-theme reasoning logs
│       ├── theme_0.md
│       ├── theme_1.md
│       ├── ...
│       └── theme_6.md
│
└── narrative/                         # StoryTeller outputs
    └── duck_weather_report.md         # The final report (~4,000 words)
```

---

## Adapting for Your Own Project

To build your own Versifai project, copy the Silly Weather example and
replace the domain content:

1. **Copy the example:**
   ```bash
   cp -r examples/silly_weather examples/my_project
   ```

2. **Edit `engineer_config.py`:**
   - Change `catalog`, `schema`, `volume_path` to your Databricks target
   - Update `join_key` to your primary join column
   - List your data sources in `known_sources`
   - Add processing hints in `source_processing_hints`

3. **Edit `research_configs/`:**
   - Write your thesis
   - Define 5-10 analysis themes with research questions
   - Specify silver datasets for pre-joined analytical tables
   - Add research references

4. **Edit `storyteller_config.py`:**
   - Define your narrative sections (one per major finding)
   - Set the style guide for your audience
   - Configure evidence thresholds

5. **Run the notebooks in order:**
   - `run_engineer.py` — Ingests raw data
   - `run_scientist.py` — Analyzes data
   - `run_storyteller.py` — Writes the report

The agent code is the same for every project. Your configs are the only thing
that changes.

---

## Key Concepts Recap

| Concept | What It Is | Where It Lives |
|---------|-----------|---------------|
| **ProjectConfig** | Data engineering instructions (catalog, schema, join keys, sources) | `engineer_config.py` |
| **ResearchConfig** | Research methodology (thesis, themes, silver datasets) | `research_configs/*.py` |
| **StorytellerConfig** | Narrative rules (sections, style, evidence thresholds) | `storyteller_config.py` |
| **AnalysisTheme** | One research question with steps, tables, and a signature chart | Inside ResearchConfig |
| **SilverDatasetSpec** | A pre-joined analytical table to build | Inside ResearchConfig |
| **NarrativeSection** | One section of the report with tone and evidence mapping | Inside StorytellerConfig |
| **Smart Resume** | Agents skip completed work on re-run | Built into all agents |
| **Tools** | The unit of agent capability (SQL, stats, charts, etc.) | `src/versifai/*/tools/` |
