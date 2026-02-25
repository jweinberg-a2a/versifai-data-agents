# Data Scientist Agent

The `DataScientistAgent` runs autonomous research analysis against validated data tables. It builds analytical datasets ("silver" tables), runs statistical tests, fits models, and produces charts and research findings.

## How It Works

The scientist operates in **theme-based analysis**. Each theme represents a research question or area of investigation (e.g., "Geographic Disparity", "Temporal Trends"). For each theme, the agent:

1. **Orientation** — Reviews available data and plans the analysis approach
2. **Silver dataset construction** — Creates derived analytical tables via SQL
3. **Statistical analysis** — Runs hypothesis tests, correlations, descriptive statistics
4. **Model fitting** — Regression, classification, or clustering as appropriate
5. **Visualization** — Produces charts and plots for each finding
6. **Findings** — Persists structured research findings with evidence

## Silver Tables

The scientist can create derived tables prefixed with `silver_` via SQL. These are read-write. All other (bronze) tables are read-only — enforced by the `SilverOnlyExecuteSQLTool`.

## Available Tools

| Tool | Description |
|------|-------------|
| `execute_sql` | SQL against Databricks (silver-only write access) |
| `list_catalog_tables` | List all tables in the target schema |
| `statistical_analysis` | Descriptive stats, correlations, distributions, hypothesis testing |
| `model_fitting` | Regression, classification, clustering with automated feature selection |
| `validate_statistics` | Cross-check statistical claims for validity |
| `literature_review` | Web-based research context gathering |
| `create_visualization` | Charts, plots, geographic maps |
| `save_finding` | Persist research findings with evidence |
| `save_note` | Save analytical notes and observations |
| `log_model` | Log model artifacts and metadata |
| `web_search` | External documentation and literature |
| `create_custom_tool` | Runtime tool creation for custom analysis |
| `ask_human` | Pause and ask the operator a question |

See the [Tool Reference](tool-inventory.md) for detailed parameter and return value tables.

## Run Outputs

Each run produces a structured output directory:

```
results/{config_name}/runs/{run_id}/
├── findings.json       # All research findings with evidence
├── charts/             # Visualization images
├── tables/             # Exported data tables
├── notes/              # Analytical notes
├── models/             # Model artifacts
└── run_metadata.json   # Run info, timing, state
```

## Usage

```python
from versifai.science_agents import DataScientistAgent, ResearchConfig

cfg = ResearchConfig(
    name="Customer Churn Analysis",
    catalog="analytics",
    schema="churn",
    results_path="/tmp/results/churn",
    themes=[...],  # Define research themes
)

agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)

# Full run
result = agent.run()

# Re-run only specific themes
result = agent.run_themes(themes=[0, 3])
```

## Evidence Standards

All findings include structured evidence with statistical rigor:

| Tier | Criteria | Example |
|------|----------|---------|
| **DEFINITIVE** | p < 0.001, large effect size, multiple methods agree | RCT with pre-registered hypothesis |
| **STRONG** | p < 0.01, meaningful effect size | Regression with significant predictors |
| **SUGGESTIVE** | p < 0.05, moderate effect size | Correlation with plausible mechanism |
| **CONTEXTUAL** | Descriptive patterns, no formal test | Geographic distribution pattern |
| **WEAK** | p > 0.05 or trivial effect size | Non-significant trend |
