# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Silly Weather — Research Analysis Pipeline
# MAGIC
# MAGIC This notebook runs the **DataScientistAgent** to investigate whether
# MAGIC duck behavior predicts rain better than professional meteorologists.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `run_engineer.py` first — the silver tables must exist
# MAGIC 2. Install versifai: `pip install versifai` (or from source)
# MAGIC 3. Set your LLM API key in the cluster environment variables
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC The agent executes 4 phases across 7 analysis themes:
# MAGIC 1. **Orientation** — Inventory tables, assess data quality
# MAGIC 2. **Silver Construction** — Build pre-joined analytical datasets
# MAGIC 3. **Theme Analysis** — Run each theme: hypothesize → test → record findings
# MAGIC 4. **Synthesis** — Cross-validate, compare to literature, summarize

# COMMAND ----------

# MAGIC %pip install versifai --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── Setup ────────────────────────────────────────────────────────────

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("silly_weather")

assert os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
    "OPENAI_API_KEY"
), "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your cluster env vars"

# COMMAND ----------

# ── Load Config ──────────────────────────────────────────────────────
#
# The ResearchConfig defines:
#   - thesis: The core argument to investigate
#   - analysis_themes: 7 themed analyses, each with research questions,
#     analysis steps, tables to produce, and signature visualizations
#   - silver_datasets: Pre-joined analytical datasets to build
#   - research_references: Published work to compare against
#
# Each theme becomes one phase of the analysis pipeline. The agent
# executes them sequentially, building on previous results.

from examples.silly_weather.research_configs.duck_rain_prediction import DUCK_RAIN

cfg = DUCK_RAIN

# ── Optional: Override for your environment ──────────────────────────
# cfg.project.catalog = "your_catalog"
# cfg.project.schema = "your_schema"
# cfg.results_volume_path = "/Volumes/your_catalog/your_schema/results"

logger.info("Research: %s", cfg.name)
logger.info("Thesis: %s", cfg.thesis[:100] + "...")
logger.info("Themes: %d", len(cfg.analysis_themes))

# COMMAND ----------

# ── Create the Agent ─────────────────────────────────────────────────
#
# The DataScientistAgent receives:
#   - cfg: The ResearchConfig with all analysis themes and methodology
#   - dbutils: Databricks utilities
#
# Tools available to the agent:
#   - execute_sql: Run SQL against Unity Catalog
#   - statistical_analysis: t-tests, ANOVA, correlation, chi-square
#   - fit_model: Train gradient boosting, logistic regression, etc.
#   - check_confounders: Detect Simpson's Paradox, selection bias
#   - validate_statistics: Verify claims match actual data
#   - create_visualization: Charts, maps, tables
#   - save_finding: Persist structured findings with evidence
#   - save_note: Record reasoning and decisions per theme

from versifai.science_agents.scientist.agent import DataScientistAgent

agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option A: Run the Full Pipeline
# MAGIC
# MAGIC This runs ALL 7 themes end-to-end. The agent has **smart resume** —
# MAGIC if it crashes midway, re-running will skip completed themes.

# COMMAND ----------

results = agent.run()
logger.info("Full pipeline complete. Result: %s", results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option B: Run Specific Themes
# MAGIC
# MAGIC If you want to re-run specific themes (e.g., after updating data),
# MAGIC use `run_themes()` with theme indices or a start point.

# COMMAND ----------

# # Skip themes 0-2, run themes 3-6 only
# results = agent.run_themes(start_theme=3)

# # Or run specific themes by index
# results = agent.run_themes(themes=[1, 5])  # Just correlation + showdown

# # Or re-run just visualizations
# results = agent.run_visualizations(chart_types=["scatter", "roc"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Results
# MAGIC
# MAGIC The agent writes structured output to the results Volume:
# MAGIC - `findings.json` — All statistical findings with evidence tiers
# MAGIC - `charts/` — PNG visualizations with metadata
# MAGIC - `tables/` — CSV summary tables
# MAGIC - `notes/` — Per-theme markdown reasoning logs

# COMMAND ----------

import json

results_path = cfg.results_volume_path

# List output files
for f in dbutils.fs.ls(results_path):
    print(f"{f.name:40s} {f.size:>10,d} bytes")

# COMMAND ----------

# Preview findings
findings_path = f"{results_path}/findings.json"
try:
    findings_text = dbutils.fs.head(findings_path, 5000)
    findings = json.loads(findings_text)
    print(f"Total findings: {len(findings)}")
    for f in findings[:3]:
        print(f"\n  Title: {f.get('title', 'N/A')}")
        print(f"  P-value: {f.get('p_value', 'N/A')}")
        print(f"  Significance: {f.get('significance', 'N/A')}")
except Exception as e:
    print(f"No findings yet (run the pipeline first): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC With findings produced, run the **StoryTellerAgent** using
# MAGIC `run_storyteller.py` to write the narrative report:
# MAGIC *"Do Ducks Predict Rain? A Rigorous Investigation"*
