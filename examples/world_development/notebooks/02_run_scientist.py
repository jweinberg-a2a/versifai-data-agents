# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # World Development — Research Analysis Pipeline
# MAGIC
# MAGIC This notebook runs the **DataScientistAgent** to investigate global
# MAGIC development patterns using World Bank indicators: GDP, life expectancy,
# MAGIC education, healthcare spending, CO2 emissions, and population.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `01_run_engineer.py` first — the Delta tables must exist
# MAGIC 2. Install versifai: `pip install versifai` (or from source)
# MAGIC 3. Create a `.env` file with your LLM API key (see Setup cell below)
# MAGIC
# MAGIC ## Before You Start
# MAGIC 1. **Update `CATALOG` and `SCHEMA`** in `research_configs/global_development.py`
# MAGIC 2. **Update the `load_dotenv()` path** in the Setup cell to point to your `.env` file
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC The agent executes 4 phases across 6 analysis themes:
# MAGIC 1. **Orientation** — Inventory tables, assess data quality and coverage
# MAGIC 2. **Silver Construction** — Build pre-joined analytical panels
# MAGIC 3. **Theme Analysis** — Run each theme: hypothesize, test, record findings
# MAGIC    - Theme 0: The Development Dashboard (descriptive)
# MAGIC    - Theme 1: The Preston Curve (GDP vs life expectancy)
# MAGIC    - Theme 2: Education and Economic Growth (comparative)
# MAGIC    - Theme 3: Healthcare Spending Returns (regression)
# MAGIC    - Theme 4: The Carbon Cost of Development (trend + confounding)
# MAGIC    - Theme 5: Convergence or Divergence? (trend)
# MAGIC 4. **Synthesis** — Cross-validate, compare to literature, summarize

# COMMAND ----------

# MAGIC %pip install ../../.. python-dotenv --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── Setup ────────────────────────────────────────────────────────────

import logging
import os

from dotenv import load_dotenv

# Load your .env file — update this path to match your workspace location
# Example: /Workspace/Users/you@company.com/versifai-data-agents/.env
load_dotenv("/Workspace/path/to/your/.env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("world_development")

assert os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"), (
    "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file"
)

# COMMAND ----------

# ── Load Config ──────────────────────────────────────────────────────

from examples.world_development.research_configs.global_development import (
    GLOBAL_DEVELOPMENT,
)

cfg = GLOBAL_DEVELOPMENT

logger.info("Research: %s", cfg.name)
logger.info("Thesis: %s", cfg.thesis[:100] + "...")
logger.info("Themes: %d", len(cfg.analysis_themes))

# COMMAND ----------

# ── Create the Agent ─────────────────────────────────────────────────

from versifai.science_agents.scientist.agent import DataScientistAgent

agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)

logger.info("Run ID: %s", agent._run_id)
logger.info("Run path: %s", agent._run_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option A: Run the Full Pipeline
# MAGIC
# MAGIC Runs ALL 6 themes end-to-end. Each run creates an isolated directory
# MAGIC under `results/runs/{run_id}/`.
# MAGIC
# MAGIC If a run crashes midway, use `resume=True` when creating the agent
# MAGIC to continue the latest run instead of starting a new one.

# COMMAND ----------

results = agent.run()
logger.info("Full pipeline complete. Result: %s", results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option B: Resume a Crashed Run
# MAGIC
# MAGIC Pass `resume=True` to pick up the latest run where it left off.

# COMMAND ----------

# agent = DataScientistAgent(cfg=cfg, dbutils=dbutils, resume=True)
# results = agent.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option C: Run Specific Themes

# COMMAND ----------

# # Skip themes 0-2, run themes 3-5 only
# results = agent.run_themes(start_theme=3)

# # Or run specific themes by index
# results = agent.run_themes(themes=[1, 4])  # Preston Curve + Carbon Cost

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

results_path = agent._run_path  # Run-isolated output directory

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
# MAGIC `03_run_storyteller.py` to write the narrative report:
# MAGIC *"The Shape of Global Progress"*
