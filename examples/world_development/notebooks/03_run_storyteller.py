# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # World Development — Narrative Report Pipeline
# MAGIC
# MAGIC This notebook runs the **StoryTellerAgent** to transform research findings
# MAGIC into a narrative report: *"The Shape of Global Progress"*
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `01_run_engineer.py` first — Delta tables must exist
# MAGIC 2. Run `02_run_scientist.py` first — findings, charts, and tables must exist
# MAGIC 3. Install versifai: `pip install versifai` (or from source)
# MAGIC 4. Create a `.env` file with your LLM API key (see Setup cell below)
# MAGIC
# MAGIC ## Before You Start
# MAGIC 1. **Update `CATALOG` and `SCHEMA`** in `storyteller_config.py`
# MAGIC 2. **Update the `load_dotenv()` path** in the Setup cell to point to your `.env` file
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC The agent executes 5 phases:
# MAGIC 1. **Inventory** — Scan findings, charts, tables from the scientist
# MAGIC 2. **Evidence Evaluation** — Score finding strength (DEFINITIVE to WEAK)
# MAGIC 3. **Section Writing** — Write each of 8 sections using curated evidence
# MAGIC 4. **Coherence Pass** — Fix transitions, consistency, tone
# MAGIC 5. **Finalization** — Assemble TOC, bibliography, export Markdown

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

from examples.world_development.storyteller_config import WORLD_DEVELOPMENT_STORY

cfg = WORLD_DEVELOPMENT_STORY

logger.info("Narrative: %s", cfg.name)
logger.info("Sections: %d", len(cfg.narrative_sections))
logger.info("Reading from: %s", cfg.research_results_path)
logger.info("Writing to: %s", cfg.narrative_output_path)

# COMMAND ----------

# ── Verify Prerequisites ─────────────────────────────────────────────
# The storyteller resolves the scientist's run directory automatically
# via its dependency config. Check that findings exist.

from versifai.core.run_manager import resolve_dependency

try:
    scientist_path = resolve_dependency(cfg.dependencies[0])
    logger.info("Found scientist run at: %s", scientist_path)
except (FileNotFoundError, IndexError) as e:
    logger.warning("Scientist run not found: %s — run 02_run_scientist.py first!", e)

for subdir in ["findings.json", "charts", "tables"]:
    path = f"{cfg.research_results_path}/{subdir}"
    try:
        info = dbutils.fs.ls(path)
        print(
            f"  Found: {path} ({len(info)} items)" if isinstance(info, list) else f"  Found: {path}"
        )
    except Exception:
        print(f"  MISSING: {path} — run 02_run_scientist.py first!")

# COMMAND ----------

# ── Create the Agent ─────────────────────────────────────────────────

from versifai.story_agents.storyteller.agent import StoryTellerAgent

agent = StoryTellerAgent(cfg=cfg, dbutils=dbutils)

logger.info("Run ID: %s", agent._run_id)
logger.info("Narrative run path: %s", agent._narrative_run_path)
logger.info("Reading scientist results from: %s", agent._research_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option A: Run the Full Pipeline

# COMMAND ----------

results = agent.run()
logger.info("Narrative complete. Result: %s", results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option B: Run Specific Sections or Editor Pass

# COMMAND ----------

# # Rewrite specific sections
# results = agent.run_sections(sections=[0, 3, 5])  # Hook, Healthcare, Convergence

# # Run editor pass with custom instructions
# results = agent.run_editor(
#     instructions="Strengthen the transition from the Preston Curve section "
#     "into the Education section. The logical flow needs work."
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the Final Report

# COMMAND ----------

report_path = f"{agent._narrative_run_path}/{cfg.output_format.filename}"
try:
    report_text = dbutils.fs.head(report_path, 10000)
    print(report_text)
except Exception as e:
    print(f"Report not found: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC You've now run the complete Versifai pipeline on real World Bank data:
# MAGIC
# MAGIC | Stage | Notebook | Agent | Output |
# MAGIC |-------|----------|-------|--------|
# MAGIC | 0. Download | `00_download_data.py` | (script) | 6 ZIP files in Volume |
# MAGIC | 1. Ingest | `01_run_engineer.py` | DataEngineerAgent | 7 Delta tables in Unity Catalog |
# MAGIC | 2. Analyze | `02_run_scientist.py` | DataScientistAgent | findings.json, charts/, tables/ |
# MAGIC | 3. Narrate | `03_run_storyteller.py` | StoryTellerAgent | world_development_report.md |
# MAGIC
# MAGIC The report is at:
# MAGIC `{narrative_output_path}/world_development_report.md`
