# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # World Development — Narrative Report Pipeline
# MAGIC
# MAGIC This notebook runs the **StoryTellerAgent** to transform research findings
# MAGIC into a narrative report: *"The Shape of Global Progress"*
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `01_run_engineer.py` first — silver tables must exist
# MAGIC 2. Run `02_run_scientist.py` first — findings, charts, and tables must exist
# MAGIC 3. Install versifai: `pip install versifai` (or from source)
# MAGIC 4. Set your LLM API key in the cluster environment variables
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC The agent executes 5 phases:
# MAGIC 1. **Inventory** — Scan findings, charts, tables from the scientist
# MAGIC 2. **Evidence Evaluation** — Score finding strength (DEFINITIVE to WEAK)
# MAGIC 3. **Section Writing** — Write each of 8 sections using curated evidence
# MAGIC 4. **Coherence Pass** — Fix transitions, consistency, tone
# MAGIC 5. **Finalization** — Assemble TOC, bibliography, export Markdown

# COMMAND ----------

# MAGIC %pip install versifai --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── Setup ────────────────────────────────────────────────────────────

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("world_development")

assert os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"), (
    "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your cluster env vars"
)

# COMMAND ----------

# ── Load Config ──────────────────────────────────────────────────────

from examples.world_development.storyteller_config import WORLD_DEVELOPMENT_STORY

cfg = WORLD_DEVELOPMENT_STORY

# ── Optional: Override for your environment ──────────────────────────
# cfg.project.catalog = "your_catalog"
# cfg.project.schema = "your_schema"
# cfg.research_results_path = "/Volumes/your_catalog/your_schema/results"
# cfg.narrative_output_path = "/Volumes/your_catalog/your_schema/narrative"

logger.info("Narrative: %s", cfg.name)
logger.info("Sections: %d", len(cfg.narrative_sections))
logger.info("Reading from: %s", cfg.research_results_path)
logger.info("Writing to: %s", cfg.narrative_output_path)

# COMMAND ----------

# ── Verify Prerequisites ─────────────────────────────────────────────

findings_path = f"{cfg.research_results_path}/findings.json"
charts_dir = f"{cfg.research_results_path}/charts"
tables_dir = f"{cfg.research_results_path}/tables"

for path in [findings_path, charts_dir, tables_dir]:
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

report_path = f"{cfg.narrative_output_path}/{cfg.output_format.filename}"
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
