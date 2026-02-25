# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Silly Weather — Narrative Report Pipeline
# MAGIC
# MAGIC This notebook runs the **StoryTellerAgent** to transform research findings
# MAGIC into a narrative report: *"Do Ducks Predict Rain? A Rigorous Investigation"*
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `run_engineer.py` first — silver tables must exist
# MAGIC 2. Run `run_scientist.py` first — findings, charts, and tables must exist
# MAGIC 3. Install versifai: `pip install versifai` (or from source)
# MAGIC 4. Set your LLM API key in the cluster environment variables
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC The agent executes 5 phases:
# MAGIC 1. **Inventory** — Scan findings, charts, tables; map coverage per section
# MAGIC 2. **Evidence Evaluation** — Score finding strength, curate bill of materials
# MAGIC 3. **Section Writing** — Write each of 8 sections using curated evidence
# MAGIC 4. **Coherence Pass** — Fix transitions, consistency, tone progression
# MAGIC 5. **Finalization** — Assemble TOC, bibliography, export final Markdown

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
# The StorytellerConfig defines:
#   - narrative_sections: 8 sections, each mapped to research themes
#   - style_guide: Voice, audience, tone, anti-patterns
#   - evidence_threshold: Rules for when evidence is strong enough to cite
#   - output_format: Markdown filename, TOC, appendices
#
# The StoryTeller READS from DataScientist outputs (findings.json,
# charts/, tables/, notes/). It does not re-run analysis — it
# synthesizes and writes the narrative.

from examples.silly_weather.storyteller_config import DUCK_STORY

cfg = DUCK_STORY

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
# Check that the DataScientist outputs exist before starting.

findings_path = f"{cfg.research_results_path}/findings.json"
charts_dir = f"{cfg.research_results_path}/charts"
tables_dir = f"{cfg.research_results_path}/tables"

for path in [findings_path, charts_dir, tables_dir]:
    try:
        info = dbutils.fs.ls(path)
        print(f"  Found: {path} ({len(info)} items)" if isinstance(info, list) else f"  Found: {path}")
    except Exception:
        print(f"  MISSING: {path} — run run_scientist.py first!")

# COMMAND ----------

# ── Create the Agent ─────────────────────────────────────────────────
#
# The StoryTellerAgent receives:
#   - cfg: The StorytellerConfig with narrative arc and style rules
#   - dbutils: Databricks utilities
#
# Tools available to the agent:
#   - read_findings: Parse findings.json from DataScientist
#   - read_chart / read_table: Load research outputs
#   - evaluate_evidence: Score finding strength (DEFINITIVE → WEAK)
#   - write_narrative: Compose and assemble report sections
#   - cite_source: Format citations and bibliography
#   - execute_sql: Live queries for numbers and context
#   - create_visualization: New charts if needed for narrative

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
# results = agent.run_sections(sections=[0, 4, 5])  # Hook, Showdown, Ensemble

# # Run editor pass with custom instructions
# results = agent.run_editor(
#     instructions="Tighten the transitions between sections 2 and 3. "
#     "The shift from Duck Barometer to Ice Cream Confounder feels abrupt."
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
# MAGIC You've now run the complete Versifai pipeline:
# MAGIC
# MAGIC | Stage | Notebook | Agent | Output |
# MAGIC |-------|----------|-------|--------|
# MAGIC | 1. Ingest | `run_engineer.py` | DataEngineerAgent | Delta tables in Unity Catalog |
# MAGIC | 2. Analyze | `run_scientist.py` | DataScientistAgent | findings.json, charts/, tables/ |
# MAGIC | 3. Narrate | `run_storyteller.py` | StoryTellerAgent | Markdown report |
# MAGIC
# MAGIC The report is at: `{narrative_output_path}/duck_weather_report.md`
