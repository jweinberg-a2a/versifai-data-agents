# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Silly Weather — Data Engineering Pipeline
# MAGIC
# MAGIC This notebook runs the **DataEngineerAgent** to ingest raw weather, duck,
# MAGIC and ice cream data from a Databricks Volume into Delta tables in Unity Catalog.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Upload your raw CSV/JSON files to the Volume path defined in the config
# MAGIC 2. Install versifai: `pip install versifai` (or from source)
# MAGIC 3. Set your LLM API key in the cluster environment variables
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC 1. **Discovery** — Scans the Volume for raw files
# MAGIC 2. **Profiling** — Profiles each file to understand structure
# MAGIC 3. **Schema Design** — Designs Delta table schemas
# MAGIC 4. **Transform & Load** — Cleans and loads data into Unity Catalog
# MAGIC 5. **Quality Check** — Validates all loaded tables

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

# Verify API key is available
assert os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
    "OPENAI_API_KEY"
), "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your cluster env vars"

# COMMAND ----------

# ── Load Config ──────────────────────────────────────────────────────
#
# The config defines EVERYTHING the engineer needs to know:
#   - Where raw data lives (volume_path)
#   - Where to write tables (catalog.schema)
#   - How to join tables (join_key)
#   - What data sources to expect (known_sources)
#   - How to process multi-file sources (source_processing_hints)
#
# You can customize any field by editing engineer_config.py or by
# overriding fields inline below.

from examples.silly_weather.engineer_config import SILLY_WEATHER

cfg = SILLY_WEATHER

# ── Optional: Override config fields for your environment ────────────
# cfg.catalog = "your_catalog"
# cfg.schema = "your_schema"
# cfg.volume_path = "/Volumes/your_catalog/your_schema/your_volume"

logger.info("Project: %s", cfg.name)
logger.info("Target: %s.%s", cfg.catalog, cfg.schema)
logger.info("Volume: %s", cfg.volume_path)

# COMMAND ----------

# ── Create the Agent ─────────────────────────────────────────────────
#
# The DataEngineerAgent receives:
#   - cfg: The ProjectConfig with all domain knowledge
#   - dbutils: Databricks utilities (for display, widgets, etc.)
#
# The agent is GENERIC — all domain-specific behavior comes from the config.

from versifai.data_agents.engineer.agent import DataEngineerAgent

agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Discovery & Ingestion
# MAGIC
# MAGIC The agent will:
# MAGIC 1. Scan the Volume for files
# MAGIC 2. Match files to known sources using `source_processing_hints`
# MAGIC 3. Profile each file to understand columns, types, and distributions
# MAGIC 4. Design schemas and CREATE TABLE DDL
# MAGIC 5. Transform and load data into Delta tables

# COMMAND ----------

# ── Run the main pipeline ────────────────────────────────────────────
# This runs Discovery → Profiling → Schema Design → Transform & Load.
# The agent will call ask_human() if it encounters ambiguous decisions.

results = agent.run(source_path=cfg.volume_path)

logger.info("Pipeline complete. Result: %s", results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Column Renaming (Optional)
# MAGIC
# MAGIC Standardize column names to snake_case across all tables.

# COMMAND ----------

rename_results = agent.run_rename()
logger.info("Rename complete. Result: %s", rename_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Build Data Catalog
# MAGIC
# MAGIC Creates a `data_catalog` table describing all tables, columns, and types.

# COMMAND ----------

catalog_results = agent.run_catalog()
logger.info("Catalog complete. Result: %s", catalog_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4: Quality Check
# MAGIC
# MAGIC The agent validates all tables: row counts, NULL rates, join key coverage,
# MAGIC and cross-table consistency.

# COMMAND ----------

quality_results = agent.run_quality_check()
logger.info("Quality check complete. Result: %s", quality_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Results
# MAGIC
# MAGIC List all tables created in our schema.

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {cfg.catalog}.{cfg.schema}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC With data loaded, run the **DataScientistAgent** using `run_scientist.py`
# MAGIC to analyze whether ducks actually predict rain.
