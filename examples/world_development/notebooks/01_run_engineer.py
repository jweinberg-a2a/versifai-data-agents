# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # World Development — Data Engineering Pipeline
# MAGIC
# MAGIC This notebook runs the **DataEngineerAgent** to ingest World Bank
# MAGIC Development Indicator ZIP archives from a Databricks Volume into
# MAGIC Delta tables in Unity Catalog.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `00_download_data.py` first — 6 ZIP files must be in the Volume
# MAGIC 2. Install versifai: `pip install versifai` (or from source)
# MAGIC 3. Set your LLM API key in the cluster environment variables
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC 1. **Discovery** — Scans the Volume for ZIP archives
# MAGIC 2. **Extraction** — Unpacks each ZIP to find data and metadata CSVs
# MAGIC 3. **Profiling** — Profiles each CSV (skipping 4 metadata rows)
# MAGIC 4. **Schema Design** — Designs Delta table schemas, handles wide-to-long pivot
# MAGIC 5. **Transform & Load** — Pivots wide format to long, loads into Unity Catalog
# MAGIC 6. **Quality Check** — Validates all loaded tables

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
#
# The config defines EVERYTHING the engineer needs to know:
#   - Where raw ZIP files live (volume_path)
#   - Where to write tables (catalog.schema)
#   - How to join tables (country_code)
#   - What indicators to expect (known_sources)
#   - How to handle wide-format CSVs (source_processing_hints)

from examples.world_development.engineer_config import WORLD_DEVELOPMENT

cfg = WORLD_DEVELOPMENT

# ── Optional: Override config fields for your environment ────────────
# cfg.catalog = "your_catalog"
# cfg.schema = "your_schema"
# cfg.volume_path = "/Volumes/your_catalog/your_schema/your_volume"

logger.info("Project: %s", cfg.name)
logger.info("Target: %s.%s", cfg.catalog, cfg.schema)
logger.info("Volume: %s", cfg.volume_path)

# COMMAND ----------

# ── Create the Agent ─────────────────────────────────────────────────

from versifai.data_agents.engineer.agent import DataEngineerAgent

agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Discovery & Ingestion
# MAGIC
# MAGIC The agent will:
# MAGIC 1. Scan the Volume for ZIP archives
# MAGIC 2. Extract each ZIP to find the data CSV and metadata CSV
# MAGIC 3. Skip the 4 metadata rows at the top of each data CSV
# MAGIC 4. Pivot the wide-format data (years as columns) to long format
# MAGIC 5. Load each indicator into its own Delta table

# COMMAND ----------

results = agent.run()
logger.info("Pipeline complete. Result: %s", results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Column Renaming
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
# MAGIC Validates join key coverage, NULL rates, value ranges, and cross-table consistency.

# COMMAND ----------

quality_results = agent.run_quality_check()
logger.info("Quality check complete. Result: %s", quality_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Results
# MAGIC
# MAGIC List all tables created in our schema. You should see 7 silver tables
# MAGIC plus a `data_catalog` table.

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {cfg.catalog}.{cfg.schema}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC With data loaded, run the **DataScientistAgent** using
# MAGIC `02_run_scientist.py` to analyze global development patterns.
