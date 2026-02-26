# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # World Development — Download Data
# MAGIC
# MAGIC This notebook downloads 6 World Bank Development Indicator datasets as
# MAGIC ZIP archives and saves them to a Databricks Volume. These files will be
# MAGIC processed by the DataEngineerAgent in `01_run_engineer.py`.
# MAGIC
# MAGIC **Data source:** [World Bank Open Data API](https://data.worldbank.org/)
# MAGIC — free, no authentication required.
# MAGIC
# MAGIC **Indicators downloaded:**
# MAGIC | Code | Indicator |
# MAGIC |------|-----------|
# MAGIC | NY.GDP.PCAP.CD | GDP per capita (current US$) |
# MAGIC | SP.DYN.LE00.IN | Life expectancy at birth (years) |
# MAGIC | SE.PRM.NENR | School enrollment, primary (% net) |
# MAGIC | SH.XPD.CHEX.PC.CD | Health expenditure per capita (US$) |
# MAGIC | EN.ATM.CO2E.PC | CO2 emissions (metric tons per capita) |
# MAGIC | SP.POP.TOTL | Population total |
# MAGIC
# MAGIC ## Before You Start
# MAGIC
# MAGIC 1. **Replace `my_catalog`** below with your Unity Catalog name
# MAGIC 2. **Replace `world_development`** with your target schema name (or keep it)
# MAGIC 3. **Create a `.env` file** in your project root with your LLM API key
# MAGIC    (needed by notebooks 01-03, not this one):
# MAGIC    ```
# MAGIC    ANTHROPIC_API_KEY=sk-ant-...
# MAGIC    ```
# MAGIC
# MAGIC These same variables (`CATALOG`, `SCHEMA`) must also be updated in:
# MAGIC - `engineer_config.py`
# MAGIC - `research_configs/global_development.py`
# MAGIC - `storyteller_config.py`

# COMMAND ----------

# ── Configuration ──────────────────────────────────────────────────
#
# !! UPDATE THESE to match your Databricks workspace !!
#
CATALOG = "my_catalog"  # <-- Replace with your Unity Catalog name
SCHEMA = "world_development"  # <-- Replace with your target schema name
VOLUMES = ["raw_data", "staging", "results"]
RAW_DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"

# World Bank indicator codes → friendly filenames
INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "SP.DYN.LE00.IN": "life_expectancy",
    "SE.PRM.NENR": "school_enrollment",
    "SH.XPD.CHEX.PC.CD": "health_expenditure",
    "EN.ATM.CO2E.PC": "co2_emissions",
    "SP.POP.TOTL": "population",
}

# COMMAND ----------

# ── Create Volume if needed ────────────────────────────────────────

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
for vol in VOLUMES:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{vol}")
    print(f"  Volume ready: /Volumes/{CATALOG}/{SCHEMA}/{vol}")
print(f"\nAll {len(VOLUMES)} volumes created.")

# COMMAND ----------

# ── Download each indicator as a ZIP ───────────────────────────────
#
# The World Bank API returns ZIP archives containing:
#   1. API_{code}_DS2_en_csv_v2_{id}.csv  — main data (wide format)
#   2. Metadata_Country_{...}.csv         — country classifications
#   3. Metadata_Indicator_{...}.csv       — indicator description
#
# We save the raw ZIPs. The DataEngineerAgent will extract and process them.

import os
import urllib.request

for code, name in INDICATORS.items():
    dest = os.path.join(RAW_DATA_PATH, f"{name}.zip")

    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  Already exists: {name}.zip ({size:,d} bytes)")
        continue

    url = f"https://api.worldbank.org/v2/en/indicator/{code}?downloadformat=csv"
    print(f"  Downloading {name} ({code})...")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  Saved: {name}.zip ({size:,d} bytes)")

# COMMAND ----------

# ── Verify downloads ───────────────────────────────────────────────

print(f"\nFiles in {RAW_DATA_PATH}:\n")
total_bytes = 0
for f in sorted(os.listdir(RAW_DATA_PATH)):
    size = os.path.getsize(os.path.join(RAW_DATA_PATH, f))
    total_bytes += size
    print(f"  {f:40s} {size:>10,d} bytes")

print(f"\n  {'TOTAL':40s} {total_bytes:>10,d} bytes")
print(f"\nAll {len(INDICATORS)} indicators downloaded. Ready for 01_run_engineer.py.")
