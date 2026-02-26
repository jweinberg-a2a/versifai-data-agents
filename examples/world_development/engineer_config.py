"""
Engineer config for the World Development Indicators project.

This config tells the DataEngineerAgent everything it needs to know about
ingesting World Bank Development Indicator ZIP archives from a Databricks
Volume, extracting and profiling the wide-format CSVs, pivoting to long
format, and loading into Delta tables in Unity Catalog.

Data source: World Bank Open Data API
    https://data.worldbank.org/indicator
    Download format: ZIP archives containing wide-format CSVs

Usage in a Databricks notebook::

    from examples.world_development.engineer_config import WORLD_DEVELOPMENT
    agent = DataEngineerAgent(cfg=WORLD_DEVELOPMENT, dbutils=dbutils)
    agent.run()
"""

from __future__ import annotations

from versifai.data_agents.engineer.config import (
    AlternativeKeyConfig,
    DataSourceHint,
    JoinKeyConfig,
    MetadataColumnConfig,
    ProjectConfig,
    SourceFileHint,
    SourceProcessingHint,
)

# ═══════════════════════════════════════════════════════════════════════
# USER SETTINGS — Update these to match your Databricks environment
# ═══════════════════════════════════════════════════════════════════════

CATALOG = "my_catalog"
SCHEMA = "world_development"

# ═══════════════════════════════════════════════════════════════════════
# Wide-format pivot notes — shared across all indicator source hints
# ═══════════════════════════════════════════════════════════════════════

_PIVOT_NOTES = (
    "The main data CSV (filename starts with 'API_') has 4 metadata rows at the "
    "top — skip them. The header row has columns: 'Country Name', 'Country Code', "
    "'Indicator Name', 'Indicator Code', then one column per year (1960, 1961, ..., 2023). "
    "This is WIDE FORMAT — each row is one country with year-values spread across columns. "
    "You MUST pivot to LONG FORMAT with columns: country_name, country_code, year, and the "
    "indicator value column. Use pandas melt() or SQL UNPIVOT. Drop rows where the value "
    "is empty or NaN (they represent missing data)."
)

_METADATA_NOTES = (
    "The 'Metadata_Country_*' CSV has country classifications: Region, IncomeGroup, "
    "SpecialNotes, TableName. Load this ONCE (it is identical across all 6 ZIPs). "
    "The 'Metadata_Indicator_*' CSV describes the indicator — you can skip this file."
)


# ═══════════════════════════════════════════════════════════════════════
# Project config — one per project
# ═══════════════════════════════════════════════════════════════════════

WORLD_DEVELOPMENT = ProjectConfig(
    # ── Identity ────────────────────────────────────────────────
    name="World Development Indicators: The Shape of Global Progress",
    description=(
        "Ingestion and harmonization of 6 World Bank Development Indicators "
        "(GDP per capita, life expectancy, school enrollment, health expenditure, "
        "CO2 emissions, population) covering 217 countries from 1960 to 2023. "
        "Data arrives as wide-format CSVs inside ZIP archives downloaded from "
        "the World Bank Open Data API."
    ),
    domain_expertise="Development Economics, Global Health, Environmental Science",
    analyst_specialty="International Development Data Analyst",
    # ── Unity Catalog target ────────────────────────────────────
    catalog=CATALOG,
    schema=SCHEMA,
    # ── Data source location ────────────────────────────────────
    #    Run 00_download_data.py first to populate these Volumes.
    volume_path=f"/Volumes/{CATALOG}/{SCHEMA}/raw_data",
    staging_path=f"/Volumes/{CATALOG}/{SCHEMA}/staging",
    # ── Primary join key ────────────────────────────────────────
    #    ISO 3166-1 alpha-3 country code — the canonical key
    #    linking all indicator tables and country metadata.
    join_key=JoinKeyConfig(
        column_name="country_code",
        data_type="STRING",
        description=(
            "ISO 3166-1 alpha-3 country code (e.g., 'USA', 'GBR', 'CHN'). "
            "Present in every World Bank CSV as 'Country Code'. This is the "
            "canonical key for joining all indicator tables."
        ),
        width=3,
        validation_rule="Must be exactly 3 uppercase letters matching [A-Z]{3}",
        expected_entity_count=217,
        related_columns=[
            {
                "name": "country_name",
                "description": "Full country name (e.g., 'United States')",
                "required": True,
            },
        ],
    ),
    # ── Alternative keys ────────────────────────────────────────
    #    Region and income group enable stratified analysis.
    alternative_keys=[
        AlternativeKeyConfig(
            column_name="region",
            description="World Bank region for regional aggregations",
            data_type="STRING",
            grain="region",
        ),
        AlternativeKeyConfig(
            column_name="income_group",
            description=(
                "World Bank income classification: "
                "Low income, Lower middle income, Upper middle income, High income"
            ),
            data_type="STRING",
            grain="income_group",
        ),
    ],
    # ── Metadata columns ────────────────────────────────────────
    metadata_columns=[
        MetadataColumnConfig(
            name="source_file_name",
            data_type="STRING",
            description="Original filename the row was loaded from",
        ),
        MetadataColumnConfig(
            name="indicator_code",
            data_type="STRING",
            description="World Bank indicator code (e.g., NY.GDP.PCAP.CD)",
        ),
        MetadataColumnConfig(
            name="load_timestamp",
            data_type="TIMESTAMP",
            description="When this row was loaded into the catalog",
            nullable=False,
        ),
    ],
    # ── Known data sources ──────────────────────────────────────
    known_sources=[
        DataSourceHint(
            name="GDP Per Capita",
            description=(
                "GDP per capita in current US dollars (NY.GDP.PCAP.CD). "
                "World Bank national accounts data and OECD National Accounts."
            ),
            keywords=["GDP", "NY.GDP.PCAP.CD", "gdp_per_capita", "national_accounts"],
        ),
        DataSourceHint(
            name="Life Expectancy",
            description=(
                "Life expectancy at birth in years (SP.DYN.LE00.IN). "
                "UN Population Division World Population Prospects."
            ),
            keywords=["life_expectancy", "SP.DYN.LE00.IN", "mortality", "longevity"],
        ),
        DataSourceHint(
            name="School Enrollment",
            description=(
                "Net primary school enrollment rate (SE.PRM.NENR). UNESCO Institute for Statistics."
            ),
            keywords=["school_enrollment", "SE.PRM.NENR", "education", "primary"],
        ),
        DataSourceHint(
            name="Health Expenditure",
            description=(
                "Current health expenditure per capita in US dollars (SH.XPD.CHEX.PC.CD). "
                "World Health Organization Global Health Expenditure database."
            ),
            keywords=[
                "health_expenditure",
                "SH.XPD.CHEX.PC.CD",
                "healthcare",
                "spending",
            ],
        ),
        DataSourceHint(
            name="CO2 Emissions",
            description=(
                "CO2 emissions in metric tons per capita (EN.ATM.CO2E.PC). "
                "Climate Watch and World Resources Institute (CAIT)."
            ),
            keywords=["co2_emissions", "EN.ATM.CO2E.PC", "carbon", "climate"],
        ),
        DataSourceHint(
            name="Population",
            description=(
                "Total population count (SP.POP.TOTL). "
                "UN Population Division, Census reports, and Eurostat."
            ),
            keywords=["population", "SP.POP.TOTL", "census", "demographics"],
        ),
    ],
    # ── Documentation URLs ──────────────────────────────────────
    documentation_urls={
        "world_bank_indicators": [
            "https://data.worldbank.org/indicator",
        ],
        "world_bank_api": [
            "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation",
        ],
    },
    # ── Source processing hints ──────────────────────────────────
    #    Each ZIP archive is a source. The first one also loads
    #    the shared country metadata table.
    source_processing_hints=[
        SourceProcessingHint(
            source_pattern="gdp_per_capita",
            description="GDP per capita (current US$) — World Bank indicator NY.GDP.PCAP.CD",
            multi_table=True,
            files=[
                SourceFileHint(
                    file_pattern="API_NY.GDP.PCAP.CD",
                    target_table="gdp_per_capita",
                    description="GDP per capita by country and year",
                    used_in="theme_0, theme_1, theme_2, theme_4, theme_5",
                ),
                SourceFileHint(
                    file_pattern="Metadata_Country",
                    target_table="country_metadata",
                    description="Country classifications: region, income group, special notes",
                    used_in="All themes (join for stratification)",
                ),
            ],
            notes=f"{_PIVOT_NOTES}\n\n{_METADATA_NOTES}",
        ),
        SourceProcessingHint(
            source_pattern="life_expectancy",
            description="Life expectancy at birth (years) — World Bank indicator SP.DYN.LE00.IN",
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="API_SP.DYN.LE00.IN",
                    target_table="life_expectancy",
                    description="Life expectancy by country and year",
                    used_in="theme_0, theme_1, theme_3, theme_5",
                ),
            ],
            notes=(
                f"{_PIVOT_NOTES}\n\n"
                "Country metadata is loaded from the gdp_per_capita source — "
                "skip the Metadata_Country file in this ZIP."
            ),
        ),
        SourceProcessingHint(
            source_pattern="school_enrollment",
            description="Primary school enrollment (% net) — World Bank indicator SE.PRM.NENR",
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="API_SE.PRM.NENR",
                    target_table="school_enrollment",
                    description="Net primary enrollment rate by country and year",
                    used_in="theme_0, theme_2",
                ),
            ],
            notes=(
                f"{_PIVOT_NOTES}\n\n"
                "Enrollment can exceed 100% due to late enrollees (UNESCO methodology). "
                "Country metadata is loaded from the gdp_per_capita source — "
                "skip the Metadata_Country file."
            ),
        ),
        SourceProcessingHint(
            source_pattern="health_expenditure",
            description=(
                "Health expenditure per capita (US$) — World Bank indicator SH.XPD.CHEX.PC.CD"
            ),
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="API_SH.XPD.CHEX.PC.CD",
                    target_table="health_expenditure",
                    description="Health expenditure per capita by country and year",
                    used_in="theme_0, theme_3",
                ),
            ],
            notes=(
                f"{_PIVOT_NOTES}\n\n"
                "Health expenditure data begins around year 2000 — earlier years are mostly "
                "missing. Country metadata is loaded from the gdp_per_capita source — "
                "skip the Metadata_Country file."
            ),
        ),
        SourceProcessingHint(
            source_pattern="co2_emissions",
            description=(
                "CO2 emissions per capita (metric tons) — World Bank indicator EN.ATM.CO2E.PC"
            ),
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="API_EN.ATM.CO2E.PC",
                    target_table="co2_emissions",
                    description="CO2 emissions per capita by country and year",
                    used_in="theme_0, theme_4",
                ),
            ],
            notes=(
                f"{_PIVOT_NOTES}\n\n"
                "CO2 data typically lags by 2-3 years. "
                "Country metadata is loaded from the gdp_per_capita source — "
                "skip the Metadata_Country file."
            ),
        ),
        SourceProcessingHint(
            source_pattern="population",
            description="Total population — World Bank indicator SP.POP.TOTL",
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="API_SP.POP.TOTL",
                    target_table="population",
                    description="Total population by country and year",
                    used_in="theme_0, theme_1 (bubble sizing)",
                ),
            ],
            notes=(
                f"{_PIVOT_NOTES}\n\n"
                "Country metadata is loaded from the gdp_per_capita source — "
                "skip the Metadata_Country file."
            ),
        ),
    ],
    # ── Column naming ────────────────────────────────────────────
    naming_convention="snake_case",
    naming_rules=(
        "All columns must be snake_case. Country columns: country_name, country_code. "
        "After pivoting wide format: 'year' (INT) column and a value column named "
        "after the indicator (e.g., gdp_per_capita_usd, life_expectancy_years). "
        "Metadata columns: region, income_group."
    ),
    column_naming_examples=(
        "'Country Name' -> country_name\n"
        "'Country Code' -> country_code\n"
        "'Indicator Name' -> (drop — redundant, each table is one indicator)\n"
        "'Indicator Code' -> indicator_code (metadata column)\n"
        "Year columns (1960..2023) -> pivot to: year (INT) + value column\n"
        "'Region' -> region\n"
        "'IncomeGroup' -> income_group\n"
        "'SpecialNotes' -> special_notes"
    ),
    grain_detection_guidance=(
        "Country-level: Look for 'Country Code' or 3-letter ISO codes (e.g., USA, GBR)\n"
        "Country-year: After pivoting wide-format data, grain is (country_code, year)\n"
        "Region: If aggregating, use 'Region' from country metadata\n"
        "Income group: If aggregating, use 'IncomeGroup' from country metadata\n"
        "WARNING: World Bank data includes aggregate entities (e.g., 'World', "
        "'East Asia & Pacific') alongside individual countries. Country codes for "
        "aggregates contain digits or are non-standard — filter them during analysis."
    ),
    geographic_grain="country",
    grain_description="Individual country as classified by the World Bank (217 economies)",
)
