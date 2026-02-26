"""
Engineer config for the Silly Weather project.

This config tells the DataEngineerAgent everything it needs to know about
ingesting, profiling, and loading weather data from a Databricks Volume
into Delta tables in Unity Catalog.

Usage in a Databricks notebook::

    from examples.silly_weather.engineer_config import SILLY_WEATHER
    agent = DataEngineerAgent(cfg=SILLY_WEATHER, dbutils=dbutils)
    agent.run(source_path=SILLY_WEATHER.volume_path)
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
# Project config — one per project
# ═══════════════════════════════════════════════════════════════════════

SILLY_WEATHER = ProjectConfig(
    # ── Identity ────────────────────────────────────────────────
    name="Silly Weather: Do Ducks Predict Rain Better Than Meteorologists?",
    description=(
        "An absurd-but-illustrative analysis of weather station data, "
        "duck pond observation logs, and ice cream sales records to "
        "investigate the pressing question: can ducks predict rain?"
    ),
    domain_expertise="Meteorology (amateur), Ornithology (dubious), Ice Cream Science (serious)",
    analyst_specialty="Waterfowl-Atmospheric Pattern Recognition",
    # ── Unity Catalog target ────────────────────────────────────
    #    Replace these with YOUR catalog and schema names.
    catalog="my_catalog",
    schema="silly_weather",
    # ── Data source location ────────────────────────────────────
    #    This is the Volume path where raw data files live.
    #    Upload your CSVs here before running the engineer.
    volume_path="/Volumes/my_catalog/silly_weather/raw_data",
    staging_path="/Volumes/my_catalog/silly_weather/staging",
    # ── Primary join key ────────────────────────────────────────
    #    The canonical column used to join all tables together.
    join_key=JoinKeyConfig(
        column_name="station_id",
        data_type="STRING",
        description=(
            "NOAA weather station identifier. Every table must include this "
            "column so we can join weather observations with duck sightings "
            "and ice cream sales at the same location."
        ),
        width=11,
        validation_rule="Must match pattern 'USW\\d{8}' or 'USC\\d{8}'",
        expected_entity_count=500,
        related_columns=[
            {
                "name": "station_name",
                "description": "Human-readable station name (e.g., 'CHICAGO O HARE INTL AP')",
                "required": False,
            },
            {
                "name": "state_code",
                "description": "Two-letter US state code",
                "required": True,
            },
            {
                "name": "latitude",
                "description": "Station latitude in decimal degrees",
                "required": False,
            },
            {
                "name": "longitude",
                "description": "Station longitude in decimal degrees",
                "required": False,
            },
        ],
    ),
    # ── Alternative keys ────────────────────────────────────────
    #    Some tables have a different grain than station_id.
    alternative_keys=[
        AlternativeKeyConfig(
            column_name="county_fips",
            description="County FIPS code for joining with census/geographic data",
            data_type="STRING",
            grain="county",
        ),
        AlternativeKeyConfig(
            column_name="pond_id",
            description="Unique duck pond identifier for pond-level observations",
            data_type="STRING",
            grain="pond",
        ),
    ],
    # ── Metadata columns ────────────────────────────────────────
    #    These columns are added to EVERY table automatically.
    metadata_columns=[
        MetadataColumnConfig(
            name="source_file_name",
            data_type="STRING",
            description="Original filename the row was loaded from",
        ),
        MetadataColumnConfig(
            name="source_year",
            data_type="INT",
            description="The observation year extracted from the data",
        ),
        MetadataColumnConfig(
            name="load_timestamp",
            data_type="TIMESTAMP",
            description="When this row was loaded into the catalog",
            nullable=False,
        ),
    ],
    # ── Known data sources ──────────────────────────────────────
    #    Hints that help the agent recognize files in the Volume.
    known_sources=[
        DataSourceHint(
            name="NOAA Daily Weather Summaries",
            description=(
                "Daily temperature (TMAX, TMIN, TAVG), precipitation (PRCP), "
                "snowfall (SNOW), and wind speed (AWND) for US weather stations."
            ),
            keywords=["GHCND", "daily_summary", "NOAA", "weather"],
        ),
        DataSourceHint(
            name="Duck Pond Observation Logs",
            description=(
                "Citizen-science duck pond logs recording daily quack frequency, "
                "feather-fluffing intensity, pond departure time, and formation "
                "flight patterns. Totally real data."
            ),
            keywords=["duck", "pond", "quack", "observation"],
        ),
        DataSourceHint(
            name="Ice Cream Sales Records",
            description=(
                "Daily ice cream sales by flavor and location from the National "
                "Ice Cream Retailers Association (NICRA). Definitely not made up."
            ),
            keywords=["ice_cream", "sales", "flavor", "scoop"],
        ),
        DataSourceHint(
            name="Meteorologist Forecast Accuracy",
            description=(
                "Historical forecast accuracy scores from local TV meteorologists. "
                "Includes confidence levels and whether they used a green screen."
            ),
            keywords=["forecast", "accuracy", "meteorologist", "prediction"],
        ),
    ],
    # ── Documentation URLs ──────────────────────────────────────
    documentation_urls={
        "noaa_data_docs": [
            "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
        ],
        "duck_behavior": [
            "https://en.wikipedia.org/wiki/Duck",
        ],
    },
    # ── Source processing hints ──────────────────────────────────
    #    Detailed instructions for multi-file sources.
    source_processing_hints=[
        SourceProcessingHint(
            source_pattern="noaa_weather",
            description="NOAA daily weather summaries — one CSV per year",
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="daily_summary",
                    target_table="silver_daily_weather",
                    description="Daily weather observations (temp, precip, wind, snow)",
                    used_in="All themes",
                ),
            ],
            notes=(
                "Files are one-per-year CSVs. Combine all years into a single table. "
                "Temperature values are in tenths of degrees Celsius — divide by 10. "
                "Precipitation is in tenths of mm."
            ),
        ),
        SourceProcessingHint(
            source_pattern="duck_observations",
            description="Duck pond observation logs with behavioral metrics",
            multi_table=True,
            files=[
                SourceFileHint(
                    file_pattern="quack_frequency",
                    target_table="silver_quack_frequency",
                    description="Daily quack counts per pond per hour",
                    used_in="theme_1, theme_2",
                ),
                SourceFileHint(
                    file_pattern="feather_index",
                    target_table="silver_feather_fluffing",
                    description="Daily feather fluffing intensity (0-10 scale)",
                    used_in="theme_2, theme_3",
                ),
                SourceFileHint(
                    file_pattern="formation_flights",
                    target_table="silver_formation_flights",
                    description="Daily V-formation flight sightings and direction",
                    used_in="theme_4",
                ),
            ],
            notes=(
                "Duck observations are recorded by volunteer 'Quack Watchers' at each pond. "
                "pond_id maps to station_id via the pond_station_crosswalk file. "
                "Quack frequency is measured in QPH (quacks per hour)."
            ),
        ),
        SourceProcessingHint(
            source_pattern="ice_cream",
            description="Daily ice cream sales by location",
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="daily_sales",
                    target_table="silver_ice_cream_sales",
                    description="Scoops sold by flavor, location, and day",
                    used_in="theme_3, theme_5",
                ),
            ],
        ),
        SourceProcessingHint(
            source_pattern="forecast_accuracy",
            description="Meteorologist forecast accuracy records",
            multi_table=False,
            files=[
                SourceFileHint(
                    file_pattern="forecast",
                    target_table="silver_forecast_accuracy",
                    description="Daily forecast accuracy scores per meteorologist",
                    used_in="theme_5, theme_6",
                ),
            ],
        ),
    ],
    # ── Column naming ────────────────────────────────────────────
    naming_convention="snake_case",
    naming_rules=(
        "All columns must be snake_case. Temperature columns: temp_max_c, temp_min_c, "
        "temp_avg_c (Celsius, not tenths). Precipitation: precip_mm. "
        "Duck columns: quacks_per_hour, fluff_index, formation_direction."
    ),
    # ── Domain-specific column renaming examples ──────────────────
    column_naming_examples=(
        "`TMAX` → `temp_max_c` (daily max temp in Celsius)\n"
        "`PRCP` → `precip_mm` (daily precipitation in mm)\n"
        "`AWND` → `avg_wind_speed_mps` (average wind speed in m/s)\n"
        "`QPH` → `quacks_per_hour` (quack frequency per hour)\n"
        "`FFI` → `fluff_index` (feather fluffing intensity 0-10)"
    ),
    # ── Grain detection guidance ──────────────────────────────────
    grain_detection_guidance=(
        "Station-level: Look for station_id or NOAA station identifiers (USW/USC prefix)\n"
        "Pond-level: Look for pond_id identifiers\n"
        "County-level: Look for county_fips or FIPS codes\n"
        "If data has both station_id AND pond_id, it's a linkage/crosswalk table"
    ),
    # ── Geographic grain ────────────────────────────────────────
    geographic_grain="station",
    grain_description="US weather station with nearby duck pond (within 5 miles)",
)
