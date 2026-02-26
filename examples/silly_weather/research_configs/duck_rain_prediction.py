"""
Research config: Do Ducks Predict Rain Better Than Meteorologists?

This config drives the DataScientistAgent through a 7-theme analysis
investigating whether duck behavior (quacking, feather fluffing,
formation flights) correlates with weather events better than
professional meteorologist forecasts.

Usage in a Databricks notebook::

    from examples.silly_weather.research_configs.duck_rain_prediction import DUCK_RAIN
    agent = DataScientistAgent(cfg=DUCK_RAIN, dbutils=dbutils)
    agent.run()
"""

from __future__ import annotations

from versifai.data_agents.engineer.config import ProjectConfig
from versifai.science_agents.scientist.config import (
    AnalysisTheme,
    ResearchConfig,
    ResearchReference,
    SilverDatasetSpec,
)

# ═══════════════════════════════════════════════════════════════════════
# Shared project config reference
# ═══════════════════════════════════════════════════════════════════════

_project = ProjectConfig(
    catalog="my_catalog",
    schema="silly_weather",
)

# ═══════════════════════════════════════════════════════════════════════
# Silver dataset specs — pre-joined analytical datasets
# ═══════════════════════════════════════════════════════════════════════

_silver_datasets = [
    SilverDatasetSpec(
        name="silver_weather_duck_daily",
        description=(
            "Daily weather observations joined with duck behavioral metrics. "
            "One row per station per day. This is the primary analytical dataset."
        ),
        source_tables=[
            "silver_daily_weather",
            "silver_quack_frequency",
            "silver_feather_fluffing",
        ],
        join_key="station_id",
        time_column="observation_date",
        data_notes=(
            "Join weather to duck data via station_id + observation_date. "
            "Duck data comes from the nearest pond (via pond_station_crosswalk). "
            "Expect ~15% NULL duck observations on weekdays (volunteers are at work)."
        ),
    ),
    SilverDatasetSpec(
        name="silver_duck_forecast_comparison",
        description=(
            "Side-by-side comparison of duck behavioral signals vs meteorologist "
            "forecast accuracy for the same station and day."
        ),
        source_tables=[
            "silver_quack_frequency",
            "silver_feather_fluffing",
            "silver_formation_flights",
            "silver_forecast_accuracy",
        ],
        join_key="station_id",
        time_column="observation_date",
        data_notes=(
            "Join all duck signals + forecast accuracy on station_id + observation_date. "
            "Meteorologist forecasts are for 'next day' — so lag by 1 day when joining."
        ),
    ),
    SilverDatasetSpec(
        name="silver_ice_cream_weather",
        description=(
            "Ice cream sales joined with weather data for confounding analysis. "
            "Tests whether the 'duck signal' is just a proxy for temperature."
        ),
        source_tables=[
            "silver_ice_cream_sales",
            "silver_daily_weather",
        ],
        join_key="station_id",
        time_column="observation_date",
        data_notes="Ice cream sales are aggregated to station-day level before joining.",
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Analysis themes — the 7-theme research arc
# ═══════════════════════════════════════════════════════════════════════

_themes = [
    # ── Theme 0: Data inventory & baseline ─────────────────────────
    AnalysisTheme(
        id="theme_0",
        title="The Quack Census",
        question=(
            "What does our data look like? How many stations, ponds, days, "
            "and ducks are we working with?"
        ),
        analysis_type="descriptive",
        sequence=0,
        required_tables=[
            "silver_daily_weather",
            "silver_quack_frequency",
            "silver_feather_fluffing",
            "silver_formation_flights",
            "silver_ice_cream_sales",
            "silver_forecast_accuracy",
        ],
        analysis_steps=[
            "Count stations, date ranges, and row counts for each silver table",
            "Profile distributions of key variables (temp, precip, quacks_per_hour, fluff_index)",
            "Check join coverage — what % of weather days have duck observations?",
            "Identify seasonal patterns in quacking (do ducks quack more in spring?)",
            "Build the baseline: average rain days per station per month",
        ],
        tables_to_produce=[
            "data_inventory_summary",
            "variable_distributions",
            "seasonal_quack_patterns",
        ],
        signature_visualization=(
            "A calendar heatmap showing quack frequency overlaid with precipitation "
            "events for the top 5 stations. Each cell = one day. Color = quacks_per_hour. "
            "Rain days get a tiny raindrop marker. Readers should immediately see if "
            "quacking spikes before rain."
        ),
        punchline="We have N stations, M ponds, and K million quacks to work with.",
    ),
    # ── Theme 1: The Quack-Rain Correlation ────────────────────────
    AnalysisTheme(
        id="theme_1",
        title="Quack Before the Storm",
        question=(
            "Is there a statistically significant correlation between quack "
            "frequency and next-day precipitation?"
        ),
        analysis_type="correlation",
        sequence=1,
        required_tables=[
            "silver_weather_duck_daily",
        ],
        analysis_steps=[
            "Compute Pearson and Spearman correlation between quacks_per_hour(t) and precip_mm(t+1)",
            "Run time-lagged cross-correlation for lags 0-3 days",
            "Test significance with permutation test (10,000 shuffles)",
            "Stratify by season — is the correlation stronger in spring vs winter?",
            "Control for temperature (partial correlation) — ducks may just be chatty on warm days",
        ],
        tables_to_produce=[
            "quack_precip_correlation_matrix",
            "lagged_cross_correlation_results",
            "seasonal_correlation_breakdown",
        ],
        signature_visualization=(
            "A lag-correlation plot (x-axis = lag days, y-axis = correlation coefficient) "
            "with confidence bands. One line per season. If ducks truly predict rain, "
            "we should see peak correlation at lag=1 (quacking today predicts rain tomorrow)."
        ),
        punchline=(
            "Quack frequency at lag-1 shows r=X.XX correlation with next-day rain "
            "(p=X.XXX) — [significant/not significant]."
        ),
        data_notes=(
            "Use silver_weather_duck_daily. Create lag columns: quacks_lag1, quacks_lag2, quacks_lag3. "
            "Precipitation threshold for 'rain day': precip_mm > 2.54 (0.1 inch)."
        ),
    ),
    # ── Theme 2: The Fluff Factor ──────────────────────────────────
    AnalysisTheme(
        id="theme_2",
        title="The Fluff Factor",
        question=(
            "Does feather-fluffing intensity predict storm severity better than "
            "quack frequency alone?"
        ),
        analysis_type="comparative",
        sequence=2,
        required_tables=[
            "silver_weather_duck_daily",
        ],
        analysis_steps=[
            "Build a 'Duck Barometer Index' (DBI) combining quack frequency + fluff index",
            "Compare DBI predictive power vs quack-only and fluff-only (AUC-ROC for rain/no-rain)",
            "Test if fluff_index correlates with storm SEVERITY (precip amount, not just occurrence)",
            "Check for threshold effects — is there a fluff_index cutoff that screams 'BIG STORM'?",
            "Run logistic regression: rain_tomorrow ~ quacks + fluff + quacks*fluff",
        ],
        tables_to_produce=[
            "duck_barometer_index_construction",
            "model_comparison_auc",
            "fluff_severity_correlation",
            "logistic_regression_coefficients",
        ],
        signature_visualization=(
            "ROC curves comparing three models: (1) quack-only, (2) fluff-only, "
            "(3) combined DBI. All three curves on one plot with AUC values in the legend. "
            "This is the money chart — if the combined model dominates, the Duck Barometer is real."
        ),
        punchline=(
            "The combined Duck Barometer Index achieves AUC=X.XX, outperforming "
            "quack-only (X.XX) and fluff-only (X.XX)."
        ),
    ),
    # ── Theme 3: The Ice Cream Confounder ──────────────────────────
    AnalysisTheme(
        id="theme_3",
        title="The Ice Cream Confounder",
        question=(
            "Is the duck-rain signal just a proxy for temperature? (Classic: "
            "ice cream sales correlate with drowning, but both are caused by hot weather.)"
        ),
        analysis_type="comparative",
        sequence=3,
        required_tables=[
            "silver_weather_duck_daily",
            "silver_ice_cream_weather",
        ],
        analysis_steps=[
            "Compute correlation: ice_cream_scoops ~ temp_avg_c (the known confounder)",
            "Compute partial correlation: quacks ~ next_day_rain | temp_avg_c",
            "Run mediation analysis: temp -> quacks -> rain (is temp the hidden variable?)",
            "Check if duck signal survives after controlling for temp, humidity, pressure",
            "Test Simpson's Paradox: does the correlation flip when stratified by temperature band?",
        ],
        tables_to_produce=[
            "confounder_analysis_summary",
            "partial_correlation_results",
            "mediation_path_coefficients",
            "simpsons_paradox_check",
        ],
        visualization_notes=(
            "A scatter plot matrix is tempting but low-value here. Instead, produce a "
            "coefficient waterfall chart showing how the quack-rain correlation changes "
            "as each confounder is added to the model."
        ),
        punchline=(
            "After controlling for temperature, the duck signal [survives/collapses] — "
            "partial r = X.XX (p = X.XXX)."
        ),
        data_notes=(
            "Join silver_ice_cream_weather with silver_weather_duck_daily on station_id + date. "
            "Use temperature bands: cold (<10C), mild (10-20C), warm (20-30C), hot (>30C)."
        ),
    ),
    # ── Theme 4: V-Formation Tornado Warning ───────────────────────
    AnalysisTheme(
        id="theme_4",
        title="V-Formation Tornado Warning",
        question=(
            "Do sudden spikes in V-formation flights predict severe weather events "
            "(thunderstorms, tornadoes, hail)?"
        ),
        analysis_type="comparative",
        sequence=4,
        required_tables=[
            "silver_formation_flights",
            "silver_daily_weather",
        ],
        analysis_steps=[
            "Define 'severe weather day': precip_mm > 25 OR wind_speed > 50 kph",
            "Compute formation_flight_count for 1-3 days before severe events vs normal days",
            "Run Mann-Whitney U test: flight counts (pre-severe) vs (pre-normal)",
            "Check if formation DIRECTION matters — do ducks flee toward safety?",
            "Build a confusion matrix: duck-warns-severe vs actual-severe (sensitivity/specificity)",
        ],
        tables_to_produce=[
            "severe_event_classification",
            "pre_event_flight_comparison",
            "formation_direction_analysis",
            "duck_tornado_confusion_matrix",
        ],
        signature_visualization=(
            "A before-after event study plot. X-axis: days relative to severe event (-3 to +3). "
            "Y-axis: mean formation flight count. One line for severe events, one for matched "
            "non-severe controls. Error bars are 95% CI. If ducks flee before storms, we should "
            "see a spike at day -1."
        ),
        punchline=(
            "Formation flights increase X% in the 24 hours before severe weather — "
            "sensitivity = X%, specificity = X%."
        ),
    ),
    # ── Theme 5: Ducks vs Meteorologists ───────────────────────────
    AnalysisTheme(
        id="theme_5",
        title="Duck vs Doppler",
        question=(
            "Who predicts rain better: ducks or professional meteorologists? The ultimate showdown."
        ),
        analysis_type="comparative",
        sequence=5,
        required_tables=[
            "silver_duck_forecast_comparison",
        ],
        analysis_steps=[
            "Compute meteorologist accuracy: % of days where forecast matched actual weather",
            "Compute duck accuracy: % of days where DBI > threshold matched actual rain",
            "Optimize DBI threshold for maximum F1 score",
            "Compare precision, recall, F1, and AUC between duck and meteorologist",
            "Stratify by difficulty: easy days (clear/heavy rain) vs hard days (borderline)",
            "Run McNemar's test: are the two predictors significantly different?",
        ],
        tables_to_produce=[
            "head_to_head_accuracy",
            "precision_recall_comparison",
            "difficulty_stratified_results",
            "mcnemars_test_result",
        ],
        signature_visualization=(
            "A paired precision-recall curve with duck predictions in orange and "
            "meteorologist predictions in blue. Include F1 iso-lines. The champion "
            "is whichever curve reaches farther into the upper-right corner."
        ),
        punchline=(
            "Ducks achieve F1=X.XX vs meteorologists at F1=X.XX — [ducks win / "
            "meteorologists win / it's a tie]."
        ),
    ),
    # ── Theme 6: The Grand Unified Duck Theory ─────────────────────
    AnalysisTheme(
        id="theme_6",
        title="The Grand Unified Duck Theory",
        question=(
            "Can we build a production-grade weather prediction model using "
            "ALL duck signals together, and does it add value beyond existing forecasts?"
        ),
        analysis_type="comparative",
        sequence=6,
        required_tables=[
            "silver_duck_forecast_comparison",
            "silver_weather_duck_daily",
        ],
        analysis_steps=[
            "Feature engineer all duck signals: quacks, fluff, formation count, formation direction",
            "Train gradient-boosted model: rain_tomorrow ~ all_duck_features",
            "Train ensemble model: rain_tomorrow ~ all_duck_features + meteorologist_forecast",
            "Compare AUC: duck-only vs meteorologist-only vs ensemble",
            "SHAP analysis: which duck feature matters most?",
            "Cross-validate with time-series split (no data leakage!)",
        ],
        tables_to_produce=[
            "feature_importance_shap",
            "model_comparison_cv_results",
            "ensemble_vs_individual_auc",
            "final_model_performance",
        ],
        signature_visualization=(
            "A SHAP beeswarm plot showing feature importance for the ensemble model. "
            "Each dot is one day. X-axis: SHAP value. Color: feature value (high/low). "
            "This reveals which duck behaviors drive the model's predictions."
        ),
        punchline=(
            "The ensemble model (ducks + meteorologist) achieves AUC=X.XX, beating "
            "the meteorologist alone by X points — proving ducks are a [valuable / "
            "useless / hilariously marginal] addition to weather forecasting."
        ),
        data_notes=(
            "CRITICAL: Use time-series cross-validation. Train on months 1..N, "
            "test on month N+1. Never shuffle — weather data has temporal autocorrelation."
        ),
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Research references
# ═══════════════════════════════════════════════════════════════════════

_references = [
    ResearchReference(
        title="Animal Behavior as Weather Predictors: A Meta-Analysis",
        url="https://en.wikipedia.org/wiki/Weather_lore",
        description="Survey of folklore and scientific evidence for animal-based weather prediction",
        keywords=["animal behavior", "weather prediction", "folklore"],
    ),
    ResearchReference(
        title="NOAA Global Historical Climatology Network Daily",
        url="https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
        description="Primary source for US daily weather observations",
        keywords=["NOAA", "GHCN-D", "weather stations"],
    ),
    ResearchReference(
        title="The Spurious Correlations Problem in Data Science",
        url="https://tylervigen.com/spurious-correlations",
        description="Why correlated data does not imply causation (see: ice cream and drowning)",
        keywords=["spurious correlation", "confounding", "causation"],
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Assembled config
# ═══════════════════════════════════════════════════════════════════════

DUCK_RAIN = ResearchConfig(
    name="Do Ducks Predict Rain Better Than Meteorologists?",
    thesis=(
        "Duck behavioral signals (quack frequency, feather-fluffing intensity, "
        "and V-formation flight patterns) contain genuine meteorological information "
        "that, when combined into a Duck Barometer Index, provides statistically "
        "significant 24-hour precipitation forecasts — possibly rivaling professional "
        "meteorologist accuracy for specific weather events."
    ),
    # ── New fields: agent identity & domain context ──────────────
    agent_role="Waterfowl-Atmospheric Research Scientist",
    domain_context=(
        "## Data Quirks\n\n"
        "- NOAA temperatures in raw files are in tenths of degrees Celsius — "
        "divide by 10 before analysis.\n"
        "- Precipitation is in tenths of mm in raw files.\n"
        "- Duck observation data has ~15% NULLs on weekdays (volunteer observers at work).\n"
        "- Quack frequency (QPH) typically ranges 0-200; values above 500 may be recording errors.\n"
        "- Fluff index ranges 0-10; values should be integers.\n"
        "- Station IDs follow pattern USW/USC + 8 digits.\n\n"
        "## Expected Value Ranges\n\n"
        "- Temperature: -40C to 50C\n"
        "- Precipitation: 0-300mm (daily)\n"
        "- Quacks per hour: 0-200 (typical), 0-500 (extreme)\n"
        "- Fluff index: 0-10 (integer scale)\n"
        "- Formation flight count: 0-50 per day\n"
    ),
    analysis_method_guidance={
        "simulation": (
            "**Simulation Analysis Approach**:\n"
            "1. Build the Duck Barometer Index (DBI) from validated components.\n"
            "2. Calibrate DBI thresholds against historical weather data.\n"
            "3. Run bootstrap resampling (1000 iterations) to quantify uncertainty.\n"
            "4. Compare DBI predictions against NOAA actual observations.\n"
            "5. Cross-validate with time-series split (never shuffle weather data).\n"
            "6. Document all assumptions and threshold choices."
        ),
    },
    visualization_guidance=(
        "OUTPUT PHILOSOPHY:\n"
        "- Tables are the primary analytical output. Every theme MUST produce well-formatted "
        "summary tables with statistics, p-values, and effect sizes.\n"
        "- Signature visualizations are the ONE dense, irreplaceable chart per theme.\n"
        "- Keep the tone fun but the statistics rigorous. Silly topic, serious methods.\n"
        "- Use duck-themed color palettes where possible (orange, teal, brown).\n"
    ),
    project=_project,
    results_volume_path="/Volumes/my_catalog/silly_weather/results",
    analysis_themes=_themes,
    silver_datasets=_silver_datasets,
    research_references=_references,
    max_turns=150,
    max_turns_per_phase=120,
    max_turns_per_theme=100,
    chart_style="seaborn-v0_8-whitegrid",
    chart_dpi=150,
    color_palette="Set2",
)
