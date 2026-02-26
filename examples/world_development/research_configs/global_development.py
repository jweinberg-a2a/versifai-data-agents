"""
Research config for the World Development Indicators analysis.

This config defines the DataScientistAgent's research agenda: 6 analysis
themes investigating global development using World Bank data. The themes
cover descriptive analysis, correlation, comparative, regression, trend,
and confounding analysis — demonstrating the full breadth of the agent's
statistical toolkit.

Usage in a Databricks notebook::

    from examples.world_development.research_configs.global_development import (
        GLOBAL_DEVELOPMENT,
    )
    agent = DataScientistAgent(cfg=GLOBAL_DEVELOPMENT, dbutils=dbutils)
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
# Shared project reference (catalog/schema only — the engineer config
# has the full ProjectConfig; here we just need the catalog coordinates)
# ═══════════════════════════════════════════════════════════════════════

_project = ProjectConfig(
    catalog="my_catalog",
    schema="world_development",
)


# ═══════════════════════════════════════════════════════════════════════
# Silver datasets — pre-joined analytical tables the agent should build
# ═══════════════════════════════════════════════════════════════════════

_silver_datasets = [
    # ── Master panel: all 6 indicators + metadata ───────────────
    SilverDatasetSpec(
        name="silver_development_panel",
        description=(
            "Master country-year panel joining all 6 indicators with country metadata. "
            "One row per country per year. This is the primary analytical dataset."
        ),
        source_tables=[
            "silver_gdp_per_capita",
            "silver_life_expectancy",
            "silver_school_enrollment",
            "silver_health_expenditure",
            "silver_co2_emissions",
            "silver_population",
            "silver_country_metadata",
        ],
        join_key="country_code",
        time_column="year",
        data_notes=(
            "JOIN all 6 indicator tables on (country_code, year) using FULL OUTER JOIN "
            "to preserve all observations. LEFT JOIN silver_country_metadata on "
            "country_code only (metadata has no year dimension). Not every indicator is "
            "available for every country-year — expect NULLs. GDP coverage is best from "
            "1960; health expenditure starts ~2000; school enrollment is patchy before 1970."
        ),
    ),
    # ── Recent panel (2000-2023): best data coverage ────────────
    SilverDatasetSpec(
        name="silver_development_recent",
        description=(
            "Subset of the development panel restricted to 2000-2023, where most "
            "indicators have good coverage. Includes income_group and region for grouping."
        ),
        source_tables=["silver_development_panel"],
        join_key="country_code",
        time_column="year",
        data_notes=(
            "Filter silver_development_panel WHERE year >= 2000. Exclude aggregate entities "
            "— keep only individual countries (those with a non-NULL, non-empty region in "
            "silver_country_metadata). Aggregates like 'World', 'East Asia & Pacific', etc. "
            "have region = NULL or are classified differently."
        ),
    ),
    # ── Long-run panel: balanced for convergence analysis ───────
    SilverDatasetSpec(
        name="silver_development_long_run",
        description=(
            "Long-run panel (1960-2023) for convergence analysis. Only includes countries "
            "with GDP and life expectancy data in both 1960 and 2020 (balanced panel)."
        ),
        source_tables=[
            "silver_gdp_per_capita",
            "silver_life_expectancy",
            "silver_country_metadata",
        ],
        join_key="country_code",
        time_column="year",
        data_notes=(
            "Join GDP and life expectancy on (country_code, year). Filter to countries "
            "that have non-NULL gdp_per_capita AND life_expectancy for BOTH 1960 and 2020. "
            "This creates a balanced panel suitable for convergence analysis. Expect "
            "~100-120 countries to survive the filter."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Analysis themes — 6 research themes, one per analysis type
# ═══════════════════════════════════════════════════════════════════════

_themes = [
    # ── Theme 0: The Development Dashboard (descriptive) ────────
    AnalysisTheme(
        id="theme_0",
        title="The Development Dashboard",
        question=(
            "What does the global development landscape look like? How many countries, "
            "what time coverage, and what are the headline distributions?"
        ),
        analysis_type="descriptive",
        sequence=0,
        required_tables=[
            "silver_gdp_per_capita",
            "silver_life_expectancy",
            "silver_school_enrollment",
            "silver_health_expenditure",
            "silver_co2_emissions",
            "silver_population",
            "silver_country_metadata",
        ],
        analysis_steps=[
            "Count countries, year ranges, and row counts for each silver table",
            "Profile distributions of all 6 indicators for the most recent available year",
            "Compute summary statistics by income group (Low, Lower middle, Upper middle, High)",
            "Compute summary statistics by region (7 World Bank regions)",
            "Identify data coverage gaps: which indicator-country-year combinations are missing?",
            "Rank top 10 and bottom 10 countries by GDP per capita, life expectancy, and CO2",
        ],
        tables_to_produce=[
            "data_inventory_summary",
            "indicator_distributions_latest",
            "income_group_summary",
            "region_summary",
            "coverage_gap_matrix",
        ],
        signature_visualization=(
            "A small-multiples grid (2x3) showing the global distribution of each indicator "
            "for the most recent year. Each panel is a histogram colored by income group. "
            "This single figure gives the reader an instant sense of global inequality "
            "across all six dimensions."
        ),
        visualization_notes=(
            "Do NOT create 6 separate histograms — use a single custom visualization "
            "with subplots. Color consistently by income group across all panels."
        ),
        punchline=(
            "We have N countries across M years with K total observations. "
            "The gap between richest and poorest is X:1 in GDP, Y years in life expectancy."
        ),
        data_notes=(
            "Use the individual silver tables (not the panel) for accurate per-indicator "
            "counts. For distributions, use the most recent year with >150 countries reporting."
        ),
    ),
    # ── Theme 1: The Preston Curve (correlation) ────────────────
    AnalysisTheme(
        id="theme_1",
        title="The Preston Curve",
        question=(
            "What is the relationship between GDP per capita and life expectancy? "
            "Does the classic concave Preston Curve hold in modern data?"
        ),
        analysis_type="correlation",
        sequence=1,
        required_tables=["silver_development_recent"],
        analysis_steps=[
            "Compute Pearson and Spearman correlation between log(GDP per capita) and life expectancy",
            "Fit a log-linear regression: life_expectancy ~ log(gdp_per_capita)",
            "Test for concavity: fit quadratic on log(GDP) and check significance of the squared term",
            (
                "Identify outliers — countries that deviate significantly from the curve "
                "(high GDP but low life expectancy, or vice versa)"
            ),
            "Stratify by income group — does the curve slope differ within groups?",
            "Compare the curve shape across decades (2000, 2010, 2020) — is it shifting upward?",
        ],
        tables_to_produce=[
            "preston_curve_regression_coefficients",
            "outlier_countries_above_below_curve",
            "decade_comparison_slopes",
            "income_group_stratification",
        ],
        signature_visualization=(
            "The Preston Curve scatter plot: x-axis is log(GDP per capita), y-axis is life "
            "expectancy. Each dot is a country (most recent year), sized by population, "
            "colored by income group. Overlay the fitted log-linear curve. Label notable "
            "outliers (e.g., USA, Cuba, Qatar, Sierra Leone). This is the iconic chart "
            "of development economics."
        ),
        visualization_notes=(
            "Use population for bubble sizing so China and India dominate visually. "
            "Label 6-8 interesting outliers by name. Use log scale on x-axis."
        ),
        punchline=(
            "GDP explains X% of variation in life expectancy (R-squared=X.XX), but the "
            "relationship is log-linear — doubling GDP from $2K to $4K buys more years "
            "than doubling from $40K to $80K."
        ),
        data_notes=(
            "Use silver_development_recent. GDP per capita should be log-transformed for "
            "regression. Filter out World Bank aggregate entities. Population is needed "
            "for bubble sizing in the signature visualization."
        ),
    ),
    # ── Theme 2: Education and Economic Growth (comparative) ────
    AnalysisTheme(
        id="theme_2",
        title="Education and Economic Growth",
        question=(
            "Do countries with higher primary school enrollment achieve faster economic "
            "growth? How does the education-growth relationship vary by income group?"
        ),
        analysis_type="comparative",
        sequence=2,
        required_tables=["silver_development_recent"],
        analysis_steps=[
            (
                "Divide countries into enrollment tertiles (low/medium/high) based on their "
                "average school enrollment rate over 2000-2023"
            ),
            "Compare GDP per capita across enrollment tertiles using ANOVA and post-hoc tests",
            "Compute effect sizes (Cohen's d) for pairwise tertile comparisons",
            "Run partial correlation: enrollment vs GDP growth, controlling for initial GDP level",
            "Stratify the comparison by income group to check for interaction effects",
            "Test for reverse causality: does GDP predict subsequent enrollment better than vice versa?",
        ],
        tables_to_produce=[
            "enrollment_tertile_gdp_comparison",
            "anova_results",
            "income_group_interaction",
            "partial_correlation_results",
        ],
        signature_visualization=(
            "Grouped box plots: GDP per capita by enrollment tertile, faceted by income group. "
            "Shows the education-growth relationship and how it varies by development level."
        ),
        punchline=(
            "Countries in the top enrollment tertile have X times higher GDP per capita "
            "than bottom-tertile countries (p < 0.001), but the effect is confounded by "
            "initial wealth — partial correlation after controlling for baseline GDP is r=X.XX."
        ),
        data_notes=(
            "Use silver_development_recent. School enrollment has many NULLs — compute "
            "tertiles from the country-level mean across available years, not from individual "
            "year observations. Minimum 5 years of enrollment data per country to be included."
        ),
    ),
    # ── Theme 3: Healthcare Spending Returns (regression) ───────
    AnalysisTheme(
        id="theme_3",
        title="Healthcare Spending and Life Expectancy",
        question=(
            "Does healthcare spending predict life expectancy beyond what GDP alone explains? "
            "Are there diminishing returns to health investment?"
        ),
        analysis_type="correlation",
        sequence=3,
        required_tables=["silver_development_recent"],
        analysis_steps=[
            "Fit linear regression: life_expectancy ~ log(health_expenditure_per_capita)",
            "Fit multiple regression: life_expectancy ~ log(gdp_per_capita) + log(health_expenditure)",
            "Compare R-squared: does adding health expenditure improve on GDP alone?",
            "Test for diminishing returns: add quadratic term and check significance",
            (
                "Identify high-efficiency countries — those with life expectancy above the "
                "regression line at low spending levels (e.g., Costa Rica, Cuba, Thailand)"
            ),
            "Stratify by income group to check if the spending-outcome curve differs",
        ],
        tables_to_produce=[
            "spending_regression_coefficients",
            "model_comparison_r_squared",
            "high_efficiency_countries",
            "income_group_regression_slopes",
        ],
        signature_visualization=(
            "Scatter plot: log(health expenditure per capita) vs life expectancy. "
            "Dots sized by population, colored by income group. Overlay regression line. "
            "Label high-efficiency outliers. Add a second panel showing the marginal "
            "effect curve (predicted life expectancy gain per additional $100 spending "
            "at different spending levels)."
        ),
        visualization_notes=(
            "The marginal effect curve is the key insight — show how the first $100 per "
            "capita buys years of life, while wealthy nations spend thousands for weeks."
        ),
        punchline=(
            "Health expenditure adds X percentage points of explanatory power beyond GDP "
            "alone (R-squared: GDP-only=X.XX vs GDP+spending=X.XX). The marginal return "
            "to spending drops sharply above $X per capita."
        ),
        data_notes=(
            "Use silver_development_recent. Health expenditure data starts ~2000 so this "
            "is appropriate. Log-transform both GDP and health expenditure (right-skewed). "
            "Exclude countries with fewer than 5 years of health expenditure data."
        ),
    ),
    # ── Theme 4: The Carbon Cost of Development (trend) ─────────
    AnalysisTheme(
        id="theme_4",
        title="The Carbon Cost of Development",
        question=(
            "Is economic development inherently carbon-intensive? Does the Environmental "
            "Kuznets Curve — where emissions peak then decline with wealth — hold?"
        ),
        analysis_type="trend",
        sequence=4,
        required_tables=["silver_development_recent", "silver_development_long_run"],
        analysis_steps=[
            "Compute cross-sectional correlation: log(GDP per capita) vs CO2 per capita",
            (
                "Test the Environmental Kuznets Curve: fit CO2 ~ log(GDP) + log(GDP)^2 and "
                "check if the squared term is significantly negative (inverted-U shape)"
            ),
            (
                "Track carbon intensity trends: compute CO2/GDP ratio over time by income group "
                "and test whether it is declining (decoupling)"
            ),
            (
                "Plot individual country CO2 trajectories for 5-6 illustrative countries "
                "(USA, UK, China, India, Germany, Brazil) to show divergent paths"
            ),
            (
                "Test whether high-income countries are achieving absolute decoupling "
                "(GDP up, total CO2 down) vs relative decoupling (CO2/GDP down, total CO2 up)"
            ),
        ],
        tables_to_produce=[
            "co2_gdp_correlation",
            "kuznets_curve_regression",
            "carbon_intensity_by_income_group",
            "country_trajectory_summary",
            "decoupling_classification",
        ],
        signature_visualization=(
            "Two-panel figure. Left: cross-sectional scatter of log(GDP) vs CO2 per capita "
            "with Kuznets curve overlay. Right: spaghetti plot of CO2 per capita trajectories "
            "over time for 6 illustrative countries, colored by income group."
        ),
        punchline=(
            "The Kuznets inverted-U is [significant/not significant] (p=X.XX). "
            "High-income countries show [absolute/relative] decoupling, but global emissions "
            "continue to rise as middle-income nations industrialize."
        ),
        data_notes=(
            "Use silver_development_recent for cross-sectional analysis. Use "
            "silver_development_long_run for time trends. CO2 data lags by 2-3 years. "
            "For the country trajectory plot, pick countries that illustrate different "
            "development-emissions paths."
        ),
    ),
    # ── Theme 5: Convergence or Divergence? (trend) ─────────────
    AnalysisTheme(
        id="theme_5",
        title="Convergence or Divergence?",
        question=(
            "Are developing nations catching up to the developed world in GDP per capita "
            "and life expectancy? Is global inequality increasing or decreasing?"
        ),
        analysis_type="trend",
        sequence=5,
        required_tables=["silver_development_long_run"],
        analysis_steps=[
            (
                "Compute sigma-convergence: coefficient of variation of log(GDP per capita) "
                "across countries by decade (1960s, 1970s, ..., 2020s). Declining CV = convergence."
            ),
            (
                "Compute beta-convergence: regress GDP growth rate (1960-2020) on initial "
                "log(GDP) level. Negative coefficient = poorer countries growing faster."
            ),
            "Repeat both convergence tests for life expectancy",
            (
                "Compute the 90/10 ratio for GDP and life expectancy by decade — how far "
                "apart are the top and bottom 10% of countries?"
            ),
            (
                "Stratify convergence by income group — is convergence happening within "
                "groups, between groups, or both?"
            ),
        ],
        tables_to_produce=[
            "sigma_convergence_by_decade",
            "beta_convergence_regression",
            "ratio_90_10_trends",
            "income_group_convergence",
        ],
        signature_visualization=(
            "Dual-axis time series: left axis shows the coefficient of variation of "
            "log(GDP per capita) by decade (sigma-convergence), right axis shows the "
            "CV of life expectancy. If the lines diverge, GDP and health are telling "
            "different convergence stories."
        ),
        visualization_notes=(
            "Use decade averages, not single years, to smooth volatility. "
            "Annotate major events: oil crises, Soviet collapse, 2008 crisis, COVID."
        ),
        punchline=(
            "Life expectancy shows [clear/moderate/no] convergence (CV declined from X to Y), "
            "while GDP per capita shows [clear/moderate/no] convergence — development "
            "depends critically on what you measure."
        ),
        data_notes=(
            "Use silver_development_long_run (balanced panel) for convergence analysis. "
            "The balanced panel requirement means only ~100-120 countries are included — "
            "note this as a limitation (countries with missing 1960 data are excluded, "
            "biasing toward more established economies)."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Research references — published work the agent should consider
# ═══════════════════════════════════════════════════════════════════════

_references = [
    ResearchReference(
        title="The Preston Curve: GDP and Life Expectancy",
        url="https://en.wikipedia.org/wiki/Preston_curve",
        description=(
            "Samuel Preston's 1975 finding that life expectancy is concavely related "
            "to GDP per capita. The curve has shifted upward over decades — countries "
            "today live longer at every income level than in the 1970s."
        ),
        keywords=["Preston", "GDP", "life expectancy", "log-linear"],
    ),
    ResearchReference(
        title="The Environmental Kuznets Curve",
        url="https://en.wikipedia.org/wiki/Kuznets_curve",
        description=(
            "The hypothesis that environmental degradation first rises then falls "
            "with economic development (inverted-U shape). Evidence is mixed — it "
            "holds for some pollutants (SO2, particulates) but not clearly for CO2."
        ),
        keywords=["Kuznets", "environment", "CO2", "inverted-U"],
    ),
    ResearchReference(
        title="Convergence in Economics",
        url="https://en.wikipedia.org/wiki/Convergence_(economics)",
        description=(
            "The hypothesis that poorer economies grow faster than richer ones, "
            "leading to income convergence. Sigma-convergence (declining cross-country "
            "dispersion) and beta-convergence (negative growth-initial income relationship) "
            "are the two standard tests."
        ),
        keywords=["convergence", "sigma", "beta", "growth", "inequality"],
    ),
    ResearchReference(
        title="World Bank Data Catalog — World Development Indicators",
        url="https://data.worldbank.org/indicator",
        description=(
            "The official World Bank data portal with 1,400+ indicators for 217 "
            "economies. All data used in this analysis is sourced from WDI."
        ),
        keywords=["World Bank", "WDI", "indicators", "data source"],
    ),
    ResearchReference(
        title="Our World in Data — Global Health",
        url="https://ourworldindata.org/",
        description=(
            "Research and interactive visualizations on global development topics. "
            "Excellent reference for contextualizing World Bank indicators and "
            "understanding long-run trends."
        ),
        keywords=["OWID", "global health", "visualization", "context"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Assembled ResearchConfig
# ═══════════════════════════════════════════════════════════════════════

GLOBAL_DEVELOPMENT = ResearchConfig(
    name="Does Economic Development Drive Human Wellbeing?",
    thesis=(
        "Economic development (GDP per capita) is strongly correlated with life "
        "expectancy and health outcomes, but the relationship is log-linear with "
        "sharply diminishing returns. Education correlates with growth but causality "
        "is confounded. Healthcare spending efficiency varies enormously across "
        "countries. Carbon emissions track development, though high-income nations "
        "show signs of decoupling. Whether the world is converging or diverging "
        "depends critically on what you measure — life expectancy shows convergence "
        "while GDP tells a more ambiguous story."
    ),
    agent_role="Development Economist and Global Health Researcher",
    domain_context=(
        "## Data Quirks\n\n"
        "- World Bank data uses 3-letter ISO country codes; aggregates (e.g., 'WLD', 'EAS') "
        "must be excluded from country-level analysis.\n"
        "- GDP per capita is in current US dollars — for growth and cross-temporal analysis, "
        "use log transforms rather than raw levels.\n"
        "- Health expenditure data starts ~2000; earlier years are mostly missing.\n"
        "- School enrollment can exceed 100% (late enrollees counted; UNESCO methodology).\n"
        "- CO2 emissions data lags by 2-3 years.\n"
        "- Population includes both countries and aggregate regions — filter carefully.\n\n"
        "## Expected Value Ranges\n\n"
        "- GDP per capita: $200 (Burundi) to $100,000+ (Luxembourg)\n"
        "- Life expectancy: 50-85 years\n"
        "- School enrollment: 30-110% (net enrollment rate)\n"
        "- Health expenditure: $10-$12,000 per capita\n"
        "- CO2 emissions: 0.05-40 metric tons per capita\n"
        "- Population: 10,000 (Tuvalu) to 1.4 billion (India/China)\n"
    ),
    analysis_method_guidance={
        "correlation": (
            "Always log-transform GDP per capita and health expenditure (right-skewed). "
            "Report both Pearson (linear) and Spearman (monotonic) correlations. "
            "For the Preston Curve and healthcare spending, test log-linear vs linear fits. "
            "Size scatter plot points by population to weight visual impact correctly. "
            "Color by income_group or region for immediate visual stratification."
        ),
        "trend": (
            "For convergence, use balanced panels (same countries across all time points). "
            "Report sigma-convergence (CV of log GDP across countries) by decade. "
            "For beta-convergence, regress long-run growth on initial income level. "
            "Use decade averages rather than single-year snapshots to smooth volatility. "
            "Annotate structural breaks (oil crises, Soviet collapse, 2008 crisis, COVID)."
        ),
    },
    visualization_guidance=(
        "OUTPUT PHILOSOPHY:\n"
        "- Tables are the primary analytical output. Every theme MUST produce summary "
        "tables with statistics, p-values, effect sizes, and confidence intervals.\n"
        "- Signature visualizations are the ONE iconic chart per theme.\n"
        "- Use a professional, Economist-magazine style. Muted color palette.\n"
        "- Always size scatter points by population — China and India should dominate visually.\n"
        "- Label outlier countries by name — readers want to know which dots are theirs.\n"
    ),
    project=_project,
    results_volume_path="/Volumes/my_catalog/world_development/results",
    analysis_themes=_themes,
    silver_datasets=_silver_datasets,
    research_references=_references,
    max_turns=150,
    max_turns_per_phase=120,
    max_turns_per_theme=100,
    chart_style="seaborn-v0_8-whitegrid",
    chart_dpi=150,
    color_palette="tab10",
)
