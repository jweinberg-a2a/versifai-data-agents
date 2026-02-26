"""
Storyteller config for the World Development Indicators report.

This config defines the StoryTellerAgent's narrative structure: 8 sections
that transform the DataScientist's findings on global development into
an evidence-grounded analytical report.

Usage in a Databricks notebook::

    from examples.world_development.storyteller_config import WORLD_DEVELOPMENT_STORY
    agent = StoryTellerAgent(cfg=WORLD_DEVELOPMENT_STORY, dbutils=dbutils)
    agent.run()
"""

from __future__ import annotations

from versifai.data_agents.engineer.config import ProjectConfig
from versifai.story_agents.storyteller.config import (
    EvidenceThreshold,
    NarrativeSection,
    OutputFormat,
    StorytellerConfig,
    StyleGuide,
)

# ═══════════════════════════════════════════════════════════════════════
# Style guide — voice, audience, and editorial rules
# ═══════════════════════════════════════════════════════════════════════

_style = StyleGuide(
    voice="third-person analytical",
    audience="Policy analysts, development economists, and informed general readers",
    reading_level="professional",
    citation_style="inline",
    document_type="Analytical white paper",
    purpose=(
        "Present a data-driven narrative about global development using World Bank "
        "indicators. Ground every claim in statistical evidence. Make the story "
        "accessible without sacrificing rigor."
    ),
    tone_guidance=(
        "Authoritative but accessible. Write like The Economist or Our World in Data — "
        "precise, evidence-first, globally aware. Acknowledge complexity and nuance. "
        "Avoid both Pollyanna optimism and doom-and-gloom framing — let the data speak."
    ),
    anti_patterns=(
        "- NO: Vague claims like 'the world is getting better/worse' without data\n"
        "- NO: Conflating correlation with causation\n"
        "- NO: Ignoring the role of data coverage gaps in shaping conclusions\n"
        "- NO: Cherry-picking countries that support a narrative\n"
        "- NO: Using current USD for temporal comparisons without noting inflation\n"
        "- NO: Treating World Bank income groups as fixed categories (they change)\n"
    ),
)

# ═══════════════════════════════════════════════════════════════════════
# Narrative sections — 8 sections building the argument
# ═══════════════════════════════════════════════════════════════════════

_sections = [
    # ── Section 0: The Shape of Progress (hook) ─────────────────
    NarrativeSection(
        id="section_hook",
        title="The Shape of Progress",
        purpose="Ground the reader in the scale and scope of the data, establish the questions",
        source_theme_ids=["theme_0"],
        tone="analytical",
        max_words=800,
        key_evidence="Data inventory, income group distributions, 90/10 ratios",
        narrative_guidance=(
            "Open with a striking comparison: the GDP gap between the richest and poorest "
            "countries, or the life expectancy gap. Then introduce the 6 indicators and "
            "217 countries. Frame the five questions the report will answer. End with the "
            "thesis: development data reveals both progress and persistent inequality."
        ),
        transition_from="",
        transition_to="The most iconic relationship in development economics is where we begin.",
        sequence=0,
    ),
    # ── Section 1: When Wealth Buys Years (Preston Curve) ───────
    NarrativeSection(
        id="section_preston",
        title="When Wealth Buys Years",
        purpose="Present the GDP-life expectancy relationship and the Preston Curve",
        source_theme_ids=["theme_1"],
        tone="analytical",
        max_words=1200,
        key_evidence="Preston Curve regression, R-squared, outlier analysis, decade shifts",
        narrative_guidance=(
            "Lead with the Preston Curve scatter plot. Explain the log-linear shape — "
            "why doubling GDP from $2K to $4K buys more years than $40K to $80K. "
            "Discuss the outliers: which countries beat their wealth level (Costa Rica, Cuba) "
            "and which underperform (USA, South Africa)? Show how the curve has shifted "
            "upward over decades — countries are living longer at every income level."
        ),
        transition_from=(
            "With data in hand, we begin with the most iconic relationship in "
            "development economics."
        ),
        transition_to=(
            "If wealth buys health with diminishing returns, what about direct health investment?"
        ),
        sequence=1,
    ),
    # ── Section 2: Schools, Skills, and Growth ──────────────────
    NarrativeSection(
        id="section_education",
        title="Schools, Skills, and Growth",
        purpose="Present the education-growth relationship and its limitations",
        source_theme_ids=["theme_2"],
        tone="analytical",
        max_words=1000,
        key_evidence="Enrollment tertile comparisons, ANOVA, partial correlations",
        narrative_guidance=(
            "Present the enrollment-growth comparison. Be honest about the limitations: "
            "primary enrollment is a crude proxy for human capital, and reverse causality "
            "is a real concern (rich countries can afford more schools). Highlight where "
            "the relationship holds and where it breaks down."
        ),
        transition_from="Wealth buys health. Does education buy wealth?",
        transition_to=(
            "Education and GDP are intertwined. But one investment is more direct: "
            "healthcare spending."
        ),
        sequence=2,
    ),
    # ── Section 3: Diminishing Returns (healthcare) ─────────────
    NarrativeSection(
        id="section_healthcare",
        title="Diminishing Returns",
        purpose="Show how healthcare spending maps to outcomes with decreasing efficiency",
        source_theme_ids=["theme_3"],
        tone="analytical",
        max_words=1200,
        key_evidence=(
            "Linear vs log-linear R-squared, marginal returns analysis, "
            "high-efficiency country identification"
        ),
        narrative_guidance=(
            "This is the policy-relevant section. Show that the first $100 per capita "
            "buys enormous health gains, but wealthy nations spend thousands for marginal "
            "improvements. Highlight high-efficiency countries (Costa Rica, Cuba, Thailand) "
            "that achieve rich-country life expectancy at a fraction of the cost. "
            "Address whether spending adds explanatory power beyond GDP alone."
        ),
        transition_from=(
            "If wealth and education shape development, what does direct health investment achieve?"
        ),
        transition_to=(
            "Progress in health and wealth comes with a cost the atmosphere is keeping track of."
        ),
        sequence=3,
    ),
    # ── Section 4: The Carbon Crossroads ────────────────────────
    NarrativeSection(
        id="section_carbon",
        title="The Carbon Crossroads",
        purpose="Present the CO2-GDP relationship and test the Environmental Kuznets Curve",
        source_theme_ids=["theme_4"],
        tone="analytical",
        max_words=1200,
        key_evidence=(
            "CO2-GDP correlation, Kuznets curve test, carbon intensity trends, country trajectories"
        ),
        narrative_guidance=(
            "Present the two-panel figure: cross-section and time series. Walk through "
            "the Kuznets curve evidence. Show the divergent country trajectories — some "
            "countries are decoupling GDP from CO2 (UK, Germany), others are still on "
            "the upslope (China, India). End with the carbon intensity trend."
        ),
        transition_from="Health and wealth improve. But development has an atmospheric price tag.",
        transition_to=(
            "Across all these dimensions — wealth, health, education, environment — "
            "the central question remains: is the gap narrowing?"
        ),
        sequence=4,
    ),
    # ── Section 5: A Narrowing Gap? (convergence) ───────────────
    NarrativeSection(
        id="section_convergence",
        title="A Narrowing Gap?",
        purpose="Present convergence/divergence evidence for GDP and life expectancy",
        source_theme_ids=["theme_5"],
        tone="analytical",
        max_words=1500,
        key_evidence=(
            "Sigma-convergence time series, beta-convergence regressions, 90/10 ratios by decade"
        ),
        narrative_guidance=(
            "This is the synthesis section — it brings together the threads. Present the "
            "dual-axis convergence chart. The key insight: life expectancy converges while "
            "GDP sends mixed signals. Discuss why — health technology diffuses faster than "
            "economic institutions. Be honest about the data: the convergence result "
            "depends heavily on which countries and years you include."
        ),
        transition_from="Rich countries got rich first. Are poor countries catching up?",
        transition_to="The data has spoken — and its message is more nuanced than any headline.",
        sequence=5,
    ),
    # ── Section 6: Conclusion ───────────────────────────────────
    NarrativeSection(
        id="section_conclusion",
        title="What Development Data Reveals",
        purpose="Synthesize all findings into a coherent narrative and state the verdict",
        source_theme_ids=["theme_0", "theme_1", "theme_3", "theme_4", "theme_5"],
        tone="analytical",
        max_words=1000,
        key_evidence="Summary statistics from all themes",
        narrative_guidance=(
            "Summarize the five findings: wealth buys health (diminishing returns), "
            "education correlates with growth (but causality is unclear), healthcare "
            "spending shows extreme diminishing returns, carbon tracks development "
            "(Kuznets curve inconclusive), and convergence depends on what you measure. "
            "Acknowledge limitations: current USD, aggregate entities, coverage gaps. "
            "End with what this demonstrates about autonomous data analysis."
        ),
        transition_from="The analysis is complete.",
        transition_to="",
        sequence=6,
    ),
    # ── Section 7: Methodology & Reproducibility ────────────────
    NarrativeSection(
        id="section_methodology",
        title="Methodology and Reproducibility",
        purpose="Technical appendix with all SQL, model specs, and data sources",
        source_theme_ids=[
            "theme_0",
            "theme_1",
            "theme_2",
            "theme_3",
            "theme_4",
            "theme_5",
        ],
        tone="analytical",
        max_words=2000,
        key_evidence="All SQL queries, model specifications, data source citations",
        narrative_guidance=(
            "Include every SQL query, regression specification, and data processing step. "
            "Document the World Bank API download process. List all indicator codes. "
            "This section must allow someone to reproduce the entire analysis without "
            "the AI agent — just by following the instructions."
        ),
        transition_from="",
        transition_to="",
        sequence=7,
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Evidence thresholds & output format
# ═══════════════════════════════════════════════════════════════════════

_evidence = EvidenceThreshold(
    min_significance_for_lead="high",
    min_significance_for_support="medium",
    require_effect_size=True,
    max_unsupported_claims=0,
)

_output = OutputFormat(
    format="markdown",
    filename="world_development_report.md",
    include_toc=True,
    include_methodology_appendix=True,
    include_data_sources_appendix=True,
    chart_reference_style="relative_path",
)

# ═══════════════════════════════════════════════════════════════════════
# Assembled StorytellerConfig
# ═══════════════════════════════════════════════════════════════════════

WORLD_DEVELOPMENT_STORY = StorytellerConfig(
    name="The Shape of Global Progress",
    thesis=(
        "Global development data reveals that wealth buys health with diminishing "
        "returns, education correlates with growth but causality is elusive, healthcare "
        "spending efficiency varies enormously, carbon emissions track development, "
        "and convergence between rich and poor nations depends on what dimension you measure."
    ),
    domain_writing_rules=(
        "EVIDENCE-FIRST ANALYTICAL TONE: Every claim must cite a specific statistic. "
        "Acknowledge when data limitations affect conclusions. Avoid both optimism bias "
        "and doom framing. Let readers draw their own policy conclusions — the report's "
        "job is to present evidence, not advocate."
    ),
    citation_source_guidance=(
        "World Bank technical documentation, peer-reviewed development economics literature, "
        "Our World in Data, OECD reports, and classic texts on growth theory and "
        "the Preston Curve."
    ),
    research_results_path="/Volumes/my_catalog/world_development/results",
    narrative_output_path="/Volumes/my_catalog/world_development/narrative",
    project=ProjectConfig(catalog="my_catalog", schema="world_development"),
    narrative_sections=_sections,
    evidence_threshold=_evidence,
    style_guide=_style,
    output_format=_output,
    citation_urls=[
        "https://data.worldbank.org/indicator",
        "https://en.wikipedia.org/wiki/Preston_curve",
        "https://en.wikipedia.org/wiki/Convergence_(economics)",
        "https://en.wikipedia.org/wiki/Kuznets_curve",
        "https://ourworldindata.org/",
    ],
)
