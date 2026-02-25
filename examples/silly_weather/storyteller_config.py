"""
Storyteller config for the Silly Weather project.

This config drives the StoryTellerAgent to produce a narrative report
from the DataScientist's findings about duck-based weather prediction.

Usage in a Databricks notebook::

    from examples.silly_weather.storyteller_config import DUCK_STORY
    agent = StoryTellerAgent(cfg=DUCK_STORY, dbutils=dbutils)
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
# Style guide — voice and tone rules
# ═══════════════════════════════════════════════════════════════════════

_style = StyleGuide(
    voice="third-person with dry humor",
    audience="Data scientists and curious duck enthusiasts",
    reading_level="professional (with occasional quack puns)",
    citation_style="inline",
    document_type="Research white paper (tongue-in-cheek)",
    purpose=(
        "Present a rigorous statistical analysis of duck behavior as weather "
        "predictors, written with a straight face despite the absurd premise. "
        "The humor comes from treating the topic seriously, not from jokes."
    ),
    tone_guidance=(
        "Deadpan scientific. Write as if this were a real Nature paper that "
        "happens to be about ducks. Use proper statistical language. Let the "
        "absurdity of the subject matter provide the comedy."
    ),
    anti_patterns=(
        "- NO: Puns in section headings (one per section max, and only in subheadings)\n"
        "- NO: Exclamation marks in scientific claims\n"
        "- NO: 'Interestingly' or 'Surprisingly' — let the data surprise the reader\n"
        "- NO: Hedging on clearly significant results\n"
        "- NO: Overclaiming on clearly insignificant results\n"
        "- NO: Breaking the deadpan tone with 'just kidding' asides\n"
    ),
)

# ═══════════════════════════════════════════════════════════════════════
# Narrative sections — the story arc
# ═══════════════════════════════════════════════════════════════════════

_sections = [
    NarrativeSection(
        id="section_hook",
        title="When Ducks Quack, Should We Listen?",
        purpose="Hook the reader with the absurd premise, then ground it in real science",
        source_theme_ids=["theme_0"],
        tone="analytical",
        max_words=800,
        key_evidence="Data inventory stats, seasonal quack patterns",
        narrative_guidance=(
            "Open with the scale of the data: N stations, M ponds, K million quacks. "
            "Briefly mention the long tradition of animal-based weather folklore. "
            "End with the research question stated formally."
        ),
        transition_from="",
        transition_to="With data in hand, we begin with the simplest question: do ducks quack more before rain?",
        sequence=0,
    ),
    NarrativeSection(
        id="section_correlation",
        title="The Quack-Storm Nexus",
        purpose="Present the core correlation between quacking and precipitation",
        source_theme_ids=["theme_1"],
        tone="analytical",
        max_words=1200,
        key_evidence="Lag-correlation results, seasonal breakdown, significance tests",
        narrative_guidance=(
            "Lead with the lag-correlation plot. Walk through the seasonal differences. "
            "Be precise about effect sizes and p-values. If the correlation is weak, "
            "say so — this is science, not marketing."
        ),
        transition_from="The data exists. Does it speak?",
        transition_to="Quacking alone may not tell the full story. Enter the Fluff Factor.",
        sequence=1,
    ),
    NarrativeSection(
        id="section_fluff",
        title="Beyond Quacking: The Duck Barometer",
        purpose="Introduce the multi-signal Duck Barometer Index",
        source_theme_ids=["theme_2"],
        tone="analytical",
        max_words=1200,
        key_evidence="ROC comparison, DBI construction, logistic regression",
        narrative_guidance=(
            "Show how combining signals outperforms any single signal. "
            "The ROC curve comparison is the hero visual. If the combined model "
            "doesn't outperform, be honest — sometimes simple is better."
        ),
        transition_from="Quacking correlates with rain. But ducks do more than quack.",
        transition_to="Before we declare victory, we must address the elephant in the room. Or rather, the ice cream.",
        sequence=2,
    ),
    NarrativeSection(
        id="section_confounder",
        title="The Ice Cream Defense",
        purpose="Address confounding — is this just a temperature artifact?",
        source_theme_ids=["theme_3"],
        tone="analytical",
        max_words=1000,
        key_evidence="Partial correlations, mediation analysis, Simpson's Paradox check",
        narrative_guidance=(
            "This is the credibility section. Walk through each confounder check "
            "methodically. Show the coefficient waterfall. If the signal survives "
            "temperature control, celebrate (quietly). If it doesn't, acknowledge it."
        ),
        transition_from="The Duck Barometer looks promising. But correlation is not causation.",
        transition_to="With confounders addressed, we turn to the ultimate test: the head-to-head showdown.",
        sequence=3,
    ),
    NarrativeSection(
        id="section_showdown",
        title="Duck vs. Doppler: The Showdown",
        purpose="Head-to-head comparison with professional meteorologists",
        source_theme_ids=["theme_5"],
        tone="analytical",
        max_words=1500,
        key_evidence="Precision-recall curves, F1 scores, McNemar's test, difficulty stratification",
        narrative_guidance=(
            "This is the climax. Present the paired PR curves side by side. "
            "Discuss where ducks excel (if anywhere) and where they fail. "
            "The difficulty stratification is key — ducks might only beat "
            "meteorologists on easy days. Be ruthlessly honest."
        ),
        transition_from="The duck signal is real (or not). But can it compete with professionals?",
        transition_to="Individual comparisons tell part of the story. What if we combined forces?",
        sequence=4,
    ),
    NarrativeSection(
        id="section_ensemble",
        title="The Grand Unified Duck Theory",
        purpose="Present the ensemble model and SHAP analysis",
        source_theme_ids=["theme_6"],
        tone="analytical",
        max_words=1500,
        key_evidence="Ensemble AUC, SHAP beeswarm, feature importance, CV results",
        narrative_guidance=(
            "Show that combining duck signals WITH meteorologist forecasts "
            "may (or may not) improve predictions. The SHAP plot reveals which "
            "duck features actually matter. Close with the final verdict."
        ),
        transition_from="If ducks can't beat meteorologists, can they help them?",
        transition_to="We've built models and run tests. What does it all mean?",
        sequence=5,
    ),
    NarrativeSection(
        id="section_conclusion",
        title="Lessons from the Pond",
        purpose="Synthesize findings, acknowledge limitations, and reflect",
        source_theme_ids=["theme_0", "theme_1", "theme_5", "theme_6"],
        tone="analytical",
        max_words=1000,
        key_evidence="Summary statistics from all themes",
        narrative_guidance=(
            "Summarize the journey: data → correlation → model → showdown → ensemble. "
            "State the final verdict clearly. Acknowledge limitations (volunteer bias, "
            "geographic coverage, the fact that this entire premise is ridiculous). "
            "End with what this example teaches about the Versifai framework."
        ),
        transition_from="The analysis is complete.",
        transition_to="",
        sequence=6,
    ),
    NarrativeSection(
        id="section_methodology",
        title="Methodology & Reproducibility",
        purpose="Technical appendix with all SQL, model specs, and data sources",
        source_theme_ids=["theme_0", "theme_1", "theme_2", "theme_3", "theme_4", "theme_5", "theme_6"],
        tone="analytical",
        max_words=2000,
        key_evidence="All SQL queries, model hyperparameters, data source citations",
        narrative_guidance=(
            "Include every SQL query used, model hyperparameters, and data processing "
            "steps. This section should allow someone to reproduce the entire analysis "
            "without the AI agent — just by following the instructions."
        ),
        transition_from="",
        transition_to="",
        sequence=7,
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Evidence thresholds
# ═══════════════════════════════════════════════════════════════════════

_evidence = EvidenceThreshold(
    min_significance_for_lead="high",
    min_significance_for_support="medium",
    require_effect_size=True,
    max_unsupported_claims=0,
)

# ═══════════════════════════════════════════════════════════════════════
# Output format
# ═══════════════════════════════════════════════════════════════════════

_output = OutputFormat(
    format="markdown",
    filename="duck_weather_report.md",
    include_toc=True,
    include_methodology_appendix=True,
    include_data_sources_appendix=True,
    chart_reference_style="relative_path",
)

# ═══════════════════════════════════════════════════════════════════════
# Assembled config
# ═══════════════════════════════════════════════════════════════════════

DUCK_STORY = StorytellerConfig(
    name="Do Ducks Predict Rain? A Rigorous Investigation",
    thesis=(
        "Duck behavioral signals contain genuine meteorological information "
        "that, when properly modeled, provides statistically significant "
        "24-hour precipitation forecasts."
    ),
    research_results_path="/Volumes/my_catalog/silly_weather/results",
    narrative_output_path="/Volumes/my_catalog/silly_weather/narrative",
    project=ProjectConfig(
        catalog="my_catalog",
        schema="silly_weather",
    ),
    narrative_sections=_sections,
    evidence_threshold=_evidence,
    style_guide=_style,
    output_format=_output,
    citation_urls=[
        "https://en.wikipedia.org/wiki/Weather_lore",
        "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
        "https://tylervigen.com/spurious-correlations",
    ],
)
