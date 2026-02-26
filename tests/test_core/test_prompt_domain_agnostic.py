"""Tests that prompts are domain-agnostic and config fields are properly injected.

Verifies:
1. New config fields (agent_role, domain_context, analysis_method_guidance, etc.)
   are properly injected into prompts when populated.
2. Default (empty) config values produce coherent prompts without domain content.
3. No residual hardcoded CMS/healthcare domain references remain in prompts.
"""

from __future__ import annotations

import pytest

from versifai.data_agents.engineer.config import JoinKeyConfig, ProjectConfig
from versifai.science_agents.scientist.config import (
    AnalysisTheme,
    ResearchConfig,
    ResearchQuestion,
    SilverDatasetSpec,
)
from versifai.story_agents.storyteller.config import StorytellerConfig, StyleGuide

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_project():
    """Minimal ProjectConfig with no domain content."""
    return ProjectConfig(
        name="Test Project",
        catalog="test_cat",
        schema="test_schema",
        join_key=JoinKeyConfig(column_name="entity_id", description="Primary entity key"),
    )


@pytest.fixture
def minimal_research_config(minimal_project):
    """Minimal ResearchConfig with no domain content."""
    return ResearchConfig(
        name="Test Research",
        thesis="Testing hypothesis",
        project=minimal_project,
        results_volume_path="/tmp/results",
    )


@pytest.fixture
def domain_research_config(minimal_project):
    """ResearchConfig with domain-specific fields populated."""
    return ResearchConfig(
        name="Duck Weather Research",
        thesis="Ducks predict rain",
        project=minimal_project,
        results_volume_path="/tmp/results",
        agent_role="Waterfowl Research Scientist",
        domain_context=(
            "## Data Quirks\n\n- Quack frequency ranges 0-200 QPH\n- Fluff index ranges 0-10\n"
        ),
        analysis_method_guidance={
            "simulation": (
                "**Custom Simulation**:\n1. Build Duck Barometer Index\n2. Calibrate thresholds\n"
            ),
        },
        visualization_guidance="PRIORITIZE scatter plots of duck behavior vs weather.",
    )


@pytest.fixture
def domain_project_config():
    """ProjectConfig with domain-specific fields populated."""
    return ProjectConfig(
        name="Duck Weather Engineering",
        catalog="test_cat",
        schema="test_schema",
        join_key=JoinKeyConfig(
            column_name="station_id",
            description="NOAA station identifier",
            validation_rule="Must match pattern USW/USC + 8 digits",
        ),
        column_naming_examples=("`TMAX` → `temp_max_c`\n`PRCP` → `precip_mm`"),
        grain_detection_guidance=(
            "Station-level: Look for station_id\nPond-level: Look for pond_id"
        ),
    )


@pytest.fixture
def minimal_storyteller_config(minimal_project):
    """Minimal StorytellerConfig with no domain content."""
    return StorytellerConfig(
        name="Test Story",
        thesis="Testing narrative",
        project=minimal_project,
        research_results_path="/tmp/results",
        narrative_output_path="/tmp/narrative",
    )


@pytest.fixture
def domain_storyteller_config(minimal_project):
    """StorytellerConfig with domain-specific fields populated."""
    return StorytellerConfig(
        name="Duck Weather Story",
        thesis="Ducks predict rain",
        project=minimal_project,
        research_results_path="/tmp/results",
        narrative_output_path="/tmp/narrative",
        domain_writing_rules=("**MAINTAIN DEADPAN TONE**: Treat duck forecasting seriously."),
        citation_source_guidance=(
            "NOAA documentation, ornithology journals, meteorology textbooks."
        ),
    )


# ---------------------------------------------------------------------------
# Data Scientist Prompt Tests
# ---------------------------------------------------------------------------


class TestScientistPromptDomainAgnostic:
    """Verify scientist prompts are domain-agnostic."""

    def test_system_prompt_uses_agent_role(self, domain_research_config):
        """agent_role field is injected into system prompt."""
        from versifai.science_agents.scientist.prompts import build_scientist_system_prompt

        prompt = build_scientist_system_prompt(domain_research_config)
        assert "Waterfowl Research Scientist" in prompt
        assert "Health Policy Researcher" not in prompt

    def test_system_prompt_default_agent_role(self, minimal_research_config):
        """Default agent_role 'Data Scientist' is used when not specified."""
        from versifai.science_agents.scientist.prompts import build_scientist_system_prompt

        prompt = build_scientist_system_prompt(minimal_research_config)
        assert "Data Scientist AI agent" in prompt

    def test_domain_context_injected(self, domain_research_config):
        """domain_context is injected when populated."""
        from versifai.science_agents.scientist.prompts import build_scientist_system_prompt

        prompt = build_scientist_system_prompt(domain_research_config)
        assert "Domain-Specific Guidance" in prompt
        assert "Quack frequency ranges 0-200 QPH" in prompt
        assert "Fluff index ranges 0-10" in prompt

    def test_no_domain_context_when_empty(self, minimal_research_config):
        """No domain context section when field is empty."""
        from versifai.science_agents.scientist.prompts import build_scientist_system_prompt

        prompt = build_scientist_system_prompt(minimal_research_config)
        # Should NOT have the domain-specific guidance header
        # (it's only injected conditionally)
        assert "CMS Data Quirk" not in prompt
        assert "Suppressed Values" not in prompt

    def test_visualization_guidance_injected(self, domain_research_config):
        """visualization_guidance is injected when populated."""
        from versifai.science_agents.scientist.prompts import build_scientist_system_prompt

        prompt = build_scientist_system_prompt(domain_research_config)
        assert "PRIORITIZE scatter plots of duck behavior" in prompt

    def test_no_hardcoded_cms_references(self, minimal_research_config):
        """System prompt with empty config has no CMS/healthcare references."""
        from versifai.science_agents.scientist.prompts import build_scientist_system_prompt

        prompt = build_scientist_system_prompt(minimal_research_config)
        # These were hardcoded before the refactoring
        assert "CMS" not in prompt
        assert "Medicare" not in prompt
        assert "MA penetration" not in prompt
        assert "SVI" not in prompt
        assert "star rating" not in prompt.lower()
        assert "FIPS" not in prompt

    def test_analysis_method_guidance_override(self, domain_research_config):
        """analysis_method_guidance overrides default simulation text."""
        from versifai.science_agents.scientist.prompts import build_theme_analysis_prompt

        theme = AnalysisTheme(
            id="test_theme",
            title="Test Theme",
            question="Test question?",
            analysis_type="simulation",
            sequence=1,
            required_tables=["test_table"],
            analysis_steps=["Step 1"],
            punchline="Test punchline",
        )
        prompt = build_theme_analysis_prompt(domain_research_config, theme)
        assert "Custom Simulation" in prompt
        assert "Build Duck Barometer Index" in prompt

    def test_default_simulation_guidance(self, minimal_research_config):
        """Default simulation guidance when no override is provided."""
        from versifai.science_agents.scientist.prompts import build_theme_analysis_prompt

        theme = AnalysisTheme(
            id="test_theme",
            title="Test Theme",
            question="Test question?",
            analysis_type="simulation",
            sequence=1,
            required_tables=["test_table"],
            analysis_steps=["Step 1"],
            punchline="Test punchline",
        )
        prompt = build_theme_analysis_prompt(minimal_research_config, theme)
        assert "VALIDATION: reproduce known published outputs" in prompt
        # Should NOT have CMS-specific references
        assert "CMS Tukey" not in prompt
        assert "CMS cut points" not in prompt

    def test_orientation_prompt_no_hardcoded_tables(self, minimal_research_config):
        """Orientation prompt does not reference specific dataset names."""
        from versifai.science_agents.scientist.prompts import build_orientation_prompt

        prompt = build_orientation_prompt(minimal_research_config)
        assert "star_ratings" not in prompt
        assert "scc_enrollment" not in prompt
        assert "ma_penetration" not in prompt
        assert "Census/ACS" not in prompt

    def test_silver_prompt_uses_join_key_validation(self, domain_research_config):
        """Silver construction prompt uses join key validation rule from config."""
        from versifai.science_agents.scientist.prompts import (
            build_silver_construction_prompt,
        )

        # Set validation rule on the project join key
        domain_research_config.project.join_key.validation_rule = (
            "Must match pattern USW/USC + 8 digits"
        )
        spec = SilverDatasetSpec(
            name="silver_test",
            description="Test dataset",
            source_tables=["table_a", "table_b"],
        )
        prompt = build_silver_construction_prompt(domain_research_config, spec)
        assert "Must match pattern USW/USC + 8 digits" in prompt
        # Should not have hardcoded FIPS reference
        assert "FIPS codes must be 5-character" not in prompt

    def test_validation_prompt_no_hardcoded_temporal_rules(self, minimal_research_config):
        """Validation prompt does not hardcode enrollment/star years."""
        from versifai.science_agents.scientist.prompts import build_validation_prompt

        theme = AnalysisTheme(
            id="test_theme",
            title="Test Theme",
            question="Test?",
            analysis_type="comparative",
            sequence=1,
            required_tables=["test"],
            analysis_steps=["Step 1"],
            punchline="Test",
        )
        prompt = build_validation_prompt(
            minimal_research_config,
            theme,
            findings_for_theme=[],
            chart_files=[],
            table_files=[],
            theme_notes="",
        )
        assert "2023-2025" not in prompt
        assert "2024-2026" not in prompt

    def test_synthesis_prompt_no_domain_references(self, minimal_research_config):
        """Synthesis prompt is generic."""
        from versifai.science_agents.scientist.prompts import build_synthesis_prompt

        prompt = build_synthesis_prompt(minimal_research_config, "Test findings")
        assert "social vulnerability were equalized" not in prompt
        assert "Theme 1" not in prompt
        assert "Theme 6" not in prompt
        assert "lift-based rating system" not in prompt

    def test_theme_prompt_no_hardcoded_temporal(self, minimal_research_config):
        """Theme analysis prompt does not include hardcoded temporal rules."""
        from versifai.science_agents.scientist.prompts import build_theme_analysis_prompt

        theme = AnalysisTheme(
            id="test_theme",
            title="Test Theme",
            question="Test?",
            analysis_type="descriptive",
            sequence=1,
            required_tables=["test"],
            analysis_steps=["Step 1"],
            punchline="Test",
        )
        prompt = build_theme_analysis_prompt(minimal_research_config, theme)
        assert "Enrollment/service area data spans" not in prompt
        assert "2023-2025" not in prompt


# ---------------------------------------------------------------------------
# Data Engineer Prompt Tests
# ---------------------------------------------------------------------------


class TestEngineerPromptDomainAgnostic:
    """Verify engineer prompts are domain-agnostic."""

    def test_system_prompt_uses_grain_detection(self, domain_project_config):
        """grain_detection_guidance is injected into system prompt."""
        from versifai.data_agents.engineer.prompts import build_system_prompt

        prompt = build_system_prompt(domain_project_config)
        assert "Station-level: Look for station_id" in prompt
        assert "Pond-level: Look for pond_id" in prompt

    def test_system_prompt_uses_column_naming_examples(self, domain_project_config):
        """column_naming_examples is injected into system prompt."""
        from versifai.data_agents.engineer.prompts import build_system_prompt

        prompt = build_system_prompt(domain_project_config)
        assert "`TMAX` → `temp_max_c`" in prompt

    def test_system_prompt_no_hardcoded_examples_when_empty(self, minimal_project):
        """Default prompts don't have domain-specific column examples."""
        from versifai.data_agents.engineer.prompts import build_system_prompt

        prompt = build_system_prompt(minimal_project)
        assert "EP_POV150" not in prompt
        assert "E_TOTPOP" not in prompt
        assert "RPL_THEMES" not in prompt

    def test_system_prompt_no_hardcoded_grain(self, minimal_project):
        """Default prompts don't have hardcoded H-number/FIPS grain references."""
        from versifai.data_agents.engineer.prompts import build_system_prompt

        prompt = build_system_prompt(minimal_project)
        assert "H-numbers (e.g., H5216)" not in prompt
        assert "H5216-001" not in prompt

    def test_grain_detection_in_source_prompt(self, domain_project_config):
        """grain_detection_guidance appears in source processing prompt."""
        from versifai.data_agents.engineer.prompts import build_source_processing_prompt

        prompt = build_source_processing_prompt(
            domain_project_config,
            source_name="test_source",
            source_path="/tmp/test",
            file_list="test.csv",
            suggested_table="test_table",
        )
        assert "Station-level: Look for station_id" in prompt

    def test_rename_prompt_no_hardcoded_examples(self, minimal_project):
        """Rename prompt with empty config doesn't have SVI/CMS examples."""
        from versifai.data_agents.engineer.prompts import build_rename_prompt

        prompt = build_rename_prompt(minimal_project)
        assert "ep_pov150" not in prompt
        assert "e_totpop" not in prompt
        assert "rpl_themes" not in prompt
        assert "spl_theme1" not in prompt

    def test_rename_prompt_uses_domain_examples(self, domain_project_config):
        """Rename prompt uses column_naming_examples when populated."""
        from versifai.data_agents.engineer.prompts import build_rename_prompt

        prompt = build_rename_prompt(domain_project_config)
        assert "`TMAX` → `temp_max_c`" in prompt

    def test_catalog_prompt_no_hardcoded_sources(self, minimal_project):
        """Catalog prompt doesn't hardcode CDC SVI, CMS Star Ratings, etc."""
        from versifai.data_agents.engineer.prompts import build_catalog_prompt

        prompt = build_catalog_prompt(minimal_project)
        assert "CDC SVI" not in prompt
        assert "CMS Star Ratings" not in prompt
        assert "Census ACS" not in prompt


# ---------------------------------------------------------------------------
# StoryTeller Prompt Tests
# ---------------------------------------------------------------------------


class TestStorytellerPromptDomainAgnostic:
    """Verify storyteller prompts are domain-agnostic."""

    def test_system_prompt_uses_domain_writing_rules(self, domain_storyteller_config):
        """domain_writing_rules is injected into system prompt."""
        from versifai.story_agents.storyteller.prompts import build_storyteller_system_prompt

        prompt = build_storyteller_system_prompt(domain_storyteller_config)
        assert "MAINTAIN DEADPAN TONE" in prompt
        assert "duck forecasting seriously" in prompt

    def test_system_prompt_uses_citation_guidance(self, domain_storyteller_config):
        """citation_source_guidance is injected into system prompt."""
        from versifai.story_agents.storyteller.prompts import build_storyteller_system_prompt

        prompt = build_storyteller_system_prompt(domain_storyteller_config)
        assert "NOAA documentation" in prompt
        assert "ornithology journals" in prompt

    def test_system_prompt_default_writing_rules(self, minimal_storyteller_config):
        """Default writing rules when domain_writing_rules is empty."""
        from versifai.story_agents.storyteller.prompts import build_storyteller_system_prompt

        prompt = build_storyteller_system_prompt(minimal_storyteller_config)
        assert "STATE FINDINGS DIRECTLY" in prompt

    def test_system_prompt_default_citation_guidance(self, minimal_storyteller_config):
        """Default citation guidance when citation_source_guidance is empty."""
        from versifai.story_agents.storyteller.prompts import build_storyteller_system_prompt

        prompt = build_storyteller_system_prompt(minimal_storyteller_config)
        assert "peer-reviewed research" in prompt
        assert "government reports" in prompt

    def test_system_prompt_no_hardcoded_cms(self, minimal_storyteller_config):
        """System prompt without domain fields has no CMS/healthcare references."""
        from versifai.story_agents.storyteller.prompts import build_storyteller_system_prompt

        prompt = build_storyteller_system_prompt(minimal_storyteller_config)
        assert "CMS Stars methodology" not in prompt
        assert "CMS technical documents" not in prompt
        assert "GAO/OIG" not in prompt
        assert "Health Affairs" not in prompt
        assert "KFF" not in prompt
        assert "NBER" not in prompt
        assert "high-SVI and low-SVI counties" not in prompt


# ---------------------------------------------------------------------------
# Config Field Tests
# ---------------------------------------------------------------------------


class TestNewConfigFieldDefaults:
    """Verify new config fields have backward-compatible defaults."""

    def test_research_config_defaults(self):
        """New ResearchConfig fields default to empty."""
        cfg = ResearchConfig()
        assert cfg.agent_role == "Data Scientist"
        assert cfg.domain_context == ""
        assert cfg.analysis_method_guidance == {}

    def test_project_config_defaults(self):
        """New ProjectConfig fields default to empty."""
        cfg = ProjectConfig()
        assert cfg.column_naming_examples == ""
        assert cfg.grain_detection_guidance == ""

    def test_storyteller_config_defaults(self):
        """New StorytellerConfig fields default to empty."""
        cfg = StorytellerConfig()
        assert cfg.domain_writing_rules == ""
        assert cfg.citation_source_guidance == ""

    def test_silver_dataset_spec_default_join_key(self):
        """SilverDatasetSpec.join_key defaults to empty string, not 'county_fips_code'."""
        spec = SilverDatasetSpec(
            name="test",
            description="test",
            source_tables=["t"],
        )
        assert spec.join_key == ""
