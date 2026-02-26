"""
Storyteller configuration framework for the StoryTellerAgent.

Defines the generic building blocks — StorytellerConfig, NarrativeSection,
StyleGuide, EvidenceThreshold, OutputFormat — with no hardcoded domain content.
All domain-specific content lives in the configs/ directory as assembled
instances.

Usage in a notebook::

    # Custom assembly
    from versifai.story_agents.storyteller.config import StorytellerConfig, NarrativeSection
    cfg = StorytellerConfig(
        name="My Narrative",
        thesis="...",
        narrative_sections=[NarrativeSection(id="intro", ...)],
    )
    agent = StoryTellerAgent(cfg=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from versifai.core.run_manager import AgentDependency
    from versifai.data_agents.engineer.config import ProjectConfig


# ---------------------------------------------------------------------------
# Building blocks — standalone Lego pieces
# ---------------------------------------------------------------------------


@dataclass
class NarrativeSection:
    """One section of the narrative report."""

    id: str  # "section_stakes"
    title: str  # "The $16 Billion Question"
    purpose: str  # What this section accomplishes
    source_theme_ids: list[str]  # ["theme_0"] — which DataScientist themes feed it
    tone: str = "analytical"  # analytical / persuasive / urgent
    max_words: int = 1500
    key_evidence: str = ""  # What specific evidence to lead with
    charts_to_include: list[str] = field(default_factory=list)
    charts_to_reconfigure: list[str] = field(default_factory=list)
    narrative_guidance: str = ""  # Detailed writing instructions
    transition_from: str = ""  # How to connect from previous section
    transition_to: str = ""  # Setup for next section
    sequence: int = 0


@dataclass
class EvidenceThreshold:
    """Rules for when evidence is strong enough to cite."""

    min_significance_for_lead: str = "high"
    min_significance_for_support: str = "medium"
    require_effect_size: bool = True
    max_unsupported_claims: int = 0


@dataclass
class StyleGuide:
    """Voice, audience, and writing rules for the narrative."""

    voice: str = "third-person analytical"
    audience: str = ""
    reading_level: str = "professional"
    citation_style: str = "inline"

    # Document purpose and tone
    document_type: str = ""
    purpose: str = ""
    tone_guidance: str = ""
    anti_patterns: str = ""


@dataclass
class OutputFormat:
    """How the final document is assembled and exported."""

    format: str = "markdown"
    filename: str = "narrative_report.md"
    include_toc: bool = True
    include_methodology_appendix: bool = True
    include_data_sources_appendix: bool = True
    chart_reference_style: str = "relative_path"  # relative_path or absolute


# ---------------------------------------------------------------------------
# StorytellerConfig — the container that the StoryTellerAgent consumes
# ---------------------------------------------------------------------------


@dataclass
class StorytellerConfig:
    """
    Configuration for a narrative report project.

    The StoryTellerAgent uses this to drive its writing workflow.
    Assemble one from building blocks (sections, style guide, evidence rules)
    and pass it to the agent. The agent code is generic — all domain
    knowledge lives in the config instance.
    """

    # ── Report identity ────────────────────────────────────────
    name: str = ""
    thesis: str = ""

    # ── Paths — where DataScientist outputs live and where narrative goes
    research_results_path: str = ""
    narrative_output_path: str = ""

    # ── Project config (catalog, schema, join key, etc.) ─────────
    project: ProjectConfig = field(default_factory=lambda: _default_project())

    # ── Narrative arc ─────────────────────────────────────────────
    narrative_sections: list[NarrativeSection] = field(default_factory=list)

    # ── Evidence & style ──────────────────────────────────────────
    evidence_threshold: EvidenceThreshold = field(default_factory=EvidenceThreshold)
    style_guide: StyleGuide = field(default_factory=StyleGuide)
    output_format: OutputFormat = field(default_factory=OutputFormat)

    # ── Known external references ─────────────────────────────────
    citation_urls: list[str] = field(default_factory=list)

    # ── Domain-specific editorial guidance ─────────────────────────
    # Injected as a writing rule (e.g. "IDENTIFY FLAWS DIRECTLY: This paper
    # exists to point out critical structural flaws in..."). When empty,
    # a generic "state findings directly" rule is used.
    domain_writing_rules: str = ""

    # ── Preferred citation sources ─────────────────────────────────
    # What kinds of sources to prioritize (e.g. "CMS technical documents,
    # GAO/OIG reports, Health Affairs, KFF, NBER"). When empty, generic
    # "peer-reviewed research, government reports" is used.
    citation_source_guidance: str = ""

    # ── Turn budgets ──────────────────────────────────────────────
    max_turns_per_section: int = 60
    max_turns_per_phase: int = 80
    coherence_pass_max_turns: int = 40

    # ── Editor pass budgets ────────────────────────────────────────
    editor_max_turns_per_section: int = 40
    editor_max_turns_overview: int = 30

    # ── Run isolation ─────────────────────────────────────────────
    run_id: str = ""  # Empty = write directly to narrative_output_path (backward compat)

    # ── Dependencies on prior agent runs ──────────────────────────
    dependencies: list[AgentDependency] = field(default_factory=list)

    # ── Compatibility properties ─────────────────────────────────

    @property
    def results_volume_path(self) -> str:
        """Alias for CreateVisualizationTool compatibility.

        CreateVisualizationTool reads cfg.results_volume_path to determine
        where to write charts/ and tables/. For the storyteller, new charts
        and tables are co-located with the DataScientist's outputs.
        """
        return self.research_results_path

    # ── Prompt-text properties ────────────────────────────────────

    @property
    def sections_text(self) -> str:
        """Formatted section list for prompt injection."""
        lines = []
        for s in sorted(self.narrative_sections, key=lambda x: x.sequence):
            themes = ", ".join(s.source_theme_ids)
            lines.append(
                f"  {s.sequence}. **{s.title}** (id={s.id}, themes=[{themes}], "
                f"tone={s.tone}, max_words={s.max_words})\n"
                f"     Purpose: {s.purpose}\n"
                f"     Key evidence: {s.key_evidence or '(agent decides)'}"
            )
        return "\n".join(lines)

    @property
    def style_prompt_text(self) -> str:
        """Style guide formatted for prompt injection."""
        sg = self.style_guide
        return (
            f"Document type: {sg.document_type}\n"
            f"Purpose: {sg.purpose}\n\n"
            f"Voice: {sg.voice}\n"
            f"Audience: {sg.audience}\n"
            f"Reading level: {sg.reading_level}\n"
            f"Citation style: {sg.citation_style}\n\n"
            f"Tone: {sg.tone_guidance}\n\n"
            f"Anti-patterns (DO NOT DO THESE):\n{sg.anti_patterns}"
        )

    @property
    def evidence_prompt_text(self) -> str:
        """Evidence rules formatted for prompt injection."""
        et = self.evidence_threshold
        return (
            f"- Lead evidence must be: {et.min_significance_for_lead} significance\n"
            f"- Supporting evidence must be: {et.min_significance_for_support} significance\n"
            f"- Require effect size for quantitative claims: {et.require_effect_size}\n"
            f"- Maximum unsupported claims per section: {et.max_unsupported_claims}"
        )


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------


def _default_project() -> ProjectConfig:
    """Lazy default — returns an empty ProjectConfig.

    Callers should always pass an explicit project config; this exists
    only to satisfy the dataclass default_factory contract.
    """
    from versifai.data_agents.engineer.config import ProjectConfig

    return ProjectConfig()
