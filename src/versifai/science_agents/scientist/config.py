"""
Research configuration framework for the DataScientistAgent.

Defines the generic building blocks — ResearchConfig, AnalysisTheme,
SilverDatasetSpec, ResearchReference — with no hardcoded domain content.
All domain-specific content lives in the configs/ directory as assembled
instances.

Usage in a notebook::

    # Custom assembly
    from versifai.science_agents.scientist.config import ResearchConfig, AnalysisTheme
    cfg = ResearchConfig(
        name="My Analysis",
        thesis="...",
        analysis_themes=[AnalysisTheme(id="theme_0", ...)],
    )
    agent = DataScientistAgent(cfg=cfg)
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
class ResearchQuestion:
    """A specific research question the agent should investigate."""
    id: str
    question: str
    required_tables: list[str] = field(default_factory=list)
    analysis_type: str = "descriptive"  # "descriptive", "comparative", "correlation", "trend", "simulation"


@dataclass
class SilverDatasetSpec:
    """Defines a silver-layer pre-joined dataset the agent should build."""
    name: str
    description: str
    source_tables: list[str] = field(default_factory=list)
    join_key: str = "county_fips_code"
    time_column: str = "source_year"
    data_notes: str = ""  # Domain-specific hints about data quirks, how to combine tables, etc.


@dataclass
class ResearchReference:
    """A published research paper or report relevant to the thesis."""
    title: str
    url: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class AnalysisTheme:
    """
    A major analysis theme with detailed methodology, required tables,
    and the argument it builds toward the thesis.

    Each theme becomes one phase of the research analysis pipeline.

    OUTPUT PHILOSOPHY:
    - tables_to_produce: The PRIMARY analytical output. Every theme must
      produce well-formatted summary tables with statistics, p-values,
      and effect sizes. These carry the argument.
    - signature_visualization: The ONE dense, multi-dimensional, irreplaceable
      chart for this theme (if any). Not every theme gets one — only 6 themes
      have prescribed signature visualizations.
    - visualization_notes: Guidance on what NOT to visualize, and encouragement
      to create custom discovery-driven visuals when the agent finds something
      unexpected in the data.
    """
    id: str
    title: str
    question: str
    analysis_steps: list[str] = field(default_factory=list)
    tables_to_produce: list[str] = field(default_factory=list)
    signature_visualization: str = ""  # The ONE dense, irreplaceable chart (if any)
    visualization_notes: str = ""  # What NOT to chart + discovery encouragement
    punchline: str = ""
    required_tables: list[str] = field(default_factory=list)
    data_notes: str = ""  # Domain-specific hints about data quirks, how to combine tables, etc.
    analysis_type: str = "comparative"
    sequence: int = 1


# ---------------------------------------------------------------------------
# ResearchConfig — the container that the DataScientistAgent consumes
# ---------------------------------------------------------------------------


@dataclass
class ResearchConfig:
    """
    Configuration for a research analysis project.

    The DataScientistAgent uses this to drive its analysis workflow.
    Assemble one from building blocks (themes, silver datasets, references)
    and pass it to the agent. The agent code is generic — all domain
    knowledge lives in the config instance.

    Usage::

        cfg = ResearchConfig(
            name="My Analysis",
            thesis="...",
            analysis_themes=[theme_a, theme_b],
            silver_datasets=[silver_county_master, silver_contract_year],
            research_references=[ref_kff, ref_cms],
        )
        agent = DataScientistAgent(cfg=cfg)
        agent.run()
    """

    # ── Research identity ────────────────────────────────────────
    name: str = ""
    thesis: str = ""

    # ── Visualization philosophy ─────────────────────────────────
    visualization_guidance: str = ""

    # ── Project config (catalog, schema, join key, etc.) ─────────
    project: ProjectConfig = field(default_factory=lambda: _default_project())

    # ── Results storage ──────────────────────────────────────────
    results_volume_path: str = ""

    # ── Analysis content (assembled from building blocks) ────────
    analysis_themes: list[AnalysisTheme] = field(default_factory=list)
    silver_datasets: list[SilverDatasetSpec] = field(default_factory=list)
    research_references: list[ResearchReference] = field(default_factory=list)

    # ── Agent behaviour ──────────────────────────────────────────
    max_turns: int = 150
    max_turns_per_phase: int = 120
    max_turns_per_theme: int = 120

    # ── Run isolation ─────────────────────────────────────────────
    run_id: str = ""  # Empty = write directly to results_volume_path (backward compat)

    # ── Dependencies on prior agent runs ──────────────────────────
    dependencies: list[AgentDependency] = field(default_factory=list)

    # ── MLflow integration ────────────────────────────────────────
    mlflow_experiment: str = ""       # e.g. "/Shared/versifai/expansion_propensity"
    mlflow_registry_name: str = ""    # e.g. "catalog.schema.model_name"

    # ── Visualization settings ───────────────────────────────────
    chart_style: str = "seaborn-v0_8-whitegrid"
    chart_dpi: int = 150
    color_palette: str = "viridis"

    # ── Derived properties ───────────────────────────────────────

    @property
    def analysis_themes_text(self) -> str:
        """Formatted text listing all analysis themes for prompts."""
        lines = []
        for theme in sorted(self.analysis_themes, key=lambda t: t.sequence):
            lines.append(
                f"- **Theme {theme.sequence}: {theme.title}** "
                f"[{theme.analysis_type}]: {theme.question}"
            )
        return "\n".join(lines)

    @property
    def analysis_themes_detail_text(self) -> str:
        """Detailed theme descriptions including steps, tables, visuals, and punchlines."""
        sections = []
        for theme in sorted(self.analysis_themes, key=lambda t: t.sequence):
            steps = "\n".join(f"   {i}. {s}" for i, s in enumerate(theme.analysis_steps, 1))
            tables = "\n".join(f"   - {t}" for t in theme.tables_to_produce)
            sig_viz = (
                f"\n**Signature Visualization**:\n{theme.signature_visualization}"
                if theme.signature_visualization else ""
            )
            viz_notes = (
                f"\n**Visualization Notes**: {theme.visualization_notes}"
                if theme.visualization_notes else ""
            )
            sections.append(
                f"### Theme {theme.sequence}: {theme.title}\n"
                f"**Question**: {theme.question}\n"
                f"**Analysis**:\n{steps}\n"
                f"**Tables to Produce**:\n{tables}"
                f"{sig_viz}"
                f"{viz_notes}\n"
                f"**Punchline**: {theme.punchline}"
            )
        return "\n\n".join(sections)

    @property
    def delivery_sequence_text(self) -> str:
        """Delivery sequencing guidance for the synthesis phase."""
        lines = []
        for theme in sorted(self.analysis_themes, key=lambda t: t.sequence):
            lines.append(f"{theme.sequence}. **{theme.title}** — {theme.punchline}")
        return "\n".join(lines)

    @property
    def research_questions_text(self) -> str:
        """Backward-compatible text — derives from analysis_themes."""
        return self.analysis_themes_text

    @property
    def silver_datasets_text(self) -> str:
        lines = []
        for ds in self.silver_datasets:
            tables = ", ".join(ds.source_tables)
            lines.append(f"- **{ds.name}**: {ds.description} (joins: {tables})")
        return "\n".join(lines)

    @property
    def research_references_text(self) -> str:
        if not self.research_references:
            return "No reference documents configured."
        lines = []
        for ref in self.research_references:
            url_part = f" — {ref.url}" if ref.url else ""
            lines.append(f"- **{ref.title}**{url_part}\n  {ref.description}")
        return "\n".join(lines)

    @property
    def all_required_tables(self) -> list[str]:
        """Union of required_tables from all themes + source processing hint targets."""
        tables: set[str] = set()
        for theme in self.analysis_themes:
            tables.update(theme.required_tables)
        tables.update(self.project.expected_tables)
        return sorted(tables)


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
