"""
Project configuration framework for the DataEngineerAgent.

Defines the generic building blocks — ProjectConfig, JoinKeyConfig,
SourceProcessingHint, etc. — with no hardcoded domain content.
All domain-specific content lives in the configs/ directory as assembled
instances.

Usage in a notebook::

    # Custom assembly
    from versifai.data_agents.engineer.config import ProjectConfig, JoinKeyConfig
    cfg = ProjectConfig(
        name="My Project",
        catalog="my_catalog",
        schema="my_schema",
        volume_path="/Volumes/my_catalog/my_schema/raw_data",
        join_key=JoinKeyConfig(column_name="patient_id"),
    )
    agent = DataEngineerAgent(cfg=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Building blocks — standalone Lego pieces
# ---------------------------------------------------------------------------


@dataclass
class JoinKeyConfig:
    """Defines the primary join key for cross-table joins."""

    column_name: str = ""
    data_type: str = "STRING"
    description: str = ""
    width: int = 0
    validation_rule: str = ""
    expected_entity_count: int = 0

    # Additional geographic/dimensional columns the agent should look for
    related_columns: list[dict] = field(default_factory=list)


@dataclass
class AlternativeKeyConfig:
    """An alternative join key for tables at a non-county grain."""

    column_name: str
    description: str
    data_type: str = "STRING"
    grain: str = ""  # e.g., "contract", "plan", "state"


@dataclass
class MetadataColumnConfig:
    """Defines metadata columns automatically added to every table."""

    name: str
    data_type: str
    description: str
    nullable: bool = False


@dataclass
class DataSourceHint:
    """Optional hint about a known data source — helps the agent recognize it."""

    name: str
    description: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class SourceFileHint:
    """Describes a specific file expected within a source directory/archive."""

    file_pattern: str  # filename substring to match (e.g., "Part C Cut Point")
    target_table: str  # table name to create (e.g., "silver_star_cut_points")
    description: str  # what this file contains
    used_in: str = ""  # which themes/silver tables use this


@dataclass
class SourceProcessingHint:
    """
    Per-source processing instructions for a specific data source directory.

    When multi_table=True, the agent should create a SEPARATE table for each
    file matching the hints, rather than combining everything into one table.
    """

    source_pattern: str  # directory name pattern to match (e.g., "star_rating")
    description: str  # overall description of this source
    multi_table: bool = False  # True if this source should produce multiple tables
    files: list[SourceFileHint] = field(default_factory=list)
    notes: str = ""  # additional processing notes


# ---------------------------------------------------------------------------
# ProjectConfig — the container that the DataEngineerAgent consumes
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    """
    Configuration for a data engineering project.

    The DataEngineerAgent uses this to drive its data processing pipeline.
    Assemble one from building blocks (join keys, source hints, metadata)
    and pass it to the agent. The agent code is generic — all domain
    knowledge lives in the config instance.
    """

    # ── Project identity ────────────────────────────────────────
    name: str = ""
    description: str = ""
    domain_expertise: str = ""
    analyst_specialty: str = ""

    # ── Unity Catalog target ────────────────────────────────────
    catalog: str = ""
    schema: str = ""

    # ── Data source location ────────────────────────────────────
    volume_path: str = ""
    staging_path: str = ""

    # ── Join key ────────────────────────────────────────────────
    join_key: JoinKeyConfig = field(default_factory=JoinKeyConfig)

    # ── Alternative keys (for non-primary-grain tables) ─────────
    alternative_keys: list[AlternativeKeyConfig] = field(default_factory=list)

    # ── Geographic grain ────────────────────────────────────────
    geographic_grain: str = ""
    grain_description: str = ""

    # ── Staging auto-flush ───────────────────────────────────────
    staging_flush_threshold_rows: int = 30_000_000

    # ── Metadata columns ────────────────────────────────────────
    metadata_columns: list[MetadataColumnConfig] = field(default_factory=list)

    # ── Column naming conventions ───────────────────────────────
    naming_convention: str = "snake_case"
    naming_rules: str = ""

    # ── Known data sources (optional hints) ─────────────────────
    known_sources: list[DataSourceHint] = field(default_factory=list)

    # ── Documentation URLs for web search ─────────────────────
    documentation_urls: dict[str, list[str]] = field(default_factory=dict)

    # ── Source processing hints ───────────────────────────────────
    source_processing_hints: list[SourceProcessingHint] = field(default_factory=list)

    # ── Derived properties ──────────────────────────────────────

    @property
    def full_schema(self) -> str:
        """Fully qualified schema name."""
        return f"{self.catalog}.{self.schema}"

    def get_source_hint(self, source_name: str) -> SourceProcessingHint | None:
        """Find a matching SourceProcessingHint for a given source directory name."""
        name_lower = source_name.lower()
        for hint in self.source_processing_hints:
            if hint.source_pattern.lower() in name_lower:
                return hint
        return None

    def format_source_hint(self, source_name: str) -> str:
        """Format the source processing hint as prompt text, or empty string if none."""
        hint = self.get_source_hint(source_name)
        if not hint:
            return ""
        lines = [
            "\n## IMPORTANT: Multi-Table Source Processing Instructions",
            f"\n**{hint.description}**\n",
        ]
        if hint.multi_table:
            lines.append(
                "This source contains MULTIPLE file types. Each file type must "
                "become its OWN SEPARATE TABLE. Do NOT combine them into one table.\n"
            )
        lines.append("### Files to Extract and Process:\n")
        for f in hint.files:
            lines.append(
                f"| **{f.file_pattern}** → `{f.target_table}` | "
                f"{f.description} | Used in: {f.used_in} |"
            )
        lines.append("")
        if hint.notes:
            lines.append(f"### Processing Notes:\n{hint.notes}\n")
        lines.append(
            "### Workflow:\n"
            "1. Extract ALL archives in this source directory.\n"
            "2. Inventory ALL extracted files — match each to a file type above "
            "using the file_pattern.\n"
            "3. **SKIP files that don't match any pattern** — do not create tables "
            "for unrecognized files.\n"
            "4. For EACH file type: group matching files across all years, "
            "profile the most recent, design one schema, and batch process "
            "all years into that table.\n"
            "5. You will make multiple write_to_catalog calls — one per target table."
        )
        return "\n".join(lines)

    @property
    def known_sources_text(self) -> str:
        """Formatted text block listing known data sources for prompts."""
        if not self.known_sources:
            return "No known data sources configured. Discover and identify them from the files."
        lines = []
        for src in self.known_sources:
            lines.append(f"- **{src.name}**: {src.description}")
        return "\n".join(lines)

    @property
    def metadata_columns_text(self) -> str:
        """Formatted text block listing metadata columns for prompts."""
        lines = []
        for col in self.metadata_columns:
            lines.append(f"- `{col.name}` — {col.description}")
        return "\n".join(lines)

    @property
    def alternative_keys_text(self) -> str:
        """Formatted text about recognized alternative keys for prompts."""
        if not self.alternative_keys:
            return ""
        lines = []
        for key in self.alternative_keys:
            lines.append(f"- `{key.column_name}` ({key.grain}-level): {key.description}")
        return "\n".join(lines)

    @property
    def alternative_key_names(self) -> list[str]:
        """List of alternative key column names."""
        return [k.column_name for k in self.alternative_keys]

    @property
    def join_key_related_text(self) -> str:
        """Formatted text about related geographic columns."""
        lines = []
        for col in self.join_key.related_columns:
            req = " (required)" if col.get("required") else ""
            lines.append(f"- Include `{col['name']}` ({col['description']}){req} when available.")
        return "\n".join(lines)

    @property
    def metadata_column_names(self) -> list[str]:
        """List of metadata column name strings."""
        return [col.name for col in self.metadata_columns]

    @property
    def expected_tables(self) -> list[str]:
        """All tables the pipeline MUST produce (from source processing hints)."""
        tables: set[str] = set()
        for hint in self.source_processing_hints:
            for f in hint.files:
                tables.add(f.target_table)
        return sorted(tables)
