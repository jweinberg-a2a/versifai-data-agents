"""
Agent state tracking: source processing status and overall progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SourceStatus(str, Enum):
    """Processing phases for a data source."""

    DISCOVERED = "discovered"
    EXTRACTING = "extracting"
    PROFILING = "profiling"
    SCHEMA_DESIGNING = "schema_designing"
    TRANSFORMING = "transforming"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    NEEDS_HUMAN_INPUT = "needs_human_input"


@dataclass
class SourceState:
    """Tracks the processing state of a single data source."""

    source_name: str
    status: SourceStatus = SourceStatus.DISCOVERED
    turns_used: int = 0
    errors: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    files_processed: int = 0
    files_total: int = 0
    rows_loaded: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    table_name: str = ""

    def record_error(self, error: str) -> None:
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")

    def record_decision(self, decision: str) -> None:
        self.decisions.append(f"[{datetime.now().isoformat()}] {decision}")


@dataclass
class AgentState:
    """
    Global state of the agent across all data sources.

    Persisted so the agent can resume if interrupted.
    """

    sources: dict[str, SourceState] = field(default_factory=dict)
    current_source: str | None = None
    total_turns: int = 0
    started_at: datetime | None = None
    phase: str = "discovery"  # discovery | processing | validation | complete

    def get_or_create_source(self, name: str) -> SourceState:
        if name not in self.sources:
            self.sources[name] = SourceState(source_name=name)
        return self.sources[name]

    def set_current_source(self, name: str) -> SourceState:
        self.current_source = name
        source = self.get_or_create_source(name)
        if source.started_at is None:
            source.started_at = datetime.now()
        return source

    def mark_source_complete(self, name: str, table_name: str, rows: int) -> None:
        source = self.sources[name]
        source.status = SourceStatus.LOADED
        source.completed_at = datetime.now()
        source.table_name = table_name
        source.rows_loaded = rows

    def mark_source_failed(self, name: str, error: str) -> None:
        source = self.sources[name]
        source.status = SourceStatus.FAILED
        source.record_error(error)
        source.completed_at = datetime.now()

    @property
    def pending_sources(self) -> list[str]:
        return [
            name
            for name, s in self.sources.items()
            if s.status not in (SourceStatus.LOADED, SourceStatus.FAILED)
        ]

    @property
    def completed_sources(self) -> list[str]:
        return [name for name, s in self.sources.items() if s.status == SourceStatus.LOADED]

    @property
    def failed_sources(self) -> list[str]:
        return [name for name, s in self.sources.items() if s.status == SourceStatus.FAILED]

    def summary(self) -> str:
        lines = [
            f"Agent State — Phase: {self.phase}, Total Turns: {self.total_turns}",
            f"  Sources: {len(self.sources)} total, "
            f"{len(self.completed_sources)} loaded, "
            f"{len(self.failed_sources)} failed, "
            f"{len(self.pending_sources)} pending",
        ]
        for name, s in self.sources.items():
            marker = ">>>" if name == self.current_source else "   "
            lines.append(
                f"  {marker} {name}: {s.status.value} "
                f"({s.files_processed}/{s.files_total} files, "
                f"{s.rows_loaded} rows, {s.turns_used} turns)"
            )
        return "\n".join(lines)
