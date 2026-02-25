"""
Shared configuration building blocks for the Versifai agent framework.

Provides base config classes and settings that can be used across all agent
families without introducing cross-package dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class CatalogConfig:
    """Shared Databricks Unity Catalog connection settings.

    Used by all agent families that need to read from or write to
    Unity Catalog tables and volumes.
    """

    catalog: str = ""
    schema: str = ""
    volume_path: str = ""
    staging_path: str = ""

    @property
    def databricks_host(self) -> str:
        """Databricks workspace host, read from environment."""
        return os.getenv("DATABRICKS_HOST", "")

    @property
    def databricks_token(self) -> str:
        """Databricks access token, read from environment."""
        return os.getenv("DATABRICKS_TOKEN", "")


@dataclass
class AgentSettings:
    """Tunable parameters for agent behaviour.

    Sensible defaults are provided. Override as needed when constructing
    agent instances.
    """

    max_agent_turns: int = 200
    max_turns_per_source: int = 120
    max_acceptance_iterations: int = 3
    sample_rows: int = 10
    profile_sample_size: int = 500


# ---------------------------------------------------------------------------
# Module-level convenience constants — importable without instantiation.
# ---------------------------------------------------------------------------
_DEFAULTS = AgentSettings()
MAX_AGENT_TURNS: int = _DEFAULTS.max_agent_turns
MAX_TURNS_PER_SOURCE: int = _DEFAULTS.max_turns_per_source
MAX_ACCEPTANCE_ITERATIONS: int = _DEFAULTS.max_acceptance_iterations
SAMPLE_ROWS: int = _DEFAULTS.sample_rows
PROFILE_SAMPLE_SIZE: int = _DEFAULTS.profile_sample_size
