"""
Run management utilities for agent output isolation and inter-agent dependencies.

Provides run-isolated output directories so each agent execution gets its own
timestamped folder. Also defines AgentDependency for declaring which prior
agent run's outputs a config consumes.

Usage::

    from versifai.core.run_manager import generate_run_id, init_run_directory

    run_id = generate_run_id()
    run_path = init_run_directory("/Volumes/.../results/geographic_disparity", run_id)
    # -> /Volumes/.../results/geographic_disparity/runs/20260223_143012_a1b2/

    # Resolve a dependency to the latest run
    from versifai.core.run_manager import AgentDependency, resolve_dependency
    dep = AgentDependency(
        agent_type="scientist",
        config_name="geographic_disparity",
        base_path="/Volumes/.../results/geographic_disparity",
    )
    path = resolve_dependency(dep)
    # -> /Volumes/.../results/geographic_disparity/runs/20260223_143012_a1b2/
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a unique run ID: YYYYMMDD_HHMMSS_XXXX.

    Format: timestamp (to second) + 4 random hex chars for uniqueness.
    Lexicographic sort = chronological sort.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)  # 4 hex chars
    return f"{ts}_{suffix}"


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def init_run_directory(base_path: str, run_id: str) -> str:
    """Create a run-isolated output directory.

    Creates ``base_path/runs/{run_id}/`` with standard subdirectories:
    charts/, tables/, notes/, models/.

    Returns the full run directory path.
    """
    run_path = os.path.join(base_path, "runs", run_id)
    for subdir in ("charts", "tables", "notes", "models"):
        os.makedirs(os.path.join(run_path, subdir), exist_ok=True)
    return run_path


def resolve_run_path(base_path: str, run_id: str | None = None) -> str:
    """Resolve a run directory path.

    If ``run_id`` is given, returns ``base_path/runs/{run_id}/``.
    If None, finds the most recent run by lexicographic sort of run IDs.

    Raises FileNotFoundError if no runs exist.
    """
    if run_id:
        path = os.path.join(base_path, "runs", run_id)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Run directory not found: {path}"
            )
        return path

    runs_dir = os.path.join(base_path, "runs")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(
            f"No runs directory found at {runs_dir}"
        )

    run_ids = sorted(
        d for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    )
    if not run_ids:
        raise FileNotFoundError(
            f"No runs found in {runs_dir}"
        )

    return os.path.join(runs_dir, run_ids[-1])


def list_runs(base_path: str) -> list[dict]:
    """List all runs under ``base_path/runs/``.

    Returns a list of dicts: [{run_id, path, created, metadata}].
    """
    runs_dir = os.path.join(base_path, "runs")
    if not os.path.isdir(runs_dir):
        return []

    results = []
    for run_id in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_id)
        if not os.path.isdir(run_path):
            continue

        entry: dict = {
            "run_id": run_id,
            "path": run_path,
        }

        # Try to read metadata
        meta_path = os.path.join(run_path, "run_metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    entry["metadata"] = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def write_run_metadata(
    run_path: str,
    config_name: str,
    run_id: str,
    agent_type: str = "",
    extra: dict | None = None,
) -> str:
    """Write run_metadata.json to the run directory.

    Called at run start with started_at, and updated at run end.
    Returns the metadata file path.
    """
    meta_path = os.path.join(run_path, "run_metadata.json")

    # Load existing metadata if updating
    metadata: dict = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Set/update fields
    metadata["run_id"] = run_id
    metadata["config_name"] = config_name
    if agent_type:
        metadata["agent_type"] = agent_type
    if "started_at" not in metadata:
        metadata["started_at"] = datetime.now().isoformat()

    if extra:
        metadata.update(extra)

    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return meta_path


# ---------------------------------------------------------------------------
# Run state tracking (for stop / resume)
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Tracks execution state of an agent run for stop/resume support.

    Persisted inside ``run_metadata.json`` under the ``"state"`` key so
    there is a single file to read on resume.

    Typical lifecycle::

        state = RunState(entry_point="run")
        state.mark_phase_start("orientation")
        state.mark_phase_complete("orientation")
        state.mark_phase_start("silver")
        state.mark_item_complete("silver", "silver_county_master")
        ...
        state.mark_completed()      # or mark_interrupted() / mark_failed()
    """

    status: str = "running"          # running | completed | failed | interrupted
    entry_point: str = "run"         # run | run_themes | run_visualizations | ...
    current_phase: str = ""
    current_item: str = ""
    completed_phases: list[str] = field(default_factory=list)
    completed_items: dict[str, list[str]] = field(default_factory=lambda: {})
    error: str = ""
    updated_at: str = ""

    # -- Mutations --------------------------------------------------------

    def mark_phase_start(self, phase: str, item: str = "") -> None:
        self.current_phase = phase
        self.current_item = item
        self.updated_at = datetime.now().isoformat()

    def mark_item_complete(self, phase: str, item: str) -> None:
        if phase not in self.completed_items:
            self.completed_items[phase] = []
        if item not in self.completed_items[phase]:
            self.completed_items[phase].append(item)
        self.current_item = ""
        self.updated_at = datetime.now().isoformat()

    def mark_phase_complete(self, phase: str) -> None:
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
        self.current_phase = ""
        self.current_item = ""
        self.updated_at = datetime.now().isoformat()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.updated_at = datetime.now().isoformat()

    def mark_interrupted(self) -> None:
        self.status = "interrupted"
        self.updated_at = datetime.now().isoformat()

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "entry_point": self.entry_point,
            "current_phase": self.current_phase,
            "current_item": self.current_item,
            "completed_phases": list(self.completed_phases),
            "completed_items": {k: list(v) for k, v in self.completed_items.items()},
            "error": self.error or None,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunState:
        return cls(
            status=data.get("status", "running"),
            entry_point=data.get("entry_point", "run"),
            current_phase=data.get("current_phase", ""),
            current_item=data.get("current_item", ""),
            completed_phases=list(data.get("completed_phases", [])),
            completed_items={
                k: list(v) for k, v in data.get("completed_items", {}).items()
            },
            error=data.get("error") or "",
            updated_at=data.get("updated_at", ""),
        )


def save_run_state(run_path: str, state: RunState) -> None:
    """Persist *state* into ``run_metadata.json`` under the ``"state"`` key.

    Reads the existing metadata, merges the state, and writes back.
    """
    meta_path = os.path.join(run_path, "run_metadata.json")
    metadata: dict = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    metadata["state"] = state.to_dict()
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_run_state(run_path: str) -> RunState | None:
    """Load :class:`RunState` from ``run_metadata.json``.

    Returns ``None`` if the file doesn't exist, has no ``"state"`` key,
    or is corrupt.
    """
    meta_path = os.path.join(run_path, "run_metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    state_data = metadata.get("state")
    if not state_data or not isinstance(state_data, dict):
        return None
    return RunState.from_dict(state_data)


# ---------------------------------------------------------------------------
# Inter-agent dependency
# ---------------------------------------------------------------------------

@dataclass
class AgentDependency:
    """Declares that a config consumes outputs from a previous agent run.

    Used by orchestrators to resolve the concrete path to a prior run's
    outputs (findings, charts, tables, notes) at startup time.

    Example::

        AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path="/Volumes/.../research_results/geographic_disparity",
        )
    """

    agent_type: str          # "engineer", "scientist", "storyteller"
    config_name: str         # "geographic_disparity", "kill_zone_H5216"
    run_id: str = ""         # Specific run ID. Empty = resolve latest.
    base_path: str = ""      # Base results path for that agent.
    outputs: list[str] = field(default_factory=list)


def resolve_dependency(dep: AgentDependency) -> str:
    """Resolve a dependency to its concrete run directory path.

    If ``dep.run_id`` is set, returns ``dep.base_path/runs/{dep.run_id}/``.
    If empty, finds the latest run under ``dep.base_path/runs/``.
    If no runs directory exists, returns ``dep.base_path`` directly
    (backward compatibility with pre-run-isolation outputs).
    """
    if not dep.base_path:
        raise ValueError(
            f"AgentDependency for '{dep.config_name}' has no base_path set."
        )

    # If a specific run_id is requested
    if dep.run_id:
        run_path = os.path.join(dep.base_path, "runs", dep.run_id)
        if os.path.isdir(run_path):
            return run_path
        raise FileNotFoundError(
            f"Run '{dep.run_id}' not found at {run_path}"
        )

    # Try to find the latest run
    runs_dir = os.path.join(dep.base_path, "runs")
    if os.path.isdir(runs_dir):
        run_ids = sorted(
            d for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d))
        )
        if run_ids:
            return os.path.join(runs_dir, run_ids[-1])

    # No runs directory — fall back to base_path (pre-isolation outputs)
    return dep.base_path
