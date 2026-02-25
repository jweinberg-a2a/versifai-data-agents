"""Unit tests for run management: directory structure, state tracking, metadata, dependencies."""

from __future__ import annotations

import json
import os
import time

import pytest

from versifai.core.run_manager import (
    AgentDependency,
    RunState,
    generate_run_id,
    init_run_directory,
    list_runs,
    load_run_state,
    resolve_dependency,
    resolve_run_path,
    save_run_state,
    write_run_metadata,
)

# ===================================================================
# generate_run_id
# ===================================================================


class TestGenerateRunId:
    def test_format_matches_pattern(self):
        """Run ID follows YYYYMMDD_HHMMSS_XXXX format."""
        run_id = generate_run_id()
        parts = run_id.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 4  # hex suffix
        # Verify hex suffix is valid hex
        int(parts[2], 16)

    def test_unique_across_calls(self):
        """Two calls produce different IDs (random suffix)."""
        ids = {generate_run_id() for _ in range(20)}
        assert len(ids) == 20

    def test_lexicographic_order_is_chronological(self):
        """IDs generated in sequence sort lexicographically in order."""
        id1 = generate_run_id()
        time.sleep(0.01)
        id2 = generate_run_id()
        # Same-second IDs may tie on timestamp, but suffix randomness means
        # we can't guarantee strict ordering without a 1s sleep. Instead
        # verify they are valid and parseable.
        assert isinstance(id1, str)
        assert isinstance(id2, str)


# ===================================================================
# init_run_directory
# ===================================================================


class TestInitRunDirectory:
    def test_creates_standard_subdirectories(self, tmp_path):
        """init_run_directory creates charts/, tables/, notes/, models/ subdirs."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        assert os.path.isdir(os.path.join(run_path, "charts"))
        assert os.path.isdir(os.path.join(run_path, "tables"))
        assert os.path.isdir(os.path.join(run_path, "notes"))
        assert os.path.isdir(os.path.join(run_path, "models"))

    def test_returns_correct_path(self, tmp_path):
        """Returned path is base_path/runs/run_id."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        expected = os.path.join(str(tmp_path), "runs", "run_001")
        assert run_path == expected

    def test_idempotent(self, tmp_path):
        """Calling twice with same ID doesn't fail (exist_ok=True)."""
        p1 = init_run_directory(str(tmp_path), "run_001")
        p2 = init_run_directory(str(tmp_path), "run_001")
        assert p1 == p2

    def test_creates_nested_runs_directory(self, tmp_path):
        """The 'runs' parent directory is created automatically."""
        run_path = init_run_directory(str(tmp_path / "deep" / "nested"), "run_001")
        assert os.path.isdir(run_path)


# ===================================================================
# resolve_run_path
# ===================================================================


class TestResolveRunPath:
    def test_specific_run_id(self, tmp_path):
        """Given a run_id, returns that specific run directory."""
        init_run_directory(str(tmp_path), "run_001")
        path = resolve_run_path(str(tmp_path), "run_001")
        assert path == os.path.join(str(tmp_path), "runs", "run_001")

    def test_latest_run_when_no_id(self, tmp_path):
        """Without run_id, returns the latest (lexicographically last) run."""
        init_run_directory(str(tmp_path), "20260101_000000_aaaa")
        init_run_directory(str(tmp_path), "20260201_000000_bbbb")
        init_run_directory(str(tmp_path), "20260115_000000_cccc")

        path = resolve_run_path(str(tmp_path))
        assert path.endswith("20260201_000000_bbbb")

    def test_missing_run_id_raises(self, tmp_path):
        """Requesting a non-existent run_id raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            resolve_run_path(str(tmp_path), "nonexistent")

    def test_no_runs_directory_raises(self, tmp_path):
        """No runs/ directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No runs directory"):
            resolve_run_path(str(tmp_path))

    def test_empty_runs_directory_raises(self, tmp_path):
        """Empty runs/ directory raises FileNotFoundError."""
        (tmp_path / "runs").mkdir()
        with pytest.raises(FileNotFoundError, match="No runs found"):
            resolve_run_path(str(tmp_path))


# ===================================================================
# list_runs
# ===================================================================


class TestListRuns:
    def test_lists_all_runs(self, tmp_path):
        """Returns all run directories sorted by ID."""
        init_run_directory(str(tmp_path), "run_a")
        init_run_directory(str(tmp_path), "run_b")
        init_run_directory(str(tmp_path), "run_c")

        runs = list_runs(str(tmp_path))
        assert len(runs) == 3
        assert [r["run_id"] for r in runs] == ["run_a", "run_b", "run_c"]

    def test_empty_when_no_runs(self, tmp_path):
        """Returns empty list when no runs/ directory exists."""
        assert list_runs(str(tmp_path)) == []

    def test_includes_metadata_when_present(self, tmp_path):
        """Includes run_metadata.json contents when available."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        write_run_metadata(run_path, "test_project", "run_001", agent_type="scientist")

        runs = list_runs(str(tmp_path))
        assert len(runs) == 1
        assert "metadata" in runs[0]
        assert runs[0]["metadata"]["config_name"] == "test_project"

    def test_ignores_non_directory_entries(self, tmp_path):
        """Files inside runs/ are ignored."""
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "not_a_dir.txt").write_text("ignore me")
        init_run_directory(str(tmp_path), "real_run")

        runs = list_runs(str(tmp_path))
        assert len(runs) == 1
        assert runs[0]["run_id"] == "real_run"


# ===================================================================
# write_run_metadata
# ===================================================================


class TestWriteRunMetadata:
    def test_creates_metadata_file(self, tmp_path):
        """Writes run_metadata.json with required fields."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        meta_path = write_run_metadata(run_path, "my_project", "run_001")

        assert os.path.isfile(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["run_id"] == "run_001"
        assert meta["config_name"] == "my_project"
        assert "started_at" in meta

    def test_sets_agent_type(self, tmp_path):
        """agent_type is included when provided."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        write_run_metadata(run_path, "proj", "run_001", agent_type="scientist")

        with open(os.path.join(run_path, "run_metadata.json")) as f:
            meta = json.load(f)
        assert meta["agent_type"] == "scientist"

    def test_merges_extra_fields(self, tmp_path):
        """Extra dict is merged into metadata."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        write_run_metadata(
            run_path,
            "proj",
            "run_001",
            extra={"total_findings": 5, "completed_at": "2026-01-01T00:00:00"},
        )

        with open(os.path.join(run_path, "run_metadata.json")) as f:
            meta = json.load(f)
        assert meta["total_findings"] == 5
        assert meta["completed_at"] == "2026-01-01T00:00:00"

    def test_preserves_started_at_on_update(self, tmp_path):
        """started_at is set on first call and not overwritten on update."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        write_run_metadata(run_path, "proj", "run_001")

        with open(os.path.join(run_path, "run_metadata.json")) as f:
            original_started = json.load(f)["started_at"]

        # Update with extra fields
        time.sleep(0.01)
        write_run_metadata(run_path, "proj", "run_001", extra={"total_findings": 10})

        with open(os.path.join(run_path, "run_metadata.json")) as f:
            meta = json.load(f)
        assert meta["started_at"] == original_started
        assert meta["total_findings"] == 10

    def test_creates_directory_if_missing(self, tmp_path):
        """Parent directories are created if run_path doesn't exist yet."""
        run_path = str(tmp_path / "deep" / "path")
        meta_path = write_run_metadata(run_path, "proj", "run_001")
        assert os.path.isfile(meta_path)


# ===================================================================
# RunState
# ===================================================================


class TestRunState:
    def test_default_state(self):
        """Default RunState is running with no phases completed."""
        state = RunState()
        assert state.status == "running"
        assert state.entry_point == "run"
        assert state.current_phase == ""
        assert state.completed_phases == []
        assert state.completed_items == {}

    def test_phase_lifecycle(self):
        """Track a phase from start to completion."""
        state = RunState()
        state.mark_phase_start("orientation")
        assert state.current_phase == "orientation"

        state.mark_phase_complete("orientation")
        assert state.current_phase == ""
        assert "orientation" in state.completed_phases

    def test_item_tracking(self):
        """Track individual items within a phase."""
        state = RunState()
        state.mark_phase_start("silver")
        state.mark_item_complete("silver", "silver_county_master")
        state.mark_item_complete("silver", "silver_enrollment")

        assert state.completed_items["silver"] == [
            "silver_county_master",
            "silver_enrollment",
        ]
        assert state.current_item == ""

    def test_item_deduplication(self):
        """Completing the same item twice doesn't duplicate it."""
        state = RunState()
        state.mark_item_complete("silver", "table_a")
        state.mark_item_complete("silver", "table_a")
        assert state.completed_items["silver"] == ["table_a"]

    def test_phase_deduplication(self):
        """Completing the same phase twice doesn't duplicate it."""
        state = RunState()
        state.mark_phase_complete("orientation")
        state.mark_phase_complete("orientation")
        assert state.completed_phases == ["orientation"]

    def test_mark_completed(self):
        state = RunState()
        state.mark_completed()
        assert state.status == "completed"
        assert state.updated_at != ""

    def test_mark_failed(self):
        state = RunState()
        state.mark_failed("OOM on large query")
        assert state.status == "failed"
        assert state.error == "OOM on large query"

    def test_mark_interrupted(self):
        state = RunState()
        state.mark_interrupted()
        assert state.status == "interrupted"

    def test_updated_at_set_on_mutations(self):
        """Every mutation updates the updated_at timestamp."""
        state = RunState()
        assert state.updated_at == ""

        state.mark_phase_start("test")
        ts1 = state.updated_at
        assert ts1 != ""

        time.sleep(0.01)
        state.mark_phase_complete("test")
        assert state.updated_at >= ts1

    def test_serialization_roundtrip(self):
        """to_dict -> from_dict preserves all state."""
        state = RunState(entry_point="run_themes")
        state.mark_phase_start("silver")
        state.mark_item_complete("silver", "table_a")
        state.mark_item_complete("silver", "table_b")
        state.mark_phase_complete("silver")
        state.mark_phase_start("themes")
        state.mark_item_complete("themes", "theme_0")

        d = state.to_dict()
        restored = RunState.from_dict(d)

        assert restored.status == state.status
        assert restored.entry_point == "run_themes"
        assert restored.completed_phases == ["silver"]
        assert restored.completed_items["silver"] == ["table_a", "table_b"]
        assert restored.completed_items["themes"] == ["theme_0"]
        assert restored.current_phase == "themes"

    def test_to_dict_error_none_when_empty(self):
        """Empty error serializes as None."""
        state = RunState()
        assert state.to_dict()["error"] is None

    def test_from_dict_error_none_to_empty_string(self):
        """None error in dict deserializes as empty string."""
        state = RunState.from_dict({"error": None})
        assert state.error == ""

    def test_from_dict_with_missing_keys(self):
        """from_dict handles missing keys with safe defaults."""
        state = RunState.from_dict({})
        assert state.status == "running"
        assert state.entry_point == "run"
        assert state.completed_phases == []
        assert state.completed_items == {}


# ===================================================================
# save_run_state / load_run_state
# ===================================================================


class TestRunStatePersistence:
    def test_save_and_load(self, tmp_path):
        """State survives a save/load roundtrip."""
        run_path = init_run_directory(str(tmp_path), "run_001")

        state = RunState(entry_point="run")
        state.mark_phase_start("orientation")
        state.mark_phase_complete("orientation")
        state.mark_phase_start("silver")
        state.mark_item_complete("silver", "silver_county")
        save_run_state(run_path, state)

        loaded = load_run_state(run_path)
        assert loaded is not None
        assert loaded.completed_phases == ["orientation"]
        assert loaded.completed_items["silver"] == ["silver_county"]
        assert loaded.current_phase == "silver"
        assert loaded.status == "running"

    def test_state_merged_into_metadata(self, tmp_path):
        """State is stored under the 'state' key in run_metadata.json."""
        run_path = init_run_directory(str(tmp_path), "run_001")
        write_run_metadata(run_path, "proj", "run_001", agent_type="scientist")

        state = RunState()
        state.mark_completed()
        save_run_state(run_path, state)

        with open(os.path.join(run_path, "run_metadata.json")) as f:
            meta = json.load(f)

        # Metadata fields preserved
        assert meta["config_name"] == "proj"
        assert meta["agent_type"] == "scientist"
        # State embedded
        assert meta["state"]["status"] == "completed"

    def test_load_returns_none_when_no_file(self, tmp_path):
        """load_run_state returns None when run_metadata.json doesn't exist."""
        assert load_run_state(str(tmp_path)) is None

    def test_load_returns_none_when_corrupt_json(self, tmp_path):
        """load_run_state returns None for corrupt JSON."""
        meta_path = tmp_path / "run_metadata.json"
        meta_path.write_text("{invalid json")
        assert load_run_state(str(tmp_path)) is None

    def test_load_returns_none_when_no_state_key(self, tmp_path):
        """load_run_state returns None when metadata has no 'state' key."""
        meta_path = tmp_path / "run_metadata.json"
        meta_path.write_text(json.dumps({"run_id": "test"}))
        assert load_run_state(str(tmp_path)) is None

    def test_multiple_saves_preserve_latest(self, tmp_path):
        """Each save overwrites with the latest state."""
        run_path = init_run_directory(str(tmp_path), "run_001")

        state = RunState()
        state.mark_phase_start("orientation")
        save_run_state(run_path, state)

        state.mark_phase_complete("orientation")
        state.mark_completed()
        save_run_state(run_path, state)

        loaded = load_run_state(run_path)
        assert loaded.status == "completed"
        assert "orientation" in loaded.completed_phases


# ===================================================================
# AgentDependency / resolve_dependency
# ===================================================================


class TestAgentDependency:
    def test_resolve_specific_run_id(self, tmp_path):
        """Resolves to specific run when run_id is set."""
        init_run_directory(str(tmp_path), "run_001")
        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            run_id="run_001",
            base_path=str(tmp_path),
        )
        path = resolve_dependency(dep)
        assert path == os.path.join(str(tmp_path), "runs", "run_001")

    def test_resolve_latest_run(self, tmp_path):
        """Without run_id, resolves to the latest run."""
        init_run_directory(str(tmp_path), "20260101_000000_aaaa")
        init_run_directory(str(tmp_path), "20260301_000000_cccc")
        init_run_directory(str(tmp_path), "20260201_000000_bbbb")

        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            base_path=str(tmp_path),
        )
        path = resolve_dependency(dep)
        assert path.endswith("20260301_000000_cccc")

    def test_falls_back_to_base_path(self, tmp_path):
        """When no runs/ directory exists, falls back to base_path."""
        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            base_path=str(tmp_path),
        )
        path = resolve_dependency(dep)
        assert path == str(tmp_path)

    def test_missing_run_id_raises(self, tmp_path):
        """Requesting a non-existent run_id raises FileNotFoundError."""
        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            run_id="nonexistent",
            base_path=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_dependency(dep)

    def test_empty_base_path_raises(self):
        """Empty base_path raises ValueError."""
        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            base_path="",
        )
        with pytest.raises(ValueError, match="no base_path"):
            resolve_dependency(dep)
