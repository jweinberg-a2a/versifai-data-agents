"""Inter-agent handoff tests.

Verifies that:
1. DataScientist outputs (findings, charts, tables, notes, metadata) have the
   correct structure in their run directory.
2. StoryTeller can discover and read DataScientist outputs via dependency resolution.
3. Run directory structure, run_metadata.json, and RunState all persist correctly
   across the agent handoff boundary.
4. Resume semantics: an interrupted run can be recovered.

These tests do NOT invoke the LLM — they exercise the filesystem and config
wiring that connects agents together.
"""

from __future__ import annotations

import json
import os

import pytest

from versifai.core.run_manager import (
    AgentDependency,
    RunState,
    generate_run_id,
    init_run_directory,
    load_run_state,
    resolve_dependency,
    save_run_state,
    write_run_metadata,
)

# ---------------------------------------------------------------------------
# Helpers — simulate what each agent writes to disk
# ---------------------------------------------------------------------------


def _simulate_scientist_run(base_path: str, run_id: str) -> str:
    """Create a realistic DataScientist run directory with outputs.

    Mimics what DataScientistAgent.run() produces on disk:
    - findings.json with structured findings
    - charts/ with PNG files
    - tables/ with CSV files
    - notes/ with per-theme notes
    - run_metadata.json with state
    """
    run_path = init_run_directory(base_path, run_id)

    # Write findings.json
    findings = [
        {
            "research_question_id": "theme_0",
            "title": "SVI correlates with lower enrollment",
            "finding": "Counties with high SVI show 15% lower MA penetration.",
            "evidence": "t-test p=0.0003, Cohen's d=0.82, N=3142",
            "significance": "high",
            "p_value": 0.0003,
            "effect_size": 0.82,
            "visualization_path": "charts/theme0_svi_scatter.png",
            "timestamp": "2026-02-25T10:00:00",
            "index": 0,
        },
        {
            "research_question_id": "theme_0",
            "title": "Urban-rural gap in access",
            "finding": "Rural counties have 8% lower enrollment rates.",
            "evidence": "Mann-Whitney U p=0.012, r=0.45, N=3142",
            "significance": "medium",
            "p_value": 0.012,
            "effect_size": 0.45,
            "visualization_path": "charts/theme0_urban_rural_box.png",
            "timestamp": "2026-02-25T10:30:00",
            "index": 1,
        },
        {
            "research_question_id": "theme_1",
            "title": "No significant income effect",
            "finding": "Median income does not predict enrollment after controlling for SVI.",
            "evidence": "Regression p=0.73, R²=0.004, N=3142",
            "significance": "low",
            "p_value": 0.73,
            "effect_size": 0.004,
            "timestamp": "2026-02-25T11:00:00",
            "index": 2,
        },
    ]
    with open(os.path.join(run_path, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)

    # Write chart files
    for fname in ["theme0_svi_scatter.png", "theme0_urban_rural_box.png", "theme1_income.png"]:
        with open(os.path.join(run_path, "charts", fname), "wb") as f:
            f.write(b"\x89PNG_FAKE")  # Fake PNG header

    # Write table files
    csv_content = "county_fips,svi_score,ma_penetration\n01001,0.45,0.28\n01003,0.32,0.42\n"
    for fname in ["theme0_summary.csv", "theme1_regression.csv"]:
        with open(os.path.join(run_path, "tables", fname), "w") as f:
            f.write(csv_content)

    # Write notes
    for theme_id in ["theme_0", "theme_1"]:
        note_path = os.path.join(run_path, "notes", f"{theme_id}_notes.md")
        with open(note_path, "w") as f:
            f.write(f"# {theme_id} Research Notes\n\nSQL: SELECT * FROM silver_county...\n")

    # Write metadata
    write_run_metadata(
        run_path,
        "geographic_disparity",
        run_id,
        agent_type="scientist",
    )

    # Write completed state
    state = RunState(entry_point="run")
    state.mark_phase_complete("orientation")
    state.mark_phase_complete("silver")
    state.mark_item_complete("themes", "theme_0")
    state.mark_item_complete("themes", "theme_1")
    state.mark_phase_complete("themes")
    state.mark_phase_complete("synthesis")
    state.mark_completed()
    save_run_state(run_path, state)

    # Update metadata with final counts
    write_run_metadata(
        run_path,
        "geographic_disparity",
        run_id,
        extra={
            "completed_at": "2026-02-25T12:00:00",
            "total_findings": len(findings),
            "total_charts": 3,
        },
    )

    return run_path


def _simulate_interrupted_scientist_run(base_path: str, run_id: str) -> str:
    """Create a partially completed DataScientist run (interrupted mid-analysis)."""
    run_path = init_run_directory(base_path, run_id)

    # Only theme_0 findings written
    findings = [
        {
            "research_question_id": "theme_0",
            "title": "Partial finding",
            "finding": "Only theme_0 was analyzed before interruption.",
            "evidence": "t-test p=0.01",
            "significance": "high",
            "timestamp": "2026-02-25T10:00:00",
            "index": 0,
        },
    ]
    with open(os.path.join(run_path, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)

    # One chart
    with open(os.path.join(run_path, "charts", "theme0_chart.png"), "wb") as f:
        f.write(b"\x89PNG_FAKE")

    # Write interrupted state
    write_run_metadata(run_path, "geographic_disparity", run_id, agent_type="scientist")

    state = RunState(entry_point="run")
    state.mark_phase_complete("orientation")
    state.mark_phase_complete("silver")
    state.mark_item_complete("themes", "theme_0")
    state.mark_phase_start("themes", "theme_1")  # Was working on theme_1
    state.mark_interrupted()
    save_run_state(run_path, state)

    return run_path


# ===================================================================
# Test class 1: Scientist output structure
# ===================================================================


class TestScientistOutputStructure:
    """Verify that scientist run directory has the expected structure."""

    @pytest.fixture()
    def scientist_run(self, tmp_path):
        return _simulate_scientist_run(str(tmp_path / "research_results"), "run_001")

    def test_run_directory_exists(self, scientist_run):
        assert os.path.isdir(scientist_run)

    def test_findings_json_present(self, scientist_run):
        findings_path = os.path.join(scientist_run, "findings.json")
        assert os.path.isfile(findings_path)

    def test_findings_json_valid_structure(self, scientist_run):
        """Each finding has required fields."""
        with open(os.path.join(scientist_run, "findings.json")) as f:
            findings = json.load(f)

        assert isinstance(findings, list)
        assert len(findings) == 3

        required_keys = {"research_question_id", "title", "finding", "evidence", "significance"}
        for finding in findings:
            missing = required_keys - set(finding.keys())
            assert not missing, f"Finding missing keys: {missing}"

    def test_findings_have_unique_indices(self, scientist_run):
        """Each finding has a unique index."""
        with open(os.path.join(scientist_run, "findings.json")) as f:
            findings = json.load(f)
        indices = [f["index"] for f in findings]
        assert len(indices) == len(set(indices))

    def test_charts_directory_has_png_files(self, scientist_run):
        charts = os.listdir(os.path.join(scientist_run, "charts"))
        png_files = [f for f in charts if f.endswith(".png")]
        assert len(png_files) == 3

    def test_tables_directory_has_csv_files(self, scientist_run):
        tables = os.listdir(os.path.join(scientist_run, "tables"))
        csv_files = [f for f in tables if f.endswith(".csv")]
        assert len(csv_files) == 2

    def test_notes_directory_has_per_theme_notes(self, scientist_run):
        notes = os.listdir(os.path.join(scientist_run, "notes"))
        assert len(notes) >= 2
        assert any("theme_0" in n for n in notes)
        assert any("theme_1" in n for n in notes)

    def test_metadata_has_required_fields(self, scientist_run):
        with open(os.path.join(scientist_run, "run_metadata.json")) as f:
            meta = json.load(f)

        assert meta["agent_type"] == "scientist"
        assert meta["config_name"] == "geographic_disparity"
        assert "started_at" in meta
        assert "completed_at" in meta
        assert meta["total_findings"] == 3
        assert meta["total_charts"] == 3

    def test_state_is_completed(self, scientist_run):
        state = load_run_state(scientist_run)
        assert state is not None
        assert state.status == "completed"
        assert "orientation" in state.completed_phases
        assert "silver" in state.completed_phases
        assert "themes" in state.completed_phases
        assert "synthesis" in state.completed_phases

    def test_visualization_paths_in_findings_point_to_real_files(self, scientist_run):
        """Visualization paths referenced in findings actually exist on disk."""
        with open(os.path.join(scientist_run, "findings.json")) as f:
            findings = json.load(f)

        for finding in findings:
            viz_path = finding.get("visualization_path")
            if viz_path:
                full_path = os.path.join(scientist_run, viz_path)
                assert os.path.isfile(full_path), (
                    f"Finding '{finding['title']}' references {viz_path} but file doesn't exist"
                )


# ===================================================================
# Test class 2: StoryTeller picks up Scientist outputs
# ===================================================================


class TestStorytellerReadsScientistOutputs:
    """Verify the storyteller can resolve and read scientist run outputs."""

    @pytest.fixture()
    def scientist_base(self, tmp_path):
        base = str(tmp_path / "research_results")
        _simulate_scientist_run(base, "20260225_100000_abcd")
        return base

    def test_dependency_resolves_to_latest_run(self, scientist_base):
        """StoryTeller dependency resolution finds the scientist run."""
        dep = AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path=scientist_base,
        )
        resolved = resolve_dependency(dep)
        assert "20260225_100000_abcd" in resolved
        assert os.path.isdir(resolved)

    def test_findings_readable_from_resolved_path(self, scientist_base):
        """StoryTeller can read findings.json from the resolved dependency path."""
        dep = AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path=scientist_base,
        )
        resolved = resolve_dependency(dep)

        findings_path = os.path.join(resolved, "findings.json")
        assert os.path.isfile(findings_path)

        with open(findings_path) as f:
            findings = json.load(f)
        assert len(findings) == 3
        assert findings[0]["significance"] == "high"

    def test_charts_discoverable_from_resolved_path(self, scientist_base):
        """StoryTeller can list charts from the resolved path."""
        dep = AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path=scientist_base,
        )
        resolved = resolve_dependency(dep)

        charts_dir = os.path.join(resolved, "charts")
        charts = [f for f in os.listdir(charts_dir) if f.endswith(".png")]
        assert len(charts) == 3

    def test_tables_discoverable_from_resolved_path(self, scientist_base):
        dep = AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path=scientist_base,
        )
        resolved = resolve_dependency(dep)

        tables_dir = os.path.join(resolved, "tables")
        tables = [f for f in os.listdir(tables_dir) if f.endswith(".csv")]
        assert len(tables) == 2

    def test_notes_readable_from_resolved_path(self, scientist_base):
        dep = AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path=scientist_base,
        )
        resolved = resolve_dependency(dep)

        notes_dir = os.path.join(resolved, "notes")
        notes = os.listdir(notes_dir)
        assert len(notes) >= 2

        # Notes contain actual content
        for note_file in notes:
            with open(os.path.join(notes_dir, note_file)) as f:
                content = f.read()
            assert len(content) > 0

    def test_scientist_state_visible_to_storyteller(self, scientist_base):
        """StoryTeller can check the scientist's run state to verify completion."""
        dep = AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path=scientist_base,
        )
        resolved = resolve_dependency(dep)

        state = load_run_state(resolved)
        assert state is not None
        assert state.status == "completed"


# ===================================================================
# Test class 3: Multiple runs — latest resolution
# ===================================================================


class TestMultipleRunResolution:
    """When multiple runs exist, dependency resolution picks the latest."""

    def test_resolves_latest_of_three_runs(self, tmp_path):
        base = str(tmp_path / "research")
        _simulate_scientist_run(base, "20260101_000000_aaaa")
        _simulate_scientist_run(base, "20260301_000000_cccc")
        _simulate_scientist_run(base, "20260201_000000_bbbb")

        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            base_path=base,
        )
        resolved = resolve_dependency(dep)
        assert "20260301_000000_cccc" in resolved

    def test_specific_run_id_overrides_latest(self, tmp_path):
        """Explicit run_id pins to that specific run, not latest."""
        base = str(tmp_path / "research")
        _simulate_scientist_run(base, "20260101_000000_aaaa")
        _simulate_scientist_run(base, "20260301_000000_cccc")

        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            run_id="20260101_000000_aaaa",
            base_path=base,
        )
        resolved = resolve_dependency(dep)
        assert "20260101_000000_aaaa" in resolved

    def test_each_run_has_independent_state(self, tmp_path):
        """Each run directory has its own state, unaffected by others."""
        base = str(tmp_path / "research")
        run1 = _simulate_scientist_run(base, "run_001")
        run2 = _simulate_interrupted_scientist_run(base, "run_002")

        state1 = load_run_state(run1)
        state2 = load_run_state(run2)

        assert state1.status == "completed"
        assert state2.status == "interrupted"


# ===================================================================
# Test class 4: Resume from interrupted run
# ===================================================================


class TestResumeFromInterrupted:
    """Verify interrupted runs can be detected and resumed."""

    def test_interrupted_run_detectable(self, tmp_path):
        base = str(tmp_path / "research")
        run_path = _simulate_interrupted_scientist_run(base, "run_001")

        state = load_run_state(run_path)
        assert state.status == "interrupted"
        assert state.current_phase == "themes"
        assert state.current_item == "theme_1"

    def test_completed_items_preserved_for_resume(self, tmp_path):
        """Completed items are available so resume can skip them."""
        base = str(tmp_path / "research")
        run_path = _simulate_interrupted_scientist_run(base, "run_001")

        state = load_run_state(run_path)
        # theme_0 was completed before interruption
        assert "theme_0" in state.completed_items.get("themes", [])
        # orientation and silver phases were fully completed
        assert "orientation" in state.completed_phases
        assert "silver" in state.completed_phases

    def test_resume_updates_state_back_to_running(self, tmp_path):
        """On resume, status changes from interrupted back to running."""
        base = str(tmp_path / "research")
        run_path = _simulate_interrupted_scientist_run(base, "run_001")

        state = load_run_state(run_path)
        assert state.status == "interrupted"

        # Simulate resume: set back to running and continue
        state.status = "running"
        state.updated_at = ""
        state.mark_phase_start("themes", "theme_1")
        save_run_state(run_path, state)

        reloaded = load_run_state(run_path)
        assert reloaded.status == "running"
        assert reloaded.current_item == "theme_1"

    def test_failed_run_preserves_error(self, tmp_path):
        """Failed run records the error string for debugging."""
        base = str(tmp_path / "research")
        run_path = init_run_directory(base, "run_fail")
        write_run_metadata(run_path, "test", "run_fail")

        state = RunState()
        state.mark_phase_start("silver")
        state.mark_failed("OOM: DataFrame exceeded 64GB memory limit")
        save_run_state(run_path, state)

        loaded = load_run_state(run_path)
        assert loaded.status == "failed"
        assert "OOM" in loaded.error
        assert "64GB" in loaded.error


# ===================================================================
# Test class 5: Full pipeline handoff simulation
# ===================================================================


class TestFullPipelineHandoff:
    """Simulate the complete Engineer -> Scientist -> StoryTeller pipeline."""

    def test_end_to_end_handoff(self, tmp_path):
        """Full pipeline: scientist writes, storyteller resolves and reads."""
        # --- Scientist produces outputs ---
        scientist_base = str(tmp_path / "research_results")
        run_id = generate_run_id()
        scientist_path = _simulate_scientist_run(scientist_base, run_id)

        # --- StoryTeller resolves dependency ---
        dep = AgentDependency(
            agent_type="scientist",
            config_name="geographic_disparity",
            base_path=scientist_base,
        )
        resolved_research_path = resolve_dependency(dep)
        assert resolved_research_path == scientist_path

        # --- StoryTeller creates its own run ---
        narrative_base = str(tmp_path / "narrative_output")
        story_run_id = generate_run_id()
        narrative_path = init_run_directory(narrative_base, story_run_id)

        # StoryTeller reads findings from scientist
        with open(os.path.join(resolved_research_path, "findings.json")) as f:
            findings = json.load(f)
        assert len(findings) == 3

        # StoryTeller writes its own sections
        for section_id in ["introduction", "findings", "conclusion"]:
            section_path = os.path.join(narrative_path, f"section_{section_id}.md")
            with open(section_path, "w") as f:
                f.write(f"# {section_id.title()}\n\nContent based on {len(findings)} findings.\n")

        # StoryTeller writes its own metadata
        write_run_metadata(
            narrative_path,
            "geographic_disparity",
            story_run_id,
            agent_type="storyteller",
        )
        story_state = RunState(entry_point="run")
        story_state.mark_phase_complete("inventory")
        story_state.mark_phase_complete("evidence_evaluation")
        for sid in ["introduction", "findings", "conclusion"]:
            story_state.mark_item_complete("sections", sid)
        story_state.mark_phase_complete("sections")
        story_state.mark_phase_complete("coherence")
        story_state.mark_phase_complete("finalization")
        story_state.mark_completed()
        save_run_state(narrative_path, story_state)

        # --- Verify both runs exist independently ---
        scientist_state = load_run_state(resolved_research_path)
        storyteller_state = load_run_state(narrative_path)

        assert scientist_state.status == "completed"
        assert storyteller_state.status == "completed"

        # Scientist has its themes
        assert "theme_0" in scientist_state.completed_items.get("themes", [])
        # StoryTeller has its sections
        assert "introduction" in storyteller_state.completed_items.get("sections", [])

        # Verify section files exist
        assert os.path.isfile(os.path.join(narrative_path, "section_introduction.md"))
        assert os.path.isfile(os.path.join(narrative_path, "section_conclusion.md"))

    def test_storyteller_detects_incomplete_scientist(self, tmp_path):
        """If scientist run is interrupted, storyteller can detect that."""
        scientist_base = str(tmp_path / "research")
        _simulate_interrupted_scientist_run(scientist_base, "run_001")

        dep = AgentDependency(
            agent_type="scientist",
            config_name="test",
            base_path=scientist_base,
        )
        resolved = resolve_dependency(dep)
        state = load_run_state(resolved)

        assert state.status == "interrupted"
        # StoryTeller could warn or refuse to proceed with partial data

    def test_findings_significance_distribution(self, tmp_path):
        """Verify findings have a realistic significance distribution for curation."""
        scientist_base = str(tmp_path / "research")
        run_path = _simulate_scientist_run(scientist_base, "run_001")

        with open(os.path.join(run_path, "findings.json")) as f:
            findings = json.load(f)

        sig_counts = {}
        for f in findings:
            sig = f["significance"]
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

        # There should be at least one high-significance finding for leads
        assert sig_counts.get("high", 0) >= 1
        # Evidence tiering needs a mix of significance levels
        assert len(sig_counts) >= 2, "Findings should have varied significance levels"
