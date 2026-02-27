"""
StoryTellerAgent — transforms DataScientist outputs into a narrative report.

Reads findings, charts, tables, and notes from the research results volume,
evaluates evidence strength, and weaves everything into a single compelling
Markdown document.

Uses the same BaseAgent ReAct loop as the DataScientistAgent.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from versifai.core.agent import BaseAgent
from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.run_manager import (
    RunState,
    init_run_directory,
    load_run_state,
    resolve_dependency,
    save_run_state,
    write_run_metadata,
)
from versifai.core.tools.catalog_writer import ListCatalogTablesTool, SilverOnlyExecuteSQLTool
from versifai.core.tools.dynamic_tool_builder import DynamicToolBuilderTool
from versifai.core.tools.registry import ToolRegistry
from versifai.core.tools.save_note import SaveNoteTool
from versifai.core.tools.view_chart import ViewChartTool
from versifai.core.tools.visualization import CreateVisualizationTool
from versifai.core.tools.web_scraper import WebScraperTool
from versifai.core.tools.web_search import WebSearchTool
from versifai.story_agents.storyteller.config import StorytellerConfig
from versifai.story_agents.storyteller.prompts import (
    build_coherence_prompt,
    build_editor_review_prompt,
    build_editor_system_prompt,
    build_evidence_evaluation_prompt,
    build_finalization_prompt,
    build_inventory_prompt,
    build_section_prompt,
    build_storyteller_system_prompt,
)
from versifai.story_agents.storyteller.tools.cite_source import CiteSourceTool
from versifai.story_agents.storyteller.tools.evaluate_evidence import EvaluateEvidenceTool
from versifai.story_agents.storyteller.tools.read_chart import ReadChartTool
from versifai.story_agents.storyteller.tools.read_findings import ReadFindingsTool
from versifai.story_agents.storyteller.tools.read_table import ReadTableTool
from versifai.story_agents.storyteller.tools.write_narrative import WriteNarrativeTool

logger = logging.getLogger("agent.storyteller")


class StoryTellerAgent(BaseAgent):
    """
    Autonomous narrative report writer powered by Claude.

    Reads DataScientist outputs and produces a compelling, evidence-grounded
    Markdown document organized into configured narrative sections.

    Supports smart resume: if interrupted, re-launching picks up from
    the last completed section. Previously written sections are loaded
    from disk and skipped.

    Workflow:
      1. Inventory — scan research outputs, map coverage per section
      2. Evidence Evaluation — score finding strength, build bill of materials
      3. Section Writing — write each section using curated evidence
      4. Coherence Pass — fix transitions, consistency, completeness
      5. Finalization — assemble document, add TOC, bibliography

    Usage in a Databricks notebook::

        from versifai.story_agents.storyteller.agent import StoryTellerAgent
        from versifai.story_agents.storyteller.config import StorytellerConfig

        cfg = StorytellerConfig()
        agent = StoryTellerAgent(cfg=cfg, dbutils=dbutils)
        agent.run()
    """

    def __init__(
        self,
        cfg: StorytellerConfig | None = None,
        dbutils=None,
    ) -> None:
        if cfg is None:
            raise ValueError(
                "cfg is required. Pass a StorytellerConfig instance. "
                "See examples/ for sample configurations."
            )
        self._cfg = cfg

        display = AgentDisplay(dbutils=dbutils)
        memory = AgentMemory()
        llm = LLMClient(
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            api_key=cfg.llm.api_key or None,
            api_base=cfg.llm.api_base or None,
            extended_context=cfg.llm.extended_context,
        )
        registry = ToolRegistry()

        super().__init__(display=display, memory=memory, llm=llm, registry=registry)

        self._dbutils = dbutils

        # Resolve run paths (isolated or direct)
        if cfg.run_id:
            self._narrative_run_path = init_run_directory(cfg.narrative_output_path, cfg.run_id)
        else:
            self._narrative_run_path = cfg.narrative_output_path

        # Resolve research results path from dependencies
        self._research_path = cfg.research_results_path
        if cfg.dependencies:
            for dep in cfg.dependencies:
                if dep.agent_type == "scientist":
                    self._research_path = resolve_dependency(dep)
                    break

        # Run state — initialised per entry point, persisted for resume
        self._run_state: RunState | None = None

        # Storyteller-specific tools — use resolved paths
        self._read_findings_tool = ReadFindingsTool(
            results_path=self._research_path,
        )
        self._read_chart_tool = ReadChartTool(
            charts_path=os.path.join(self._research_path, "charts"),
            notes_path=os.path.join(self._research_path, "notes"),
        )
        self._read_table_tool = ReadTableTool(
            tables_path=os.path.join(self._research_path, "tables"),
        )
        self._write_narrative_tool = WriteNarrativeTool(
            output_path=self._narrative_run_path,
            output_filename=cfg.output_format.filename,
            include_toc=cfg.output_format.include_toc,
            report_title=cfg.name or "Research Analysis Report",
        )
        self._evaluate_evidence_tool = EvaluateEvidenceTool(
            min_significance_for_lead=cfg.evidence_threshold.min_significance_for_lead,
            min_significance_for_support=cfg.evidence_threshold.min_significance_for_support,
        )
        self._cite_source_tool = CiteSourceTool()
        self._note_tool = SaveNoteTool(
            notes_path=os.path.join(self._research_path, "notes"),
        )
        self._view_chart_tool = ViewChartTool(
            charts_path=os.path.join(self._research_path, "charts"),
            tables_path=os.path.join(self._research_path, "tables"),
        )

        self._register_tools()

        # Build system prompt
        self._system_prompt = build_storyteller_system_prompt(cfg)

    def _register_tools(self) -> None:
        """Register the storyteller's tool set."""
        cfg = self._cfg
        web_scraper = WebScraperTool()

        tools = [
            # Reading DataScientist outputs
            self._read_findings_tool,
            self._read_chart_tool,
            self._read_table_tool,
            # Writing narrative + notes
            self._write_narrative_tool,
            self._note_tool,
            # Evidence evaluation
            self._evaluate_evidence_tool,
            # Citations
            self._cite_source_tool,
            # Shared tools (reused from scientist/engineer)
            self._view_chart_tool,
            SilverOnlyExecuteSQLTool(),
            ListCatalogTablesTool(cfg=cfg.project),
            CreateVisualizationTool(
                cfg=cfg,  # type: ignore[arg-type]
                display=self._display,
                notes_path=os.path.join(self._research_path, "notes"),
            ),
            WebSearchTool(cfg=cfg.project),
            web_scraper,
            DynamicToolBuilderTool(registry=self._registry),
        ]
        for tool in tools:
            self._registry.register(tool)

    # ------------------------------------------------------------------
    # Phase runner — thin override to inject progress_path
    # ------------------------------------------------------------------

    def _run_phase(self, prompt: str, max_turns: int) -> str:  # type: ignore[override]
        """Run a phase, providing the storyteller progress path."""
        progress_path = os.path.join(self._narrative_run_path, "progress.txt")
        return super()._run_phase(prompt, max_turns, progress_path=progress_path)

    # ------------------------------------------------------------------
    # State persistence helpers
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist ``self._run_state`` to ``run_metadata.json`` if active."""
        if self._run_state and self._cfg.run_id:
            save_run_state(self._narrative_run_path, self._run_state)

    def _dump_progress_on_crash(self) -> None:
        """Flush the display log to progress.txt on interrupt/failure."""
        try:
            progress_path = os.path.join(self._narrative_run_path, "progress.txt")
            self._display.dump_progress(progress_path)
        except Exception:
            pass  # best-effort

    # ------------------------------------------------------------------
    # Pre-flight: scan research outputs
    # ------------------------------------------------------------------

    def _scan_research_outputs(self) -> dict:
        """Scan the DataScientist results directory for available outputs.

        Returns an inventory dict used by all prompt-building functions.
        """
        results_path = self._research_path
        inventory: dict = {
            "findings_count": 0,
            "findings": [],
            "charts": [],
            "tables": [],
            "notes": [],
        }

        # Findings
        findings_path = os.path.join(results_path, "findings.json")
        if os.path.isfile(findings_path):
            try:
                with open(findings_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    inventory["findings"] = data
                    inventory["findings_count"] = len(data)
            except (json.JSONDecodeError, OSError):
                pass

        # Charts
        charts_dir = os.path.join(results_path, "charts")
        if os.path.isdir(charts_dir):
            inventory["charts"] = sorted(
                f for f in os.listdir(charts_dir) if f.lower().endswith((".png", ".html", ".svg"))
            )

        # Tables
        tables_dir = os.path.join(results_path, "tables")
        if os.path.isdir(tables_dir):
            inventory["tables"] = sorted(
                f for f in os.listdir(tables_dir) if f.lower().endswith(".csv")
            )

        # Notes
        notes_dir = os.path.join(results_path, "notes")
        if os.path.isdir(notes_dir):
            inventory["notes"] = sorted(os.listdir(notes_dir))

        return inventory

    # ------------------------------------------------------------------
    # Smart resume: scan existing narrative sections on disk
    # ------------------------------------------------------------------

    def _scan_completed_sections(self) -> dict[str, str]:
        """Scan the narrative output directory for previously written sections.

        Returns a dict of section_id -> content for sections that exist on disk.
        These are loaded into the write_narrative tool so the agent can read
        them back, and sections that are complete can be skipped.
        """
        completed: dict[str, str] = {}
        output_path = self._narrative_run_path

        if not os.path.isdir(output_path):
            return completed

        for filename in os.listdir(output_path):
            if filename.startswith("section_") and filename.endswith(".md"):
                section_id = filename[:-3]  # strip .md
                filepath = os.path.join(output_path, filename)
                try:
                    with open(filepath) as f:
                        content = f.read()
                    if content.strip():
                        completed[section_id] = content
                except OSError:
                    pass

        return completed

    def _load_existing_sections(self, completed: dict[str, str]) -> None:
        """Load previously written sections into the write_narrative tool.

        This allows the coherence pass to read all sections (including
        previously written ones) and the assembler to include them.
        """
        # Build a section_id -> sequence mapping from config
        section_seq = {s.id: s.sequence for s in self._cfg.narrative_sections}

        for section_id, content in completed.items():
            seq = section_seq.get(section_id, 999)
            self._write_narrative_tool._sections[section_id] = content
            self._write_narrative_tool._section_sequence[section_id] = seq
            if section_id not in self._write_narrative_tool._section_order:
                self._write_narrative_tool._section_order.append(section_id)

        # Sort the order list
        self._write_narrative_tool._section_order.sort(
            key=lambda sid: self._write_narrative_tool._section_sequence.get(sid, 999)
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        instructions: str = "",
        rerun: bool = False,
        focus_visuals: list[str] | None = None,
    ) -> dict:
        """
        Run the full storytelling pipeline.

        By default, scans for previously written sections and skips them
        (smart resume). Set ``rerun=True`` to force a fresh start.

        Args:
            instructions: Optional high-level guidance prepended to every
                          phase prompt (e.g., "Focus on the bullwhip effect").
            rerun: If True, ignore existing sections and rewrite everything.
            focus_visuals: Optional shortlist of chart/table filenames the
                agent should prioritize when selecting visuals for sections.
                e.g., ``["theme0_stars_vs_svi.png", "theme3_measure_heatmap.png"]``.
                The agent still picks 1-2 per section but starts from this
                curated pool instead of all available charts.

        Returns a summary dict with sections written, word counts, etc.
        """
        cfg = self._cfg
        self._instructions = instructions
        self._display.phase("STORYTELLER AGENT STARTING")
        self._display.step(f"Project: {cfg.name}")
        self._display.step(f"Thesis: {cfg.thesis[:100]}...")
        self._display.step(f"Sections: {len(cfg.narrative_sections)}")
        self._display.step(f"Results source: {self._research_path}")
        self._display.step(f"Output: {self._narrative_run_path}")
        if cfg.run_id:
            self._display.step(f"Run ID: {cfg.run_id}")
        self._display.step(f"Tools: {self._registry.tool_names + ['ask_human']}")
        if focus_visuals:
            self._display.step(f"Focus visuals: {len(focus_visuals)} pre-selected")
        if rerun:
            self._display.step("Mode: FULL RE-RUN (ignoring existing sections)")
        else:
            self._display.step("Mode: SMART RESUME (skipping completed sections)")

        # Ensure output directories exist
        os.makedirs(self._narrative_run_path, exist_ok=True)
        os.makedirs(os.path.join(self._research_path, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self._research_path, "tables"), exist_ok=True)
        os.makedirs(os.path.join(self._research_path, "notes"), exist_ok=True)

        # Write run metadata if using run isolation
        if cfg.run_id:
            write_run_metadata(
                self._narrative_run_path,
                cfg.name,
                cfg.run_id,
                agent_type="storyteller",
            )

        # Initialise or resume run state
        if cfg.run_id:
            if not rerun:
                existing = load_run_state(self._narrative_run_path)
                if existing and existing.status in ("running", "interrupted", "failed"):
                    self._run_state = existing
                    self._run_state.status = "running"
                    self._display.step(f"Resuming previous run (was {existing.status})")
                    self._display.step(f"  Completed phases: {existing.completed_phases}")
            if self._run_state is None:
                self._run_state = RunState(entry_point="run")
            self._save_state()

        try:
            # ── Pre-flight: scan research outputs ─────────────────────
            inventory = self._scan_research_outputs()
            self._display.step(
                f"Research outputs: {inventory['findings_count']} findings, "
                f"{len(inventory['charts'])} charts, "
                f"{len(inventory['tables'])} tables, "
                f"{len(inventory['notes'])} note files"
            )

            if inventory["findings_count"] == 0:
                self._display.error("No findings found. Run the DataScientist agent first.")
                return self._build_summary()

            # ── Smart resume: scan existing sections ──────────────────
            completed_sections: dict[str, str] = {}
            if not rerun:
                completed_sections = self._scan_completed_sections()
                if completed_sections:
                    self._load_existing_sections(completed_sections)
                    self._display.step("--- Narrative State ---")
                    for sid, content in sorted(completed_sections.items()):
                        words = len(content.split())
                        self._display.step(f"  {sid}: {words} words (DONE)")
                    self._display.step("---")

            # ── Phase 1: Inventory ────────────────────────────────────
            if self._run_state and "inventory" in self._run_state.completed_phases:
                self._display.step("Phase 1: Inventory — SKIPPED (completed in prior run)")
            else:
                if self._run_state:
                    self._run_state.mark_phase_start("inventory")
                    self._save_state()
                self._display.phase("Phase 1: Research Output Inventory")
                self._run_phase(
                    prompt=self._inject_instructions(
                        build_inventory_prompt(cfg, inventory, focus_visuals=focus_visuals)
                    ),
                    max_turns=cfg.max_turns_per_phase,
                )
                if self._run_state:
                    self._run_state.mark_phase_complete("inventory")
                    self._save_state()

            # ── Phase 2: Evidence Evaluation ──────────────────────────
            if self._run_state and "evidence" in self._run_state.completed_phases:
                self._display.step("Phase 2: Evidence — SKIPPED (completed in prior run)")
            else:
                if self._run_state:
                    self._run_state.mark_phase_start("evidence")
                    self._save_state()
                self._display.phase("Phase 2: Evidence Evaluation")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                self._run_phase(
                    prompt=self._inject_instructions(
                        build_evidence_evaluation_prompt(
                            cfg, inventory, focus_visuals=focus_visuals
                        )
                    ),
                    max_turns=cfg.max_turns_per_phase,
                )
                if self._run_state:
                    self._run_state.mark_phase_complete("evidence")
                    self._save_state()

            # ── Phase 3: Section-by-Section Writing ───────────────────
            sections = sorted(cfg.narrative_sections, key=lambda s: s.sequence)
            if self._run_state:
                self._run_state.mark_phase_start("sections")
                self._save_state()
            self._display.phase(f"Phase 3: Section Writing ({len(sections)} sections)")

            sections_skipped = 0
            for i, section in enumerate(sections, 1):
                # Smart resume: skip sections that already exist on disk
                if not rerun and section.id in completed_sections:
                    words = len(completed_sections[section.id].split())
                    self._display.step(
                        f"  Section {i}/{len(sections)} "
                        f"{section.title} — SKIPPED ({words} words on disk)"
                    )
                    sections_skipped += 1
                    continue

                self._display.phase(f"Section {i}/{len(sections)}: {section.title}")
                carryover = self._memory.get_carryover_context()
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                prompt = build_section_prompt(cfg, section, inventory, focus_visuals=focus_visuals)
                if carryover:
                    prompt = f"## Context From Prior Sections\n{carryover}\n\n---\n\n{prompt}"

                self._run_phase(
                    prompt=self._inject_instructions(prompt),
                    max_turns=cfg.max_turns_per_section,
                )
                if self._run_state:
                    self._run_state.mark_item_complete("sections", section.id)
                    self._save_state()
                self._display.success(f"Completed section: {section.title}")

            if self._run_state:
                self._run_state.mark_phase_complete("sections")
                self._save_state()

            # ── Phase 4: Coherence Pass ───────────────────────────────
            # Skip only if ALL sections were skipped (nothing new)
            if sections_skipped == len(sections):
                self._display.step("Phase 4: Coherence — SKIPPED (no new sections written)")
                if self._run_state:
                    self._run_state.mark_phase_complete("coherence")
                    self._save_state()
            else:
                if self._run_state:
                    self._run_state.mark_phase_start("coherence")
                    self._save_state()
                self._display.phase("Phase 4: Coherence Pass")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                self._run_phase(
                    prompt=self._inject_instructions(build_coherence_prompt(cfg)),
                    max_turns=cfg.coherence_pass_max_turns,
                )
                if self._run_state:
                    self._run_state.mark_phase_complete("coherence")
                    self._save_state()

            # ── Phase 5: Finalization ─────────────────────────────────
            if self._run_state:
                self._run_state.mark_phase_start("finalization")
                self._save_state()
            self._display.phase("Phase 5: Finalization")
            self._memory.reset_for_new_source()
            self._consecutive_errors = 0
            self._missing_param_tracker.clear()

            self._run_phase(
                prompt=self._inject_instructions(build_finalization_prompt(cfg)),
                max_turns=cfg.max_turns_per_phase,
            )
            if self._run_state:
                self._run_state.mark_phase_complete("finalization")
                self._run_state.mark_completed()
                self._save_state()

        except KeyboardInterrupt:
            self._display.warning("Agent interrupted by user.")
            if self._run_state:
                self._run_state.mark_interrupted()
                self._save_state()
            self._dump_progress_on_crash()
        except Exception as e:
            self._display.error(f"Storyteller failed: {e}")
            logger.exception("StoryTellerAgent top-level failure")
            if self._run_state:
                self._run_state.mark_failed(str(e))
                self._save_state()
            self._dump_progress_on_crash()

        # Update run metadata on completion
        if cfg.run_id:
            write_run_metadata(
                self._narrative_run_path,
                cfg.name,
                cfg.run_id,
                agent_type="storyteller",
                extra={
                    "completed_at": datetime.now().isoformat(),
                    "sections_written": self._write_narrative_tool.sections_written,
                    "citations": len(self._cite_source_tool._citations),
                },
            )

        summary = self._build_summary()
        self._display.phase("STORYTELLER COMPLETE")
        self._display.step(f"Sections written: {self._write_narrative_tool.sections_written}")
        self._display.step(f"Citations: {len(self._cite_source_tool._citations)}")
        self._display.step(f"Notes: {len(self._note_tool._notes)}")
        self._display.step(f"LLM usage: {self._llm.usage_summary}")
        return summary

    def run_sections(
        self,
        sections: list[int] | None = None,
        coherence: bool = True,
        instructions: str = "",
    ) -> dict:
        """
        Re-run specific sections only (like DataScientist.run_themes).

        Always rewrites the specified sections (no smart resume).
        Loads existing sections from disk so coherence pass and assembly
        have the full document.

        Args:
            sections: Section sequence numbers to run.
                      e.g., ``sections=[0, 3, 5]``. None = all sections.
            coherence: Whether to run coherence pass after sections.
            instructions: High-level guidance for the agent.

        Usage::

            agent = StoryTellerAgent(cfg=cfg, dbutils=dbutils)
            agent.run_sections(sections=[0, 1])     # rewrite first two sections
            agent.run_sections(coherence=False)       # all sections, skip coherence
        """
        cfg = self._cfg
        self._instructions = instructions
        self._display.phase("STORYTELLER — SECTIONS ONLY")

        os.makedirs(self._narrative_run_path, exist_ok=True)
        os.makedirs(os.path.join(self._research_path, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self._research_path, "tables"), exist_ok=True)
        os.makedirs(os.path.join(self._research_path, "notes"), exist_ok=True)

        # Fresh run state for this entry point
        if cfg.run_id:
            self._run_state = RunState(entry_point="run_sections")
            self._save_state()

        try:
            inventory = self._scan_research_outputs()
            self._display.step(
                f"Research outputs: {inventory['findings_count']} findings, "
                f"{len(inventory['charts'])} charts"
            )

            if inventory["findings_count"] == 0:
                self._display.error("No findings found.")
                return self._build_summary()

            # Load existing sections so coherence/assembly has full context
            completed = self._scan_completed_sections()
            if completed:
                self._load_existing_sections(completed)
                self._display.step(f"Loaded {len(completed)} existing sections from disk")

            self._system_prompt = build_storyteller_system_prompt(cfg)

            all_sections = sorted(cfg.narrative_sections, key=lambda s: s.sequence)
            if sections is not None:
                section_set = set(sections)
                to_run = [s for s in all_sections if s.sequence in section_set]
            else:
                to_run = all_sections

            if self._run_state:
                self._run_state.mark_phase_start("sections")
                self._save_state()
            self._display.phase(f"Writing {len(to_run)} of {len(all_sections)} sections")

            for _i, section in enumerate(to_run, 1):
                self._display.phase(f"Section {section.sequence}: {section.title}")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                self._run_phase(
                    prompt=self._inject_instructions(build_section_prompt(cfg, section, inventory)),
                    max_turns=cfg.max_turns_per_section,
                )
                if self._run_state:
                    self._run_state.mark_item_complete("sections", section.id)
                    self._save_state()
                self._display.success(f"Completed: {section.title}")

            if self._run_state:
                self._run_state.mark_phase_complete("sections")
                self._save_state()

            if coherence:
                if self._run_state:
                    self._run_state.mark_phase_start("coherence")
                    self._save_state()
                self._display.phase("Coherence Pass")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                self._run_phase(
                    prompt=self._inject_instructions(build_coherence_prompt(cfg)),
                    max_turns=cfg.coherence_pass_max_turns,
                )
                if self._run_state:
                    self._run_state.mark_phase_complete("coherence")
                    self._save_state()

            # Assemble
            if self._run_state:
                self._run_state.mark_phase_start("finalization")
                self._save_state()
            self._display.phase("Finalization")
            self._memory.reset_for_new_source()
            self._consecutive_errors = 0
            self._missing_param_tracker.clear()

            self._run_phase(
                prompt=self._inject_instructions(build_finalization_prompt(cfg)),
                max_turns=cfg.max_turns_per_phase,
            )
            if self._run_state:
                self._run_state.mark_phase_complete("finalization")
                self._run_state.mark_completed()
                self._save_state()

        except KeyboardInterrupt:
            self._display.warning("Interrupted.")
            if self._run_state:
                self._run_state.mark_interrupted()
                self._save_state()
            self._dump_progress_on_crash()
        except Exception as e:
            self._display.error(f"Failed: {e}")
            logger.exception("run_sections failure")
            if self._run_state:
                self._run_state.mark_failed(str(e))
                self._save_state()
            self._dump_progress_on_crash()

        summary = self._build_summary()
        self._display.phase("SECTION WRITING COMPLETE")
        self._display.step(f"Sections written: {self._write_narrative_tool.sections_written}")
        self._display.step(f"LLM usage: {self._llm.usage_summary}")
        return summary

    # ------------------------------------------------------------------
    # Editor entry point — review / validation pass
    # ------------------------------------------------------------------

    def run_editor(
        self,
        instructions: str = "",
    ) -> dict:
        """
        Run an editorial review pass on the completed narrative.

        Reads all existing sections, then works with the human operator
        to diagnose issues and apply targeted revisions. Uses ``ask_human``
        proactively at defined checkpoints — this is a HITL workflow by
        design.

        This is the validation step for the storyteller, analogous to
        ``DataScientistAgent.run_validation()``.

        Args:
            instructions: Editorial guidance for the review. e.g.,
                ``"The bullwhip section is too technical for policymakers"``
                or ``"Tighten recommendations — each should be actionable
                in a single sentence."``.  If empty, the editor runs an
                open-ended review.

        Usage::

            agent = StoryTellerAgent(cfg=cfg, dbutils=dbutils)

            # Guided review
            agent.run_editor(
                instructions="The bullwhip section is too technical. "
                             "Simplify for a policymaker audience."
            )

            # Open-ended review
            agent.run_editor()
        """
        cfg = self._cfg
        self._instructions = instructions
        self._display.phase("STORYTELLER — EDITORIAL REVIEW")
        self._display.step(f"Project: {cfg.name}")
        if instructions:
            self._display.step(f"Instructions: {instructions[:200]}")
        else:
            self._display.step("No specific instructions — full editorial review")

        # Ensure directories exist
        os.makedirs(self._narrative_run_path, exist_ok=True)
        os.makedirs(os.path.join(self._research_path, "notes"), exist_ok=True)

        # Fresh run state for this entry point
        if cfg.run_id:
            self._run_state = RunState(entry_point="run_editor")
            self._save_state()

        try:
            # ── Pre-flight: scan research outputs ─────────────────
            inventory = self._scan_research_outputs()
            self._display.step(
                f"Research outputs: {inventory['findings_count']} findings, "
                f"{len(inventory['charts'])} charts, "
                f"{len(inventory['tables'])} tables"
            )

            # ── Load existing sections (REQUIRED) ─────────────────
            completed_sections = self._scan_completed_sections()
            if not completed_sections:
                self._display.error(
                    "No sections found on disk. Run the storyteller pipeline "
                    "first with agent.run() to write the initial narrative."
                )
                return self._build_summary()

            self._load_existing_sections(completed_sections)

            # Build section summaries for the prompt
            section_seq = {s.id: (s.sequence, s.title) for s in cfg.narrative_sections}
            section_summaries: list[dict] = []
            self._display.step("--- Sections Under Review ---")
            for sid, content in completed_sections.items():
                words = len(content.split())
                seq, config_title = section_seq.get(sid, (999, ""))
                # Try to extract title from the section content
                title = config_title
                for line in content.split("\n"):
                    if line.startswith("## "):
                        title = line.replace("## ", "").strip()
                        break
                if not title:
                    title = sid.replace("section_", "").replace("_", " ").title()
                section_summaries.append(
                    {
                        "section_id": sid,
                        "title": title,
                        "word_count": words,
                    }
                )
                self._display.step(f"  {sid}: {title} ({words} words)")
            section_summaries.sort(key=lambda s: section_seq.get(s["section_id"], (999,))[0])
            self._display.step("---")

            # ── Switch to editor system prompt ────────────────────
            self._system_prompt = build_editor_system_prompt(cfg)

            # ── Fresh memory for the editor session ───────────────
            self._memory.reset_for_new_source()
            self._consecutive_errors = 0
            self._missing_param_tracker.clear()

            # ── Single editor phase ───────────────────────────────
            if self._run_state:
                self._run_state.mark_phase_start("editor")
                self._save_state()
            self._display.phase("Editorial Review (HITL — expect ask_human pauses)")
            max_turns = cfg.editor_max_turns_overview + cfg.editor_max_turns_per_section * len(
                section_summaries
            )
            self._display.step(f"Turn budget: {max_turns}")

            prompt = build_editor_review_prompt(
                cfg=cfg,
                inventory=inventory,
                instructions=instructions,
                section_summaries=section_summaries,
            )

            self._run_phase(
                prompt=prompt,
                max_turns=max_turns,
            )
            if self._run_state:
                self._run_state.mark_phase_complete("editor")
                self._run_state.mark_completed()
                self._save_state()

        except KeyboardInterrupt:
            self._display.warning("Editor interrupted by user.")
            if self._run_state:
                self._run_state.mark_interrupted()
                self._save_state()
            self._dump_progress_on_crash()
        except Exception as e:
            self._display.error(f"Editor failed: {e}")
            logger.exception("StoryTellerAgent run_editor failure")
            if self._run_state:
                self._run_state.mark_failed(str(e))
                self._save_state()
            self._dump_progress_on_crash()

        summary = self._build_summary()
        self._display.phase("EDITORIAL REVIEW COMPLETE")
        self._display.step(f"Sections updated: {self._write_narrative_tool.sections_written}")
        self._display.step(f"LLM usage: {self._llm.usage_summary}")
        return summary

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict:
        return {
            "sections_written": self._write_narrative_tool.sections_written,
            "citations": len(self._cite_source_tool._citations),
            "notes": len(self._note_tool._notes),
            "output_path": self._narrative_run_path,
            "run_id": self._cfg.run_id or "",
            "llm_usage": self._llm.usage_summary,
        }
