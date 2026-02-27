"""
DataScientistAgent — autonomous research analysis agent.

Reads bronze Unity Catalog tables, builds silver-layer pre-joined datasets,
runs thesis-driven analysis, creates compelling visualizations, and exports
structured findings to a results volume.

Uses the same ReAct loop pattern as the DataEngineerAgent.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

from versifai.core.agent import BaseAgent
from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.run_manager import (
    RunState,
    generate_run_id,
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
from versifai.science_agents.scientist.prompts import (
    build_orientation_prompt,
    build_scientist_system_prompt,
    build_silver_construction_prompt,
    build_synthesis_prompt,
    build_theme_analysis_prompt,
    build_validation_prompt,
)
from versifai.science_agents.scientist.tools.check_confounders import CheckConfoundersTool
from versifai.science_agents.scientist.tools.literature_review import LiteratureReviewTool
from versifai.science_agents.scientist.tools.model_fitting import ModelFittingTool
from versifai.science_agents.scientist.tools.save_finding import SaveFindingTool
from versifai.science_agents.scientist.tools.statistical_analysis import StatisticalAnalysisTool
from versifai.science_agents.scientist.tools.validate_silver import ValidateSilverTool
from versifai.science_agents.scientist.tools.validate_statistics import ValidateStatisticsTool

if TYPE_CHECKING:
    from versifai.science_agents.scientist.config import AnalysisTheme, ResearchConfig

logger = logging.getLogger("agent.scientist")


class DataScientistAgent(BaseAgent):
    """
    Autonomous research analysis agent powered by Claude.

    Workflow:
      1. Orientation — inventory bronze tables, understand available data
      2. Silver Construction — build pre-joined analytical datasets
      3. Research Analysis — investigate each research question
      4. Synthesis — compile findings, create executive summary

    Usage in a Databricks notebook::

        from versifai.science_agents.scientist.agent import DataScientistAgent
        from versifai.science_agents.scientist.config import ResearchConfig

        cfg = ResearchConfig()
        agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)
        agent.run()
    """

    def __init__(
        self,
        cfg: ResearchConfig | None = None,
        dbutils=None,
    ) -> None:
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
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

        # Resolve run path — always isolated
        self._run_id = cfg.run_id or generate_run_id()
        self._run_path = init_run_directory(cfg.results_volume_path, self._run_id)
        write_run_metadata(
            self._run_path, config_name=cfg.name, run_id=self._run_id, agent_type="scientist"
        )
        logger.info("Run ID: %s", self._run_id)
        logger.info("Run path: %s", self._run_path)

        # Run state — initialised per entry point, persisted for resume
        self._run_state: RunState | None = None

        # Tools — use self._run_path for all output paths
        self._finding_tool = SaveFindingTool(results_path=self._run_path)
        self._note_tool = SaveNoteTool(
            notes_path=os.path.join(self._run_path, "notes"),
        )
        self._viz_tool = CreateVisualizationTool(
            cfg=cfg,
            display=self._display,
            notes_path=os.path.join(self._run_path, "notes"),
            results_path=self._run_path,
        )
        self._view_chart_tool = ViewChartTool(
            charts_path=os.path.join(self._run_path, "charts"),
            tables_path=os.path.join(self._run_path, "tables"),
        )
        self._register_tools()

        # Pre-build system prompt
        self._system_prompt = build_scientist_system_prompt(cfg)

    def _register_tools(self) -> None:
        """Register the scientist's tool set."""
        cfg = self._cfg
        # Shared web scraper — used directly as a tool AND by literature review
        self._web_scraper = WebScraperTool()
        tools = [
            # Data access (silver-only writes, bronze is read-only)
            SilverOnlyExecuteSQLTool(),
            ListCatalogTablesTool(cfg=cfg.project),
            # Statistical analysis & modeling
            StatisticalAnalysisTool(),
            ModelFittingTool(),
            CheckConfoundersTool(),
            # Validation gates
            ValidateSilverTool(),
            ValidateStatisticsTool(),
            # Web search & documentation
            WebSearchTool(cfg=cfg.project),
            # Web scraping & literature
            self._web_scraper,
            LiteratureReviewTool(cfg=cfg, web_scraper=self._web_scraper),
            # Visualization & output
            self._finding_tool,
            self._note_tool,
            self._viz_tool,
            self._view_chart_tool,
            # Utility
            DynamicToolBuilderTool(registry=self._registry),
        ]
        # Conditionally register MLflow model logging
        if cfg.mlflow_experiment:
            from versifai.science_agents.scientist.tools.log_model import LogModelTool

            tools.append(LogModelTool(cfg=cfg))
        for tool in tools:
            self._registry.register(tool)

    # ------------------------------------------------------------------
    # Pre-flight catalog discovery
    # ------------------------------------------------------------------

    def _discover_catalog_tables(self) -> tuple[set[str], dict[str, list[tuple[str, str]]]]:
        """Query Unity Catalog to find what tables exist and their schemas.

        Returns:
            - set of simple table names (not fully qualified)
            - dict mapping table name -> list of (column_name, data_type) tuples

        This runs BEFORE any agent phases. The schemas are injected into
        prompts so the agent never has to guess column names.
        """
        self._display.step("Discovering available catalog tables...")
        result = self._registry.execute("list_catalog_tables")
        if not result.success:
            logger.warning(f"Catalog discovery failed: {result.error}")
            return set(), {}

        tables = set()
        for table_info in result.data.get("tables", []):
            name = (
                table_info.get("tableName")
                or table_info.get("name")
                or table_info.get("table_name", "")
            )
            if name:
                tables.add(name.lower())

        # Fetch schema for every table via DESCRIBE TABLE
        schemas: dict[str, list[tuple[str, str]]] = {}
        cfg = self._cfg
        for table_name in sorted(tables):
            self._display.step(f"  Fetching schema: {table_name}")
            desc_result = self._registry.execute(
                "execute_sql",
                sql=f"DESCRIBE TABLE {cfg.project.full_schema}.{table_name}",
            )
            if desc_result.success:
                cols = []
                for row in desc_result.data.get("rows", []):
                    col_name = row.get("col_name", "")
                    data_type = row.get("data_type", "")
                    # DESCRIBE TABLE includes partition/metadata separator rows
                    # that start with # — skip those
                    if col_name and not col_name.startswith("#"):
                        cols.append((col_name, data_type))
                schemas[table_name] = cols
            else:
                logger.warning(f"Could not describe {table_name}: {desc_result.error}")

        return tables, schemas

    def _log_table_coverage(self, available_tables: set[str]) -> None:
        """Log which tables each theme and silver dataset can access.

        Purely informational — all themes and datasets are processed regardless
        of how many required tables are present. The agent adapts to whatever
        data is available.
        """
        cfg = self._cfg
        for ds in cfg.silver_datasets:
            present = [t for t in ds.source_tables if t.lower() in available_tables]
            missing = [t for t in ds.source_tables if t.lower() not in available_tables]
            if missing:
                self._display.step(
                    f"  Silver '{ds.name}': {len(present)}/{len(ds.source_tables)} "
                    f"source tables available (missing: {', '.join(missing)})"
                )
            else:
                self._display.step(
                    f"  Silver '{ds.name}': all {len(present)} source tables available"
                )

        for theme in sorted(cfg.analysis_themes, key=lambda t: t.sequence):
            required = set(t.lower() for t in theme.required_tables)  # type: ignore[assignment]
            present = required & available_tables  # type: ignore[assignment]
            missing = required - available_tables  # type: ignore[assignment]
            if missing:
                self._display.step(
                    f"  Theme {theme.sequence} ({theme.title}): "
                    f"{len(present)}/{len(required)} tables available "
                    f"(missing: {', '.join(missing)})"
                )
            else:
                self._display.step(
                    f"  Theme {theme.sequence} ({theme.title}): all {len(present)} tables available"
                )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        instructions: str = "",
        rerun_analysis: bool = False,
    ) -> dict:
        """
        Run the full research analysis pipeline.

        By default, scans for existing state (silver tables, findings,
        charts) and resumes from where the pipeline left off. Set
        ``rerun_analysis=True`` to force a complete fresh start.

        Args:
            instructions: Optional high-level guidance prepended to every
                          phase prompt. e.g., "Focus on 2023 data only."
            rerun_analysis: If True, ignore all existing state and re-run
                            every phase from scratch. Default is False
                            (smart resume).

        Returns a summary dict with findings, charts, and statistics.

        Usage in a Databricks notebook::

            agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)
            agent.run()                             # resumes from last state
            agent.run(rerun_analysis=True)          # full fresh start
            agent.run(instructions="Focus on 2024") # resume with guidance
        """
        cfg = self._cfg
        self._instructions = instructions
        self._display.phase("DATA SCIENTIST AGENT STARTING")
        self._display.step(f"Project: {cfg.name}")
        self._display.step(f"Thesis: {cfg.thesis[:100]}...")
        self._display.step(f"Analysis themes: {len(cfg.analysis_themes)}")
        self._display.step(f"Results: {self._run_path}")
        if cfg.run_id:
            self._display.step(f"Run ID: {cfg.run_id}")
        self._display.step(f"Tools: {self._registry.tool_names + ['ask_human']}")
        if rerun_analysis:
            self._display.step("Mode: FULL RE-RUN (ignoring existing state)")
        else:
            self._display.step("Mode: SMART RESUME (skipping completed phases)")

        # Ensure results directories exist
        os.makedirs(os.path.join(self._run_path, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "tables"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "notes"), exist_ok=True)

        # Write run metadata if using run isolation
        if cfg.run_id:
            write_run_metadata(self._run_path, cfg.name, cfg.run_id, agent_type="scientist")

        # Initialise or resume run state
        if cfg.run_id:
            if not rerun_analysis:
                existing = load_run_state(self._run_path)
                if existing and existing.status in ("running", "interrupted", "failed"):
                    self._run_state = existing
                    self._run_state.status = "running"
                    self._display.step(f"Resuming previous run (was {existing.status})")
                    self._display.step(f"  Completed phases: {existing.completed_phases}")
            if self._run_state is None:
                self._run_state = RunState(entry_point="run")
            save_run_state(self._run_path, self._run_state)

        # Resolve dependencies
        self._resolved_deps: dict[str, str] = {}
        if cfg.dependencies:
            for dep in cfg.dependencies:
                resolved = resolve_dependency(dep)
                self._resolved_deps[dep.config_name] = resolved
                self._display.step(f"Dependency '{dep.config_name}' -> {resolved}")

        try:
            # ── Pre-flight: discover what tables actually exist ─────
            available_tables, table_schemas = self._discover_catalog_tables()
            self._display.step(
                f"Found {len(available_tables)} tables in catalog: "
                f"{', '.join(sorted(available_tables)) if available_tables else '(none)'}"
            )

            if not available_tables:
                self._display.error(
                    "No tables found in catalog. Run the Data Engineer agent first."
                )
                return self._build_summary()

            # Log table coverage — purely informational, nothing is skipped
            self._log_table_coverage(available_tables)

            # Rebuild system prompt with schemas injected (cached)
            # The agent queries data_catalog on demand via execute_sql
            self._system_prompt = build_scientist_system_prompt(
                cfg,
                table_schemas=table_schemas,
            )

            # ── Scan existing pipeline state ───────────────────────
            state = None
            if not rerun_analysis:
                state = self._scan_pipeline_state(available_tables)
                completed_silver = state["completed_silver"]
                completed_themes = state["completed_themes"]
                pending_silver = [
                    ds.name for ds in cfg.silver_datasets if ds.name not in completed_silver
                ]
                all_themes = sorted(cfg.analysis_themes, key=lambda t: t.sequence)
                pending_themes = [t for t in all_themes if t.id not in completed_themes]

                self._display.step("--- Pipeline State ---")
                if completed_silver:
                    self._display.step(
                        f"  Silver DONE ({len(completed_silver)}): "
                        f"{', '.join(sorted(completed_silver))}"
                    )
                if pending_silver:
                    self._display.step(
                        f"  Silver TODO ({len(pending_silver)}): {', '.join(pending_silver)}"
                    )
                if completed_themes:
                    self._display.step(
                        f"  Themes DONE ({len(completed_themes)}): "
                        f"{', '.join(sorted(completed_themes))}"
                    )
                if pending_themes:
                    self._display.step(
                        f"  Themes TODO ({len(pending_themes)}): "
                        f"{', '.join(t.id for t in pending_themes)}"
                    )
                if state["has_findings"]:
                    self._display.step(f"  Existing findings: {state['findings_count']}")
                    # Load existing findings so synthesis has full picture
                    self._load_existing_findings(state["existing_findings"])
                if state.get("existing_notes"):
                    note_themes = sorted(state["existing_notes"].keys())
                    self._display.step(
                        f"  Existing notes ({len(note_themes)}): {', '.join(note_themes)}"
                    )
                self._display.step("---")

            # ── Phase 1: Orientation ────────────────────────────────
            # Always run — lightweight context-setting phase
            if self._run_state:
                self._run_state.mark_phase_start("orientation")
                self._save_state()
            self._display.phase("Phase 1: Orientation")
            self._run_phase(
                prompt=self._inject_instructions(
                    build_orientation_prompt(cfg, available_tables, table_schemas)
                ),
                max_turns=cfg.max_turns_per_phase,
            )
            if self._run_state:
                self._run_state.mark_phase_complete("orientation")
                self._save_state()

            # ── Phase 2: Silver Construction ────────────────────────
            if self._run_state:
                self._run_state.mark_phase_start("silver")
                self._save_state()
            self._display.phase("Phase 2: Silver Dataset Construction")
            for i, dataset_spec in enumerate(cfg.silver_datasets, 1):
                # Smart resume: skip silver tables that already exist
                if state and dataset_spec.name in state["completed_silver"]:
                    self._display.step(
                        f"  Silver {i}/{len(cfg.silver_datasets)} "
                        f"{dataset_spec.name} — SKIPPED (already exists)"
                    )
                    continue

                self._display.phase(
                    f"Silver Dataset {i}/{len(cfg.silver_datasets)}: {dataset_spec.name}"
                )
                carryover = self._memory.get_carryover_context()
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                prompt = build_silver_construction_prompt(
                    cfg,
                    dataset_spec,
                    available_tables,
                    table_schemas,
                )
                if carryover:
                    prompt = f"## Context From Orientation\n{carryover}\n\n---\n\n{prompt}"

                self._run_phase(
                    prompt=self._inject_instructions(prompt),
                    max_turns=cfg.max_turns_per_phase,
                )
                self._memory.log_source_summary(
                    f"silver_{dataset_spec.name}",
                    f"Built silver dataset: {dataset_spec.name}",
                )
                if self._run_state:
                    self._run_state.mark_item_complete("silver", dataset_spec.name)
                    self._save_state()

            if self._run_state:
                self._run_state.mark_phase_complete("silver")
                self._save_state()

            # ── Phase 3: Theme-Driven Research Analysis ────────────
            all_themes = sorted(cfg.analysis_themes, key=lambda t: t.sequence)

            # If silver tables were just built, refresh catalog knowledge
            # so theme prompts include the new silver schemas
            if state and state["completed_silver"] != {ds.name for ds in cfg.silver_datasets}:
                refreshed, refreshed_schemas = self._discover_catalog_tables()
                if refreshed:
                    available_tables = refreshed
                    table_schemas = refreshed_schemas

            if self._run_state:
                self._run_state.mark_phase_start("themes")
                self._save_state()
            self._display.phase(
                f"Phase 3: Theme-Driven Research Analysis ({len(all_themes)} themes)"
            )
            themes_skipped = 0
            for _i, theme in enumerate(all_themes, 1):
                # Smart resume: skip themes that already have findings
                if state and theme.id in state["completed_themes"]:
                    self._display.step(
                        f"  Theme {theme.sequence}/{len(all_themes)} "
                        f"{theme.title} — SKIPPED (findings exist)"
                    )
                    themes_skipped += 1
                    continue

                self._display.phase(
                    f"Theme {theme.sequence}/{len(all_themes)}: "
                    f"{theme.title} — {theme.question[:80]}..."
                )
                carryover = self._memory.get_carryover_context()
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                # Load any prior-session notes for this theme
                theme_notes = self._note_tool.get_theme_notes_from_disk(theme.id)
                prompt = build_theme_analysis_prompt(
                    cfg,
                    theme,
                    available_tables,
                    table_schemas,
                    existing_notes=theme_notes,
                )
                if carryover:
                    prompt = f"## Context From Prior Work\n{carryover}\n\n---\n\n{prompt}"

                # Track findings count before this theme
                findings_before = len(self._finding_tool.findings)

                self._run_phase(
                    prompt=self._inject_instructions(prompt),
                    max_turns=cfg.max_turns_per_theme,
                )

                # Nudge agent if it forgot to call save_finding
                theme_findings = len(self._finding_tool.findings) - findings_before
                if theme_findings == 0:
                    self._display.warning(
                        f"Theme {theme.sequence} produced 0 findings — "
                        f"prompting agent to record findings"
                    )
                    self._nudge_save_finding(theme)

                self._memory.log_source_summary(
                    f"analysis_{theme.id}",
                    f"Analyzed Theme {theme.sequence}: {theme.title}",
                )
                if self._run_state:
                    self._run_state.mark_item_complete("themes", theme.id)
                    self._save_state()
                self._display.success(f"Completed Theme {theme.sequence}: {theme.title}")

            if self._run_state:
                self._run_state.mark_phase_complete("themes")
                self._save_state()

            # ── Phase 3b: Retrospective Findings ────────────────────
            # If themes produced sparse findings, review charts/notes
            # and create structured findings from existing output.
            self._retrospective_findings(all_themes, available_tables, table_schemas)

            # ── Phase 4: Synthesis ──────────────────────────────────
            # Skip synthesis only if ALL themes were skipped (nothing new)
            new_findings = (
                len(self._finding_tool.findings) - state["findings_count"]
                if state
                else len(self._finding_tool.findings)
            )
            if state and themes_skipped == len(all_themes) and new_findings == 0:
                self._display.step("Phase 4: Synthesis — SKIPPED (no new findings)")
                self._display.step(
                    "All phases complete. Nothing to do. Use "
                    "rerun_analysis=True to force a fresh start."
                )
                if self._run_state:
                    self._run_state.mark_phase_complete("synthesis")
                    self._save_state()
            else:
                if self._run_state:
                    self._run_state.mark_phase_start("synthesis")
                    self._save_state()
                self._display.phase("Phase 4: Synthesis")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                findings_text = self._finding_tool.findings_summary_text()
                prompt = build_synthesis_prompt(cfg, findings_text)
                self._run_phase(prompt=prompt, max_turns=cfg.max_turns_per_phase)

                # Export findings JSON
                findings_path = self._finding_tool.export_findings_json()
                self._display.success(f"Findings exported to {findings_path}")
                if self._run_state:
                    self._run_state.mark_phase_complete("synthesis")
                    self._save_state()

            # Mark run as completed
            if self._run_state:
                self._run_state.mark_completed()
                self._save_state()

        except KeyboardInterrupt:
            self._display.warning("Agent interrupted by user.")
            if self._run_state:
                self._run_state.mark_interrupted()
                self._save_state()
            self._dump_progress_on_crash()
        except Exception as e:
            self._display.error(f"Agent failed: {e}")
            logger.exception("Scientist agent top-level failure")
            if self._run_state:
                self._run_state.mark_failed(str(e))
                self._save_state()
            self._dump_progress_on_crash()

        # Always export findings (even after crash/interrupt) so nothing is lost
        if self._finding_tool.findings:
            try:
                findings_path = self._finding_tool.export_findings_json()
                self._display.success(f"Findings exported to {findings_path}")
            except Exception as export_err:
                self._display.warning(f"Could not export findings: {export_err}")

        # Update run metadata on completion
        if cfg.run_id:
            write_run_metadata(
                self._run_path,
                cfg.name,
                cfg.run_id,
                extra={
                    "completed_at": datetime.now().isoformat(),
                    "total_findings": len(self._finding_tool.findings),
                    "total_charts": len(self._viz_tool.charts_created),
                },
            )

        # Final summary
        summary = self._build_summary()
        self._display.phase("RESEARCH ANALYSIS COMPLETE")
        self._display.step(f"Total findings: {len(self._finding_tool.findings)}")
        self._display.step(f"Charts created: {len(self._viz_tool.charts_created)}")
        self._display.step(f"LLM usage: {self._llm.usage_summary}")

        return summary

    def run_visualizations(
        self,
        chart_types: list[str] | None = None,
        instructions: str = "",
    ) -> dict:
        """
        Re-generate visualizations and output tables from existing data.

        Pre-scans Unity Catalog for silver/bronze tables and the results
        volume for existing charts and tables. Then runs a focused prompt
        per theme that uses existing data to create new outputs.

        Args:
            chart_types: Only regenerate these output types.
                         e.g., ``["choropleth", "dual_choropleth"]`` for
                         maps only, ``["table"]`` for summary tables only,
                         or ``None`` for everything.
            instructions: High-level guidance for the agent. e.g.,
                          "Recreate all choropleths with RdYlGn scale.
                          Also create a summary table per theme."

        Usage in a Databricks notebook::

            agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)
            agent.run_visualizations()                                  # all outputs
            agent.run_visualizations(chart_types=["choropleth"])        # maps only
            agent.run_visualizations(chart_types=["table"])             # tables only
            agent.run_visualizations(
                chart_types=["choropleth", "dual_choropleth"],
                instructions="Every map must cover all ~3,200 US counties."
            )
        """
        cfg = self._cfg
        self._instructions = instructions
        focus = chart_types or []
        focus_label = ", ".join(focus) if focus else "all"
        self._display.phase(f"VISUALIZATION REFRESH — {focus_label}")

        os.makedirs(os.path.join(self._run_path, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "tables"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "notes"), exist_ok=True)

        # Fresh run state for this entry point
        if cfg.run_id:
            self._run_state = RunState(entry_point="run_visualizations")
            self._save_state()

        try:
            # ── Pre-scan ──────────────────────────────────────────────
            available_tables, table_schemas = self._discover_catalog_tables()
            silver_tables = sorted(t for t in available_tables if t.startswith("silver_"))
            bronze_tables = sorted(
                t for t in available_tables if not t.startswith("silver_") and t != "data_catalog"
            )
            self._display.step(
                f"Silver tables: {', '.join(silver_tables) if silver_tables else '(none)'}"
            )

            if not silver_tables:
                self._display.error("No silver tables found. Run the full pipeline first.")
                return self._build_summary()

            # Scan existing outputs (charts + tables)
            existing_outputs = self._scan_existing_outputs()
            existing_charts = existing_outputs.get("charts", [])
            existing_tables = existing_outputs.get("tables", [])
            if existing_charts:
                self._display.step(
                    f"Existing charts ({len(existing_charts)}): "
                    f"{', '.join(existing_charts[:10])}"
                    + (
                        f"... +{len(existing_charts) - 10} more"
                        if len(existing_charts) > 10
                        else ""
                    )
                )
            if existing_tables:
                self._display.step(
                    f"Existing tables ({len(existing_tables)}): "
                    f"{', '.join(existing_tables[:10])}"
                    + (
                        f"... +{len(existing_tables) - 10} more"
                        if len(existing_tables) > 10
                        else ""
                    )
                )

            self._system_prompt = build_scientist_system_prompt(
                cfg,
                table_schemas=table_schemas,
            )

            # ── Per-theme visualization ───────────────────────────────
            all_themes = sorted(cfg.analysis_themes, key=lambda t: t.sequence)
            if self._run_state:
                self._run_state.mark_phase_start("visualizations")
                self._save_state()
            for theme in all_themes:
                self._display.phase(f"Viz: Theme {theme.sequence} — {theme.title}")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                # Load prior-session notes for context
                theme_notes = self._note_tool.get_theme_notes_from_disk(theme.id)
                prompt = self._build_viz_refresh_prompt(
                    theme,
                    silver_tables,
                    bronze_tables,
                    existing_charts,
                    existing_tables,
                    focus,
                    existing_notes=theme_notes,
                )
                self._run_phase(
                    prompt=self._inject_instructions(prompt),
                    max_turns=30,
                )
                if self._run_state:
                    self._run_state.mark_item_complete("visualizations", theme.id)
                    self._save_state()
                self._display.success(f"Viz complete: Theme {theme.sequence}")

            if self._run_state:
                self._run_state.mark_phase_complete("visualizations")
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
            logger.exception("run_visualizations failure")
            if self._run_state:
                self._run_state.mark_failed(str(e))
                self._save_state()
            self._dump_progress_on_crash()

        summary = self._build_summary()
        self._display.phase("VISUALIZATION REFRESH COMPLETE")
        self._display.step(f"Charts created: {len(self._viz_tool.charts_created)}")
        self._display.step(f"LLM usage: {self._llm.usage_summary}")
        return summary

    def _scan_existing_outputs(self) -> dict[str, list[str]]:
        """Scan the results volume for existing charts, tables, and notes."""
        results = {}
        for subdir in ("charts", "tables", "notes"):
            path = os.path.join(self._run_path, subdir)
            if os.path.isdir(path):
                files = sorted(os.listdir(path))
                if files:
                    results[subdir] = files
        return results

    def _build_viz_refresh_prompt(
        self,
        theme,
        silver_tables: list[str],
        bronze_tables: list[str],
        existing_charts: list[str],
        existing_result_tables: list[str],
        focus_chart_types: list[str],
        existing_notes: str = "",
    ) -> str:
        """Build a focused prompt for regenerating visualizations and tables.

        Includes the full theme context (data_notes, analysis_steps, punchline)
        so the agent knows what SQL to write and what story each output tells.
        """
        cfg = self._cfg
        proj = cfg.project

        silver_list = "\n".join(f"- `{proj.full_schema}.{t}`" for t in silver_tables)
        bronze_list = "\n".join(f"- `{proj.full_schema}.{t}`" for t in bronze_tables)

        existing_section = ""
        if existing_charts or existing_result_tables:
            existing_section = "\n### Existing Outputs (already generated)\n"
            if existing_charts:
                existing_section += (
                    "**Charts:**\n" + "\n".join(f"- `{c}`" for c in existing_charts) + "\n"
                )
            if existing_result_tables:
                existing_section += (
                    "**Tables:**\n" + "\n".join(f"- `{t}`" for t in existing_result_tables) + "\n"
                )
            existing_section += "\nYou may replace these or create new ones.\n"

        focus_instruction = ""
        if focus_chart_types:
            types_str = ", ".join(focus_chart_types)
            focus_instruction = (
                f"\n**FOCUS**: Only create {types_str} outputs. Skip all other types.\n"
            )
        else:
            focus_instruction = (
                "\nCreate ALL prescribed visualizations AND summary tables for this theme.\n"
            )

        if theme.signature_visualization:
            viz_text = (
                f"{theme.signature_visualization}\n\n"
                f"**Visualization Notes**: {theme.visualization_notes}"
                if theme.visualization_notes
                else theme.signature_visualization
            )
        elif theme.visualization_notes:
            viz_text = (
                "No prescribed signature visualization for this theme.\n\n"
                f"**Visualization Notes**: {theme.visualization_notes}"
            )
        else:
            viz_text = "(No prescribed visualizations — recreate from existing data.)"

        # Full theme context so the agent knows the story and data relationships
        data_notes_section = ""
        if theme.data_notes:
            data_notes_section = f"\n### Data Notes\n{theme.data_notes}\n"

        # Prior session research notes
        notes_section = ""
        if existing_notes:
            notes_section = (
                "\n### Prior Session Notes\n"
                "These notes from a previous analysis session contain data quality "
                "observations, quirks, and methodology decisions relevant to this "
                "theme's visualizations.\n\n"
                f"{existing_notes}\n"
            )

        steps_text = "\n".join(f"   {i}. {step}" for i, step in enumerate(theme.analysis_steps, 1))

        # Which tables this theme primarily uses
        theme_tables = ""
        if theme.required_tables:
            present = [
                t
                for t in theme.required_tables
                if t.lower() in {s.lower() for s in silver_tables + bronze_tables}
            ]
            theme_tables = "\n### Primary Tables for This Theme\n" + "\n".join(
                f"- `{proj.full_schema}.{t}`" for t in present
            )

        return f"""\
## Visualization Refresh — Theme {theme.sequence}: {theme.title}

**Core Question**: {theme.question}
**Analysis Type**: {theme.analysis_type}
**Punchline**: {theme.punchline}
{data_notes_section}{notes_section}
### Analysis Steps (for context — do NOT re-run, just use for viz design)
{steps_text}

The silver tables and bronze data already exist. Your ONLY job is to create
visualizations and output tables. Do NOT re-run statistical analysis, do NOT
rebuild silver tables, do NOT save findings. Just create the charts and tables.
{theme_tables}

### Available Silver Tables
{silver_list}

### Available Bronze Tables
{bronze_list}
{existing_section}
{focus_instruction}
### ★ Signature Visualization (REQUIRED — follow spec exactly)
{viz_text}

### Instructions

1. For each visualization, write a SQL query that fetches the needed data
   from the silver or bronze tables above. Use the Data Notes and Analysis
   Steps above to understand what JOINs and calculations are needed.
2. Call `create_visualization` with `sql_query`, `theme_id='{theme.id}'`,
   and `interpretation` (2-3 sentences on what the chart shows and how to
   read it). ALWAYS use sql_query (never pass data from execute_sql).
   theme_id + interpretation auto-log reproducibility metadata and chart
   documentation to the notes file.
3. For choropleths: the sql_query must return `{proj.join_key.column_name}`
   (5-digit FIPS) plus a value column. Set fips_column="{proj.join_key.column_name}".
   The map should cover ALL ~3,200 US counties, not a subset.
4. For dual_choropleth: return FIPS + TWO value columns.
5. Use descriptive titles that state the finding, not just label axes.
   Good: "Counties with High SVI Show 23% Lower MA Penetration"
   Bad: "SVI vs MA Penetration"
6. Move quickly — this is a visualization refresh, not a full analysis.
   Run a quick SELECT to verify the query works, then create the chart.
"""

    def run_themes(
        self,
        start_theme: int = 0,
        themes: list[int] | None = None,
        synthesize: bool = True,
        instructions: str = "",
    ) -> dict:
        """
        Re-run ONLY the theme analysis phase (and optionally synthesis).

        Skips orientation and silver construction — assumes silver tables
        already exist in Unity Catalog. Useful for re-running analysis with
        updated prompts, regenerating visualizations, or continuing from a
        specific theme.

        Args:
            start_theme: Theme sequence number to start from (0 = all themes).
                         e.g., ``start_theme=3`` skips Themes 0-2.
                         Ignored if ``themes`` is provided.
            themes: Explicit list of theme sequence numbers to run.
                    e.g., ``themes=[1, 4, 7]`` runs only those three themes.
                    Overrides ``start_theme`` when provided.
            synthesize: Whether to run the synthesis phase after themes.
            instructions: High-level guidance for the agent. e.g.,
                          "Focus on generating choropleths for every finding."

        Usage in a Databricks notebook::

            agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)
            agent.run_themes()                    # all themes + synthesis
            agent.run_themes(start_theme=4)       # themes 4-8 + synthesis
            agent.run_themes(themes=[1, 4, 7])    # only these three themes
            agent.run_themes(themes=[3])           # single theme re-run
            agent.run_themes(synthesize=False)     # themes only, no synthesis
            agent.run_themes(instructions="Prioritize geographic maps.")
        """
        cfg = self._cfg
        self._instructions = instructions
        self._display.phase("DATA SCIENTIST AGENT — THEMES ONLY")
        self._display.step(f"Project: {cfg.name}")
        self._display.step(f"Thesis: {cfg.thesis[:100]}...")
        if themes is not None:
            self._display.step(f"Themes: {themes}")
        else:
            self._display.step(f"Start theme: {start_theme}")

        os.makedirs(os.path.join(self._run_path, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "tables"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "notes"), exist_ok=True)

        # Fresh run state for this entry point
        if cfg.run_id:
            self._run_state = RunState(entry_point="run_themes")
            self._save_state()

        try:
            # ── Pre-flight (still needed for schemas + data catalog) ──
            available_tables, table_schemas = self._discover_catalog_tables()
            self._display.step(f"Found {len(available_tables)} tables in catalog")
            if not available_tables:
                self._display.error("No tables found. Run the Data Engineer first.")
                return self._build_summary()

            self._system_prompt = build_scientist_system_prompt(
                cfg,
                table_schemas=table_schemas,
            )

            # ── Theme Analysis ────────────────────────────────────────
            all_themes = sorted(cfg.analysis_themes, key=lambda t: t.sequence)
            if themes is not None:
                theme_set = set(themes)
                themes_to_run = [t for t in all_themes if t.sequence in theme_set]
            else:
                themes_to_run = [t for t in all_themes if t.sequence >= start_theme]
            if self._run_state:
                self._run_state.mark_phase_start("themes")
                self._save_state()
            self._display.phase(
                f"Theme Analysis ({len(themes_to_run)} of {len(all_themes)} themes)"
            )

            for _i, theme in enumerate(themes_to_run, 1):
                self._display.phase(
                    f"Theme {theme.sequence}/{len(all_themes)}: "
                    f"{theme.title} — {theme.question[:80]}..."
                )
                carryover = self._memory.get_carryover_context()
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                # Load any prior-session notes for this theme
                theme_notes = self._note_tool.get_theme_notes_from_disk(theme.id)
                prompt = build_theme_analysis_prompt(
                    cfg,
                    theme,
                    available_tables,
                    table_schemas,
                    existing_notes=theme_notes,
                )
                if carryover:
                    prompt = f"## Context From Prior Work\n{carryover}\n\n---\n\n{prompt}"

                # Track findings count before this theme
                findings_before = len(self._finding_tool.findings)

                self._run_phase(
                    prompt=self._inject_instructions(prompt),
                    max_turns=cfg.max_turns_per_theme,
                )

                # Nudge agent if it forgot to call save_finding
                theme_findings = len(self._finding_tool.findings) - findings_before
                if theme_findings == 0:
                    self._display.warning(
                        f"Theme {theme.sequence} produced 0 findings — "
                        f"prompting agent to record findings"
                    )
                    self._nudge_save_finding(theme)

                self._memory.log_source_summary(
                    f"analysis_{theme.id}",
                    f"Analyzed Theme {theme.sequence}: {theme.title}",
                )
                if self._run_state:
                    self._run_state.mark_item_complete("themes", theme.id)
                    self._save_state()
                self._display.success(f"Completed Theme {theme.sequence}: {theme.title}")

            if self._run_state:
                self._run_state.mark_phase_complete("themes")
                self._save_state()

            # ── Retrospective Findings ─────────────────────────────────
            self._retrospective_findings(themes_to_run, available_tables, table_schemas)

            # ── Synthesis (optional) ──────────────────────────────────
            if synthesize:
                if self._run_state:
                    self._run_state.mark_phase_start("synthesis")
                    self._save_state()
                self._display.phase("Synthesis")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                findings_text = self._finding_tool.findings_summary_text()
                prompt = build_synthesis_prompt(cfg, findings_text)
                self._run_phase(
                    prompt=self._inject_instructions(prompt),
                    max_turns=cfg.max_turns_per_phase,
                )

                findings_path = self._finding_tool.export_findings_json()
                self._display.success(f"Findings exported to {findings_path}")
                if self._run_state:
                    self._run_state.mark_phase_complete("synthesis")
                    self._save_state()

            if self._run_state:
                self._run_state.mark_completed()
                self._save_state()

        except KeyboardInterrupt:
            self._display.warning("Agent interrupted by user.")
            if self._run_state:
                self._run_state.mark_interrupted()
                self._save_state()
            self._dump_progress_on_crash()
        except Exception as e:
            self._display.error(f"Agent failed: {e}")
            logger.exception("Scientist agent run_themes failure")
            if self._run_state:
                self._run_state.mark_failed(str(e))
                self._save_state()
            self._dump_progress_on_crash()

        # Always export findings (even after crash/interrupt) so nothing is lost
        if self._finding_tool.findings:
            try:
                findings_path = self._finding_tool.export_findings_json()
                self._display.success(f"Findings exported to {findings_path}")
            except Exception as export_err:
                self._display.warning(f"Could not export findings: {export_err}")

        summary = self._build_summary()
        self._display.phase("THEME ANALYSIS COMPLETE")
        self._display.step(f"Total findings: {len(self._finding_tool.findings)}")
        self._display.step(f"Charts created: {len(self._viz_tool.charts_created)}")
        self._display.step(f"LLM usage: {self._llm.usage_summary}")
        return summary

    def run_validation(
        self,
        themes: list[int] | None = None,
        instructions: str = "",
    ) -> dict:
        """
        Validate and improve all analysis outputs against original specs.

        Audits every theme's findings, charts, and tables for correctness,
        data quality, visualization quality, and completeness. Rebuilds
        anything that is wrong or unconvincing. Stores all validation
        findings into notes.

        The agent has FULL tool access — it can re-run queries, rebuild
        charts with advanced types, update findings, and create new outputs.

        Args:
            themes: Specific theme sequence numbers to validate.
                    e.g., ``themes=[1, 3, 7]``. None = all themes.
            instructions: High-level guidance for the agent.

        Usage in a Databricks notebook::

            agent = DataScientistAgent(cfg=cfg, dbutils=dbutils)
            agent.run_validation()                     # validate all themes
            agent.run_validation(themes=[1, 3, 7])     # validate specific themes
            agent.run_validation(instructions="Focus on upgrading chart quality.")
        """
        cfg = self._cfg
        self._instructions = instructions
        self._display.phase("VALIDATION & QUALITY REVIEW")
        self._display.step(f"Project: {cfg.name}")
        if themes is not None:
            self._display.step(f"Themes to validate: {themes}")
        else:
            self._display.step("Validating ALL themes")

        os.makedirs(os.path.join(self._run_path, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "tables"), exist_ok=True)
        os.makedirs(os.path.join(self._run_path, "notes"), exist_ok=True)

        # Fresh run state for this entry point
        if cfg.run_id:
            self._run_state = RunState(entry_point="run_validation")
            self._save_state()

        try:
            # ── Pre-flight ────────────────────────────────────────────
            available_tables, table_schemas = self._discover_catalog_tables()
            self._display.step(f"Found {len(available_tables)} tables in catalog")

            if not available_tables:
                self._display.error("No tables found. Run the pipeline first.")
                return self._build_summary()

            self._system_prompt = build_scientist_system_prompt(
                cfg,
                table_schemas=table_schemas,
            )

            # ── Scan existing outputs ─────────────────────────────────
            existing_outputs = self._scan_existing_outputs()
            all_charts = existing_outputs.get("charts", [])
            all_tables = existing_outputs.get("tables", [])

            self._display.step(f"Existing charts: {len(all_charts)}, tables: {len(all_tables)}")

            # ── Load findings.json ────────────────────────────────────
            findings_path = os.path.join(self._run_path, "findings.json")
            all_findings: list[dict] = []
            if os.path.isfile(findings_path):
                try:
                    with open(findings_path) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_findings = data
                        # Load into the finding tool for continuity
                        self._load_existing_findings(all_findings)
                except (json.JSONDecodeError, OSError):
                    pass
            self._display.step(f"Loaded {len(all_findings)} existing findings")

            # ── Per-theme validation ──────────────────────────────────
            all_themes = sorted(cfg.analysis_themes, key=lambda t: t.sequence)
            if themes is not None:
                theme_set = set(themes)
                themes_to_validate = [t for t in all_themes if t.sequence in theme_set]
            else:
                themes_to_validate = all_themes

            if self._run_state:
                self._run_state.mark_phase_start("validation")
                self._save_state()
            self._display.phase(f"Validating {len(themes_to_validate)} of {len(all_themes)} themes")

            for theme in themes_to_validate:
                self._display.phase(f"Validate Theme {theme.sequence}: {theme.title}")
                self._memory.reset_for_new_source()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                # Gather theme-specific outputs
                theme_prefix = f"theme{theme.sequence}_"
                alt_prefix = f"t{theme.sequence}_"
                theme_id_prefix = f"{theme.id}_"
                theme_charts = [
                    c
                    for c in all_charts
                    if c.lower().startswith(theme_prefix)
                    or c.lower().startswith(alt_prefix)
                    or c.lower().startswith(theme_id_prefix)
                ]
                theme_tables = [
                    t
                    for t in all_tables
                    if t.lower().startswith(theme_prefix)
                    or t.lower().startswith(alt_prefix)
                    or t.lower().startswith(theme_id_prefix)
                ]
                theme_findings = [
                    f for f in all_findings if f.get("research_question_id", "") == theme.id
                ]
                theme_notes = self._note_tool.get_theme_notes_from_disk(theme.id)

                self._display.step(
                    f"  Findings: {len(theme_findings)}, "
                    f"Charts: {len(theme_charts)}, "
                    f"Tables: {len(theme_tables)}"
                )

                prompt = build_validation_prompt(
                    cfg=cfg,
                    theme=theme,
                    findings_for_theme=theme_findings,
                    chart_files=theme_charts,
                    table_files=theme_tables,
                    theme_notes=theme_notes,
                    instructions=instructions,
                )

                self._run_phase(
                    prompt=prompt,
                    max_turns=120,
                )
                if self._run_state:
                    self._run_state.mark_item_complete("validation", theme.id)
                    self._save_state()
                self._display.success(f"Validated Theme {theme.sequence}: {theme.title}")

            # ── Re-export updated findings ────────────────────────────
            updated_path = self._finding_tool.export_findings_json()
            self._display.success(f"Updated findings exported to {updated_path}")

            if self._run_state:
                self._run_state.mark_phase_complete("validation")
                self._run_state.mark_completed()
                self._save_state()

        except KeyboardInterrupt:
            self._display.warning("Interrupted.")
            if self._run_state:
                self._run_state.mark_interrupted()
                self._save_state()
            self._dump_progress_on_crash()
        except Exception as e:
            self._display.error(f"Validation failed: {e}")
            logger.exception("run_validation failure")
            if self._run_state:
                self._run_state.mark_failed(str(e))
                self._save_state()
            self._dump_progress_on_crash()

        # Always export findings (even after crash/interrupt) so nothing is lost
        if self._finding_tool.findings:
            try:
                findings_path = self._finding_tool.export_findings_json()
                self._display.success(f"Findings exported to {findings_path}")
            except Exception as export_err:
                self._display.warning(f"Could not export findings: {export_err}")

        summary = self._build_summary()
        self._display.phase("VALIDATION COMPLETE")
        self._display.step(f"Total findings: {len(self._finding_tool.findings)}")
        self._display.step(f"Charts created: {len(self._viz_tool.charts_created)}")
        self._display.step(f"LLM usage: {self._llm.usage_summary}")
        return summary

    # ------------------------------------------------------------------
    # Phase runner — thin override to inject progress_path
    # ------------------------------------------------------------------

    def _run_phase(self, prompt: str, max_turns: int) -> str:  # type: ignore[override]
        """Run a phase, automatically providing the scientist progress path."""
        progress_path = os.path.join(self._run_path, "notes", "progress.txt")
        return super()._run_phase(prompt, max_turns, progress_path=progress_path)

    # ------------------------------------------------------------------
    # Finding enforcement
    # ------------------------------------------------------------------

    def _nudge_save_finding(self, theme: AnalysisTheme) -> None:
        """Run a short follow-up phase to get the agent to call save_finding."""
        findings_before = len(self._finding_tool.findings)
        nudge = (
            f"You just completed analysis for Theme {theme.sequence}: "
            f"{theme.title}, but you did NOT call `save_finding`. "
            f"Every theme MUST produce at least one structured finding. "
            f"Review the analysis you just performed and call `save_finding` "
            f"now with the theme ID '{theme.id}', a title, the key "
            f"quantitative result, the supporting evidence (SQL or stats), "
            f"and a significance rating. Do this for each major insight "
            f"from the theme."
        )
        self._run_phase(
            prompt=self._inject_instructions(nudge),
            max_turns=10,
        )
        nudge_findings = len(self._finding_tool.findings) - findings_before
        if nudge_findings > 0:
            self._display.success(f"Recorded {nudge_findings} finding(s) after nudge")
        else:
            self._display.warning(f"Theme {theme.sequence} still has 0 findings")

    def _retrospective_findings(
        self,
        themes: list[AnalysisTheme],
        available_tables: set[str],
        table_schemas: dict[str, list[tuple[str, str]]],
    ) -> None:
        """Review existing output and create findings for themes that have none.

        After theme analysis completes, some themes may have no findings —
        either because the agent forgot to call save_finding, the context
        window overflowed and the phase was cut short, or the nudge failed.

        This method scans the results volume for existing charts, notes, and
        tables, identifies themes with zero findings, and runs a focused
        recovery phase where the agent reviews existing artifacts and creates
        structured findings retroactively.
        """
        cfg = self._cfg

        # Find themes with zero findings
        themes_with_findings: set[str] = set()
        for f in self._finding_tool.findings:
            rq_id = f.get("research_question_id", "")
            if rq_id:
                themes_with_findings.add(rq_id)

        sparse_themes = [t for t in themes if t.id not in themes_with_findings]

        if not sparse_themes:
            self._display.step(
                f"All {len(themes)} themes have findings — retrospective review not needed."
            )
            return

        self._display.phase(
            f"Phase 3b: Retrospective Findings ({len(sparse_themes)} themes need findings)"
        )

        # Gather available artifacts
        existing_outputs = self._scan_existing_outputs()
        all_notes = self._note_tool.load_notes_from_disk()
        chart_files = existing_outputs.get("charts", [])
        table_files = existing_outputs.get("tables", [])

        for theme in sparse_themes:
            # Collect artifacts for this theme
            theme_notes = all_notes.get(theme.id, "")
            theme_charts = [c for c in chart_files if theme.id in c.lower()]
            theme_tables = [t for t in table_files if theme.id in t.lower()]

            if not theme_notes and not theme_charts and not theme_tables:
                self._display.warning(
                    f"Theme {theme.sequence} ({theme.title}): "
                    f"no artifacts found — cannot create retrospective findings"
                )
                continue

            self._display.step(
                f"Theme {theme.sequence} ({theme.title}): "
                f"reviewing {len(theme_notes.splitlines())} note lines, "
                f"{len(theme_charts)} charts, {len(theme_tables)} tables"
            )

            # Build the retrospective prompt
            artifact_sections = []
            if theme_notes:
                # Cap notes to keep prompt size reasonable
                notes_text = theme_notes[:5000]
                if len(theme_notes) > 5000:
                    notes_text += "\n[... truncated ...]"
                artifact_sections.append(f"## Theme Notes\n```\n{notes_text}\n```")
            if theme_charts:
                artifact_sections.append(
                    "## Available Charts\n" + "\n".join(f"- `{c}`" for c in theme_charts)
                )
            if theme_tables:
                artifact_sections.append(
                    "## Available Tables\n" + "\n".join(f"- `{t}`" for t in theme_tables)
                )

            # Provide data access for verification
            proj = cfg.project
            tables_list = ", ".join(f"`{proj.full_schema}.{t}`" for t in sorted(available_tables))

            prompt = (
                f"# Retrospective Finding Recovery — Theme {theme.sequence}: "
                f"{theme.title}\n\n"
                f"## Context\n"
                f"This theme was analyzed previously but NO structured findings "
                f"were recorded via `save_finding`. Your job is to review the "
                f"existing artifacts below and create findings retroactively.\n\n"
                f"**Theme question:** {theme.question}\n"
                f"**Expected punchline:** {theme.punchline}\n\n"
                f"{''.join(artifact_sections)}\n\n"
                f"## Available Tables\n{tables_list}\n\n"
                f"## Instructions\n"
                f"1. Read the notes and chart metadata above\n"
                f"2. If needed, run quick SQL queries via `execute_sql` to "
                f"verify key numbers (p-values, effect sizes, etc.)\n"
                f"3. Call `save_finding` for EACH significant insight. Include:\n"
                f"   - `research_question_id`: '{theme.id}'\n"
                f"   - Specific quantitative evidence\n"
                f"   - Appropriate significance rating\n"
                f"4. You MUST call `save_finding` at least once.\n"
            )

            # Reset memory and run the recovery phase
            findings_before = len(self._finding_tool.findings)
            self._memory.reset_for_new_source()
            self._consecutive_errors = 0
            self._missing_param_tracker.clear()

            self._run_phase(
                prompt=self._inject_instructions(prompt),
                max_turns=15,
            )

            recovered = len(self._finding_tool.findings) - findings_before
            if recovered > 0:
                self._display.success(
                    f"Recovered {recovered} finding(s) for Theme {theme.sequence}: {theme.title}"
                )
            else:
                self._display.warning(
                    f"Theme {theme.sequence} ({theme.title}): "
                    f"retrospective recovery produced 0 findings"
                )

        # Export after retrospective recovery so nothing is lost
        if self._finding_tool.findings:
            try:
                findings_path = self._finding_tool.export_findings_json()
                self._display.step(f"Findings checkpoint after retrospective: {findings_path}")
            except Exception as e:
                self._display.warning(f"Could not checkpoint findings: {e}")

    # ------------------------------------------------------------------
    # State persistence helpers
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist ``self._run_state`` to ``run_metadata.json`` if active."""
        if self._run_state and self._cfg.run_id:
            save_run_state(self._run_path, self._run_state)

    def _dump_progress_on_crash(self) -> None:
        """Flush the display log to progress.txt on interrupt/failure."""
        try:
            progress_path = os.path.join(self._run_path, "notes", "progress.txt")
            self._display.dump_progress(progress_path)
        except Exception:
            pass  # best-effort

    # ------------------------------------------------------------------
    # Pipeline state detection (for smart resume)
    # ------------------------------------------------------------------

    def _scan_pipeline_state(self, available_tables: set[str]) -> dict:
        """Scan for existing pipeline state to determine what can be skipped.

        Checks Unity Catalog for silver tables and the results volume for
        previously exported findings. Returns a dict describing what's done.
        """
        cfg = self._cfg

        # Which silver datasets already exist as catalog tables?
        completed_silver = set()
        for ds in cfg.silver_datasets:
            if ds.name.lower() in available_tables:
                completed_silver.add(ds.name)

        # Load existing findings.json (if any)
        findings_path = os.path.join(self._run_path, "findings.json")
        completed_themes: set[str] = set()
        existing_findings: list[dict] = []

        if os.path.isfile(findings_path):
            try:
                with open(findings_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    existing_findings = data
                    for finding in existing_findings:
                        rq_id = finding.get("research_question_id", "")
                        if rq_id and rq_id != "synthesis" and rq_id != "general":
                            completed_themes.add(rq_id)
            except (json.JSONDecodeError, OSError):
                pass

        # Merge formal run state (union — if either check says done, treat as done)
        if self._run_state:
            for name in self._run_state.completed_items.get("silver", []):
                completed_silver.add(name)
            for tid in self._run_state.completed_items.get("themes", []):
                completed_themes.add(tid)

        # Scan existing output files
        existing_outputs = self._scan_existing_outputs()

        # Load existing notes from disk
        existing_notes = self._note_tool.load_notes_from_disk()

        return {
            "completed_silver": completed_silver,
            "completed_themes": completed_themes,
            "existing_findings": existing_findings,
            "findings_count": len(existing_findings),
            "has_findings": len(existing_findings) > 0,
            "existing_outputs": existing_outputs,
            "existing_notes": existing_notes,
        }

    def _load_existing_findings(self, findings: list[dict]) -> None:
        """Load previously exported findings into the finding tool for continuity."""
        self._finding_tool._findings = list(findings)
        self._display.step(f"  Loaded {len(findings)} existing findings into memory.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict:
        existing_notes = self._note_tool.load_notes_from_disk()
        return {
            "findings": self._finding_tool.findings,
            "high_significance": self._finding_tool.get_high_significance_findings(),
            "charts_created": self._viz_tool.charts_created,
            "total_findings": len(self._finding_tool.findings),
            "total_charts": len(self._viz_tool.charts_created),
            "total_note_files": len(existing_notes),
            "llm_usage": self._llm.usage_summary,
            "results_path": self._run_path,
            "run_id": self._cfg.run_id or "",
        }
