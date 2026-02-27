"""
DataEngineerAgent — the main ReAct agent loop.

Orchestrates the full pipeline:
  1. Discovery — crawl the volume, identify all data sources
  2. Per-source processing — process each source independently
  3. Acceptance loop — analyst agent validates, engineer fixes, repeat until accepted

Each data source gets its own fresh conversation with Claude, with carryover
context from previously processed sources.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from versifai.core.config import (
    MAX_ACCEPTANCE_ITERATIONS,
    MAX_TURNS_PER_SOURCE,
)
from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.tools.catalog_writer import (
    CatalogWriterTool,
    ExecuteSQLTool,
    ListCatalogTablesTool,
)
from versifai.core.tools.column_renamer import RenameColumnsTool
from versifai.core.tools.dynamic_tool_builder import DynamicToolBuilderTool
from versifai.core.tools.registry import ToolRegistry
from versifai.core.tools.web_search import WebSearchTool
from versifai.data_agents.analyst.agent import DataAnalystAgent
from versifai.data_agents.analyst.prompts import ANALYST_FEEDBACK_PROMPT_TEMPLATE
from versifai.data_agents.engineer.prompts import (
    build_catalog_prompt,
    build_discovery_prompt,
    build_incremental_source_prompt,
    build_rename_prompt,
    build_source_processing_prompt,
    build_system_prompt,
)
from versifai.data_agents.engineer.tools.data_profiler import DataProfilerTool
from versifai.data_agents.engineer.tools.data_transformer import DataTransformerTool
from versifai.data_agents.engineer.tools.doc_reader import (
    DocumentationReaderTool,
    ScanForDocumentationTool,
)
from versifai.data_agents.engineer.tools.file_extractor import FileExtractorTool
from versifai.data_agents.engineer.tools.file_reader import FileReaderTool
from versifai.data_agents.engineer.tools.schema_designer import SchemaDesignerTool
from versifai.data_agents.engineer.tools.volume_explorer import VolumeExplorerTool
from versifai.data_agents.models.state import AgentState

if TYPE_CHECKING:
    from versifai.data_agents.engineer.config import ProjectConfig

logger = logging.getLogger("agent.orchestrator")


class DataEngineerAgent:
    """
    Autonomous data engineering agent powered by Claude.

    The agent processes data sources one at a time, resetting its
    conversation memory between sources to stay within context limits.
    Cross-source knowledge (schemas, decisions) is carried forward
    via the memory's carryover context.

    Usage in a Databricks notebook::

        from versifai.data_agents.engineer.agent import DataEngineerAgent
        from versifai.data_agents.engineer.config import ProjectConfig

        cfg = ProjectConfig()  # or customize for your project
        agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)
        agent.run()
    """

    def __init__(
        self,
        cfg: ProjectConfig | None = None,
        dbutils=None,
        volume_path: str | None = None,
    ) -> None:
        if cfg is None:
            raise ValueError(
                "cfg is required. Pass a ProjectConfig instance. "
                "See examples/ for sample configurations."
            )
        self._cfg = cfg

        self._volume_path = volume_path or cfg.volume_path
        self._display = AgentDisplay(dbutils=dbutils)
        self._state = AgentState(started_at=datetime.now())
        self._memory = AgentMemory()
        self._llm = LLMClient(
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            api_key=cfg.llm.api_key or None,
            api_base=cfg.llm.api_base or None,
            extended_context=cfg.llm.extended_context,
        )

        # Build the tool registry with inter-tool dependencies
        self._schema_tool = SchemaDesignerTool(cfg=cfg)
        self._transformer_tool = DataTransformerTool(
            schema_designer_tool=self._schema_tool,
            cfg=cfg,
        )
        self._catalog_writer = CatalogWriterTool(transformer_tool=self._transformer_tool)
        # Back-reference so transformer can auto-flush to catalog
        self._transformer_tool.set_catalog_writer(self._catalog_writer)

        self._registry = ToolRegistry()
        self._register_tools()

        # Human-in-the-loop state
        self._dbutils = dbutils

        # Error tracking for self-healing
        self._error_tracker: dict[str, list[dict]] = {}
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # Track repeated missing-parameter patterns (LLM output size limits)
        self._missing_param_tracker: dict[str, dict[str, int]] = {}

        # Discovery results — populated in Phase 1
        self._discovered_sources: list[dict] = []

        # Pre-build prompts from config
        self._system_prompt = build_system_prompt(cfg)

    def _register_tools(self) -> None:
        """Register all tools the agent can use."""
        cfg = self._cfg
        self._dynamic_tool_builder = DynamicToolBuilderTool(
            registry=self._registry,
            transformer_tool=self._transformer_tool,
        )

        tools = [
            VolumeExplorerTool(),
            FileExtractorTool(),
            FileReaderTool(),
            DocumentationReaderTool(),
            ScanForDocumentationTool(),
            DataProfilerTool(),
            self._schema_tool,
            self._transformer_tool,
            self._catalog_writer,
            ExecuteSQLTool(),
            ListCatalogTablesTool(cfg=cfg),
            RenameColumnsTool(cfg=cfg),
            WebSearchTool(cfg=cfg),
            self._dynamic_tool_builder,
        ]
        for tool in tools:
            self._registry.register(tool)

        logger.debug(f"Registered {len(self._registry)} tools: {self._registry.tool_names}")

    def _get_tool_definitions(self) -> list[dict]:
        """Get all tool definitions for Claude, including ask_human."""
        tools = self._registry.to_claude_tools()

        # Add the ask_human pseudo-tool
        tools.append(
            {
                "name": "ask_human",
                "description": (
                    "Pause execution and ask the human operator a question. "
                    "Use this when you are uncertain about a data interpretation, "
                    "column mapping, file selection, or any decision that requires "
                    "human judgment. The operator is watching the notebook output "
                    "and will provide an answer."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the operator. Be specific and provide context.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context to help the operator answer.",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of suggested answers.",
                        },
                    },
                    "required": ["question"],
                },
            }
        )

        return tools

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        force_full: bool = False,
        source_path: str | None = None,
    ) -> dict:
        """
        Run the full agent pipeline:
          1. Discovery — crawl the volume, identify all data sources
          1.5. Incremental detection — check existing tables for loaded files
          2. Per-source processing — process each source independently
          3. Acceptance loop — analyst validates, engineer fixes, repeat

        Args:
            force_full: If True, skip incremental detection and reprocess
                all sources from scratch (overwrite existing tables).
            source_path: If provided, skip discovery and process only this
                directory path directly. Useful for re-processing a single
                source or adding a new source without re-crawling everything.

        Returns a summary dict with results and statistics.
        """
        cfg = self._cfg
        self._display.phase("AGENTIC DATA PROFILER STARTING")
        self._display.step(f"Project: {cfg.name}")
        self._display.step(f"Volume: {self._volume_path}")
        self._display.step(f"Target: {cfg.full_schema}")
        self._display.step(f"Tools: {self._registry.tool_names + ['ask_human']}")

        try:
            if source_path:
                # ── Shortcut: Skip discovery, process a single directory ──
                import os

                source_name = os.path.basename(source_path.rstrip("/"))
                file_count = 0
                if os.path.isdir(source_path):
                    file_count = len(
                        [
                            f
                            for f in os.listdir(source_path)
                            if os.path.isfile(os.path.join(source_path, f))
                        ]
                    )

                self._display.phase(f"Direct mode: {source_name} ({file_count} files)")
                self._display.step("Skipping discovery — processing target directory only.")

                self._discovered_sources = [
                    {
                        "name": source_name,
                        "path": source_path,
                        "file_count": file_count,
                        "files": "",
                        "mode": "full" if force_full else "check_incremental",
                    }
                ]

                if not force_full:
                    self._display.phase("Incremental Status Check")
                    self._detect_incremental_status()

            else:
                # ── Phase 1: Discovery ──────────────────────────────────
                self._display.phase("Phase 1: Discovery")
                self._run_phase(
                    prompt=build_discovery_prompt(cfg),
                    max_turns=MAX_TURNS_PER_SOURCE,
                )

                # Parse the discovery results to find sources
                self._discovered_sources = self._parse_discovery_results()

                # Always merge with filesystem fallback to catch missed directories
                fallback_sources = self._fallback_discover_sources()
                existing_paths = {s["path"] for s in self._discovered_sources}
                for fb_source in fallback_sources:
                    if fb_source["path"] not in existing_paths:
                        self._discovered_sources.append(fb_source)
                        self._display.step(f"  Added from fallback: {fb_source['name']}")

                self._display.step(
                    f"Discovered {len(self._discovered_sources)} data sources to process"
                )

                if not self._discovered_sources:
                    self._display.warning(
                        "No data sources discovered. Check the volume path and directory structure."
                    )

                # ── Phase 1.5: Incremental Detection ────────────────────
                if not force_full:
                    self._display.phase("Phase 1.5: Incremental Status Check")
                    self._detect_incremental_status()
                else:
                    self._display.step("Force full reload — skipping incremental detection.")
                    for source in self._discovered_sources:
                        source["mode"] = "full"

            # ── Phase 2: Process each source ────────────────────────
            self._display.phase("Phase 2: Source-by-Source Processing")

            for i, source in enumerate(self._discovered_sources, 1):
                source_name = source["name"]
                source_path = source["path"]
                file_count = source.get("file_count", "?")
                mode = source.get("mode", "full")

                # Calculate turn budget — multi-table sources get more
                hint = cfg.get_source_hint(source_name)
                if hint and hint.multi_table:
                    source_max_turns = MAX_TURNS_PER_SOURCE + (len(hint.files) * 15)
                else:
                    source_max_turns = MAX_TURNS_PER_SOURCE

                self._display.phase(
                    f"Source {i}/{len(self._discovered_sources)}: {source_name} "
                    f"({file_count} files) [{mode.upper()}] "
                    f"[budget: {source_max_turns} turns]"
                )

                # Reset memory for this source, keeping cross-source context
                carryover = self._memory.get_carryover_context()
                self._memory.reset_for_new_source()
                self._error_tracker.clear()
                self._consecutive_errors = 0
                self._missing_param_tracker.clear()

                # Register auto-flush target so large staging buffers
                # get written to Unity Catalog before running out of memory
                table_name = f"{cfg.full_schema}.{source_name}"
                self._transformer_tool.register_flush_target(source_name, table_name)

                # Route to the right prompt based on mode
                if mode == "check_incremental":
                    source_prompt = build_incremental_source_prompt(
                        cfg=cfg,
                        source_name=source_name,
                        source_path=source_path,
                        existing_table=source["existing_table"],
                        loaded_files=source["loaded_files"],
                    )
                else:
                    # mode == "full" or unknown — full processing
                    source_prompt = build_source_processing_prompt(
                        cfg=cfg,
                        source_name=source_name,
                        source_path=source_path,
                        file_list=source.get("files", "See directory listing"),
                        suggested_table=source_name,
                    )

                # Prepend carryover context if we have it
                if carryover:
                    source_prompt = (
                        f"## Context From Previous Sources\n{carryover}\n\n---\n\n{source_prompt}"
                    )

                # Process this source
                self._run_phase(
                    prompt=source_prompt,
                    max_turns=source_max_turns,
                )

                # Log source completion
                self._memory.log_source_summary(
                    source_name, f"Processed from {source_path} ({file_count} files) [mode={mode}]"
                )
                self._display.success(f"Completed source: {source_name}")

                # ── Per-source completeness check (multi-table retry) ──
                # For multi-table sources, verify all expected tables were
                # created. If any are missing, force the agent back in with
                # a focused prompt to finish the job.
                if hint and hint.multi_table:
                    missing = self._check_source_tables(source_name)
                    retry_attempts = 0
                    max_retries = 2

                    while missing and retry_attempts < max_retries:
                        retry_attempts += 1
                        self._display.warning(
                            f"Multi-table source {source_name} INCOMPLETE: "
                            f"missing {len(missing)} table(s): "
                            f"{', '.join(missing)}"
                        )

                        # Reset error state for the retry
                        self._error_tracker.clear()
                        self._consecutive_errors = 0
                        self._missing_param_tracker.clear()

                        retry_prompt = self._build_retry_prompt(
                            source_name=source_name,
                            source_path=source_path,
                            missing_tables=missing,
                            hint=hint,
                        )
                        retry_turns = max(30, len(missing) * 20)
                        self._display.phase(
                            f"RETRY {retry_attempts}/{max_retries}: "
                            f"{source_name} — {len(missing)} missing table(s) "
                            f"[budget: {retry_turns} turns]"
                        )
                        self._run_phase(
                            prompt=retry_prompt,
                            max_turns=retry_turns,
                        )

                        # Re-check
                        missing = self._check_source_tables(source_name)

                    if missing:
                        self._display.warning(
                            f"After {retry_attempts} retries, still missing: {', '.join(missing)}"
                        )
                    else:
                        self._display.success(
                            f"All {len(hint.files)} tables created for {source_name}."
                        )

            # ── Phase 3: Table Completeness Check ─────────────────
            self._display.phase("Phase 3: Table Completeness Check")
            processed_names = [s["name"] for s in self._discovered_sources]
            self._check_table_completeness(source_names=processed_names)

        except KeyboardInterrupt:
            self._display.warning("Agent interrupted by user.")
        except Exception as e:
            self._display.error(f"Agent failed: {e}")
            logger.exception("Agent top-level failure")

        # Final summary
        summary = self._build_summary()
        self._display.phase("PIPELINE COMPLETE")
        self._display.summary_table("Final Results", summary["tables"])
        self._display.step(f"Engineer LLM Usage: {self._llm.usage_summary}")

        return summary

    # ------------------------------------------------------------------
    # Retroactive column rename — fix cryptic names on existing tables
    # ------------------------------------------------------------------

    def run_rename(self) -> dict:
        """
        Run the retroactive column rename phase.

        Reads documentation for each existing table in Unity Catalog,
        builds a mapping from cryptic column names to descriptive names,
        and applies ALTER TABLE RENAME COLUMN statements.

        This is safe to run at any time — it's metadata-only on Delta
        tables and doesn't rewrite data.

        Usage in a Databricks notebook::

            from versifai.data_agents.engineer.agent import DataEngineerAgent
            from versifai.data_agents.engineer.config import ProjectConfig

            cfg = ProjectConfig()
            agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)
            agent.run_rename()
        """
        cfg = self._cfg
        self._display.phase("RETROACTIVE COLUMN RENAME")
        self._display.step(f"Target schema: {cfg.full_schema}")
        self._display.step(f"Documentation source: {cfg.volume_path}")
        self._display.step(f"Tools: {self._registry.tool_names + ['ask_human']}")

        try:
            self._memory.reset_for_new_source()
            self._error_tracker.clear()
            self._consecutive_errors = 0
            self._missing_param_tracker.clear()

            prompt = build_rename_prompt(cfg)
            self._run_phase(prompt=prompt, max_turns=MAX_TURNS_PER_SOURCE)

            self._display.success("Column rename phase complete.")

        except KeyboardInterrupt:
            self._display.warning("Rename interrupted by user.")
        except Exception as e:
            self._display.error(f"Rename failed: {e}")
            logger.exception("Rename phase failure")

        summary = {
            "phase": "rename",
            "total_turns": self._state.total_turns,
            "llm_usage": self._llm.usage_summary,
        }
        self._display.phase("RENAME COMPLETE")
        self._display.step(f"LLM Usage: {self._llm.usage_summary}")
        return summary

    # ------------------------------------------------------------------
    # Data catalog — document every column in every table
    # ------------------------------------------------------------------

    def run_catalog(self) -> dict:
        """
        Build or update the data_catalog table documenting every column in every table.

        Uses delta-fill: detects which tables are new or changed since the last
        run and only documents those. Existing, unchanged entries are preserved.

        Hard constraint: every table in Unity Catalog must be documented.
        The agent will verify 100% coverage before finishing.

        Safe to run at any time — incrementally updates the data_catalog table.

        Usage in a Databricks notebook::

            from versifai.data_agents.engineer.agent import DataEngineerAgent
            from versifai.data_agents.engineer.config import ProjectConfig

            cfg = ProjectConfig()
            agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)
            agent.run_catalog()
        """
        cfg = self._cfg
        self._display.phase("DATA CATALOG BUILDER")
        self._display.step(f"Target schema: {cfg.full_schema}")
        self._display.step(f"Catalog table: {cfg.full_schema}.data_catalog")
        self._display.step(f"Documentation source: {cfg.volume_path}")
        self._display.step(f"Tools: {self._registry.tool_names + ['ask_human']}")

        try:
            self._memory.reset_for_new_source()
            self._error_tracker.clear()
            self._consecutive_errors = 0
            self._missing_param_tracker.clear()

            prompt = build_catalog_prompt(cfg)
            self._run_phase(prompt=prompt, max_turns=MAX_TURNS_PER_SOURCE)

            self._display.success("Data catalog built.")

        except KeyboardInterrupt:
            self._display.warning("Catalog build interrupted by user.")
        except Exception as e:
            self._display.error(f"Catalog build failed: {e}")
            logger.exception("Catalog build failure")

        summary = {
            "phase": "catalog",
            "total_turns": self._state.total_turns,
            "llm_usage": self._llm.usage_summary,
        }
        self._display.phase("CATALOG BUILD COMPLETE")
        self._display.step(f"LLM Usage: {self._llm.usage_summary}")
        return summary

    # ------------------------------------------------------------------
    # Quality check — analyst validates using real table inventory
    # ------------------------------------------------------------------

    def run_quality_check(
        self,
        tables: list[str] | None = None,
        hints: str = "",
    ) -> dict:
        """
        Run acceptance testing on tables in Unity Catalog.

        Uses the data_catalog table (if it exists) to ground the analyst in
        real table names, real column names, and real descriptions. This
        prevents hallucination and ensures all validation queries reference
        actual schema objects.

        Should be called AFTER run_catalog() so the data_catalog is available.

        Args:
            tables: Optional list of specific table names to validate.
                If None, validates ALL tables. Use this to focus on a
                specific dataset at a time (e.g., ["scc_enrollment"]).
            hints: Optional user instructions injected into the analyst
                prompt. Use this to tell the analyst what to focus on
                or what specific issues you want checked.
                Example: "Check that enrollment has no suppressed values.
                Star ratings should have 2022-2026 coverage."

        Usage in a Databricks notebook::

            from versifai.data_agents.engineer.agent import DataEngineerAgent
            from versifai.data_agents.engineer.config import ProjectConfig

            cfg = ProjectConfig()
            agent = DataEngineerAgent(cfg=cfg, dbutils=dbutils)

            # Check everything:
            agent.run_quality_check()

            # Check specific tables with guidance:
            agent.run_quality_check(
                tables=["scc_enrollment", "star_ratings_summary"],
                hints="Verify enrollment suppression handling. "
                      "Star ratings should cover 2022-2026."
            )
        """
        cfg = self._cfg
        self._display.phase("DATA QUALITY CHECK")
        self._display.step(f"Target schema: {cfg.full_schema}")
        if tables:
            self._display.step(f"Focused on: {', '.join(tables)}")

        try:
            # Pre-build the table inventory from Unity Catalog
            table_inventory = self._build_table_inventory()

            if not table_inventory:
                self._display.error("No tables found in Unity Catalog. Nothing to validate.")
                return {"overall_status": "ERROR", "summary": "No tables found."}

            # Filter to requested tables if specified
            if tables:
                tables_lower = {t.lower() for t in tables}
                table_inventory = [
                    t for t in table_inventory if t["table_name"].lower() in tables_lower
                ]
                if not table_inventory:
                    self._display.error(f"None of the requested tables found: {tables}")
                    return {
                        "overall_status": "ERROR",
                        "summary": f"Requested tables not found: {tables}",
                    }

            self._display.step(f"Found {len(table_inventory)} tables to validate.")

            # Run the analyst with the pre-built inventory
            analyst = DataAnalystAgent(
                cfg=cfg,
                dbutils=self._dbutils,
                table_inventory=table_inventory,
                user_hints=hints,
            )
            feedback = analyst.run()

            # Display results
            verdicts = feedback.get("verdicts", {})
            if verdicts:
                verdict_rows = []
                for table_name, verdict in verdicts.items():
                    verdict_rows.append(
                        {
                            "table": table_name,
                            "status": verdict.get("status", "?"),
                            "issues": str(len(verdict.get("issues", []))),
                        }
                    )
                self._display.summary_table("Quality Check Verdicts", verdict_rows)

            overall = feedback.get("overall_status", "NEEDS_REVIEW")
            if overall == "PASS":
                self._display.success("All tables ACCEPTED.")
            else:
                self._display.warning(f"Quality check: {overall}. Review verdicts above.")

        except KeyboardInterrupt:
            self._display.warning("Quality check interrupted by user.")
            feedback = {"overall_status": "INTERRUPTED"}
        except Exception as e:
            self._display.error(f"Quality check failed: {e}")
            logger.exception("Quality check failure")
            feedback = {"overall_status": "ERROR", "summary": str(e)}

        summary = {
            "phase": "quality_check",
            "overall_status": feedback.get("overall_status", "UNKNOWN"),
            "verdicts": feedback.get("verdicts", {}),
            "total_turns": self._state.total_turns,
            "llm_usage": self._llm.usage_summary,
        }
        self._display.phase("QUALITY CHECK COMPLETE")
        self._display.step(f"LLM Usage: {self._llm.usage_summary}")
        return summary

    def _build_table_inventory(self) -> list[dict]:
        """
        Query Unity Catalog for all tables, DESCRIBE each one, and optionally
        pull descriptions from data_catalog. Returns a list of table dicts
        with real schema information the analyst can use.
        """
        cfg = self._cfg

        # Get table list
        tables_result = self._registry.execute("list_catalog_tables")
        if not tables_result.success:
            return []

        inventory = []
        for t in tables_result.data.get("tables", []):
            name = t.get("tableName") or t.get("table_name") or t.get("name", "")
            if not name or name == "data_catalog":
                continue

            full_name = f"{cfg.full_schema}.{name}"

            # DESCRIBE TABLE to get columns
            desc_result = self._registry.execute("execute_sql", sql=f"DESCRIBE TABLE {full_name}")
            columns = []
            if desc_result.success:
                for row in desc_result.data.get("rows", []):
                    col_name = row.get("col_name") or row.get("column_name", "")
                    col_type = row.get("data_type") or row.get("type", "")
                    if col_name and not col_name.startswith("#"):
                        columns.append(
                            {
                                "name": col_name,
                                "type": col_type,
                            }
                        )

            # Row count
            count_result = self._registry.execute(
                "execute_sql",
                sql=f"SELECT COUNT(*) as cnt FROM {full_name}",
            )
            row_count = 0
            if count_result.success:
                rows = count_result.data.get("rows", [])
                if rows:
                    row_count = rows[0].get("cnt", 0)

            # Determine grain from columns and extract key column types
            col_lookup = {c["name"].lower(): c for c in columns}
            has_fips = cfg.join_key.column_name in col_lookup
            has_contract = "contract_id" in col_lookup or "contract_number" in col_lookup
            if has_fips and has_contract:
                grain = "bridge"
            elif has_fips:
                grain = "county"
            elif has_contract:
                grain = "contract"
            else:
                grain = "other"

            # Extract key column types for cross-table consistency checks
            key_types = {}
            if has_fips:
                fips_col = col_lookup[cfg.join_key.column_name]
                key_types[cfg.join_key.column_name] = fips_col["type"]
            contract_col_name = (
                "contract_number"
                if "contract_number" in col_lookup
                else "contract_id"
                if "contract_id" in col_lookup
                else None
            )
            if contract_col_name:
                key_types[contract_col_name] = col_lookup[contract_col_name]["type"]

            inventory.append(
                {
                    "table_name": name,
                    "full_name": full_name,
                    "columns": columns,
                    "row_count": row_count,
                    "grain": grain,
                    "key_types": key_types,
                }
            )

        # Try to pull catalog descriptions to enrich the inventory
        catalog_result = self._registry.execute(
            "execute_sql",
            sql=(
                f"SELECT source_table, column_name, description FROM {cfg.full_schema}.data_catalog"
            ),
        )
        if catalog_result.success:
            # Build lookup: (table, column) -> description
            desc_lookup: dict[tuple[str, str], str] = {}
            for row in catalog_result.data.get("rows", []):
                key = (
                    row.get("source_table", ""),
                    row.get("column_name", ""),
                )
                desc_lookup[key] = row.get("description", "")

            # Enrich inventory with descriptions
            for table_info in inventory:
                for col in table_info["columns"]:
                    desc = desc_lookup.get(
                        (table_info["full_name"], col["name"]),
                        desc_lookup.get((table_info["table_name"], col["name"]), ""),
                    )
                    col["description"] = desc

        return inventory

    # ------------------------------------------------------------------
    # Acceptance loop — analyst reviews, engineer fixes, repeat
    # (kept for backward compatibility but no longer called from run())
    # ------------------------------------------------------------------

    def _run_acceptance_loop(self) -> None:
        """
        Run the analyst->engineer feedback loop.

        1. Analyst agent reviews all tables in Unity Catalog
        2. If all tables are ACCEPTED -> done
        3. If issues found -> engineer gets the feedback and fixes them
        4. Repeat until accepted or max iterations reached
        """
        for iteration in range(1, MAX_ACCEPTANCE_ITERATIONS + 1):
            self._display.phase(
                f"Phase 3: Acceptance Iteration {iteration}/{MAX_ACCEPTANCE_ITERATIONS}"
            )

            # ── Analyst reviews ──────────────────────────────────
            self._display.phase("Data Analyst: Reviewing Tables")
            analyst = DataAnalystAgent(cfg=self._cfg, dbutils=self._dbutils)
            feedback = analyst.run()

            overall = feedback.get("overall_status", "NEEDS_REVIEW")
            verdicts = feedback.get("verdicts", {})

            # Show the verdicts
            if verdicts:
                verdict_rows = []
                for table_name, verdict in verdicts.items():
                    verdict_rows.append(
                        {
                            "table": table_name,
                            "status": verdict.get("status", "?"),
                            "issues": str(len(verdict.get("issues", []))),
                        }
                    )
                self._display.summary_table(
                    f"Analyst Verdicts (Iteration {iteration})", verdict_rows
                )

            # Check if everything passed
            if overall == "PASS":
                self._display.success("All tables ACCEPTED by the Data Analyst. Pipeline complete!")
                return

            # Count what needs fixing
            needs_fix = {
                name: v
                for name, v in verdicts.items()
                if v.get("status") in ("NEEDS_FIX", "REJECTED")
            }

            if not needs_fix:
                # No structured verdicts but status isn't PASS — use raw response
                if feedback.get("raw_response"):
                    self._display.warning(
                        "Analyst did not produce structured verdicts. "
                        "Treating raw response as feedback."
                    )
                    needs_fix = {
                        "_raw": {
                            "status": "NEEDS_REVIEW",
                            "issues": [feedback.get("summary", "See raw response")],
                            "fixes": [],
                        }
                    }
                else:
                    self._display.success("No issues found by analyst. Acceptance complete.")
                    return

            self._display.phase(f"Data Engineer: Fixing {len(needs_fix)} table(s)")

            # ── Engineer fixes ───────────────────────────────────
            # Build a focused feedback prompt for the engineer
            feedback_json = json.dumps(feedback, indent=2, default=str)

            fix_prompt = ANALYST_FEEDBACK_PROMPT_TEMPLATE.format(analyst_feedback=feedback_json)

            # Fresh memory for the fix phase
            self._memory.reset_for_new_source()
            self._error_tracker.clear()
            self._consecutive_errors = 0

            self._run_phase(
                prompt=fix_prompt,
                max_turns=MAX_TURNS_PER_SOURCE,
            )

            self._display.success(
                f"Engineer completed fixes for iteration {iteration}. Analyst will re-review."
            )

        # If we exhausted iterations
        self._display.warning(
            f"Reached max acceptance iterations ({MAX_ACCEPTANCE_ITERATIONS}). "
            f"Some tables may still have issues. Review manually."
        )

    # ------------------------------------------------------------------
    # Discovery result parsing
    # ------------------------------------------------------------------

    def _parse_discovery_results(self) -> list[dict]:
        """
        Extract discovered data sources from the conversation history.

        Looks at tool results from explore_volume calls to identify
        subdirectories that contain data files.
        """
        sources = []
        seen_paths = set()

        for msg in self._memory.messages:
            content = msg.get("content", "")

            # Look through tool results for explore_volume data
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue

                    result_str = block.get("content", "")
                    if not isinstance(result_str, str):
                        continue

                    # Try to parse as JSON (explore_volume returns JSON)
                    try:
                        data = json.loads(result_str)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    # Look for subdirectories with files
                    if isinstance(data, dict) and "entries" in data:
                        for entry in data["entries"]:
                            if entry.get("type") == "directory":
                                path = entry.get("path", "")
                                name = entry.get("name", "")
                                if path and path not in seen_paths:
                                    seen_paths.add(path)
                                    sources.append(
                                        {
                                            "name": name,
                                            "path": path,
                                            "file_count": entry.get("file_count", "?"),
                                            "files": "",
                                        }
                                    )

                    # Also look for direct file listings (recursive explore results)
                    if isinstance(data, dict) and "path" in data and "total_files" in data:
                        path = data["path"]
                        if path not in seen_paths and data.get("total_files", 0) > 0:
                            # This is a folder with files — add it if not root
                            base_path = self._volume_path.rstrip("/")
                            if path != base_path:
                                name = path.rstrip("/").split("/")[-1]
                                file_names = [
                                    e.get("name", "")
                                    for e in data.get("entries", [])
                                    if e.get("type") == "file"
                                ]
                                seen_paths.add(path)
                                sources.append(
                                    {
                                        "name": name,
                                        "path": path,
                                        "file_count": data.get("total_files", len(file_names)),
                                        "files": ", ".join(file_names[:10]),
                                    }
                                )

        # If parsing didn't find sources, fall back to known subdirectories
        if not sources:
            self._display.warning(
                "Could not auto-parse discovery results. Falling back to direct directory listing."
            )
            sources = self._fallback_discover_sources()

        return sources

    def _fallback_discover_sources(self) -> list[dict]:
        """
        Direct filesystem fallback if parsing conversation didn't work.
        Lists subdirectories in the volume path.
        """
        import os

        try:
            entries = os.listdir(self._volume_path)
            sources = []
            for name in sorted(entries):
                full_path = os.path.join(self._volume_path, name)
                if os.path.isdir(full_path):
                    file_count = len(
                        [
                            f
                            for f in os.listdir(full_path)
                            if os.path.isfile(os.path.join(full_path, f))
                        ]
                    )
                    sources.append(
                        {
                            "name": name,
                            "path": full_path,
                            "file_count": file_count,
                            "files": "",
                        }
                    )
            return sources
        except Exception as e:
            self._display.error(f"Fallback discovery failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Table completeness check — verify expected tables exist
    # ------------------------------------------------------------------

    def _check_table_completeness(
        self,
        source_names: list[str] | None = None,
    ) -> list[str]:
        """
        Verify that expected tables exist in Unity Catalog.

        Args:
            source_names: If provided, only check expected tables from
                these sources' processing hints. If None, check ALL
                expected tables across all sources.

        Returns:
            List of missing table names (empty if all present).
        """
        tables_result = self._registry.execute("list_catalog_tables")
        if not tables_result.success:
            self._display.warning("Could not verify table completeness.")
            return []

        existing: set[str] = set()
        for t in tables_result.data.get("tables", []):
            name = t.get("tableName") or t.get("table_name") or t.get("name", "")
            if name:
                existing.add(name.lower())

        # Scope expected tables to processed sources if specified
        if source_names:
            expected: set[str] = set()
            for sn in source_names:
                hint = self._cfg.get_source_hint(sn)
                if hint:
                    for f in hint.files:
                        expected.add(f.target_table.lower())
        else:
            expected = set(t.lower() for t in self._cfg.expected_tables)

        missing = sorted(expected - existing)
        extra = existing - expected - {"data_catalog"}

        if missing:
            self._display.warning(f"MISSING TABLES ({len(missing)}): {', '.join(missing)}")
        if extra:
            self._display.step(f"Extra tables (not in expected list): {', '.join(sorted(extra))}")
        if not missing:
            self._display.success(f"All {len(expected)} expected tables exist in Unity Catalog.")
        return missing

    def _check_source_tables(
        self,
        source_name: str,
    ) -> list[str]:
        """Check which expected tables from a source's hint are still missing.

        Returns a list of missing target_table names, or empty if all exist.
        """
        hint = self._cfg.get_source_hint(source_name)
        if not hint:
            return []

        tables_result = self._registry.execute("list_catalog_tables")
        if not tables_result.success:
            return [f.target_table for f in hint.files]

        existing: set[str] = set()
        for t in tables_result.data.get("tables", []):
            name = t.get("tableName") or t.get("table_name") or t.get("name", "")
            if name:
                existing.add(name.lower())

        return [f.target_table for f in hint.files if f.target_table.lower() not in existing]

    def _build_retry_prompt(
        self,
        source_name: str,
        source_path: str,
        missing_tables: list[str],
        hint,
    ) -> str:
        """Build a focused retry prompt for missing multi-table targets.

        Tells the agent exactly which tables are missing and which file
        patterns to look for, so it can finish the job without re-doing
        work on tables that already exist.
        """
        cfg = self._cfg

        # Build per-table instructions from the hint
        table_instructions = []
        for table_name in missing_tables:
            for f in hint.files:
                if f.target_table == table_name:
                    desc = f.description or "see source documentation"
                    table_instructions.append(
                        f"- **`{table_name}`**: file pattern `{f.file_pattern}` — {desc}"
                    )
                    break

        tables_block = "\n".join(table_instructions)

        return f"""\
## RETRY: Complete remaining tables for {source_name}

You already processed `{source_name}` from `{source_path}` but only created
SOME of the expected tables. The following {len(missing_tables)} table(s)
are MISSING and MUST be created now:

{tables_block}

### Instructions
1. The files are already extracted in `{source_path}` — do NOT re-extract
   archives. Use `explore_volume` to see what's available.
2. For EACH missing table above:
   a. Find the matching data files using the file pattern shown above
   b. Profile ONE representative file (`read_file_header` + `profile_data`)
   c. Design the schema (`design_schema`) — remember to rename cryptic columns
   d. Batch transform ALL matching files (`transform_and_load` with `files=[...]`)
      — the tool handles schema drift automatically
   e. Write to Unity Catalog (`write_to_catalog`)
   f. Verify: `SELECT COUNT(*) FROM {cfg.full_schema}.<table>` — must not be 0
3. Process ALL {len(missing_tables)} missing tables listed above. Do NOT stop
   until every table exists in `{cfg.full_schema}`.
4. Do NOT re-process tables that already exist — only create the missing ones.
5. Do NOT skip tables because of errors — investigate and fix each one.
"""

    # ------------------------------------------------------------------
    # Incremental detection — check existing tables for loaded files
    # ------------------------------------------------------------------

    def _detect_incremental_status(self) -> None:
        """
        Check Unity Catalog for existing tables and annotate each discovered
        source with its processing mode: 'full' or 'check_incremental'.

        For each source directory, determines whether a corresponding table
        already exists. If so, queries `source_file_name` to find which files
        have already been loaded. The agent then compares directory contents
        against the loaded list to decide: skip (up to date) or append (new
        files found).
        """
        # Step 1: Get existing tables
        tables_result = self._registry.execute("list_catalog_tables")
        if not tables_result.success:
            self._display.warning(
                "Could not check existing tables — defaulting to full processing."
            )
            for source in self._discovered_sources:
                source["mode"] = "full"
            return

        existing_table_names: set[str] = set()
        for t in tables_result.data.get("tables", []):
            # SHOW TABLES returns tableName or table_name depending on method
            name = t.get("tableName") or t.get("table_name") or t.get("name", "")
            if name:
                existing_table_names.add(name.lower())

        cfg = self._cfg

        for source in self._discovered_sources:
            source_name = source["name"].lower()

            if source_name not in existing_table_names:
                source["mode"] = "full"
                source["loaded_files"] = []
                self._display.step(f"  {source['name']}: NEW — full processing")
                continue

            # Table exists — query which files are already loaded
            full_table = f"{cfg.full_schema}.{source_name}"
            file_result = self._registry.execute(
                "execute_sql",
                sql=f"SELECT DISTINCT source_file_name FROM {full_table}",
            )

            if not file_result.success:
                source["mode"] = "full"
                source["loaded_files"] = []
                self._display.step(
                    f"  {source['name']}: table exists but couldn't query files — full processing"
                )
                continue

            loaded_files = [
                r.get("source_file_name", "")
                for r in file_result.data.get("rows", [])
                if r.get("source_file_name")
            ]
            source["loaded_files"] = loaded_files
            source["existing_table"] = full_table
            # "check_incremental" = table exists, agent must compare
            # directory files against loaded_files and decide skip vs append
            source["mode"] = "check_incremental"
            self._display.step(
                f"  {source['name']}: EXISTS ({len(loaded_files)} files loaded)"
                f" — checking for new data"
            )

    # ------------------------------------------------------------------
    # Phase runner — the core ReAct loop
    # ------------------------------------------------------------------

    def _run_phase(self, prompt: str, max_turns: int) -> str:
        """
        Run a single agent phase with the ReAct loop.

        Returns the last text response from Claude (useful for discovery).
        """
        self._memory.add_user_message(prompt)

        last_text = ""
        turns = 0

        while turns < max_turns:
            turns += 1
            self._state.total_turns += 1

            self._display.step(f"Turn {turns}/{max_turns}")

            # REASON: Send to Claude
            try:
                response = self._llm.send(
                    messages=self._memory.messages,
                    system=self._system_prompt,
                    tools=self._get_tool_definitions(),
                )
            except Exception as e:
                self._display.error(f"LLM call failed: {e}")
                break

            # Parse actions
            actions = LLMClient.extract_actions(response)

            # Store the raw assistant response
            self._memory.add_assistant_message(response.content)

            # Process each action
            has_tool_use = False
            for action in actions:
                if action["type"] == "text":
                    last_text = action["text"]
                    self._display.thinking(action["text"])

                elif action["type"] == "tool_use":
                    has_tool_use = True
                    tool_name = action["name"]
                    tool_input = action["input"]
                    tool_use_id = action["id"]

                    self._display.tool_call(tool_name, tool_input)

                    # Handle ask_human pseudo-tool
                    if tool_name == "ask_human":
                        result_str = self._handle_ask_human(
                            question=tool_input.get("question", ""),
                            context=tool_input.get("context", ""),
                            options=tool_input.get("options"),
                        )
                        self._memory.add_tool_result(tool_use_id, result_str)
                        self._display.tool_result(tool_name, result_str)
                        continue

                    # PRE-CHECK: Detect repeated missing-parameter failures
                    # before executing the tool. If the agent keeps dropping
                    # the same parameter, it's likely an LLM output limit issue.
                    intercept = self._check_missing_param_pattern(tool_name, tool_input)
                    if intercept:
                        # Don't execute — return guidance immediately
                        result_str = intercept
                        self._consecutive_errors += 1
                        self._display.tool_result(
                            tool_name,
                            intercept[:400],
                            is_error=True,
                        )
                        self._memory.add_tool_result(
                            tool_use_id,
                            result_str,
                            is_error=True,
                        )
                        continue

                    # ACT: Execute the tool
                    result = self._registry.execute(tool_name, **tool_input)

                    # OBSERVE: Get the result
                    result_str = result.to_content_str()

                    # Track errors for self-healing
                    if not result.success:
                        self._consecutive_errors += 1
                        self._track_error(tool_name, tool_input, result.error)

                        error_history = self._get_error_context(tool_name)
                        if error_history:
                            result_str += (
                                f"\n\n[SELF-HEALING CONTEXT] "
                                f"This tool has failed {len(self._error_tracker.get(tool_name, []))} time(s). "
                                f"Previous errors: {error_history}\n"
                                f"Try a different approach, different parameters, "
                                f"or ask the human for help."
                            )

                        if self._consecutive_errors >= self._max_consecutive_errors:
                            result_str += (
                                f"\n\n[CRITICAL — {self._consecutive_errors} CONSECUTIVE ERRORS]\n"
                                f"You MUST change your approach NOW. DO NOT retry the same call.\n\n"
                                f"You are likely hitting LLM output size limits — the parameter "
                                f"you're trying to produce is too large to fit in a single response.\n\n"
                                f"SOLUTION: Use `create_custom_tool` to build a helper function that "
                                f"generates the large parameter programmatically in Python code. "
                                f"For example, create a tool that reads file headers and produces "
                                f"the column list, or that builds the data structure in a loop.\n\n"
                                f"DO NOT retry the same call. Change your approach."
                            )
                            self._display.warning(
                                f"{self._consecutive_errors} consecutive errors — "
                                f"agent should ask for help."
                            )
                    else:
                        self._consecutive_errors = 0

                    # Truncate very large results but preserve the summary
                    # so the LLM still sees compact file lists, row counts, etc.
                    if len(result_str) > 15000:
                        result_str = result_str[:15000] + "\n\n[... TRUNCATED ...]"
                        if result.summary:
                            result_str += f"\n\nSUMMARY (complete):\n{result.summary}"

                    self._display.tool_result(
                        tool_name,
                        result.summary or result_str[:400],
                        is_error=not result.success,
                    )

                    # REFLECT: Feed result back to Claude
                    self._memory.add_tool_result(
                        tool_use_id, result_str, is_error=not result.success
                    )

            # Check stop condition
            if response.stop_reason == "end_turn" and not has_tool_use:
                self._display.step("Phase complete.")
                break

        if turns >= max_turns:
            self._display.warning(f"Reached max turns ({max_turns}). Moving on.")

        return last_text

    # ------------------------------------------------------------------
    # Human-in-the-loop
    # ------------------------------------------------------------------

    def _handle_ask_human(
        self, question: str, context: str = "", options: list[str] | None = None
    ) -> str:
        """Handle the ask_human tool by pausing for human input."""
        answer = self._display.ask_human(question=question, context=context, options=options)
        return answer or "(no response)"

    # ------------------------------------------------------------------
    # Output-limit detection — catch repeated missing parameter patterns
    # ------------------------------------------------------------------

    def _check_missing_param_pattern(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> str | None:
        """
        Detect when the LLM keeps calling a tool without a required parameter.

        This happens when the parameter is too large for the LLM to produce
        in a single response — the parameter silently gets dropped. After 2
        occurrences of the same missing required parameter, intercept the call
        and return guidance to use create_custom_tool instead.

        Returns None if no problem detected, or a guidance string to return
        to the agent instead of executing the tool.
        """
        # Get the tool's required parameters from its schema
        tool_obj = self._registry.get(tool_name)
        if tool_obj is None:
            return None

        schema = tool_obj.parameters_schema
        required = set(schema.get("required", []))
        provided = set(tool_input.keys())
        missing = required - provided

        if not missing:
            # All required params present — clear any tracking for this tool
            self._missing_param_tracker.pop(tool_name, None)
            return None

        # Track the missing parameters
        tracker_key = tool_name
        if tracker_key not in self._missing_param_tracker:
            self._missing_param_tracker[tracker_key] = {}

        for param in missing:
            self._missing_param_tracker[tracker_key][param] = (
                self._missing_param_tracker[tracker_key].get(param, 0) + 1
            )

        # Check if any parameter has been missing 2+ times
        repeated = {
            p: count for p, count in self._missing_param_tracker[tracker_key].items() if count >= 2
        }

        if not repeated:
            return None  # First occurrence — let the tool's own error handle it

        # Build specific guidance based on the missing parameter's schema
        param_names = ", ".join(f"'{p}'" for p in repeated)
        counts = ", ".join(f"{p}: {c}x" for p, c in repeated.items())

        # Get the parameter descriptions for context
        props = schema.get("properties", {})
        param_details = []
        for p in repeated:
            ptype = props.get(p, {}).get("type", "unknown")
            pdesc = props.get(p, {}).get("description", "")
            param_details.append(f"  - {p} ({ptype}): {pdesc[:100]}")
        param_block = "\n".join(param_details)

        return (
            f"ERROR: Required parameter(s) {param_names} missing from '{tool_name}' "
            f"call — this has happened {counts} times.\n\n"
            f"THIS IS LIKELY AN LLM OUTPUT SIZE LIMIT. You are trying to produce a "
            f"parameter that is too large to fit in your response. The parameter keeps "
            f"getting silently dropped.\n\n"
            f"Missing parameter details:\n{param_block}\n\n"
            f"YOU MUST USE A DIFFERENT APPROACH:\n\n"
            f"**Use `create_custom_tool` to build a helper function** that generates "
            f"the large parameter in Python code and calls the target tool programmatically.\n\n"
            f"Example: Create a tool that reads the source file headers, builds the "
            f"parameter list in a loop, and returns the result. Your Python code has no "
            f"output size limits — only your direct responses do.\n\n"
            f"DO NOT attempt to call '{tool_name}' directly again with this pattern. "
            f"Delegate to code."
        )

    # ------------------------------------------------------------------
    # Error tracking for self-healing
    # ------------------------------------------------------------------

    def _track_error(self, tool_name: str, tool_input: dict, error: str) -> None:
        if tool_name not in self._error_tracker:
            self._error_tracker[tool_name] = []

        self._error_tracker[tool_name].append(
            {
                "input": {k: str(v)[:100] for k, v in tool_input.items()},
                "error": error[:300],
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._error_tracker[tool_name] = self._error_tracker[tool_name][-10:]

    def _get_error_context(self, tool_name: str) -> str:
        errors = self._error_tracker.get(tool_name, [])
        if not errors:
            return ""
        summaries = []
        for err in errors[-3:]:
            params = ", ".join(f"{k}={v}" for k, v in err["input"].items())
            summaries.append(f"  - ({params}): {err['error'][:150]}")
        return "\n".join(summaries)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict:
        tables = []
        for name, source_state in self._state.sources.items():
            tables.append(
                {
                    "source": name,
                    "status": source_state.status.value,
                    "table": source_state.table_name,
                    "rows": source_state.rows_loaded,
                    "files": f"{source_state.files_processed}/{source_state.files_total}",
                    "errors": len(source_state.errors),
                }
            )

        return {
            "phase": self._state.phase,
            "total_turns": self._state.total_turns,
            "sources_discovered": len(self._discovered_sources),
            "sources_completed": len(self._state.completed_sources),
            "tables": tables,
            "llm_usage": self._llm.usage_summary,
            "source_summaries": self._memory.source_summaries,
        }
