"""
DataAnalystAgent — acceptance testing agent for Unity Catalog tables.

Runs AFTER the DataEngineerAgent has loaded all tables. Performs rigorous
validation (schema quality, join keys, volume, data quality, cross-table joins)
and produces structured feedback for the engineer to fix.

Used by the main pipeline in an engineer->analyst feedback loop.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.tools.catalog_writer import ExecuteSQLTool, ListCatalogTablesTool
from versifai.core.tools.dynamic_tool_builder import DynamicToolBuilderTool
from versifai.core.tools.registry import ToolRegistry
from versifai.data_agents.analyst.prompts import (
    build_analyst_initial_prompt,
    build_analyst_system_prompt,
)

if TYPE_CHECKING:
    from versifai.data_agents.engineer.config import ProjectConfig

logger = logging.getLogger("agent.analyst")


class DataAnalystAgent:
    """
    Acceptance testing agent that validates tables in Unity Catalog.

    The analyst has a limited tool set — primarily SQL execution and
    catalog listing. It doesn't need file readers or transformers because
    it works only with data already in the catalog.

    Usage::

        from versifai.data_agents.engineer.config import ProjectConfig

        cfg = ProjectConfig()
        analyst = DataAnalystAgent(cfg=cfg, dbutils=dbutils)
        feedback = analyst.run()
        # feedback is a dict with verdicts per table
    """

    def __init__(
        self,
        cfg: ProjectConfig | None = None,
        dbutils=None,
        max_turns: int = 80,
        table_inventory: list[dict] | None = None,
        user_hints: str = "",
    ) -> None:
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg
        self._table_inventory = table_inventory or []
        self._user_hints = user_hints

        self._display = AgentDisplay(dbutils=dbutils)
        self._memory = AgentMemory()
        self._llm = LLMClient()
        self._max_turns = max_turns
        self._dbutils = dbutils

        # Analyst only needs SQL and catalog tools
        self._registry = ToolRegistry()
        self._register_tools()

        # Error tracking
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # Pre-build prompts from config
        self._system_prompt = build_analyst_system_prompt(cfg)
        self._initial_prompt = build_analyst_initial_prompt(
            cfg,
            table_inventory=self._table_inventory,
            user_hints=self._user_hints,
        )

    def _register_tools(self) -> None:
        """Register the analyst's limited tool set."""
        cfg = self._cfg
        tools = [
            ExecuteSQLTool(),
            ListCatalogTablesTool(cfg=cfg),
            DynamicToolBuilderTool(registry=self._registry),
        ]
        for tool in tools:
            self._registry.register(tool)

    def _get_tool_definitions(self) -> list[dict]:
        """Get tool definitions including ask_human."""
        tools = self._registry.to_claude_tools()

        tools.append(
            {
                "name": "ask_human",
                "description": (
                    "Pause execution and ask the human operator a question. "
                    "Use this when you need clarification about expected data "
                    "behavior, acceptable ranges, or ambiguous findings."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the operator.",
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

    def run(self) -> dict:
        """
        Run acceptance testing on all tables in the target schema.

        Returns a structured feedback dict with per-table verdicts:
        {
            "verdicts": { "table_name": {"status": "...", "issues": [...], "fixes": [...]} },
            "cross_table_issues": [...],
            "overall_status": "PASS | NEEDS_WORK",
            "summary": "...",
            "raw_response": "..."
        }
        """
        self._display.phase("DATA ANALYST: Acceptance Testing")
        self._display.step(f"Target schema: {self._cfg.full_schema}")
        self._display.step(f"Tools: {self._registry.tool_names + ['ask_human']}")

        self._memory.add_user_message(self._initial_prompt)

        last_text = ""
        turns = 0

        while turns < self._max_turns:
            turns += 1
            self._display.step(f"Analyst turn {turns}/{self._max_turns}")

            # Send to Claude
            try:
                response = self._llm.send(
                    messages=self._memory.messages,
                    system=self._system_prompt,
                    tools=self._get_tool_definitions(),
                )
            except Exception as e:
                self._display.error(f"Analyst LLM call failed: {e}")
                break

            actions = LLMClient.extract_actions(response)
            self._memory.add_assistant_message(response.content)

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

                    if tool_name == "ask_human":
                        result_str = self._display.ask_human(
                            question=tool_input.get("question", ""),
                            context=tool_input.get("context", ""),
                            options=tool_input.get("options"),
                        )
                        self._memory.add_tool_result(tool_use_id, result_str or "(no response)")
                        self._display.tool_result(tool_name, result_str or "(no response)")
                        continue

                    result = self._registry.execute(tool_name, **tool_input)
                    result_str = result.to_content_str()

                    if not result.success:
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= self._max_consecutive_errors:
                            result_str += (
                                f"\n\n[WARNING] {self._consecutive_errors} consecutive errors."
                            )
                    else:
                        self._consecutive_errors = 0

                    if len(result_str) > 15000:
                        result_str = result_str[:15000] + "\n\n[... TRUNCATED ...]"

                    self._display.tool_result(
                        tool_name,
                        result.summary or result_str[:400],
                        is_error=not result.success,
                    )

                    self._memory.add_tool_result(
                        tool_use_id, result_str, is_error=not result.success
                    )

            if response.stop_reason == "end_turn" and not has_tool_use:
                self._display.step("Analyst review complete.")
                break

        if turns >= self._max_turns:
            self._display.warning(f"Analyst reached max turns ({self._max_turns}).")

        # Parse the structured feedback from the last text response
        feedback = self._parse_feedback(last_text)
        feedback["raw_response"] = last_text

        self._display.step(f"Analyst LLM usage: {self._llm.usage_summary}")

        return feedback

    def _parse_feedback(self, text: str) -> dict:
        """
        Extract the structured JSON feedback from the analyst's response.

        The analyst is instructed to output a JSON block with verdicts.
        We try to find and parse it; fall back to a text-based summary.
        """
        # Try to find a JSON block in the response
        import re

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON (no code block)
        json_match = re.search(r'(\{"verdicts".*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        # Fallback: wrap the raw text as the summary
        return {
            "verdicts": {},
            "cross_table_issues": [],
            "overall_status": "NEEDS_REVIEW",
            "summary": text[:2000] if text else "No feedback produced.",
        }
