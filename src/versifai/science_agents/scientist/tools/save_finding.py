"""
Tool: save_finding

Accumulates structured research findings produced by the DataScientistAgent.
Each finding records what was discovered, the supporting evidence, and a
significance rating.  At the end of the analysis the orchestrator calls
export methods to persist everything to the results volume.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from versifai.core.tools.base import BaseTool, ToolResult


class SaveFindingTool(BaseTool):
    """
    Record a structured research finding.

    The agent calls this after each analysis step to accumulate findings
    that will be compiled into the final research report.
    """

    def __init__(self, results_path: str = "") -> None:
        super().__init__()
        self._findings: list[dict] = []
        self._results_path = results_path

    @property
    def name(self) -> str:
        return "save_finding"

    @property
    def description(self) -> str:
        return (
            "Save a structured research finding. Call this after every significant "
            "discovery during analysis. Each finding should be tied to a research "
            "question and include the evidence (SQL query or data) that supports it. "
            "Findings are accumulated in memory and exported during the synthesis phase."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "research_question_id": {
                    "type": "string",
                    "description": (
                        "ID of the research question this finding addresses "
                        "(e.g. 'rq1', 'rq2'). Use 'general' for cross-cutting findings."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Short title summarizing the finding (1 sentence).",
                },
                "finding": {
                    "type": "string",
                    "description": (
                        "The actual insight or conclusion. Be specific and quantitative "
                        "where possible (e.g. 'Counties in the top SVI quartile have "
                        "23% lower MA penetration than bottom quartile counties')."
                    ),
                },
                "evidence": {
                    "type": "string",
                    "description": (
                        "The SQL query, data summary, or statistical result that "
                        "supports this finding."
                    ),
                },
                "significance": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": (
                        "How important is this finding to the overall thesis? "
                        "'high' = directly supports/refutes the thesis, "
                        "'medium' = provides useful context, "
                        "'low' = minor or tangential observation."
                    ),
                },
                "visualization_path": {
                    "type": "string",
                    "description": (
                        "Optional path to the chart/visualization that illustrates "
                        "this finding (from create_visualization)."
                    ),
                },
            },
            "required": ["research_question_id", "title", "finding", "evidence", "significance"],
        }

    def _execute(
        self,
        research_question_id: str = "",
        title: str = "",
        finding: str = "",
        evidence: str = "",
        significance: str = "medium",
        visualization_path: str = "",
        **kwargs,
    ) -> ToolResult:
        if not research_question_id:
            return ToolResult(
                success=False,
                error="Missing required parameter 'research_question_id'.",
            )
        if not title or not finding:
            return ToolResult(
                success=False,
                error="Both 'title' and 'finding' are required.",
            )
        if significance not in ("high", "medium", "low"):
            significance = "medium"

        entry = {
            "research_question_id": research_question_id,
            "title": title,
            "finding": finding,
            "evidence": evidence,
            "significance": significance,
            "visualization_path": visualization_path,
            "timestamp": datetime.now().isoformat(),
            "index": len(self._findings),
        }
        self._findings.append(entry)

        return ToolResult(
            success=True,
            data=entry,
            summary=(
                f"Finding #{len(self._findings)} saved: '{title}' "
                f"[{significance}] for {research_question_id}."
            ),
        )

    # ------------------------------------------------------------------
    # Public helpers for the orchestrator
    # ------------------------------------------------------------------

    @property
    def findings(self) -> list[dict]:
        return self._findings

    def get_findings_for_question(self, question_id: str) -> list[dict]:
        return [f for f in self._findings if f["research_question_id"] == question_id]

    def get_high_significance_findings(self) -> list[dict]:
        return [f for f in self._findings if f["significance"] == "high"]

    def findings_summary_text(self) -> str:
        """Build a text summary of all findings for injection into prompts."""
        if not self._findings:
            return "No findings recorded yet."
        lines = []
        for f in self._findings:
            viz = f" [chart: {f['visualization_path']}]" if f.get("visualization_path") else ""
            lines.append(
                f"- [{f['significance'].upper()}] ({f['research_question_id']}) "
                f"{f['title']}: {f['finding'][:200]}{viz}"
            )
        return "\n".join(lines)

    def export_findings_json(self, path: str = "") -> str:
        """Export all findings as JSON to the given path. Returns the path written."""
        target = path or os.path.join(self._results_path, "findings.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            json.dump(self._findings, f, indent=2, default=str)
        return target
