"""
EvaluateEvidenceTool — score evidence strength for narrative use.

Classifies findings as DEFINITIVE / STRONG / SUGGESTIVE / CONTEXTUAL / WEAK
based on p-value, effect size, significance rating, and consistency.
"""

from __future__ import annotations

from typing import Any

from versifai.core.tools.base import BaseTool, ToolResult

# Evidence strength tiers
TIERS = {
    "DEFINITIVE": "p < 0.001, large effect size (d > 0.8 or r > 0.5), replicated",
    "STRONG": "p < 0.01, medium-large effect size (d > 0.5), consistent direction",
    "SUGGESTIVE": "p < 0.05, any effect size, or p < 0.01 with small effect",
    "CONTEXTUAL": "Descriptive statistic, no hypothesis test, but informative",
    "WEAK": "p > 0.05, small effect size, or unreplicated",
}


class EvaluateEvidenceTool(BaseTool):
    """Score evidence strength and curate findings for narrative sections."""

    def __init__(
        self,
        min_significance_for_lead: str = "high",
        min_significance_for_support: str = "medium",
    ) -> None:
        self._min_lead = min_significance_for_lead
        self._min_support = min_significance_for_support

    @property
    def name(self) -> str:
        return "evaluate_evidence"

    @property
    def description(self) -> str:
        return (
            "Score the strength of evidence from DataScientist findings. "
            "Operations: 'evaluate' (classify a single finding), "
            "'curate' (select best findings for a section's purpose). "
            "Returns evidence tier: DEFINITIVE / STRONG / SUGGESTIVE / CONTEXTUAL / WEAK."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["evaluate", "curate"],
                    "description": "Which operation to run.",
                },
                "finding": {
                    "type": "object",
                    "description": "A finding dict from findings.json (for 'evaluate').",
                },
                "findings": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of findings to curate (for 'curate').",
                },
                "purpose": {
                    "type": "string",
                    "description": "The section's purpose — what evidence is needed (for 'curate').",
                },
                "max_findings": {
                    "type": "integer",
                    "description": "Maximum findings to return from curate (default 5).",
                },
            },
            "required": ["operation"],
        }

    def _classify_finding(self, finding: dict) -> dict:
        """Classify a single finding into an evidence tier."""
        significance = finding.get("significance", "").lower()
        effect_size = finding.get("effect_size", "")
        p_value = finding.get("p_value")
        title = finding.get("title", "")
        finding.get("description", "")

        # Parse effect size magnitude
        effect_magnitude = "unknown"
        if isinstance(effect_size, str):
            es_lower = effect_size.lower()
            if any(w in es_lower for w in ["large", "strong", "> 0.8", ">0.8"]):
                effect_magnitude = "large"
            elif any(w in es_lower for w in ["medium", "moderate", "> 0.5", ">0.5"]):
                effect_magnitude = "medium"
            elif any(w in es_lower for w in ["small", "> 0.2", ">0.2"]):
                effect_magnitude = "small"

        # Parse p-value
        p = None
        if isinstance(p_value, (int, float)):
            p = float(p_value)
        elif isinstance(p_value, str):
            try:
                p = float(p_value.replace("<", "").replace(">", "").strip())
            except (ValueError, TypeError):
                pass

        # Classify
        tier = "CONTEXTUAL"  # default for descriptive findings

        if p is not None:
            if p < 0.001 and effect_magnitude in ("large", "medium"):
                tier = "DEFINITIVE"
            elif p < 0.01 and effect_magnitude in ("large", "medium"):
                tier = "STRONG"
            elif p < 0.05:
                tier = "SUGGESTIVE"
            elif p >= 0.05:
                tier = "WEAK"
        elif significance in ("high", "critical"):
            tier = "STRONG"
        elif significance == "medium":
            tier = "SUGGESTIVE"
        elif significance == "low":
            tier = "WEAK"

        return {
            "title": title,
            "tier": tier,
            "tier_description": TIERS[tier],
            "significance": significance,
            "effect_size": effect_size,
            "effect_magnitude": effect_magnitude,
            "p_value": p_value,
            "usable_as_lead": tier in ("DEFINITIVE", "STRONG"),
            "usable_as_support": tier in ("DEFINITIVE", "STRONG", "SUGGESTIVE"),
            "finding_title": title,
        }

    def _execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs["operation"]

        if operation == "evaluate":
            finding = kwargs.get("finding", {})
            if not finding:
                return ToolResult(success=False, error="finding dict required.")
            result = self._classify_finding(finding)
            return ToolResult(
                success=True,
                data=result,
                summary=f"'{result['title']}' → {result['tier']}",
            )

        elif operation == "curate":
            findings = kwargs.get("findings", [])
            purpose = kwargs.get("purpose", "")
            max_findings = kwargs.get("max_findings", 5)

            if not findings:
                return ToolResult(success=False, error="findings list required.")

            # Classify all
            classified = [self._classify_finding(f) for f in findings]

            # Sort by evidence strength
            tier_order = {
                "DEFINITIVE": 0,
                "STRONG": 1,
                "SUGGESTIVE": 2,
                "CONTEXTUAL": 3,
                "WEAK": 4,
            }
            classified.sort(key=lambda c: tier_order.get(c["tier"], 5))

            # Take top N
            curated = classified[:max_findings]

            lead_candidates = [c for c in curated if c["usable_as_lead"]]
            support_candidates = [c for c in curated if c["usable_as_support"]]

            return ToolResult(
                success=True,
                data={
                    "purpose": purpose,
                    "total_evaluated": len(classified),
                    "curated": curated,
                    "lead_candidates": len(lead_candidates),
                    "support_candidates": len(support_candidates),
                },
                summary=(
                    f"Curated {len(curated)} of {len(findings)} findings. "
                    f"Lead: {len(lead_candidates)}, Support: {len(support_candidates)}."
                ),
            )

        return ToolResult(success=False, error=f"Unknown operation: {operation}")
