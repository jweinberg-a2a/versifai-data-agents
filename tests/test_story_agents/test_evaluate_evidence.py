"""Tests for EvaluateEvidenceTool — evidence tiering behavioral tests."""

from __future__ import annotations

import pytest

from versifai.story_agents.storyteller.tools.evaluate_evidence import EvaluateEvidenceTool


@pytest.fixture
def tool():
    return EvaluateEvidenceTool()


class TestEvaluate:
    def test_strong_finding_high_tier(self, tool):
        """BEHAVIORAL: p < 0.001 + large effect → DEFINITIVE or STRONG."""
        finding = {
            "title": "Major geographic disparity",
            "finding": "Counties with high SVI show 20% lower enrollment",
            "p_value": 0.0001,
            "effect_size": "large (Cohen's d = 0.9)",
            "significance": "high",
        }
        result = tool.execute(operation="evaluate", finding=finding)
        assert result.success is True
        tier = result.data.get("tier", result.data.get("classification", "")).upper()
        assert tier in ("DEFINITIVE", "STRONG"), f"Expected DEFINITIVE/STRONG, got {tier}"

    def test_weak_finding_low_tier(self, tool):
        """BEHAVIORAL: p = 0.8 + no effect → WEAK."""
        finding = {
            "title": "No age effect",
            "finding": "Age does not predict enrollment",
            "p_value": 0.78,
            "effect_size": 0.003,
            "significance": "low",
        }
        result = tool.execute(operation="evaluate", finding=finding)
        assert result.success is True
        tier = result.data.get("tier", result.data.get("classification", "")).upper()
        assert tier in ("WEAK", "CONTEXTUAL"), f"Expected WEAK/CONTEXTUAL, got {tier}"

    def test_text_does_not_override_stats(self, tool):
        """BEHAVIORAL: Finding text says 'highly significant' but p = 0.45 → still WEAK.

        The tool should base classification on actual p-value and effect size,
        not on the narrative text. This prevents the LLM's own language from
        inflating evidence quality.
        """
        finding = {
            "title": "Highly significant result (NOT!)",
            "finding": "This is a HIGHLY SIGNIFICANT and groundbreaking discovery",
            "p_value": 0.45,
            "effect_size": 0.02,
            "significance": "low",
        }
        result = tool.execute(operation="evaluate", finding=finding)
        assert result.success is True
        tier = result.data.get("tier", result.data.get("classification", "")).upper()
        assert tier in ("WEAK", "CONTEXTUAL"), (
            f"Text 'highly significant' should not override p=0.45. Got tier={tier}"
        )

    def test_suggestive_tier(self, tool):
        """p < 0.05 but not highly significant → SUGGESTIVE."""
        finding = {
            "title": "Moderate effect",
            "finding": "Some effect observed",
            "p_value": 0.03,
            "effect_size": 0.3,
            "significance": "medium",
        }
        result = tool.execute(operation="evaluate", finding=finding)
        assert result.success is True
        tier = result.data.get("tier", result.data.get("classification", "")).upper()
        assert tier in ("SUGGESTIVE", "STRONG"), f"Expected SUGGESTIVE/STRONG, got {tier}"


class TestFalseClaimDetection:
    """BEHAVIORAL: Verify the tool catches misleading or contradictory evidence.

    These tests simulate scenarios where an LLM might generate an overly
    confident narrative that doesn't match the statistical evidence.
    The evidence tier should reflect the STATS, not the text.
    """

    def test_inflated_claim_with_zero_evidence(self, tool):
        """A finding claiming 'massive effect' with p=0.9 and tiny effect → WEAK."""
        finding = {
            "title": "MASSIVE breakthrough in enrollment prediction!",
            "finding": "We discovered an ENORMOUS and UNPRECEDENTED effect that PROVES "
            "a causal link between income and enrollment rates.",
            "p_value": 0.9,
            "effect_size": "negligible (d=0.01)",
            "significance": "low",
        }
        result = tool.execute(operation="evaluate", finding=finding)
        assert result.success is True
        tier = result.data["tier"].upper()
        assert tier == "WEAK", f"Inflated text with p=0.9, d=0.01 should be WEAK, got {tier}"

    def test_contradictory_significance_label(self, tool):
        """Significance label says 'high' but p-value says 0.6 → WEAK.

        This tests that the tool trusts the p-value over the significance label
        when they contradict each other.
        """
        finding = {
            "title": "Contradictory labels",
            "finding": "Something was observed.",
            "p_value": 0.6,
            "effect_size": "small (d=0.1)",
            "significance": "high",  # contradicts p=0.6
        }
        result = tool.execute(operation="evaluate", finding=finding)
        assert result.success is True
        tier = result.data["tier"].upper()
        # p=0.6 should dominate — this is WEAK regardless of the 'high' label
        assert tier == "WEAK", (
            f"p=0.6 should make this WEAK even with significance='high', got {tier}"
        )

    def test_false_causation_still_assessed_on_stats(self, tool):
        """A finding claiming causation based on correlation, but with valid stats.

        The tool shouldn't reject valid stats just because the text
        overclaims causation — it should classify based on the numbers.
        """
        finding = {
            "title": "Income CAUSES low enrollment",
            "finding": "We PROVED that income directly CAUSES lower enrollment rates",
            "p_value": 0.001,
            "effect_size": "large (d=0.85)",
            "significance": "high",
        }
        result = tool.execute(operation="evaluate", finding=finding)
        assert result.success is True
        tier = result.data["tier"].upper()
        # Stats are genuinely strong, so tier should be high
        assert tier in ("DEFINITIVE", "STRONG"), (
            f"Valid stats (p=0.001, d=0.85) should be DEFINITIVE/STRONG, got {tier}"
        )

    def test_curate_rejects_all_weak_findings(self, tool):
        """BEHAVIORAL: If all findings are weak, none should be usable as lead."""
        weak_findings = [
            {
                "title": f"Weak claim {i}",
                "finding": f"No real finding {i}",
                "p_value": 0.5 + i * 0.05,
                "effect_size": "negligible",
                "significance": "low",
            }
            for i in range(5)
        ]
        result = tool.execute(
            operation="curate",
            findings=weak_findings,
            purpose="Test lead selection",
            max_findings=5,
        )
        assert result.success is True
        data = result.data
        lead_count = data.get("lead_candidates", data.get("usable_as_lead", 0))
        if isinstance(lead_count, list):
            lead_count = len(lead_count)
        # None of these weak findings should be usable as a lead
        assert lead_count == 0, f"No weak findings should be usable as lead, got {lead_count}"


class TestCurate:
    def test_curate_ranks_strong_above_weak(self, tool):
        """BEHAVIORAL: Curate returns strong findings before weak ones."""
        findings = [
            {
                "title": "Weak finding",
                "finding": "No effect",
                "p_value": 0.8,
                "effect_size": 0.01,
                "significance": "low",
            },
            {
                "title": "Strong finding",
                "finding": "Major effect",
                "p_value": 0.0001,
                "effect_size": 0.9,
                "significance": "high",
            },
        ]
        result = tool.execute(
            operation="curate",
            findings=findings,
            purpose="Test section",
            max_findings=5,
        )
        assert result.success is True
        curated = result.data.get(
            "curated", result.data.get("findings", result.data.get("selected", []))
        )
        if len(curated) >= 2:
            # Strong finding should come first
            first_title = curated[0].get("title", curated[0].get("finding", {}).get("title", ""))
            assert "Strong" in first_title or "Major" in first_title

    def test_curate_limits_count(self, tool):
        findings = [
            {
                "title": f"Finding {i}",
                "finding": f"f{i}",
                "p_value": 0.01,
                "effect_size": 0.5,
                "significance": "high",
            }
            for i in range(10)
        ]
        result = tool.execute(
            operation="curate",
            findings=findings,
            purpose="Test",
            max_findings=3,
        )
        assert result.success is True
        curated = result.data.get(
            "curated", result.data.get("findings", result.data.get("selected", []))
        )
        assert len(curated) <= 3
