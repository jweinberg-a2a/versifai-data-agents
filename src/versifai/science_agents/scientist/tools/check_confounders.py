"""
Tool: check_confounders

Detects Simpson's Paradox and hidden confounders in relationships.

When the agent finds a weak, absent, or non-monotonic relationship between
two variables, this tool decomposes the analysis by categorical grouping
variables to determine whether the aggregate result is masking subgroup
patterns.  If subgroups show a different direction or strength than the
aggregate, Simpson's Paradox is flagged and the tool returns a structured
comparison the agent can use to create a single explanatory visualization.

This is a formal analytical skill — the agent should call it whenever an
observed relationship contradicts its hypothesis, not just note the
discrepancy and move on.
"""

from __future__ import annotations

import pandas as pd

from versifai._utils.sql_data import fetch_sql_data
from versifai.core.tools.base import BaseTool, ToolResult


class CheckConfoundersTool(BaseTool):
    """
    Decompose a relationship by categorical variables to detect Simpson's
    Paradox and hidden confounders.

    WHEN TO USE:
    - After computing a correlation or group comparison that is unexpectedly
      weak (|r| < 0.1), absent, or non-monotonic.
    - When an effect is suspiciously strong and might be inflated by a
      confounding subgroup.
    - Whenever the aggregate result contradicts your hypothesis.

    HOW IT WORKS:
    1. Computes the aggregate relationship (correlation or group means).
    2. For each grouping column, computes the same relationship within
       each subgroup.
    3. Compares subgroup directions/magnitudes to the aggregate.
    4. Flags Simpson's Paradox if the subgroup pattern differs materially
       from the aggregate.

    WHAT TO DO WITH THE RESULT:
    - If paradox_detected is True, create ONE visualization showing the
      aggregate vs decomposed result side-by-side, and save it as a
      high-significance finding.
    - If no paradox is detected, the weak result is genuine — report it
      and move on.
    """

    @property
    def name(self) -> str:
        return "check_confounders"

    @property
    def description(self) -> str:
        return (
            "Detect Simpson's Paradox and hidden confounders in a relationship. "
            "Call this when a correlation or group comparison is unexpectedly "
            "weak, absent, non-monotonic, or contradicts your hypothesis.\n\n"
            "Provide a dataset with a predictor column, an outcome column, and "
            "one or more categorical grouping columns to decompose by. The tool "
            "computes the aggregate relationship, then re-computes it within "
            "each subgroup. If subgroups show a different pattern than the "
            "aggregate, Simpson's Paradox is flagged.\n\n"
            "Returns structured results including: aggregate stats, per-subgroup "
            "stats, whether paradox was detected, which grouping variable "
            "matters most, and a recommended visualization approach.\n\n"
            "ALWAYS use sql_query to provide data — pass a SELECT statement and the "
            "tool fetches ALL rows directly. Do NOT call execute_sql first.\n\n"
            "IMPORTANT: Keep grouping_columns to 1-2 variables — the most likely "
            "confounders based on domain knowledge. Do not decompose by everything."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": (
                        "A SELECT SQL query to fetch the data. The tool executes it "
                        "directly and retrieves ALL rows — no row limit."
                    ),
                },
                "outcome_column": {
                    "type": "string",
                    "description": (
                        "The dependent / outcome variable (e.g., 'overall_rating', "
                        "'measure_score'). Must be numeric."
                    ),
                },
                "predictor_column": {
                    "type": "string",
                    "description": (
                        "The independent / predictor variable (e.g., 'weighted_svi'). "
                        "Can be numeric (correlation computed) or categorical "
                        "(group means compared)."
                    ),
                },
                "grouping_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Categorical columns to decompose by (e.g., ['plan_type', "
                        "'region']). The tool will check each one independently. "
                        "Keep to 1-2 most likely confounders."
                    ),
                },
            },
            "required": ["sql_query", "outcome_column", "predictor_column", "grouping_columns"],
        }

    def _execute(
        self,
        sql_query: str = "",
        data: list[dict] | None = None,
        outcome_column: str = "",
        predictor_column: str = "",
        grouping_columns: list[str] | None = None,
        **kwargs,
    ) -> ToolResult:
        if not outcome_column:
            return ToolResult(success=False, error="Missing 'outcome_column'.")
        if not predictor_column:
            return ToolResult(success=False, error="Missing 'predictor_column'.")
        if not grouping_columns:
            return ToolResult(success=False, error="Missing 'grouping_columns'.")

        # Resolve data from SQL query (preferred) or inline data (backward compat / tests)
        if sql_query:
            df = fetch_sql_data(sql_query)
            if df is None:
                return ToolResult(
                    success=False,
                    error="sql_query execution failed — check the SQL syntax.",
                )
        elif data:
            df = pd.DataFrame(data)
        else:
            return ToolResult(
                success=False,
                error="Provide 'sql_query' (a SELECT statement) to fetch data.",
            )
        if df.empty:
            return ToolResult(success=False, error="Data is empty.")

        # Validate columns exist
        for col in [outcome_column, predictor_column] + grouping_columns:
            if col not in df.columns:
                return ToolResult(
                    success=False,
                    error=f"Column '{col}' not found. Available: {list(df.columns)}",
                )

        # Determine whether predictor is numeric or categorical
        predictor_is_numeric = pd.api.types.is_numeric_dtype(df[predictor_column])

        # Compute aggregate relationship
        aggregate = self._compute_relationship(
            df,
            outcome_column,
            predictor_column,
            predictor_is_numeric,
        )

        # Decompose by each grouping column
        decompositions = []
        for group_col in grouping_columns:
            decomp = self._decompose_by_group(
                df,
                outcome_column,
                predictor_column,
                group_col,
                predictor_is_numeric,
            )
            decompositions.append(decomp)

        # Detect Simpson's Paradox across all decompositions
        any_paradox = False
        strongest_paradox = None
        strongest_score = 0.0

        for decomp in decompositions:
            if decomp.get("paradox_detected"):
                any_paradox = True
                score = decomp.get("paradox_strength", 0.0)
                if score > strongest_score:
                    strongest_score = score
                    strongest_paradox = decomp

        # Build recommendation
        if any_paradox and strongest_paradox:
            recommendation = (
                f"Simpson's Paradox detected when decomposing by "
                f"'{strongest_paradox['grouping_column']}'. "
                f"The aggregate relationship ({aggregate['summary']}) "
                f"masks subgroup patterns that tell a different story. "
                f"Create ONE visualization: a grouped/faceted chart showing "
                f"the {outcome_column} vs {predictor_column} relationship "
                f"separately for each {strongest_paradox['grouping_column']} "
                f"category, with the aggregate trend overlaid for contrast. "
                f"Save this as a high-significance finding."
            )
        else:
            recommendation = (
                f"No Simpson's Paradox detected. The aggregate relationship "
                f"({aggregate['summary']}) holds consistently across subgroups. "
                f"The result is genuine — report it as-is."
            )

        result = {
            "aggregate": aggregate,
            "decompositions": decompositions,
            "paradox_detected": any_paradox,
            "strongest_confounder": (
                strongest_paradox["grouping_column"] if strongest_paradox else None
            ),
            "recommendation": recommendation,
        }

        summary_label = "PARADOX DETECTED" if any_paradox else "No paradox found"
        return ToolResult(
            success=True,
            data=result,
            summary=(
                f"Confounder check: {summary_label}. "
                f"Aggregate: {aggregate['summary']}. "
                + (
                    f"Key confounder: {strongest_paradox['grouping_column']}."
                    if strongest_paradox
                    else "Relationship is consistent across subgroups."
                )
            ),
        )

    # ------------------------------------------------------------------
    # Internal: compute relationship (correlation or group means)
    # ------------------------------------------------------------------

    def _compute_relationship(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictor: str,
        predictor_numeric: bool,
        label: str = "aggregate",
    ) -> dict:
        """Compute the relationship between predictor and outcome."""
        from scipy import stats as sp_stats

        clean = df[[predictor, outcome]].dropna()
        n = len(clean)
        if n < 3:
            return {
                "label": label,
                "n": n,
                "method": "insufficient_data",
                "summary": f"Insufficient data (n={n})",
            }

        if predictor_numeric:
            # Correlation
            r, p = sp_stats.spearmanr(clean[predictor], clean[outcome])
            r = float(r)
            p = float(p)
            strength = self._interpret_r(abs(r))
            direction = "positive" if r > 0 else "negative" if r < 0 else "none"
            return {
                "label": label,
                "n": n,
                "method": "spearman_correlation",
                "r": round(r, 4),
                "p_value": round(p, 6),
                "r_squared": round(r**2, 4),
                "direction": direction,
                "strength": strength,
                "summary": f"r={r:.3f} ({strength} {direction}, n={n})",
            }
        else:
            # Group means comparison
            groups = clean.groupby(predictor)[outcome]
            group_stats = {}
            for name, group in groups:
                if len(group) >= 1:
                    group_stats[str(name)] = {
                        "n": int(len(group)),
                        "mean": round(float(group.mean()), 4),
                        "median": round(float(group.median()), 4),
                        "std": round(float(group.std()), 4) if len(group) > 1 else 0.0,
                    }
            means = [v["mean"] for v in group_stats.values()]
            spread = max(means) - min(means) if means else 0.0
            return {
                "label": label,
                "n": n,
                "method": "group_means",
                "groups": group_stats,
                "spread": round(spread, 4),
                "summary": f"Group spread={spread:.3f} across {len(group_stats)} groups (n={n})",
            }

    # ------------------------------------------------------------------
    # Internal: decompose by a grouping variable
    # ------------------------------------------------------------------

    def _decompose_by_group(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictor: str,
        group_col: str,
        predictor_numeric: bool,
    ) -> dict:
        """Decompose the relationship within each category of group_col."""
        subgroups = {}
        group_sizes = df[group_col].value_counts()

        # Only analyze groups with enough data
        valid_groups = group_sizes[group_sizes >= 5].index.tolist()

        for group_val in valid_groups:
            subset = df[df[group_col] == group_val]
            rel = self._compute_relationship(
                subset,
                outcome,
                predictor,
                predictor_numeric,
                label=f"{group_col}={group_val}",
            )
            rel["group_value"] = str(group_val)
            rel["group_size"] = int(len(subset))
            rel["group_pct"] = round(len(subset) / len(df) * 100, 1)
            subgroups[str(group_val)] = rel

        # Detect paradox: do subgroups disagree with aggregate?
        aggregate = self._compute_relationship(
            df,
            outcome,
            predictor,
            predictor_numeric,
        )
        paradox = self._detect_paradox(aggregate, subgroups, predictor_numeric)

        return {
            "grouping_column": group_col,
            "n_subgroups": len(subgroups),
            "subgroups": subgroups,
            "paradox_detected": paradox["detected"],
            "paradox_strength": paradox["strength"],
            "paradox_explanation": paradox["explanation"],
        }

    # ------------------------------------------------------------------
    # Internal: paradox detection logic
    # ------------------------------------------------------------------

    def _detect_paradox(
        self,
        aggregate: dict,
        subgroups: dict[str, dict],
        predictor_numeric: bool,
    ) -> dict:
        """
        Detect whether subgroup patterns materially differ from aggregate.

        For correlations: paradox if majority of subgroups have opposite
        direction from aggregate, or if subgroups show strong effects while
        aggregate is weak.

        For group means: paradox if subgroup ordering is consistently
        different from aggregate ordering.
        """
        if not subgroups:
            return {"detected": False, "strength": 0.0, "explanation": "No subgroups to compare."}

        if predictor_numeric:
            return self._detect_correlation_paradox(aggregate, subgroups)
        else:
            return self._detect_means_paradox(aggregate, subgroups)

    def _detect_correlation_paradox(
        self,
        aggregate: dict,
        subgroups: dict[str, dict],
    ) -> dict:
        agg_r = aggregate.get("r", 0.0)
        agg_strength = abs(agg_r)

        subgroup_rs = []
        for sg in subgroups.values():
            if sg.get("method") == "spearman_correlation" and sg.get("r") is not None:
                subgroup_rs.append(
                    {
                        "label": sg["label"],
                        "r": sg["r"],
                        "n": sg["n"],
                    }
                )

        if not subgroup_rs:
            return {
                "detected": False,
                "strength": 0.0,
                "explanation": "No valid subgroup correlations.",
            }

        # Weighted by sample size
        total_n = sum(s["n"] for s in subgroup_rs)

        # Check 1: Direction reversal — majority of subgroups have opposite sign
        same_direction = sum(
            1 for s in subgroup_rs if (s["r"] > 0) == (agg_r > 0) or abs(s["r"]) < 0.05
        )
        opposite_direction = len(subgroup_rs) - same_direction

        # Check 2: Strength masking — aggregate is weak but subgroups are strong
        weighted_avg_strength = (
            sum(abs(s["r"]) * s["n"] for s in subgroup_rs) / total_n if total_n > 0 else 0.0
        )

        strength_ratio = weighted_avg_strength / max(agg_strength, 0.01)

        # Paradox conditions:
        # A) Most subgroups disagree with aggregate direction
        direction_paradox = opposite_direction > same_direction and len(subgroup_rs) >= 2
        # B) Subgroups are substantially stronger than aggregate (masking effect)
        masking_paradox = strength_ratio > 2.0 and weighted_avg_strength > 0.15

        detected = direction_paradox or masking_paradox
        strength = max(
            (opposite_direction / len(subgroup_rs)) if direction_paradox else 0.0,
            min(strength_ratio / 5.0, 1.0) if masking_paradox else 0.0,
        )

        if direction_paradox:
            explanation = (
                f"Direction reversal: aggregate r={agg_r:.3f}, but "
                f"{opposite_direction}/{len(subgroup_rs)} subgroups show the "
                f"opposite direction. A confounding variable is masking or "
                f"reversing the true relationship."
            )
        elif masking_paradox:
            explanation = (
                f"Strength masking: aggregate |r|={agg_strength:.3f} (weak), but "
                f"subgroups average |r|={weighted_avg_strength:.3f} "
                f"({strength_ratio:.1f}x stronger). A confounding variable is "
                f"diluting a real effect."
            )
        else:
            explanation = (
                f"No paradox: aggregate r={agg_r:.3f} is consistent with "
                f"subgroup patterns (avg |r|={weighted_avg_strength:.3f})."
            )

        return {"detected": detected, "strength": round(strength, 3), "explanation": explanation}

    def _detect_means_paradox(
        self,
        aggregate: dict,
        subgroups: dict[str, dict],
    ) -> dict:
        """Detect paradox in group means comparisons."""
        agg_groups = aggregate.get("groups", {})
        if len(agg_groups) < 2:
            return {"detected": False, "strength": 0.0, "explanation": "Need 2+ groups."}

        # Get aggregate ordering by mean
        agg_ranking = sorted(agg_groups.keys(), key=lambda k: agg_groups[k]["mean"])

        # Check if subgroup orderings consistently differ
        reversals = 0
        total_checks = 0

        for sg in subgroups.values():
            sg_groups = sg.get("groups", {})
            if len(sg_groups) < 2:
                continue
            sg_ranking = sorted(sg_groups.keys(), key=lambda k: sg_groups[k]["mean"])
            # Compare rankings using Kendall-style concordance
            if sg_ranking != agg_ranking:
                # Check if it's a full reversal or partial
                if sg_ranking == list(reversed(agg_ranking)):
                    reversals += 2  # Full reversal counts double
                else:
                    reversals += 1
            total_checks += 1

        if total_checks == 0:
            return {"detected": False, "strength": 0.0, "explanation": "Insufficient subgroups."}

        # Paradox if significant number of subgroups show different ordering
        strength = reversals / (total_checks * 2)
        detected = strength > 0.3

        explanation = (
            f"{'Ordering paradox' if detected else 'No paradox'}: "
            f"{reversals} ranking differences across {total_checks} subgroups "
            f"(strength={strength:.2f})."
        )

        return {"detected": detected, "strength": round(strength, 3), "explanation": explanation}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_r(r: float) -> str:
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        return "very strong"
