"""
Tool: validate_statistics

Statistical rigor gate. The agent calls this to ensure analytical conclusions
are defensible before saving findings.

Formalizes four common statistical pitfalls as explicit checks:
  - Multiple comparisons (FDR correction for batch tests)
  - Multicollinearity (VIF for regression interpretation)
  - Ecological fallacy (cross-grain relationship validation)
  - Robustness (sensitivity to outliers and alternative methods)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from versifai.core.tools.base import BaseTool, ToolResult


class ValidateStatisticsTool(BaseTool):
    """
    Validate statistical rigor of analytical findings.

    WHEN TO USE:
    - 'multiple_comparisons': After running correlations or hypothesis tests
      across many variables (e.g., 30 measures). MUST call before claiming
      any individual result is "significant."
    - 'multicollinearity': Before interpreting regression coefficients when
      using 2+ predictors. Catches unstable coefficients caused by correlated
      features.
    - 'ecological_fallacy': When drawing conclusions about one unit of
      analysis from data measured at a different grain (e.g., county-level
      SVI predicting contract-level Stars).
    - 'robustness': Before saving any high-significance finding. Verifies
      the result survives outlier removal, winsorization, and alternative
      statistical methods.
    """

    @property
    def name(self) -> str:
        return "validate_statistics"

    @property
    def description(self) -> str:
        return (
            "Statistical rigor validation. Call before saving findings to "
            "ensure conclusions are defensible.\n\n"
            "Check types:\n"
            "- 'multiple_comparisons': Takes batch p-values, applies Bonferroni "
            "and Benjamini-Hochberg FDR correction. Returns which results "
            "survive at each threshold.\n"
            "- 'multicollinearity': Computes actual VIF for each predictor. "
            "Flags VIF > 5 (concern) and VIF > 10 (severe). Stronger than "
            "the r > 0.7 proxy in assumption_check.\n"
            "- 'ecological_fallacy': Takes a relationship at two different "
            "grains (e.g., county vs contract). Flags if direction reverses "
            "or magnitude changes substantially across grains.\n"
            "- 'robustness': Re-runs a correlation/comparison 4 ways — "
            "original, outliers removed, winsorized, and alternative method. "
            "Reports whether the finding holds across all variants.\n\n"
            "Each check returns a structured verdict with pass/fail, details, "
            "and recommendations."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "check_type": {
                    "type": "string",
                    "description": (
                        "Type of validation: 'multiple_comparisons', "
                        "'multicollinearity', 'ecological_fallacy', 'robustness'."
                    ),
                },
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Data as list of row dicts. Required for "
                        "'multicollinearity', 'ecological_fallacy', 'robustness'."
                    ),
                },
                "p_values": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "For 'multiple_comparisons': list of {name: str, p_value: float} "
                        "from batch hypothesis tests or correlations."
                    ),
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance threshold (default 0.05).",
                },
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For 'multicollinearity': predictor column names.",
                },
                "outcome_column": {
                    "type": "string",
                    "description": (
                        "For 'ecological_fallacy' and 'robustness': the dependent "
                        "variable column name."
                    ),
                },
                "predictor_column": {
                    "type": "string",
                    "description": (
                        "For 'ecological_fallacy' and 'robustness': the independent "
                        "variable column name."
                    ),
                },
                "data_fine_grain": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "For 'ecological_fallacy': the same relationship measured "
                        "at the finer grain (e.g., contract-level data when 'data' "
                        "is county-level)."
                    ),
                },
            },
            "required": ["check_type"],
        }

    def _execute(
        self,
        check_type: str = "",
        data: list[dict] | None = None,
        p_values: list[dict] | None = None,
        alpha: float = 0.05,
        feature_columns: list[str] | None = None,
        outcome_column: str = "",
        predictor_column: str = "",
        data_fine_grain: list[dict] | None = None,
        **kwargs,
    ) -> ToolResult:
        if not check_type:
            return ToolResult(success=False, error="Missing 'check_type'.")

        dispatch = {
            "multiple_comparisons": self._check_multiple_comparisons,
            "multicollinearity": self._check_multicollinearity,
            "ecological_fallacy": self._check_ecological_fallacy,
            "robustness": self._check_robustness,
        }

        handler = dispatch.get(check_type)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown check_type '{check_type}'. Use: {list(dispatch.keys())}",
            )

        return handler(
            data=data,
            p_values=p_values,
            alpha=alpha,
            feature_columns=feature_columns,
            outcome_column=outcome_column,
            predictor_column=predictor_column,
            data_fine_grain=data_fine_grain,
        )

    # ------------------------------------------------------------------
    # Check: Multiple comparisons
    # ------------------------------------------------------------------

    def _check_multiple_comparisons(
        self,
        p_values=None,
        alpha=0.05,
        **kw,
    ) -> ToolResult:
        if not p_values:
            return ToolResult(
                success=False,
                error=("Missing 'p_values'. Provide a list of {name: str, p_value: float} dicts."),
            )

        n = len(p_values)
        if n < 2:
            return ToolResult(
                success=True,
                data={
                    "correction_needed": False,
                    "n_tests": n,
                    "details": [],
                },
                summary="Only 1 test — no multiple comparison correction needed.",
            )

        # Sort by p-value for BH procedure
        tests = sorted(p_values, key=lambda x: x.get("p_value", 1.0))

        # Bonferroni correction
        bonferroni_alpha = alpha / n

        # Benjamini-Hochberg FDR
        results = []
        for rank, test in enumerate(tests, 1):
            name = test.get("name", f"test_{rank}")
            p = test.get("p_value", 1.0)
            bh_threshold = (rank / n) * alpha

            results.append(
                {
                    "name": name,
                    "p_value": round(p, 6),
                    "rank": rank,
                    "uncorrected_significant": p < alpha,
                    "bonferroni_threshold": round(bonferroni_alpha, 6),
                    "bonferroni_significant": p < bonferroni_alpha,
                    "bh_threshold": round(bh_threshold, 6),
                    "bh_significant": p < bh_threshold,
                }
            )

        # Find the BH cutoff (largest rank where p <= threshold)
        bh_significant_count = 0
        for r in reversed(results):
            if r["bh_significant"]:
                bh_significant_count = r["rank"]
                break
        # Mark all tests up to and including the cutoff rank as significant
        for r in results:
            r["bh_significant"] = r["rank"] <= bh_significant_count

        uncorrected_sig = sum(1 for r in results if r["uncorrected_significant"])
        bonferroni_sig = sum(1 for r in results if r["bonferroni_significant"])
        bh_sig = sum(1 for r in results if r["bh_significant"])
        lost_to_correction = uncorrected_sig - bh_sig

        return ToolResult(
            success=True,
            data={
                "correction_needed": True,
                "n_tests": n,
                "alpha": alpha,
                "uncorrected_significant": uncorrected_sig,
                "bonferroni_significant": bonferroni_sig,
                "bonferroni_alpha": round(bonferroni_alpha, 6),
                "bh_fdr_significant": bh_sig,
                "lost_to_correction": lost_to_correction,
                "results": results,
                "recommendation": (
                    f"Of {uncorrected_sig} nominally significant results, "
                    f"{bh_sig} survive BH-FDR correction and {bonferroni_sig} "
                    f"survive Bonferroni. Use BH-FDR (less conservative) for "
                    f"exploratory analysis, Bonferroni for confirmatory claims. "
                    f"Report ONLY corrected p-values in findings."
                ),
            },
            summary=(
                f"Multiple comparisons ({n} tests): {uncorrected_sig} nominally "
                f"significant → {bh_sig} survive FDR, {bonferroni_sig} survive "
                f"Bonferroni."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Multicollinearity
    # ------------------------------------------------------------------

    def _check_multicollinearity(
        self,
        data=None,
        feature_columns=None,
        **kw,
    ) -> ToolResult:
        if not data:
            return ToolResult(success=False, error="Missing 'data'.")
        if not feature_columns or len(feature_columns) < 2:
            return ToolResult(
                success=False,
                error="Need at least 2 feature_columns for multicollinearity check.",
            )

        df = pd.DataFrame(data)
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            return ToolResult(
                success=False,
                error=f"Columns not found: {missing}. Available: {list(df.columns)}",
            )

        # Prepare numeric matrix
        X = df[feature_columns].apply(pd.to_numeric, errors="coerce").dropna()
        if len(X) < len(feature_columns) + 1:
            return ToolResult(
                success=False,
                error=(
                    f"Insufficient observations ({len(X)}) for {len(feature_columns)} features."
                ),
            )

        # Add constant for VIF calculation
        X_const = X.copy()
        X_const.insert(0, "_const", 1.0)

        # Compute VIF
        vif_results = []
        severe = []
        concern = []

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            # Fallback: compute VIF manually via R² from OLS
            variance_inflation_factor = None

        for i, col in enumerate(feature_columns):
            try:
                if variance_inflation_factor is not None:
                    vif = float(variance_inflation_factor(X_const.values, i + 1))
                else:
                    # Manual VIF: 1 / (1 - R²) from regressing col on other cols
                    from numpy.linalg import lstsq

                    others = [c for c in feature_columns if c != col]
                    y = X[col].values
                    A = X[others].values
                    A = np.column_stack([np.ones(len(A)), A])
                    coef, _, _, _ = lstsq(A, y, rcond=None)
                    y_pred = A @ coef
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    vif = 1 / (1 - r2) if r2 < 1 else float("inf")
            except Exception:
                vif = None

            status = "ok"
            if vif is not None:
                if vif > 10:
                    status = "severe"
                    severe.append(col)
                elif vif > 5:
                    status = "concern"
                    concern.append(col)

            vif_results.append(
                {
                    "feature": col,
                    "vif": round(vif, 2) if vif is not None else None,
                    "status": status,
                }
            )

        # Also compute pairwise correlations for context
        corr_matrix = X.corr()
        high_pairs = []
        for i, c1 in enumerate(feature_columns):
            for c2 in feature_columns[i + 1 :]:
                r = float(corr_matrix.loc[c1, c2])
                if abs(r) > 0.7:
                    high_pairs.append(
                        {
                            "var1": c1,
                            "var2": c2,
                            "r": round(r, 3),
                        }
                    )

        has_issues = len(severe) > 0 or len(concern) > 0
        recommendation = ""
        if severe:
            recommendation = (
                f"SEVERE multicollinearity in: {severe}. These features share "
                f"so much variance that their individual regression coefficients "
                f"are unreliable. Options: (1) remove redundant features, "
                f"(2) combine into a composite index, (3) use regularization "
                f"(Ridge/Lasso), (4) use PCA to create orthogonal components."
            )
        elif concern:
            recommendation = (
                f"Moderate multicollinearity in: {concern}. Coefficients may "
                f"be somewhat unstable. Consider whether these features measure "
                f"overlapping constructs and could be combined."
            )
        else:
            recommendation = "No multicollinearity issues. Coefficients are interpretable."

        return ToolResult(
            success=True,
            data={
                "issues_found": has_issues,
                "severe_features": severe,
                "concern_features": concern,
                "vif_results": vif_results,
                "high_correlations": high_pairs,
                "n_features": len(feature_columns),
                "n_observations": len(X),
                "recommendation": recommendation,
            },
            summary=(
                f"VIF check: {len(severe)} severe (VIF>10), "
                f"{len(concern)} concern (VIF>5) of "
                f"{len(feature_columns)} features."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Ecological fallacy
    # ------------------------------------------------------------------

    def _check_ecological_fallacy(
        self,
        data=None,
        data_fine_grain=None,
        outcome_column="",
        predictor_column="",
        **kw,
    ) -> ToolResult:
        if not data or not data_fine_grain:
            return ToolResult(
                success=False,
                error=(
                    "Both 'data' (coarse grain) and 'data_fine_grain' (fine grain) are required."
                ),
            )
        if not outcome_column or not predictor_column:
            return ToolResult(
                success=False,
                error="Both 'outcome_column' and 'predictor_column' are required.",
            )

        from scipy import stats as sp_stats

        df_coarse = pd.DataFrame(data)
        df_fine = pd.DataFrame(data_fine_grain)

        # Validate columns exist in both
        for label, df in [("coarse", df_coarse), ("fine", df_fine)]:
            for col in [outcome_column, predictor_column]:
                if col not in df.columns:
                    return ToolResult(
                        success=False,
                        error=f"Column '{col}' not found in {label} data.",
                    )

        # Compute correlation at coarse grain
        coarse_clean = df_coarse[[predictor_column, outcome_column]].dropna()
        coarse_clean = coarse_clean.apply(pd.to_numeric, errors="coerce").dropna()
        if len(coarse_clean) < 5:
            return ToolResult(
                success=False,
                error=f"Insufficient coarse-grain data (n={len(coarse_clean)}).",
            )
        r_coarse, p_coarse = sp_stats.spearmanr(
            coarse_clean[predictor_column],
            coarse_clean[outcome_column],
        )

        # Compute correlation at fine grain
        fine_clean = df_fine[[predictor_column, outcome_column]].dropna()
        fine_clean = fine_clean.apply(pd.to_numeric, errors="coerce").dropna()
        if len(fine_clean) < 5:
            return ToolResult(
                success=False,
                error=f"Insufficient fine-grain data (n={len(fine_clean)}).",
            )
        r_fine, p_fine = sp_stats.spearmanr(
            fine_clean[predictor_column],
            fine_clean[outcome_column],
        )

        # Detect ecological fallacy
        r_coarse = float(r_coarse)
        r_fine = float(r_fine)
        direction_reversal = (r_coarse > 0) != (r_fine > 0) and abs(r_fine) > 0.05
        magnitude_change = abs(abs(r_coarse) - abs(r_fine))
        substantial_change = magnitude_change > 0.2

        fallacy_detected = direction_reversal or substantial_change

        if direction_reversal:
            explanation = (
                f"DIRECTION REVERSAL: At coarse grain r={r_coarse:.3f}, but at "
                f"fine grain r={r_fine:.3f}. The relationship reverses across "
                f"aggregation levels. Conclusions drawn at the coarse level do "
                f"NOT hold at the individual/fine level."
            )
        elif substantial_change:
            explanation = (
                f"MAGNITUDE SHIFT: Coarse grain r={r_coarse:.3f}, fine grain "
                f"r={r_fine:.3f} (difference={magnitude_change:.3f}). The effect "
                f"{'inflates' if abs(r_coarse) > abs(r_fine) else 'attenuates'} "
                f"at the {'coarser' if abs(r_coarse) > abs(r_fine) else 'finer'} "
                f"level. Be cautious about the magnitude of claims."
            )
        else:
            explanation = (
                f"Relationship is consistent across grains: coarse r={r_coarse:.3f}, "
                f"fine r={r_fine:.3f}. No ecological fallacy detected."
            )

        return ToolResult(
            success=True,
            data={
                "fallacy_detected": fallacy_detected,
                "direction_reversal": direction_reversal,
                "coarse_grain": {
                    "r": round(r_coarse, 4),
                    "p_value": round(float(p_coarse), 6),
                    "n": len(coarse_clean),
                    "strength": _interpret_r(abs(r_coarse)),
                },
                "fine_grain": {
                    "r": round(r_fine, 4),
                    "p_value": round(float(p_fine), 6),
                    "n": len(fine_clean),
                    "strength": _interpret_r(abs(r_fine)),
                },
                "magnitude_change": round(magnitude_change, 4),
                "explanation": explanation,
                "recommendation": (
                    "Report findings at the grain where they were measured. "
                    "Do NOT extrapolate coarse-grain relationships to finer units. "
                    "If both grains are relevant, report both correlations."
                    if fallacy_detected
                    else "Relationship holds across grains — safe to interpret."
                ),
            },
            summary=(
                f"ECOLOGICAL FALLACY {'DETECTED' if fallacy_detected else 'not detected'}: "
                f"coarse r={r_coarse:.3f} (n={len(coarse_clean)}), "
                f"fine r={r_fine:.3f} (n={len(fine_clean)})."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Robustness
    # ------------------------------------------------------------------

    def _check_robustness(
        self,
        data=None,
        outcome_column="",
        predictor_column="",
        **kw,
    ) -> ToolResult:
        if not data:
            return ToolResult(success=False, error="Missing 'data'.")
        if not outcome_column or not predictor_column:
            return ToolResult(
                success=False,
                error="Both 'outcome_column' and 'predictor_column' required.",
            )

        from scipy import stats as sp_stats

        df = pd.DataFrame(data)
        for col in [outcome_column, predictor_column]:
            if col not in df.columns:
                return ToolResult(
                    success=False,
                    error=f"Column '{col}' not found. Available: {list(df.columns)}",
                )

        clean = (
            df[[predictor_column, outcome_column]].apply(pd.to_numeric, errors="coerce").dropna()
        )

        if len(clean) < 10:
            return ToolResult(
                success=False,
                error=f"Insufficient data for robustness check (n={len(clean)}).",
            )

        x = clean[predictor_column].values
        y = clean[outcome_column].values

        variants = []

        # Variant 1: Original (Spearman)
        r_orig, p_orig = sp_stats.spearmanr(x, y)
        variants.append(
            {
                "variant": "original_spearman",
                "r": round(float(r_orig), 4),
                "p_value": round(float(p_orig), 6),
                "n": len(x),
                "direction": "positive" if r_orig > 0 else "negative",
            }
        )

        # Variant 2: Pearson (alternative method)
        r_pearson, p_pearson = sp_stats.pearsonr(x, y)
        variants.append(
            {
                "variant": "pearson",
                "r": round(float(r_pearson), 4),
                "p_value": round(float(p_pearson), 6),
                "n": len(x),
                "direction": "positive" if r_pearson > 0 else "negative",
            }
        )

        # Variant 3: Outliers removed (IQR method on both variables)
        x_q1, x_q3 = np.percentile(x, [25, 75])
        y_q1, y_q3 = np.percentile(y, [25, 75])
        x_iqr = x_q3 - x_q1
        y_iqr = y_q3 - y_q1
        mask = (
            (x >= x_q1 - 1.5 * x_iqr)
            & (x <= x_q3 + 1.5 * x_iqr)
            & (y >= y_q1 - 1.5 * y_iqr)
            & (y <= y_q3 + 1.5 * y_iqr)
        )
        x_no_outlier = x[mask]
        y_no_outlier = y[mask]
        removed = len(x) - len(x_no_outlier)

        if len(x_no_outlier) >= 5:
            r_no, p_no = sp_stats.spearmanr(x_no_outlier, y_no_outlier)
            variants.append(
                {
                    "variant": "outliers_removed",
                    "r": round(float(r_no), 4),
                    "p_value": round(float(p_no), 6),
                    "n": len(x_no_outlier),
                    "removed": removed,
                    "direction": "positive" if r_no > 0 else "negative",
                }
            )
        else:
            variants.append(
                {
                    "variant": "outliers_removed",
                    "r": None,
                    "p_value": None,
                    "n": len(x_no_outlier),
                    "removed": removed,
                    "direction": "insufficient_data",
                }
            )

        # Variant 4: Winsorized at 5th/95th percentile
        x_win = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
        y_win = np.clip(y, np.percentile(y, 5), np.percentile(y, 95))
        r_win, p_win = sp_stats.spearmanr(x_win, y_win)
        variants.append(
            {
                "variant": "winsorized_5_95",
                "r": round(float(r_win), 4),
                "p_value": round(float(p_win), 6),
                "n": len(x_win),
                "direction": "positive" if r_win > 0 else "negative",
            }
        )

        # Determine robustness: all variants agree on direction and significance
        directions = [
            v["direction"] for v in variants if v["direction"] not in ("insufficient_data",)
        ]
        r_values = [abs(v["r"]) for v in variants if v["r"] is not None]

        direction_consistent = len(set(directions)) <= 1
        # All variants should be significant (or all not)
        sig_flags = [v["p_value"] < 0.05 for v in variants if v["p_value"] is not None]
        significance_consistent = len(set(sig_flags)) <= 1

        # Magnitude stability: max/min ratio of |r| values
        if r_values and min(r_values) > 0:
            magnitude_ratio = max(r_values) / min(r_values)
        else:
            magnitude_ratio = float("inf")
        magnitude_stable = magnitude_ratio < 2.0

        robust = direction_consistent and significance_consistent and magnitude_stable

        if robust:
            recommendation = (
                "Finding is robust: direction, significance, and magnitude are "
                "consistent across all variants (original, Pearson, outlier-free, "
                "winsorized). Safe to report as a confirmed finding."
            )
        else:
            issues = []
            if not direction_consistent:
                issues.append("direction reverses across variants")
            if not significance_consistent:
                issues.append("significance is inconsistent across variants")
            if not magnitude_stable:
                issues.append(f"magnitude is unstable (ratio={magnitude_ratio:.1f}x)")
            recommendation = (
                f"Finding is NOT robust: {'; '.join(issues)}. "
                f"Report with appropriate caveats, or investigate which "
                f"variant best represents the true relationship."
            )

        return ToolResult(
            success=True,
            data={
                "robust": robust,
                "direction_consistent": direction_consistent,
                "significance_consistent": significance_consistent,
                "magnitude_stable": magnitude_stable,
                "magnitude_ratio": (
                    round(magnitude_ratio, 2) if magnitude_ratio != float("inf") else None
                ),
                "variants": variants,
                "recommendation": recommendation,
            },
            summary=(
                f"Robustness: {'ROBUST' if robust else 'NOT ROBUST'} — "
                f"{'direction consistent' if direction_consistent else 'DIRECTION VARIES'}, "
                f"{'significance consistent' if significance_consistent else 'SIGNIFICANCE VARIES'}, "
                f"{'magnitude stable' if magnitude_stable else 'MAGNITUDE UNSTABLE'}."
            ),
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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
