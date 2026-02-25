"""
Tool: statistical_analysis

Performs rigorous statistical analyses on data provided by the agent.
Supports descriptive statistics, distribution analysis, hypothesis testing,
correlation analysis, effect size calculation, data quality assessment,
and assumption checking for statistical methods.

The agent provides data (typically from SQL query results) and specifies
what analysis to run. The tool performs computations using scipy/numpy/pandas
and returns structured results with interpretations.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from versifai.core.tools.base import BaseTool, ToolResult


class StatisticalAnalysisTool(BaseTool):
    """
    Performs statistical analyses on tabular data.

    The agent sends data (as list of dicts from SQL results) along with
    an analysis_type and relevant parameters. The tool runs the computation
    and returns structured results with test statistics, p-values,
    effect sizes, and interpretive guidance.
    """

    @property
    def name(self) -> str:
        return "statistical_analysis"

    @property
    def description(self) -> str:
        return (
            "Perform statistical analysis on data. Analysis types:\n"
            "- 'describe': Descriptive statistics (mean, median, std, skew, kurtosis, quantiles)\n"
            "- 'distribution': Distribution analysis with normality tests (Shapiro-Wilk, D'Agostino)\n"
            "- 'hypothesis_test': Statistical tests (ttest, mannwhitney, chi_square, anova, kruskal)\n"
            "- 'correlation': Correlation matrix with p-values (Pearson or Spearman)\n"
            "- 'effect_size': Effect size between groups (Cohen's d, rank-biserial)\n"
            "- 'data_quality': Missing rates, outlier detection, cardinality, value ranges\n"
            "- 'assumption_check': Check assumptions for a method (normality, homoscedasticity, etc.)\n"
            "- 'bayesian_test': Bayesian hypothesis testing — posterior distributions, credible "
            "intervals, probability of direction, ROPE analysis, Bayes factors. Methods: "
            "bayesian_ttest (group comparison), bayesian_proportion (rate comparison), "
            "bayesian_correlation. USE THIS when sample sizes are small, when you have "
            "informative priors from published research, or when you need probability "
            "statements rather than p-values.\n\n"
            "Data should be a list of row dicts from SQL results. "
            "Returns structured results with statistics, p-values, and interpretations."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "description": (
                        "Type of analysis: 'describe', 'distribution', 'hypothesis_test', "
                        "'correlation', 'effect_size', 'data_quality', 'assumption_check'."
                    ),
                },
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Data as list of row dicts from SQL query results.",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names to analyze. If omitted, all numeric columns are used.",
                },
                "group_column": {
                    "type": "string",
                    "description": "Column for grouping (used in hypothesis_test, effect_size).",
                },
                "value_column": {
                    "type": "string",
                    "description": "Column with values to test (used in hypothesis_test, effect_size).",
                },
                "method": {
                    "type": "string",
                    "description": (
                        "Specific method. For hypothesis_test: 'ttest_ind', 'ttest_rel', "
                        "'mannwhitney', 'chi_square', 'anova', 'kruskal'. "
                        "For correlation: 'pearson', 'spearman'. "
                        "For assumption_check: 'regression', 'ttest', 'anova', 'chi_square'. "
                        "For bayesian_test: 'bayesian_ttest', 'bayesian_proportion', "
                        "'bayesian_correlation'."
                    ),
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for intervals (default 0.95).",
                },
                "prior": {
                    "type": "object",
                    "description": (
                        "Bayesian prior specification (for bayesian_test only). "
                        "For bayesian_ttest: {'mean': 0, 'std': 1} (prior on mean difference). "
                        "For bayesian_proportion: {'alpha': 1, 'beta': 1} (Beta prior on each proportion). "
                        "For bayesian_correlation: {'mean': 0, 'std': 1} (prior on Fisher-z). "
                        "Use informative priors from published research when available — "
                        "especially for low-N analyses."
                    ),
                },
                "rope": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Region of Practical Equivalence [low, high] for bayesian_test. "
                        "Effects inside ROPE are 'practically zero'. "
                        "Defaults to ±0.1 * pooled SD for bayesian_ttest."
                    ),
                },
            },
            "required": ["analysis_type", "data"],
        }

    def _execute(
        self,
        analysis_type: str = "",
        data: list[dict] | None = None,
        columns: list[str] | None = None,
        group_column: str = "",
        value_column: str = "",
        method: str = "",
        confidence_level: float = 0.95,
        prior: dict | None = None,
        rope: list[float] | None = None,
        **kwargs,
    ) -> ToolResult:
        if not analysis_type:
            return ToolResult(success=False, error="Missing 'analysis_type'.")
        if not data:
            return ToolResult(success=False, error="Missing 'data'.")

        df = pd.DataFrame(data)
        if df.empty:
            return ToolResult(success=False, error="Data is empty.")

        dispatch = {
            "describe": self._describe,
            "distribution": self._distribution,
            "hypothesis_test": self._hypothesis_test,
            "correlation": self._correlation,
            "effect_size": self._effect_size,
            "data_quality": self._data_quality,
            "assumption_check": self._assumption_check,
            "bayesian_test": self._bayesian_test,
        }

        handler = dispatch.get(analysis_type)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown analysis_type '{analysis_type}'. Use: {list(dispatch.keys())}",
            )

        return handler(
            df=df,
            columns=columns,
            group_column=group_column,
            value_column=value_column,
            method=method,
            confidence_level=confidence_level,
            prior=prior,
            rope=rope,
        )

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def _describe(self, df, columns, **kwargs) -> ToolResult:
        numeric = df[columns] if columns else df.select_dtypes(include="number")
        if numeric.empty:
            return ToolResult(success=False, error="No numeric columns found in data.")

        from scipy import stats as sp_stats

        results = {}
        for col in numeric.columns:
            series = numeric[col].dropna()
            if len(series) == 0:
                continue
            results[col] = {
                "count": int(len(series)),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "q25": round(float(series.quantile(0.25)), 4),
                "q75": round(float(series.quantile(0.75)), 4),
                "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 4),
                "skewness": round(float(sp_stats.skew(series)), 4),
                "kurtosis": round(float(sp_stats.kurtosis(series)), 4),
                "cv": round(float(series.std() / series.mean()), 4) if series.mean() != 0 else None,
            }

        return ToolResult(
            success=True,
            data=results,
            summary=f"Descriptive statistics for {len(results)} columns, {len(df)} rows.",
        )

    # ------------------------------------------------------------------
    # Distribution analysis
    # ------------------------------------------------------------------

    def _distribution(self, df, columns, **kwargs) -> ToolResult:
        from scipy import stats as sp_stats

        target_cols = columns or list(df.select_dtypes(include="number").columns)
        if not target_cols:
            return ToolResult(success=False, error="No numeric columns found.")

        results = {}
        for col in target_cols:
            series = df[col].dropna()
            if len(series) < 8:
                results[col] = {"error": f"Too few values ({len(series)}) for distribution tests."}
                continue

            entry: dict[str, Any] = {
                "n": int(len(series)),
                "skewness": round(float(sp_stats.skew(series)), 4),
                "kurtosis": round(float(sp_stats.kurtosis(series)), 4),
            }

            # Shapiro-Wilk (best for n < 5000)
            if len(series) <= 5000:
                stat, p = sp_stats.shapiro(series)
                entry["shapiro_wilk"] = {
                    "statistic": round(float(stat), 6),
                    "p_value": round(float(p), 6),
                    "normal": p > 0.05,
                }

            # D'Agostino-Pearson (works for larger samples)
            if len(series) >= 20:
                stat, p = sp_stats.normaltest(series)
                entry["dagostino"] = {
                    "statistic": round(float(stat), 6),
                    "p_value": round(float(p), 6),
                    "normal": p > 0.05,
                }

            # Anderson-Darling
            try:
                ad_result = sp_stats.anderson(series, dist="norm")
                # Handle both old and new scipy API
                if hasattr(ad_result, "critical_values"):
                    entry["anderson_darling"] = {
                        "statistic": round(float(ad_result.statistic), 6),
                        "critical_values": {
                            f"{sl}%": round(float(cv), 6)
                            for sl, cv in zip(
                                ad_result.significance_level, ad_result.critical_values
                            )
                        },
                        "normal_at_5pct": ad_result.statistic < ad_result.critical_values[2],
                    }
                elif hasattr(ad_result, "pvalue"):
                    entry["anderson_darling"] = {
                        "statistic": round(float(ad_result.statistic), 6),
                        "p_value": round(float(ad_result.pvalue), 6),
                        "normal_at_5pct": ad_result.pvalue > 0.05,
                    }
            except Exception:
                pass  # Skip if Anderson-Darling not available

            # Distribution fitting - try common distributions
            best_fit = None
            best_sse = float("inf")
            for dist_name in ["norm", "lognorm", "expon", "gamma", "beta"]:
                try:
                    dist = getattr(sp_stats, dist_name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        params = dist.fit(series)
                    # KS test
                    ks_stat, ks_p = sp_stats.kstest(series, dist_name, args=params)
                    if ks_p > (best_sse if best_sse < 1 else 0):
                        best_fit = {
                            "distribution": dist_name,
                            "params": [round(float(p), 6) for p in params],
                            "ks_statistic": round(float(ks_stat), 6),
                            "ks_p_value": round(float(ks_p), 6),
                        }
                        best_sse = ks_p
                except Exception:
                    continue

            if best_fit:
                entry["best_fit"] = best_fit

            # Percentile breakpoints for binning
            entry["percentiles"] = {
                f"p{p}": round(float(series.quantile(p / 100)), 4)
                for p in [5, 10, 25, 50, 75, 90, 95]
            }

            is_normal = entry.get("shapiro_wilk", entry.get("dagostino", {})).get("normal", None)
            if is_normal is True:
                entry["recommendation"] = (
                    "Data appears normally distributed — parametric tests appropriate."
                )
            elif is_normal is False:
                entry["recommendation"] = (
                    "Data is NOT normally distributed — use non-parametric tests "
                    "(Mann-Whitney, Kruskal-Wallis) or transform the data."
                )

            results[col] = entry

        return ToolResult(
            success=True,
            data=results,
            summary=f"Distribution analysis for {len(results)} columns.",
        )

    # ------------------------------------------------------------------
    # Hypothesis testing
    # ------------------------------------------------------------------

    def _hypothesis_test(
        self, df, group_column, value_column, method, confidence_level, **kwargs
    ) -> ToolResult:
        from scipy import stats as sp_stats

        if not group_column:
            return ToolResult(
                success=False, error="'group_column' is required for hypothesis testing."
            )
        if not value_column:
            return ToolResult(
                success=False, error="'value_column' is required for hypothesis testing."
            )
        if group_column not in df.columns:
            return ToolResult(
                success=False,
                error=f"Group column '{group_column}' not found. Available: {list(df.columns)}",
            )
        if value_column not in df.columns:
            return ToolResult(
                success=False,
                error=f"Value column '{value_column}' not found. Available: {list(df.columns)}",
            )

        groups = df.groupby(group_column)[value_column].apply(lambda x: x.dropna().values)
        group_names = list(groups.index)

        if len(group_names) < 2:
            return ToolResult(
                success=False, error=f"Need at least 2 groups, found {len(group_names)}."
            )

        # Auto-select method if not specified
        if not method:
            if len(group_names) == 2:
                # Check normality to decide parametric vs non-parametric
                try:
                    normal_checks = [
                        sp_stats.shapiro(g)[1] > 0.05 for g in groups.values if len(g) >= 8
                    ]
                    method = "ttest_ind" if all(normal_checks) else "mannwhitney"
                except Exception:
                    method = "mannwhitney"
            else:
                method = "anova"

        result: dict[str, Any] = {
            "test": method,
            "group_column": group_column,
            "value_column": value_column,
            "n_groups": len(group_names),
            "groups": {},
        }

        # Group summaries
        for name in group_names:
            g = groups[name]
            result["groups"][str(name)] = {
                "n": int(len(g)),
                "mean": round(float(np.mean(g)), 4),
                "median": round(float(np.median(g)), 4),
                "std": round(float(np.std(g, ddof=1)), 4) if len(g) > 1 else 0,
            }

        alpha = 1 - confidence_level

        # Run the test
        if method == "ttest_ind" and len(group_names) == 2:
            g1, g2 = groups.values[0], groups.values[1]
            stat, p = sp_stats.ttest_ind(g1, g2, equal_var=False)
            result["statistic"] = round(float(stat), 6)
            result["p_value"] = round(float(p), 6)
            result["significant"] = p < alpha
            # Cohen's d
            pooled_std = np.sqrt((np.std(g1, ddof=1) ** 2 + np.std(g2, ddof=1) ** 2) / 2)
            d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
            result["cohens_d"] = round(float(d), 4)
            result["effect_interpretation"] = _interpret_cohens_d(abs(d))

        elif method == "ttest_rel" and len(group_names) == 2:
            g1, g2 = groups.values[0], groups.values[1]
            min_len = min(len(g1), len(g2))
            stat, p = sp_stats.ttest_rel(g1[:min_len], g2[:min_len])
            result["statistic"] = round(float(stat), 6)
            result["p_value"] = round(float(p), 6)
            result["significant"] = p < alpha

        elif method == "mannwhitney" and len(group_names) == 2:
            g1, g2 = groups.values[0], groups.values[1]
            stat, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            result["statistic"] = round(float(stat), 6)
            result["p_value"] = round(float(p), 6)
            result["significant"] = p < alpha
            # Rank-biserial correlation as effect size
            n1, n2 = len(g1), len(g2)
            r = 1 - (2 * stat) / (n1 * n2)
            result["rank_biserial_r"] = round(float(r), 4)
            result["effect_interpretation"] = _interpret_r(abs(r))

        elif method == "anova":
            stat, p = sp_stats.f_oneway(*groups.values)
            result["statistic"] = round(float(stat), 6)
            result["p_value"] = round(float(p), 6)
            result["significant"] = p < alpha
            # Eta-squared
            grand_mean = df[value_column].dropna().mean()
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values)
            ss_total = sum((v - grand_mean) ** 2 for g in groups.values for v in g)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            result["eta_squared"] = round(float(eta_sq), 4)
            result["effect_interpretation"] = _interpret_eta_squared(eta_sq)

        elif method == "kruskal":
            stat, p = sp_stats.kruskal(*groups.values)
            result["statistic"] = round(float(stat), 6)
            result["p_value"] = round(float(p), 6)
            result["significant"] = p < alpha

        elif method == "chi_square":
            # For chi-square, create contingency table
            ct = pd.crosstab(df[group_column], df[value_column])
            chi2, p, dof, expected = sp_stats.chi2_contingency(ct)
            result["statistic"] = round(float(chi2), 6)
            result["p_value"] = round(float(p), 6)
            result["degrees_of_freedom"] = int(dof)
            result["significant"] = p < alpha
            # Cramér's V
            n = ct.sum().sum()
            k = min(ct.shape) - 1
            v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0
            result["cramers_v"] = round(float(v), 4)
            result["effect_interpretation"] = _interpret_cramers_v(v)
        else:
            return ToolResult(
                success=False,
                error=f"Unknown method '{method}'. Use: ttest_ind, ttest_rel, mannwhitney, anova, kruskal, chi_square.",
            )

        result["interpretation"] = (
            f"The difference is {'statistically significant' if result['significant'] else 'NOT statistically significant'} "
            f"at the {confidence_level * 100:.0f}% confidence level (p={result['p_value']})."
        )

        return ToolResult(
            success=True,
            data=result,
            summary=f"{method} test: p={result['p_value']}, significant={result['significant']}",
        )

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def _correlation(self, df, columns, method, confidence_level, **kwargs) -> ToolResult:
        from scipy import stats as sp_stats

        method = method or "pearson"
        numeric = (
            df[columns].select_dtypes(include="number")
            if columns
            else df.select_dtypes(include="number")
        )
        if numeric.shape[1] < 2:
            return ToolResult(
                success=False, error="Need at least 2 numeric columns for correlation."
            )

        cols = list(numeric.columns)
        n = len(numeric.dropna())
        results = {
            "method": method,
            "n_observations": n,
            "correlations": [],
        }

        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                valid = numeric[[c1, c2]].dropna()
                if len(valid) < 5:
                    continue
                if method == "pearson":
                    r, p = sp_stats.pearsonr(valid[c1], valid[c2])
                else:
                    r, p = sp_stats.spearmanr(valid[c1], valid[c2])

                results["correlations"].append(
                    {
                        "var1": c1,
                        "var2": c2,
                        "r": round(float(r), 4),
                        "p_value": round(float(p), 6),
                        "significant": p < (1 - confidence_level),
                        "r_squared": round(float(r**2), 4),
                        "strength": _interpret_r(abs(r)),
                        "direction": "positive" if r > 0 else "negative",
                    }
                )

        # Sort by absolute correlation strength
        results["correlations"].sort(key=lambda x: abs(x["r"]), reverse=True)

        # Top finding
        if results["correlations"]:
            top = results["correlations"][0]
            results["strongest"] = (
                f"{top['var1']} and {top['var2']}: r={top['r']} ({top['strength']} {top['direction']})"
            )

        return ToolResult(
            success=True,
            data=results,
            summary=f"{method.title()} correlations for {len(cols)} variables ({len(results['correlations'])} pairs).",
        )

    # ------------------------------------------------------------------
    # Effect size
    # ------------------------------------------------------------------

    def _effect_size(self, df, group_column, value_column, **kwargs) -> ToolResult:
        from scipy import stats as sp_stats

        if not group_column or not value_column:
            return ToolResult(
                success=False, error="'group_column' and 'value_column' are required."
            )

        groups = df.groupby(group_column)[value_column].apply(lambda x: x.dropna().values)
        group_names = list(groups.index)

        if len(group_names) < 2:
            return ToolResult(success=False, error="Need at least 2 groups.")

        results = {"group_column": group_column, "value_column": value_column, "comparisons": []}

        for i, name1 in enumerate(group_names):
            for name2 in group_names[i + 1 :]:
                g1, g2 = groups[name1], groups[name2]
                if len(g1) < 2 or len(g2) < 2:
                    continue

                mean_diff = float(np.mean(g1) - np.mean(g2))
                pooled_std = np.sqrt((np.std(g1, ddof=1) ** 2 + np.std(g2, ddof=1) ** 2) / 2)
                d = mean_diff / pooled_std if pooled_std > 0 else 0

                # Confidence interval for Cohen's d (approximate)
                se_d = np.sqrt(
                    (len(g1) + len(g2)) / (len(g1) * len(g2)) + d**2 / (2 * (len(g1) + len(g2)))
                )
                z = sp_stats.norm.ppf(0.975)
                ci_low = d - z * se_d
                ci_high = d + z * se_d

                # Percentage difference
                base_mean = float(np.mean(g2))
                pct_diff = (mean_diff / base_mean * 100) if base_mean != 0 else None

                results["comparisons"].append(
                    {
                        "group1": str(name1),
                        "group2": str(name2),
                        "mean1": round(float(np.mean(g1)), 4),
                        "mean2": round(float(np.mean(g2)), 4),
                        "mean_difference": round(mean_diff, 4),
                        "pct_difference": round(pct_diff, 2) if pct_diff is not None else None,
                        "cohens_d": round(float(d), 4),
                        "cohens_d_ci": [round(float(ci_low), 4), round(float(ci_high), 4)],
                        "interpretation": _interpret_cohens_d(abs(d)),
                    }
                )

        return ToolResult(
            success=True,
            data=results,
            summary=f"Effect sizes for {len(results['comparisons'])} group comparisons.",
        )

    # ------------------------------------------------------------------
    # Data quality assessment
    # ------------------------------------------------------------------

    def _data_quality(self, df, columns, **kwargs) -> ToolResult:
        target_cols = columns or list(df.columns)
        results = {"n_rows": len(df), "n_columns": len(target_cols), "columns": {}}

        for col in target_cols:
            if col not in df.columns:
                continue
            series = df[col]
            entry: dict[str, Any] = {
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(series.isna().mean() * 100, 2),
                "unique_values": int(series.nunique()),
                "cardinality_ratio": round(series.nunique() / len(series), 4)
                if len(series) > 0
                else 0,
            }

            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    lower_fence = q1 - 1.5 * iqr
                    upper_fence = q3 + 1.5 * iqr
                    outliers = clean[(clean < lower_fence) | (clean > upper_fence)]
                    entry["outlier_count"] = int(len(outliers))
                    entry["outlier_pct"] = round(len(outliers) / len(clean) * 100, 2)
                    entry["range"] = [round(float(clean.min()), 4), round(float(clean.max()), 4)]
                    entry["zero_count"] = int((clean == 0).sum())
                    entry["negative_count"] = int((clean < 0).sum())
            else:
                # Categorical column
                top_values = series.value_counts().head(5)
                entry["top_values"] = {str(k): int(v) for k, v in top_values.items()}

            # Quality flags
            flags = []
            if entry["missing_pct"] > 20:
                flags.append("HIGH_MISSING")
            if entry.get("outlier_pct", 0) > 5:
                flags.append("MANY_OUTLIERS")
            if entry["cardinality_ratio"] > 0.95 and not pd.api.types.is_numeric_dtype(series):
                flags.append("NEAR_UNIQUE")
            if entry["cardinality_ratio"] < 0.01 and pd.api.types.is_numeric_dtype(series):
                flags.append("LOW_VARIANCE")
            entry["quality_flags"] = flags

            results["columns"][col] = entry

        # Overall assessment
        flagged = sum(1 for c in results["columns"].values() if c["quality_flags"])
        results["summary"] = {
            "columns_with_issues": flagged,
            "total_missing_cells": sum(c["missing_count"] for c in results["columns"].values()),
            "total_missing_pct": round(
                sum(c["missing_count"] for c in results["columns"].values())
                / (len(df) * len(target_cols))
                * 100,
                2,
            )
            if len(df) * len(target_cols) > 0
            else 0,
        }

        return ToolResult(
            success=True,
            data=results,
            summary=f"Data quality for {len(target_cols)} columns: {flagged} with issues.",
        )

    # ------------------------------------------------------------------
    # Bayesian hypothesis testing
    # ------------------------------------------------------------------

    def _bayesian_test(
        self,
        df,
        group_column,
        value_column,
        method,
        confidence_level,
        prior=None,
        rope=None,
        columns=None,
        **kwargs,
    ) -> ToolResult:
        """Bayesian hypothesis testing with posterior distributions and Bayes factors.

        Methods:
            bayesian_ttest — Compare two group means (conjugate Normal model).
            bayesian_proportion — Compare two proportions (Beta-Binomial).
            bayesian_correlation — Test correlation strength (Fisher-z).
        """

        prior = prior or {}
        n_samples = 10_000

        if not method:
            method = "bayesian_ttest"

        if method == "bayesian_ttest":
            return self._bayesian_ttest(
                df,
                group_column,
                value_column,
                prior,
                rope,
                n_samples,
                confidence_level,
            )
        elif method == "bayesian_proportion":
            return self._bayesian_proportion(
                df,
                group_column,
                value_column,
                prior,
                n_samples,
                confidence_level,
            )
        elif method == "bayesian_correlation":
            cols = columns or []
            return self._bayesian_correlation(
                df,
                cols,
                prior,
                n_samples,
                confidence_level,
            )
        else:
            return ToolResult(
                success=False,
                error=(
                    f"Unknown bayesian method '{method}'. "
                    "Use: bayesian_ttest, bayesian_proportion, bayesian_correlation."
                ),
            )

    def _bayesian_ttest(
        self,
        df,
        group_column,
        value_column,
        prior,
        rope,
        n_samples,
        confidence_level,
    ) -> ToolResult:
        """Bayesian two-group comparison using conjugate Normal-InverseGamma model."""
        from scipy import stats as sp_stats

        if not group_column or not value_column:
            return ToolResult(
                success=False,
                error="'group_column' and 'value_column' required for bayesian_ttest.",
            )

        groups = df.groupby(group_column)[value_column].apply(lambda x: x.dropna().values)
        group_names = list(groups.index)

        if len(group_names) != 2:
            return ToolResult(
                success=False,
                error=f"bayesian_ttest requires exactly 2 groups, found {len(group_names)}.",
            )

        g1, g2 = groups.values[0].astype(float), groups.values[1].astype(float)
        n1, n2 = len(g1), len(g2)

        if n1 < 2 or n2 < 2:
            return ToolResult(success=False, error="Each group needs at least 2 observations.")

        # Group statistics
        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        s1, s2 = float(np.std(g1, ddof=1)), float(np.std(g2, ddof=1))
        pooled_std = np.sqrt((s1**2 + s2**2) / 2)

        # Prior on the mean difference
        prior_mean = prior.get("mean", 0.0)
        prior_std = prior.get("std", pooled_std * 2)  # weakly informative default
        prior_type = "informative" if "mean" in prior else "weakly_informative"

        # Posterior for each group mean using conjugate normal model
        # With known-ish variance (plug-in), posterior mean is normal
        # For group 1: posterior ~ N(weighted_mean, posterior_var)
        # We use a simple approach: sample from t-distributions for each
        # group mean (accounts for variance uncertainty), then compute difference.

        # Degrees of freedom
        df1, df2 = n1 - 1, n2 - 1

        # Standard error of each mean
        se1 = s1 / np.sqrt(n1)
        se2 = s2 / np.sqrt(n2)

        # Sample posterior of each group mean from t-distribution
        post_m1 = sp_stats.t.rvs(df=df1, loc=m1, scale=se1, size=n_samples)
        post_m2 = sp_stats.t.rvs(df=df2, loc=m2, scale=se2, size=n_samples)

        # Posterior of the difference
        post_diff = post_m1 - post_m2

        # If informative prior is specified, blend via precision weighting
        if prior_type == "informative":
            data_precision = 1.0 / (se1**2 + se2**2)
            prior_precision = 1.0 / (prior_std**2)
            total_precision = data_precision + prior_precision
            posterior_mean_diff = (
                data_precision * (m1 - m2) + prior_precision * prior_mean
            ) / total_precision
            posterior_std_diff = 1.0 / np.sqrt(total_precision)
            # Resample from the blended posterior
            post_diff = sp_stats.norm.rvs(
                loc=posterior_mean_diff, scale=posterior_std_diff, size=n_samples
            )

        # HDI (highest density interval) — use percentile-based for simplicity
        alpha = 1 - confidence_level
        hdi_low = float(np.percentile(post_diff, alpha / 2 * 100))
        hdi_high = float(np.percentile(post_diff, (1 - alpha / 2) * 100))

        # Probability of direction
        prob_positive = float(np.mean(post_diff > 0))
        prob_direction = max(prob_positive, 1 - prob_positive)

        # ROPE analysis
        if rope is None:
            rope = [-0.1 * pooled_std, 0.1 * pooled_std]
        rope_probability = float(np.mean((post_diff >= rope[0]) & (post_diff <= rope[1])))

        # Effect size (posterior Cohen's d)
        post_d = post_diff / pooled_std if pooled_std > 0 else post_diff
        d_mean = float(np.mean(post_d))
        d_hdi_low = float(np.percentile(post_d, alpha / 2 * 100))
        d_hdi_high = float(np.percentile(post_d, (1 - alpha / 2) * 100))

        # Bayes Factor (Savage-Dickey density ratio at null=0)
        # BF10 = prior_density_at_0 / posterior_density_at_0
        try:
            prior_density_0 = sp_stats.norm.pdf(0, loc=prior_mean, scale=prior_std)
            # Estimate posterior density at 0 via KDE
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(post_diff)
            posterior_density_0 = float(kde(0.0)[0])
            bf10 = (
                prior_density_0 / posterior_density_0
                if posterior_density_0 > 1e-10
                else float("inf")
            )
        except Exception:
            bf10 = None

        # Frequentist comparison
        freq_stat, freq_p = sp_stats.ttest_ind(g1, g2, equal_var=False)
        freq_d = (m1 - m2) / pooled_std if pooled_std > 0 else 0

        # Interpretation
        direction_word = "greater" if prob_positive > 0.5 else "less"
        interp_parts = [
            f"{prob_direction * 100:.0f}% probability that {group_names[0]} is "
            f"{direction_word} than {group_names[1]}.",
        ]
        if rope_probability > 0.05:
            interp_parts.append(
                f"{rope_probability * 100:.0f}% probability the effect is practically zero "
                f"(inside ROPE [{rope[0]:.2f}, {rope[1]:.2f}])."
            )
        interp_parts.append(
            f"Posterior effect size d={d_mean:.3f} "
            f"({_interpret_cohens_d(abs(d_mean))}), "
            f"{confidence_level * 100:.0f}% HDI [{d_hdi_low:.3f}, {d_hdi_high:.3f}]."
        )

        result = {
            "method": "bayesian_ttest",
            "group_column": group_column,
            "value_column": value_column,
            "groups": {
                str(group_names[0]): {"n": n1, "mean": round(m1, 4), "std": round(s1, 4)},
                str(group_names[1]): {"n": n2, "mean": round(m2, 4), "std": round(s2, 4)},
            },
            "prior": {
                "type": prior_type,
                "mean": round(prior_mean, 4),
                "std": round(prior_std, 4),
            },
            "posterior": {
                "mean_difference": round(float(np.mean(post_diff)), 4),
                "std_difference": round(float(np.std(post_diff)), 4),
                f"hdi_{confidence_level * 100:.0f}": [round(hdi_low, 4), round(hdi_high, 4)],
                "probability_of_direction": round(prob_direction, 4),
                "probability_positive": round(prob_positive, 4),
                "rope": [round(rope[0], 4), round(rope[1], 4)],
                "rope_probability": round(rope_probability, 4),
                "effect_size_posterior_mean": round(d_mean, 4),
                "effect_size_interpretation": _interpret_cohens_d(abs(d_mean)),
                f"effect_size_hdi_{confidence_level * 100:.0f}": [
                    round(d_hdi_low, 4),
                    round(d_hdi_high, 4),
                ],
            },
            "bayes_factor_10": round(bf10, 2)
            if bf10 is not None and bf10 != float("inf")
            else bf10,
            "bf_interpretation": _interpret_bf(bf10) if bf10 is not None else "Could not compute.",
            "frequentist_comparison": {
                "t_statistic": round(float(freq_stat), 4),
                "p_value": round(float(freq_p), 6),
                "cohens_d": round(float(freq_d), 4),
            },
            "interpretation": " ".join(interp_parts),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=(
                f"Bayesian t-test: P(direction)={prob_direction:.2f}, "
                f"posterior d={d_mean:.3f}, BF₁₀={bf10:.1f}"
                if bf10 and bf10 != float("inf")
                else f"Bayesian t-test: P(direction)={prob_direction:.2f}, posterior d={d_mean:.3f}"
            ),
        )

    def _bayesian_proportion(
        self,
        df,
        group_column,
        value_column,
        prior,
        n_samples,
        confidence_level,
    ) -> ToolResult:
        """Bayesian comparison of two proportions using Beta-Binomial conjugate model."""
        from scipy import stats as sp_stats

        if not group_column or not value_column:
            return ToolResult(
                success=False,
                error="'group_column' and 'value_column' required for bayesian_proportion.",
            )

        groups = df.groupby(group_column)[value_column].apply(lambda x: x.dropna().values)
        group_names = list(groups.index)

        if len(group_names) != 2:
            return ToolResult(
                success=False,
                error=f"bayesian_proportion requires exactly 2 groups, found {len(group_names)}.",
            )

        g1, g2 = groups.values[0].astype(float), groups.values[1].astype(float)

        # Interpret values as successes/failures (binary or rate)
        # If values are binary (0/1), count successes
        # If values are rates (0-1), convert to pseudo-counts
        if set(np.unique(g1)).issubset({0, 1}) and set(np.unique(g2)).issubset({0, 1}):
            s1, n1 = int(g1.sum()), len(g1)
            s2, n2 = int(g2.sum()), len(g2)
        else:
            # Treat as rates — use mean and N for pseudo-counts
            n1, n2 = len(g1), len(g2)
            s1 = int(round(np.mean(g1) * n1))
            s2 = int(round(np.mean(g2) * n2))

        # Beta prior
        prior_alpha = prior.get("alpha", 1.0)
        prior_beta = prior.get("beta", 1.0)
        prior_type = (
            "informative"
            if "alpha" in prior and prior.get("alpha", 1) != 1
            else "weakly_informative"
        )

        # Posterior Beta distributions
        post_alpha1 = prior_alpha + s1
        post_beta1 = prior_beta + (n1 - s1)
        post_alpha2 = prior_alpha + s2
        post_beta2 = prior_beta + (n2 - s2)

        # Sample posteriors
        post_p1 = sp_stats.beta.rvs(post_alpha1, post_beta1, size=n_samples)
        post_p2 = sp_stats.beta.rvs(post_alpha2, post_beta2, size=n_samples)
        post_diff = post_p1 - post_p2

        # HDI
        alpha_ci = 1 - confidence_level
        hdi_low = float(np.percentile(post_diff, alpha_ci / 2 * 100))
        hdi_high = float(np.percentile(post_diff, (1 - alpha_ci / 2) * 100))

        # Probability of direction
        prob_1_greater = float(np.mean(post_p1 > post_p2))

        result = {
            "method": "bayesian_proportion",
            "group_column": group_column,
            "value_column": value_column,
            "groups": {
                str(group_names[0]): {
                    "n": n1,
                    "successes": s1,
                    "observed_rate": round(s1 / n1, 4) if n1 > 0 else 0,
                    "posterior_mean": round(float(post_alpha1 / (post_alpha1 + post_beta1)), 4),
                },
                str(group_names[1]): {
                    "n": n2,
                    "successes": s2,
                    "observed_rate": round(s2 / n2, 4) if n2 > 0 else 0,
                    "posterior_mean": round(float(post_alpha2 / (post_alpha2 + post_beta2)), 4),
                },
            },
            "prior": {
                "type": prior_type,
                "alpha": round(prior_alpha, 2),
                "beta": round(prior_beta, 2),
            },
            "posterior": {
                "mean_difference": round(float(np.mean(post_diff)), 4),
                f"hdi_{confidence_level * 100:.0f}": [round(hdi_low, 4), round(hdi_high, 4)],
                "probability_1_greater": round(prob_1_greater, 4),
            },
            "interpretation": (
                f"{prob_1_greater * 100:.0f}% probability that {group_names[0]} "
                f"has a higher rate than {group_names[1]}. "
                f"Posterior difference: {np.mean(post_diff):.4f} "
                f"({confidence_level * 100:.0f}% HDI [{hdi_low:.4f}, {hdi_high:.4f}])."
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=(
                f"Bayesian proportion test: P({group_names[0]} > {group_names[1]})="
                f"{prob_1_greater:.2f}"
            ),
        )

    def _bayesian_correlation(
        self,
        df,
        columns,
        prior,
        n_samples,
        confidence_level,
    ) -> ToolResult:
        """Bayesian correlation test using Fisher-z transform with normal prior."""
        from scipy import stats as sp_stats

        if not columns or len(columns) < 2:
            return ToolResult(
                success=False,
                error="Need at least 2 column names in 'columns' for bayesian_correlation.",
            )

        col1, col2 = columns[0], columns[1]
        valid = df[[col1, col2]].dropna()
        n = len(valid)

        if n < 5:
            return ToolResult(success=False, error=f"Too few observations ({n}) for correlation.")

        x, y = valid[col1].astype(float).values, valid[col2].astype(float).values

        # Observed correlation
        r_obs, p_freq = sp_stats.pearsonr(x, y)

        # Fisher-z transform
        z_obs = np.arctanh(r_obs) if abs(r_obs) < 0.9999 else np.sign(r_obs) * 3.8
        se_z = 1.0 / np.sqrt(n - 3) if n > 3 else 1.0

        # Prior on Fisher-z
        prior_mean = prior.get("mean", 0.0)
        prior_std = prior.get("std", 1.0)
        prior_type = "informative" if "mean" in prior else "weakly_informative"

        # Posterior (conjugate normal-normal)
        data_precision = 1.0 / (se_z**2)
        prior_precision = 1.0 / (prior_std**2)
        total_precision = data_precision + prior_precision
        post_mean_z = (data_precision * z_obs + prior_precision * prior_mean) / total_precision
        post_std_z = 1.0 / np.sqrt(total_precision)

        # Sample posterior in z-space, transform back to r-space
        post_z = sp_stats.norm.rvs(loc=post_mean_z, scale=post_std_z, size=n_samples)
        post_r = np.tanh(post_z)  # inverse Fisher-z

        # HDI
        alpha_ci = 1 - confidence_level
        hdi_low = float(np.percentile(post_r, alpha_ci / 2 * 100))
        hdi_high = float(np.percentile(post_r, (1 - alpha_ci / 2) * 100))

        # Probability of direction
        prob_positive = float(np.mean(post_r > 0))
        prob_direction = max(prob_positive, 1 - prob_positive)

        # Bayes Factor (Savage-Dickey at r=0, i.e., z=0)
        try:
            prior_density_0 = sp_stats.norm.pdf(0, loc=prior_mean, scale=prior_std)
            posterior_density_0 = sp_stats.norm.pdf(0, loc=post_mean_z, scale=post_std_z)
            bf10 = (
                prior_density_0 / posterior_density_0
                if posterior_density_0 > 1e-10
                else float("inf")
            )
        except Exception:
            bf10 = None

        result = {
            "method": "bayesian_correlation",
            "var1": col1,
            "var2": col2,
            "n": n,
            "observed_r": round(float(r_obs), 4),
            "prior": {
                "type": prior_type,
                "mean_z": round(prior_mean, 4),
                "std_z": round(prior_std, 4),
            },
            "posterior": {
                "mean_r": round(float(np.mean(post_r)), 4),
                "std_r": round(float(np.std(post_r)), 4),
                f"hdi_{confidence_level * 100:.0f}": [round(hdi_low, 4), round(hdi_high, 4)],
                "probability_of_direction": round(prob_direction, 4),
                "probability_positive": round(prob_positive, 4),
                "r_squared_posterior_mean": round(float(np.mean(post_r**2)), 4),
            },
            "bayes_factor_10": round(bf10, 2)
            if bf10 is not None and bf10 != float("inf")
            else bf10,
            "bf_interpretation": _interpret_bf(bf10) if bf10 is not None else "Could not compute.",
            "frequentist_comparison": {
                "pearson_r": round(float(r_obs), 4),
                "p_value": round(float(p_freq), 6),
            },
            "interpretation": (
                f"{prob_direction * 100:.0f}% probability that the correlation between "
                f"{col1} and {col2} is {'positive' if prob_positive > 0.5 else 'negative'}. "
                f"Posterior r={np.mean(post_r):.3f} "
                f"({confidence_level * 100:.0f}% HDI [{hdi_low:.3f}, {hdi_high:.3f}])."
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=(
                f"Bayesian correlation: posterior r={np.mean(post_r):.3f}, "
                f"P(direction)={prob_direction:.2f}, BF₁₀={bf10:.1f}"
                if bf10 and bf10 != float("inf")
                else f"Bayesian correlation: posterior r={np.mean(post_r):.3f}, P(direction)={prob_direction:.2f}"
            ),
        )

    # ------------------------------------------------------------------
    # Assumption checking
    # ------------------------------------------------------------------

    def _assumption_check(
        self, df, columns, method, value_column, group_column, **kwargs
    ) -> ToolResult:
        from scipy import stats as sp_stats

        method = method or "regression"
        target_cols = columns or list(df.select_dtypes(include="number").columns)
        results = {"method": method, "assumptions": {}}

        if method in ("regression", "ttest", "anova"):
            # Normality check
            for col in target_cols[:5]:  # Limit to avoid excessive output
                series = df[col].dropna()
                if len(series) < 8:
                    continue
                stat, p = (
                    sp_stats.shapiro(series) if len(series) <= 5000 else sp_stats.normaltest(series)
                )
                results["assumptions"][f"normality_{col}"] = {
                    "test": "shapiro_wilk" if len(series) <= 5000 else "dagostino",
                    "statistic": round(float(stat), 6),
                    "p_value": round(float(p), 6),
                    "passed": p > 0.05,
                    "note": "Normal"
                    if p > 0.05
                    else "Non-normal — consider transformation or non-parametric test",
                }

        if method in ("ttest", "anova") and group_column and value_column:
            # Homogeneity of variance (Levene's test)
            groups = [g.dropna().values for _, g in df.groupby(group_column)[value_column]]
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                stat, p = sp_stats.levene(*groups)
                results["assumptions"]["homogeneity_of_variance"] = {
                    "test": "levene",
                    "statistic": round(float(stat), 6),
                    "p_value": round(float(p), 6),
                    "passed": p > 0.05,
                    "note": "Equal variances"
                    if p > 0.05
                    else "Unequal variances — use Welch's t-test or non-parametric",
                }

        if method == "regression" and len(target_cols) >= 2:
            # Multicollinearity check (VIF approximation via correlation)
            numeric = df[target_cols].select_dtypes(include="number").dropna()
            if numeric.shape[1] >= 2:
                corr_matrix = numeric.corr()
                high_corr = []
                cols_list = list(corr_matrix.columns)
                for i, c1 in enumerate(cols_list):
                    for c2 in cols_list[i + 1 :]:
                        r = abs(corr_matrix.loc[c1, c2])
                        if r > 0.7:
                            high_corr.append({"var1": c1, "var2": c2, "r": round(float(r), 4)})
                results["assumptions"]["multicollinearity"] = {
                    "test": "correlation_threshold",
                    "threshold": 0.7,
                    "high_correlations": high_corr,
                    "passed": len(high_corr) == 0,
                    "note": (
                        "No multicollinearity detected"
                        if not high_corr
                        else f"{len(high_corr)} variable pairs have |r| > 0.7 — consider removing or combining"
                    ),
                }

            # Sample size adequacy
            n = len(numeric)
            k = numeric.shape[1] - 1  # predictors
            results["assumptions"]["sample_size"] = {
                "n": n,
                "predictors": k,
                "ratio": round(n / k, 1) if k > 0 else None,
                "passed": (n / k > 10) if k > 0 else True,
                "note": f"n/k ratio = {round(n / k, 1) if k > 0 else 'N/A'}. Should be > 10, ideally > 20.",
            }

        if method == "chi_square" and group_column and value_column:
            ct = pd.crosstab(df[group_column], df[value_column])
            min_expected = ct.sum().sum() / (ct.shape[0] * ct.shape[1])
            results["assumptions"]["expected_frequencies"] = {
                "min_expected_approx": round(min_expected, 2),
                "passed": min_expected >= 5,
                "note": "Expected frequencies OK"
                if min_expected >= 5
                else "Expected frequencies < 5 — use Fisher's exact test",
            }

        # Overall verdict
        all_passed = all(a.get("passed", True) for a in results["assumptions"].values())
        results["overall"] = {
            "all_assumptions_met": all_passed,
            "recommendation": (
                f"Assumptions are met for {method} analysis."
                if all_passed
                else "Some assumptions violated — review individual checks and consider alternatives."
            ),
        }

        return ToolResult(
            success=True,
            data=results,
            summary=f"Assumption check for {method}: {'PASSED' if all_passed else 'ISSUES FOUND'}",
        )


# ------------------------------------------------------------------
# Interpretation helpers
# ------------------------------------------------------------------


def _interpret_cohens_d(d: float) -> str:
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _interpret_r(r: float) -> str:
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "weak"
    elif r < 0.5:
        return "moderate"
    elif r < 0.7:
        return "strong"
    else:
        return "very strong"


def _interpret_eta_squared(eta: float) -> str:
    if eta < 0.01:
        return "negligible"
    elif eta < 0.06:
        return "small"
    elif eta < 0.14:
        return "medium"
    else:
        return "large"


def _interpret_cramers_v(v: float) -> str:
    if v < 0.1:
        return "negligible"
    elif v < 0.3:
        return "small"
    elif v < 0.5:
        return "medium"
    else:
        return "large"


def _interpret_bf(bf: float | None) -> str:
    """Interpret Bayes Factor using Jeffreys' scale."""
    if bf is None:
        return "Could not compute"
    if bf == float("inf"):
        return "Decisive evidence for H1"
    if bf > 100:
        return f"Decisive evidence for H1 (BF₁₀={bf:.1f})"
    elif bf > 30:
        return f"Very strong evidence for H1 (BF₁₀={bf:.1f})"
    elif bf > 10:
        return f"Strong evidence for H1 (BF₁₀={bf:.1f})"
    elif bf > 3:
        return f"Moderate evidence for H1 (BF₁₀={bf:.1f})"
    elif bf > 1:
        return f"Anecdotal evidence for H1 (BF₁₀={bf:.1f})"
    elif bf > 1 / 3:
        return f"Anecdotal evidence for H0 (BF₁₀={bf:.2f})"
    elif bf > 1 / 10:
        return f"Moderate evidence for H0 (BF₁₀={bf:.2f})"
    elif bf > 1 / 30:
        return f"Strong evidence for H0 (BF₁₀={bf:.3f})"
    else:
        return f"Very strong evidence for H0 (BF₁₀={bf:.3f})"
