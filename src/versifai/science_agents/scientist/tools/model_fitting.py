"""
Tool: fit_model

Fits statistical and machine learning models on data provided by the agent.
Supports regression, classification, clustering, time series analysis,
feature importance, and counterfactual/what-if scenario modeling.

The agent provides data (from SQL query results), specifies a model type
and target/feature columns, and the tool fits the model, returns diagnostics,
predictions, and interpretive summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from versifai._utils.sql_data import fetch_sql_data
from versifai.core.tools.base import BaseTool, ToolResult


class ModelFittingTool(BaseTool):
    """
    Fits statistical and ML models on tabular data.

    The agent provides data, a model type, and column specifications.
    The tool fits the model and returns coefficients, metrics,
    feature importance, predictions, and diagnostic information.
    """

    @property
    def name(self) -> str:
        return "fit_model"

    @property
    def description(self) -> str:
        return (
            "Fit a statistical or machine learning model on data. Model types:\n"
            "- 'linear_regression': OLS regression with diagnostics (R², coefficients, p-values, VIF)\n"
            "- 'logistic_regression': Binary classification with metrics (AUC, precision, recall)\n"
            "- 'random_forest': Ensemble model with feature importance and partial dependence\n"
            "- 'gradient_boosting': Boosted trees with feature importance\n"
            "- 'kmeans': K-means clustering with silhouette scores and cluster profiles\n"
            "- 'time_series': Trend decomposition, change-point detection, forecasting\n"
            "- 'counterfactual': What-if scenario analysis — predict outcomes under changed conditions\n"
            "- 'cross_validate': Compare multiple model types on same data\n"
            "- 'bayesian_regression': Bayesian linear regression with posterior credible intervals, "
            "probability of direction for each predictor, and posterior predictive checks. "
            "Uses PyMC (MCMC) when available, falls back to sklearn BayesianRidge. "
            "Specify informative priors via parameters.prior (from published research).\n\n"
            "ALWAYS use sql_query to provide data — pass a SELECT statement and the "
            "tool fetches ALL rows directly. Do NOT call execute_sql first.\n\n"
            "Returns model metrics, coefficients, feature importance, and predictions."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": (
                        "Model type: 'linear_regression', 'logistic_regression', 'random_forest', "
                        "'gradient_boosting', 'kmeans', 'time_series', 'counterfactual', "
                        "'cross_validate', 'bayesian_regression'."
                    ),
                },
                "sql_query": {
                    "type": "string",
                    "description": (
                        "A SELECT SQL query to fetch the data. The tool executes it "
                        "directly and retrieves ALL rows — no row limit. "
                        "Example: 'SELECT gdp, life_expectancy FROM catalog.schema.table'"
                    ),
                },
                "target_column": {
                    "type": "string",
                    "description": "Dependent variable / target column name.",
                },
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Independent variable / feature column names.",
                },
                "time_column": {
                    "type": "string",
                    "description": "Column containing time periods (for time_series model).",
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "Model-specific parameters. E.g., {'n_clusters': 4} for kmeans, "
                        "{'scenarios': [{'svi_score': 0.9, 'provider_count': 5}]} for counterfactual, "
                        "{'n_folds': 5} for cross_validate, "
                        "{'class_weight': 'balanced'} for logistic_regression with imbalanced classes."
                    ),
                },
            },
            "required": ["model_type", "sql_query"],
        }

    def _execute(
        self,
        model_type: str = "",
        sql_query: str = "",
        data: list[dict] | None = None,
        target_column: str = "",
        feature_columns: list[str] | None = None,
        time_column: str = "",
        parameters: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if not model_type:
            return ToolResult(success=False, error="Missing 'model_type'.")

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

        params = parameters or {}

        dispatch = {
            "linear_regression": self._linear_regression,
            "logistic_regression": self._logistic_regression,
            "random_forest": self._random_forest,
            "gradient_boosting": self._gradient_boosting,
            "kmeans": self._kmeans,
            "time_series": self._time_series,
            "counterfactual": self._counterfactual,
            "cross_validate": self._cross_validate,
            "bayesian_regression": self._bayesian_regression,
        }

        handler = dispatch.get(model_type)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown model_type '{model_type}'. Use: {list(dispatch.keys())}",
            )

        return handler(  # type: ignore[no-any-return, operator]
            df=df,
            target_column=target_column,
            feature_columns=feature_columns or [],
            time_column=time_column,
            params=params,
        )

    # ------------------------------------------------------------------
    # Linear regression
    # ------------------------------------------------------------------

    def _linear_regression(self, df, target_column, feature_columns, **kwargs) -> ToolResult:
        import statsmodels.api as sm
        from scipy import stats as sp_stats

        if not target_column or not feature_columns:
            return ToolResult(
                success=False, error="'target_column' and 'feature_columns' are required."
            )

        # Prepare data
        all_cols = [target_column] + feature_columns
        clean = df[all_cols].dropna()
        if len(clean) < len(feature_columns) + 2:
            return ToolResult(
                success=False,
                error=f"Too few rows ({len(clean)}) for {len(feature_columns)} features.",
            )

        y = clean[target_column].astype(float)
        X = clean[feature_columns].astype(float)
        X_const = sm.add_constant(X)

        model = sm.OLS(y, X_const).fit()

        # VIF calculation
        vif_values = {}
        if len(feature_columns) > 1:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            for i, col in enumerate(feature_columns):
                try:
                    vif_values[col] = round(variance_inflation_factor(X_const.values, i + 1), 2)
                except Exception:
                    vif_values[col] = None

        # Residual diagnostics
        residuals = model.resid
        _, norm_p = sp_stats.shapiro(residuals[: min(len(residuals), 5000)])

        # Coefficients
        coefs = []
        for name, coef, se, t, p, ci_low, ci_high in zip(
            model.params.index,
            model.params,
            model.bse,
            model.tvalues,
            model.pvalues,
            model.conf_int()[0],
            model.conf_int()[1],
        ):
            coefs.append(
                {
                    "variable": name,
                    "coefficient": round(float(coef), 6),
                    "std_error": round(float(se), 6),
                    "t_statistic": round(float(t), 4),
                    "p_value": round(float(p), 6),
                    "ci_95": [round(float(ci_low), 6), round(float(ci_high), 6)],
                    "significant": p < 0.05,
                }
            )

        # Standardized coefficients (beta weights)
        X_std = (X - X.mean()) / X.std()
        y_std = (y - y.mean()) / y.std()
        try:
            beta_model = sm.OLS(y_std, sm.add_constant(X_std)).fit()
            for idx, _col in enumerate(feature_columns):
                coefs[idx + 1]["beta_weight"] = round(float(beta_model.params.iloc[idx + 1]), 4)
        except Exception:
            pass

        result = {
            "model": "OLS Linear Regression",
            "n_observations": int(model.nobs),
            "n_features": len(feature_columns),
            "r_squared": round(float(model.rsquared), 4),
            "r_squared_adj": round(float(model.rsquared_adj), 4),
            "f_statistic": round(float(model.fvalue), 4),
            "f_p_value": round(float(model.f_pvalue), 6),
            "aic": round(float(model.aic), 2),
            "bic": round(float(model.bic), 2),
            "coefficients": coefs,
            "vif": vif_values,
            "diagnostics": {
                "residual_normality_p": round(float(norm_p), 6),
                "residuals_normal": norm_p > 0.05,
                "durbin_watson": round(float(sm.stats.stattools.durbin_watson(residuals)), 4),
            },
            "interpretation": (
                f"Model explains {model.rsquared_adj * 100:.1f}% of variance in {target_column}. "
                f"F-test p={model.f_pvalue:.4g} — model is "
                f"{'statistically significant' if model.f_pvalue < 0.05 else 'NOT significant'}."
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=f"Linear regression: R²={model.rsquared:.4f}, adj R²={model.rsquared_adj:.4f}",
        )

    # ------------------------------------------------------------------
    # Logistic regression
    # ------------------------------------------------------------------

    def _logistic_regression(self, df, target_column, feature_columns, **kwargs) -> ToolResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        if not target_column or not feature_columns:
            return ToolResult(
                success=False, error="'target_column' and 'feature_columns' are required."
            )

        clean = df[[target_column] + feature_columns].dropna()
        y = clean[target_column].astype(float)
        X = clean[feature_columns].astype(float)

        if y.nunique() != 2:
            return ToolResult(
                success=False, error=f"Target must be binary. Found {y.nunique()} unique values."
            )

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Support class_weight from parameters (e.g., "balanced" for imbalanced data)
        params = kwargs.get("params", {})
        class_weight = params.get("class_weight", None)
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=class_weight,
        )
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        # Feature importance (absolute coefficient)
        importance = sorted(
            [
                {
                    "feature": f,
                    "coefficient": round(float(c), 4),
                    "odds_ratio": round(float(np.exp(c)), 4),
                }
                for f, c in zip(feature_columns, model.coef_[0])
            ],
            key=lambda x: abs(x["coefficient"]),
            reverse=True,
        )

        result = {
            "model": "Logistic Regression",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "feature_importance": importance,
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        }

        return ToolResult(
            success=True,
            data=result,
            summary=f"Logistic regression: AUC={result['auc_roc']}, accuracy={result['accuracy']}",
        )

    # ------------------------------------------------------------------
    # Random forest
    # ------------------------------------------------------------------

    def _random_forest(self, df, target_column, feature_columns, params, **kwargs) -> ToolResult:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        if not target_column or not feature_columns:
            return ToolResult(
                success=False, error="'target_column' and 'feature_columns' are required."
            )

        clean = df[[target_column] + feature_columns].dropna()
        y = clean[target_column].astype(float)
        X = clean[feature_columns].astype(float)

        is_classification = y.nunique() <= 10
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", None)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if is_classification:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Feature importance
        importance = sorted(
            [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in zip(feature_columns, model.feature_importances_)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

        result: dict[str, Any] = {
            "model": f"Random Forest {'Classifier' if is_classification else 'Regressor'}",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_estimators": n_estimators,
            "feature_importance": importance,
        }

        if is_classification:
            from sklearn.metrics import accuracy_score, f1_score

            result["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            result["f1_score"] = round(
                float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4
            )
        else:
            result["r_squared"] = round(float(r2_score(y_test, y_pred)), 4)
            result["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            result["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

        # Partial dependence for top 3 features
        try:
            from sklearn.inspection import partial_dependence

            top_features = [f["feature"] for f in importance[:3]]
            pd_results = []
            for feat in top_features:
                feat_idx = feature_columns.index(feat)
                pdp = partial_dependence(model, X_train, features=[feat_idx], kind="average")
                pd_results.append(
                    {
                        "feature": feat,
                        "values": [round(float(v), 4) for v in pdp["values"][0][:10]],
                        "effects": [round(float(v), 4) for v in pdp["average"][0][:10]],
                    }
                )
            result["partial_dependence"] = pd_results
        except Exception:
            pass

        metric_str = (
            f"R²={result.get('r_squared', 'N/A')}"
            if not is_classification
            else f"accuracy={result.get('accuracy', 'N/A')}"
        )
        return ToolResult(
            success=True,
            data=result,
            summary=f"Random Forest: {metric_str}, top feature={importance[0]['feature']}",
        )

    # ------------------------------------------------------------------
    # Gradient boosting
    # ------------------------------------------------------------------

    def _gradient_boosting(
        self, df, target_column, feature_columns, params, **kwargs
    ) -> ToolResult:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        if not target_column or not feature_columns:
            return ToolResult(
                success=False, error="'target_column' and 'feature_columns' are required."
            )

        clean = df[[target_column] + feature_columns].dropna()
        y = clean[target_column].astype(float)
        X = clean[feature_columns].astype(float)

        is_classification = y.nunique() <= 10
        n_estimators = params.get("n_estimators", 100)
        learning_rate = params.get("learning_rate", 0.1)
        max_depth = params.get("max_depth", 3)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if is_classification:
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        importance = sorted(
            [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in zip(feature_columns, model.feature_importances_)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

        result: dict[str, Any] = {
            "model": f"Gradient Boosting {'Classifier' if is_classification else 'Regressor'}",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "hyperparameters": {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
            },
            "feature_importance": importance,
        }

        if is_classification:
            from sklearn.metrics import accuracy_score, f1_score

            result["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            result["f1_score"] = round(
                float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4
            )
        else:
            result["r_squared"] = round(float(r2_score(y_test, y_pred)), 4)
            result["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            result["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

            # Training curve (staged predictions)
            try:
                staged_r2 = []
                for i, y_stage in enumerate(model.staged_predict(X_test)):
                    if i % 10 == 0 or i == n_estimators - 1:
                        staged_r2.append(
                            {
                                "n_trees": i + 1,
                                "r_squared": round(float(r2_score(y_test, y_stage)), 4),
                            }
                        )
                result["learning_curve"] = staged_r2
            except Exception:
                pass

        metric_str = (
            f"R²={result.get('r_squared', 'N/A')}"
            if not is_classification
            else f"accuracy={result.get('accuracy', 'N/A')}"
        )
        return ToolResult(
            success=True,
            data=result,
            summary=f"Gradient Boosting: {metric_str}, top feature={importance[0]['feature']}",
        )

    # ------------------------------------------------------------------
    # K-Means clustering
    # ------------------------------------------------------------------

    def _kmeans(self, df, feature_columns, params, target_column="", **kwargs) -> ToolResult:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        features = feature_columns or list(df.select_dtypes(include="number").columns)
        if len(features) < 2:
            return ToolResult(
                success=False, error="Need at least 2 numeric features for clustering."
            )

        clean = df[features].dropna()
        if len(clean) < 10:
            return ToolResult(success=False, error="Too few rows for clustering.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean)

        n_clusters = params.get("n_clusters", None)

        # If n_clusters not specified, find optimal via silhouette
        if not n_clusters:
            max_k = min(10, len(clean) // 5)
            silhouettes = []
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels)
                silhouettes.append({"k": k, "silhouette": round(float(sil), 4)})
            n_clusters = max(silhouettes, key=lambda x: x["silhouette"])["k"]

        # Fit final model
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        # Cluster profiles
        clean_with_labels = clean.copy()
        clean_with_labels["cluster"] = labels
        profiles = []
        for c in range(n_clusters):
            cluster_data = clean_with_labels[clean_with_labels["cluster"] == c]
            profile = {
                "cluster": c,
                "size": int(len(cluster_data)),
                "pct": round(len(cluster_data) / len(clean) * 100, 1),
                "means": {feat: round(float(cluster_data[feat].mean()), 4) for feat in features},
            }
            # How this cluster differs from overall mean
            profile["deviations"] = {
                feat: round(float(cluster_data[feat].mean() - clean[feat].mean()), 4)
                for feat in features
            }
            profiles.append(profile)

        result = {
            "model": "K-Means Clustering",
            "n_clusters": n_clusters,
            "n_observations": len(clean),
            "silhouette_score": round(float(sil), 4),
            "cluster_profiles": profiles,
        }

        if "silhouettes" in dir():
            result["silhouette_by_k"] = silhouettes

        return ToolResult(
            success=True,
            data=result,
            summary=f"K-Means: {n_clusters} clusters, silhouette={sil:.4f}",
        )

    # ------------------------------------------------------------------
    # Time series analysis
    # ------------------------------------------------------------------

    def _time_series(self, df, target_column, time_column, params, **kwargs) -> ToolResult:
        from scipy import stats as sp_stats

        if not target_column:
            return ToolResult(success=False, error="'target_column' is required for time series.")
        if not time_column:
            return ToolResult(success=False, error="'time_column' is required for time series.")

        clean = df[[time_column, target_column]].dropna().sort_values(time_column)
        times = clean[time_column].values
        values = clean[target_column].astype(float).values

        if len(values) < 4:
            return ToolResult(success=False, error="Too few time periods for time series analysis.")

        # Basic trend analysis
        time_numeric = np.arange(len(values))
        slope, intercept, r, p, se = sp_stats.linregress(time_numeric, values)

        # Period-over-period changes
        changes = []
        for i in range(1, len(values)):
            change = values[i] - values[i - 1]
            pct_change = (change / values[i - 1] * 100) if values[i - 1] != 0 else None
            changes.append(
                {
                    "from": str(times[i - 1]),
                    "to": str(times[i]),
                    "change": round(float(change), 4),
                    "pct_change": round(float(pct_change), 2) if pct_change is not None else None,
                }
            )

        # Detect acceleration/deceleration
        if len(values) >= 6:
            mid = len(values) // 2
            first_half = values[:mid]
            second_half = values[mid:]
            slope1, _, _, _, _ = sp_stats.linregress(np.arange(len(first_half)), first_half)
            slope2, _, _, _, _ = sp_stats.linregress(np.arange(len(second_half)), second_half)
            if abs(slope2) > abs(slope1) * 1.2:
                trend_change = "accelerating"
            elif abs(slope2) < abs(slope1) * 0.8:
                trend_change = "decelerating"
            else:
                trend_change = "steady"
        else:
            trend_change = "insufficient data"

        # Simple forecast (linear extrapolation)
        forecast_periods = params.get("forecast_periods", 3)
        forecasts = []
        for i in range(1, forecast_periods + 1):
            future_x = len(values) - 1 + i
            predicted = intercept + slope * future_x
            # Prediction interval
            se_pred = se * np.sqrt(
                1
                + 1 / len(values)
                + (future_x - time_numeric.mean()) ** 2
                / ((time_numeric - time_numeric.mean()) ** 2).sum()
            )
            ci = 1.96 * se_pred
            forecasts.append(
                {
                    "period_ahead": i,
                    "predicted": round(float(predicted), 4),
                    "ci_lower": round(float(predicted - ci), 4),
                    "ci_upper": round(float(predicted + ci), 4),
                }
            )

        # Volatility
        pct_changes = np.diff(values) / values[:-1] * 100
        volatility = float(np.std(pct_changes)) if len(pct_changes) > 0 else 0

        result = {
            "model": "Time Series Analysis",
            "n_periods": len(values),
            "time_range": [str(times[0]), str(times[-1])],
            "trend": {
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 4),
                "r_squared": round(float(r**2), 4),
                "p_value": round(float(p), 6),
                "direction": "increasing" if slope > 0 else "decreasing",
                "significant": p < 0.05,
                "trend_change": trend_change,
            },
            "summary_stats": {
                "first_value": round(float(values[0]), 4),
                "last_value": round(float(values[-1]), 4),
                "total_change": round(float(values[-1] - values[0]), 4),
                "total_pct_change": round(float((values[-1] - values[0]) / values[0] * 100), 2)
                if values[0] != 0
                else None,
                "volatility_pct": round(volatility, 2),
            },
            "period_changes": changes,
            "forecast": forecasts,
            "interpretation": (
                f"{'Significant' if p < 0.05 else 'No significant'} "
                f"{'upward' if slope > 0 else 'downward'} trend "
                f"(slope={slope:.4f}/period, p={p:.4g}). "
                f"Trend is {trend_change}."
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=f"Time series: {'↑' if slope > 0 else '↓'} trend, R²={r**2:.4f}, {trend_change}",
        )

    # ------------------------------------------------------------------
    # Counterfactual / what-if analysis
    # ------------------------------------------------------------------

    def _counterfactual(self, df, target_column, feature_columns, params, **kwargs) -> ToolResult:
        from sklearn.ensemble import GradientBoostingRegressor

        if not target_column or not feature_columns:
            return ToolResult(
                success=False, error="'target_column' and 'feature_columns' are required."
            )

        scenarios = params.get("scenarios", [])
        if not scenarios:
            return ToolResult(
                success=False,
                error=(
                    "'parameters.scenarios' required — a list of dicts specifying "
                    "changed feature values. E.g., [{'svi_score': 0.2}, {'svi_score': 0.9}]"
                ),
            )

        clean = df[[target_column] + feature_columns].dropna()
        y = clean[target_column].astype(float)
        X = clean[feature_columns].astype(float)

        # Fit a model on actual data
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X, y)

        baseline_mean = float(y.mean())
        baseline_median = float(y.median())

        # For each scenario, predict outcomes
        scenario_results = []
        for i, scenario in enumerate(scenarios):
            # Start from the median feature values
            scenario_X = pd.DataFrame([X.median().to_dict()])
            for col, val in scenario.items():
                if col in feature_columns:
                    scenario_X[col] = val

            predicted = float(model.predict(scenario_X)[0])
            change = predicted - baseline_mean
            pct_change = (change / baseline_mean * 100) if baseline_mean != 0 else None

            scenario_results.append(
                {
                    "scenario_id": i + 1,
                    "conditions": scenario,
                    "predicted_outcome": round(predicted, 4),
                    "baseline_mean": round(baseline_mean, 4),
                    "change_from_baseline": round(change, 4),
                    "pct_change": round(pct_change, 2) if pct_change is not None else None,
                }
            )

        # Also do population-level what-if: shift each feature by 1 std
        sensitivity = []
        for feat in feature_columns:
            shifted_X = X.copy()
            std = X[feat].std()
            shifted_X[feat] = shifted_X[feat] + std
            shifted_pred = model.predict(shifted_X)
            baseline_pred = model.predict(X)
            avg_change = float(np.mean(shifted_pred - baseline_pred))
            sensitivity.append(
                {
                    "feature": feat,
                    "shift": f"+1 std ({round(float(std), 4)})",
                    "avg_outcome_change": round(avg_change, 4),
                    "pct_outcome_change": round(avg_change / baseline_mean * 100, 2)
                    if baseline_mean != 0
                    else None,
                }
            )
        sensitivity.sort(key=lambda x: abs(x["avg_outcome_change"]), reverse=True)

        result = {
            "model": "Counterfactual Analysis (Gradient Boosting)",
            "n_observations": len(clean),
            "baseline": {
                "mean": round(baseline_mean, 4),
                "median": round(baseline_median, 4),
                "model_r_squared": round(float(model.score(X, y)), 4),
            },
            "scenarios": scenario_results,
            "sensitivity_analysis": sensitivity,
            "interpretation": (
                f"Model trained on {len(clean)} observations (R²={model.score(X, y):.4f}). "
                f"Most sensitive feature: {sensitivity[0]['feature']} "
                f"(+1 std → {sensitivity[0]['pct_outcome_change']}% change in {target_column})."
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=f"Counterfactual: {len(scenario_results)} scenarios, most sensitive={sensitivity[0]['feature']}",
        )

    # ------------------------------------------------------------------
    # Bayesian regression
    # ------------------------------------------------------------------

    def _bayesian_regression(
        self, df, target_column, feature_columns, params, **kwargs
    ) -> ToolResult:
        """Bayesian linear regression with posterior credible intervals.

        Uses PyMC for full MCMC sampling when available, falls back to
        sklearn BayesianRidge otherwise.
        """
        if not target_column or not feature_columns:
            return ToolResult(
                success=False, error="'target_column' and 'feature_columns' are required."
            )

        all_cols = [target_column] + feature_columns
        clean = df[all_cols].dropna()
        if len(clean) < len(feature_columns) + 2:
            return ToolResult(
                success=False,
                error=f"Too few rows ({len(clean)}) for {len(feature_columns)} features.",
            )

        y = clean[target_column].astype(float).values
        X = clean[feature_columns].astype(float)

        # Standardize features for stable MCMC sampling
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_means = scaler.mean_
        feature_stds = scaler.scale_

        user_priors = params.get("prior", {})

        # Try PyMC first, fall back to sklearn
        try:
            return self._bayesian_regression_pymc(
                y,
                X_scaled,
                feature_columns,
                feature_means,
                feature_stds,
                user_priors,
                params,
                clean,
                target_column,
            )
        except ImportError:
            return self._bayesian_regression_sklearn(
                y,
                X_scaled,
                feature_columns,
                feature_means,
                feature_stds,
                user_priors,
                clean,
                target_column,
            )

    def _bayesian_regression_pymc(
        self,
        y,
        X_scaled,
        feature_columns,
        feature_means,
        feature_stds,
        user_priors,
        params,
        clean,
        target_column,
    ) -> ToolResult:
        """Full MCMC Bayesian regression using PyMC."""
        import arviz as az
        import pymc as pm

        n_draws = params.get("n_draws", 2000)
        n_tune = params.get("n_tune", 1000)
        n_chains = params.get("n_chains", 2)

        with pm.Model():
            # Priors for coefficients
            betas = []
            for i, feat in enumerate(feature_columns):
                if feat in user_priors:
                    # Informative prior from published research
                    # Scale the prior to standardized space
                    p = user_priors[feat]
                    prior_mean = p.get("mean", 0) * feature_stds[i]
                    prior_std = p.get("std", 10) * feature_stds[i]
                else:
                    # Weakly informative
                    prior_mean = 0
                    prior_std = 10

                beta = pm.Normal(f"beta_{feat}", mu=prior_mean, sigma=prior_std)
                betas.append(beta)

            # Intercept
            intercept = pm.Normal("intercept", mu=float(y.mean()), sigma=float(y.std()) * 2)

            # Noise
            sigma = pm.HalfCauchy("sigma", beta=float(y.std()))

            # Likelihood
            mu = intercept
            for i, beta in enumerate(betas):
                mu = mu + beta * X_scaled[:, i]

            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Sample
            trace = pm.sample(
                draws=n_draws,
                tune=n_tune,
                chains=n_chains,
                return_inferencedata=True,
                progressbar=False,
                random_seed=42,
            )

        # Extract results
        summary = az.summary(trace, hdi_prob=0.95)

        coefficients = {}
        for i, feat in enumerate(feature_columns):
            var_name = f"beta_{feat}"
            if var_name in summary.index:
                row = summary.loc[var_name]
                samples = trace.posterior[var_name].values.flatten()

                # Convert back to original scale
                orig_mean = float(row["mean"]) / feature_stds[i]
                orig_hdi_low = float(row["hdi_2.5%"]) / feature_stds[i]
                orig_hdi_high = float(row["hdi_97.5%"]) / feature_stds[i]

                prob_positive = float(np.mean(samples > 0))

                prior_info = user_priors.get(feat, {"mean": 0, "std": 10})
                coefficients[feat] = {
                    "posterior_mean": round(orig_mean, 4),
                    "posterior_sd": round(float(row["sd"]) / feature_stds[i], 4),
                    "hdi_95": [round(orig_hdi_low, 4), round(orig_hdi_high, 4)],
                    "probability_positive": round(prob_positive, 4),
                    "probability_negative": round(1 - prob_positive, 4),
                    "prior": {
                        "mean": prior_info.get("mean", 0),
                        "std": prior_info.get("std", 10),
                    },
                }

        # Model fit
        y_pred = trace.posterior["intercept"].values.flatten().mean()
        for i, feat in enumerate(feature_columns):
            y_pred = (
                y_pred + trace.posterior[f"beta_{feat}"].values.flatten().mean() * X_scaled[:, i]
            )

        # Handle y_pred being a scalar or array
        if np.isscalar(y_pred):
            ss_res = float(np.sum((y - y_pred) ** 2))
        else:
            ss_res = float(np.sum((y - np.array(y_pred)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Convergence diagnostics
        r_hat_vals = summary["r_hat"].dropna().values
        ess_vals = summary["ess_bulk"].dropna().values

        # WAIC and LOO
        model_fit = {"r_squared": round(r_squared, 4)}
        try:
            waic = az.waic(trace)
            model_fit["waic"] = round(float(waic.waic), 2)
        except Exception:
            pass
        try:
            loo = az.loo(trace)
            model_fit["loo"] = round(float(loo.loo), 2)
        except Exception:
            pass

        # OLS comparison
        import statsmodels.api as sm

        X_const = sm.add_constant(X_scaled)
        ols = sm.OLS(y, X_const).fit()

        ols_coefficients = {}
        for i, feat in enumerate(feature_columns):
            ols_coefficients[feat] = {
                "coefficient": round(float(ols.params[i + 1]) / feature_stds[i], 4),
                "p_value": round(float(ols.pvalues[i + 1]), 6),
            }

        result = {
            "model_type": "bayesian_regression",
            "engine": "pymc",
            "n_observations": len(y),
            "target": target_column,
            "features": feature_columns,
            "coefficients": coefficients,
            "model_fit": model_fit,
            "convergence": {
                "r_hat_max": round(float(r_hat_vals.max()), 4) if len(r_hat_vals) > 0 else None,
                "ess_min": round(float(ess_vals.min()), 0) if len(ess_vals) > 0 else None,
                "n_draws": n_draws,
                "n_chains": n_chains,
                "converged": bool(r_hat_vals.max() < 1.05) if len(r_hat_vals) > 0 else None,
            },
            "frequentist_comparison": {
                "ols_r_squared": round(float(ols.rsquared), 4),
                "ols_coefficients": ols_coefficients,
            },
            "interpretation": self._build_bayesian_interpretation(
                coefficients, r_squared, target_column
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=f"Bayesian regression (PyMC): R²={r_squared:.4f}, {len(feature_columns)} predictors",
        )

    def _bayesian_regression_sklearn(
        self,
        y,
        X_scaled,
        feature_columns,
        feature_means,
        feature_stds,
        user_priors,
        clean,
        target_column,
    ) -> ToolResult:
        """Fallback Bayesian regression using sklearn BayesianRidge."""
        from scipy import stats as sp_stats
        from sklearn.linear_model import BayesianRidge

        model = BayesianRidge(compute_score=True)
        model.fit(X_scaled, y)

        # Get posterior mean and variance for each coefficient
        coef_mean = model.coef_
        # BayesianRidge stores the regularization param; approximate posterior std
        # from the model's alpha (noise precision) and lambda (weight precision)
        alpha = model.alpha_  # noise precision
        lambda_ = model.lambda_  # weight precision

        # Posterior covariance approximation
        n_samples = 10_000
        coefficients = {}
        for i, feat in enumerate(feature_columns):
            # Approximate posterior std for coefficient i
            post_std_scaled = np.sqrt(1.0 / (alpha * X_scaled[:, i].dot(X_scaled[:, i]) + lambda_))
            post_mean_scaled = coef_mean[i]

            # Sample posterior
            samples = sp_stats.norm.rvs(loc=post_mean_scaled, scale=post_std_scaled, size=n_samples)

            # Convert to original scale
            orig_samples = samples / feature_stds[i]
            orig_mean = float(np.mean(orig_samples))
            orig_std = float(np.std(orig_samples))
            hdi_low = float(np.percentile(orig_samples, 2.5))
            hdi_high = float(np.percentile(orig_samples, 97.5))
            prob_positive = float(np.mean(orig_samples > 0))

            prior_info = user_priors.get(feat, {"mean": 0, "std": 10})
            coefficients[feat] = {
                "posterior_mean": round(orig_mean, 4),
                "posterior_sd": round(orig_std, 4),
                "hdi_95": [round(hdi_low, 4), round(hdi_high, 4)],
                "probability_positive": round(prob_positive, 4),
                "probability_negative": round(1 - prob_positive, 4),
                "prior": {
                    "mean": prior_info.get("mean", 0),
                    "std": prior_info.get("std", 10),
                },
            }

        r_squared = float(model.score(X_scaled, y))

        # OLS comparison
        import statsmodels.api as sm

        X_const = sm.add_constant(X_scaled)
        ols = sm.OLS(y, X_const).fit()
        ols_coefficients = {}
        for i, feat in enumerate(feature_columns):
            ols_coefficients[feat] = {
                "coefficient": round(float(ols.params[i + 1]) / feature_stds[i], 4),
                "p_value": round(float(ols.pvalues[i + 1]), 6),
            }

        result = {
            "model_type": "bayesian_regression",
            "engine": "sklearn_fallback",
            "n_observations": len(y),
            "target": target_column,
            "features": feature_columns,
            "coefficients": coefficients,
            "model_fit": {
                "r_squared": round(r_squared, 4),
                "alpha": round(float(alpha), 4),
                "lambda": round(float(lambda_), 4),
            },
            "convergence": {
                "note": "sklearn BayesianRidge uses closed-form EM — no MCMC convergence to check.",
            },
            "frequentist_comparison": {
                "ols_r_squared": round(float(ols.rsquared), 4),
                "ols_coefficients": ols_coefficients,
            },
            "interpretation": self._build_bayesian_interpretation(
                coefficients, r_squared, target_column
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=f"Bayesian regression (sklearn): R²={r_squared:.4f}, {len(feature_columns)} predictors",
        )

    @staticmethod
    def _build_bayesian_interpretation(coefficients: dict, r_squared: float, target: str) -> str:
        """Build interpretive text for Bayesian regression results."""
        parts = [f"Bayesian regression (R²={r_squared:.4f})."]

        # Find strongest predictors by probability of direction
        strong = []
        for feat, info in coefficients.items():
            prob_dir = max(info["probability_positive"], info["probability_negative"])
            if prob_dir > 0.95:
                direction = "positive" if info["probability_positive"] > 0.5 else "negative"
                strong.append((feat, prob_dir, direction, info["posterior_mean"]))

        strong.sort(key=lambda x: x[1], reverse=True)
        if strong:
            feat, prob, direction, mean = strong[0]
            parts.append(
                f"Strongest predictor: {feat} ({prob * 100:.0f}% probability of "
                f"{direction} effect, posterior mean={mean:.4f})."
            )
            if len(strong) > 1:
                others = ", ".join(f"{f} ({d})" for f, _, d, _ in strong[1:3])
                parts.append(f"Also confident: {others}.")
        else:
            parts.append("No predictor has >95% probability of direction — high uncertainty.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Cross-validation / model comparison
    # ------------------------------------------------------------------

    def _cross_validate(self, df, target_column, feature_columns, params, **kwargs) -> ToolResult:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        if not target_column or not feature_columns:
            return ToolResult(
                success=False, error="'target_column' and 'feature_columns' are required."
            )

        clean = df[[target_column] + feature_columns].dropna()
        y = clean[target_column].astype(float)
        X = clean[feature_columns].astype(float)

        n_folds = params.get("n_folds", 5)
        if len(clean) < n_folds * 5:
            n_folds = max(2, len(clean) // 5)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42
            ),
        }

        results = []
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=n_folds, scoring="r2")
                mae_scores = -cross_val_score(
                    model, X_scaled, y, cv=n_folds, scoring="neg_mean_absolute_error"
                )
                results.append(
                    {
                        "model": name,
                        "mean_r2": round(float(scores.mean()), 4),
                        "std_r2": round(float(scores.std()), 4),
                        "mean_mae": round(float(mae_scores.mean()), 4),
                        "std_mae": round(float(mae_scores.std()), 4),
                        "fold_r2_scores": [round(float(s), 4) for s in scores],
                    }
                )
            except Exception as e:
                results.append({"model": name, "error": str(e)})

        results.sort(key=lambda x: x.get("mean_r2", -999), reverse=True)  # type: ignore[arg-type, return-value]
        best = results[0]

        return ToolResult(
            success=True,
            data={
                "target": target_column,
                "features": feature_columns,
                "n_observations": len(clean),
                "n_folds": n_folds,
                "model_comparison": results,
                "best_model": best["model"],
                "recommendation": (
                    f"Best model: {best['model']} (R²={best.get('mean_r2', 'N/A')} ± {best.get('std_r2', 'N/A')}). "
                    f"Use this for production predictions and feature interpretation."
                ),
            },
            summary=f"Cross-validation: best={best['model']} R²={best.get('mean_r2', 'N/A')}",
        )
