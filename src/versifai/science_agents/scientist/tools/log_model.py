"""
Tool: log_model

Logs a trained sklearn model to MLflow with metrics, parameters, feature
importance, and source code as artifacts. Optionally registers the model in
Databricks Unity Catalog.

The agent provides data, model specification, and pre-computed metrics.
The tool re-fits the model, logs everything to MLflow, and returns the
run ID and model URI.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from versifai.core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from versifai.science_agents.scientist.config import ResearchConfig

logger = logging.getLogger("tool.log_model")


class LogModelTool(BaseTool):
    """
    Log a trained model to MLflow with full provenance.

    Re-fits the model from data, logs parameters, metrics, feature
    importance, and source code artifacts. If the config specifies
    a Unity Catalog registry name, registers the model there.
    """

    def __init__(self, cfg: "ResearchConfig") -> None:
        super().__init__()
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "log_model"

    @property
    def description(self) -> str:
        return (
            "Log a trained sklearn model to MLflow with full provenance tracking. "
            "Re-fits the model from data, logs hyperparameters, evaluation metrics, "
            "feature importance, and source code as artifacts. Optionally registers "
            "the model in Databricks Unity Catalog.\n\n"
            "Use this AFTER training and evaluating a model with fit_model. "
            "Provide the same data, model_type, and parameters used for training, "
            "plus the metrics you computed during evaluation.\n\n"
            "Supported model types: 'gradient_boosting', 'logistic_regression', "
            "'random_forest'."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": (
                        "Model type: 'gradient_boosting', 'logistic_regression', "
                        "'random_forest'."
                    ),
                },
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Training data as list of row dicts (same as fit_model).",
                },
                "target_column": {
                    "type": "string",
                    "description": "Target column name.",
                },
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature column names.",
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "Model hyperparameters. E.g., "
                        "{'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1}."
                    ),
                },
                "metrics": {
                    "type": "object",
                    "description": (
                        "Evaluation metrics to log. E.g., "
                        "{'pr_auc': 0.82, 'roc_auc': 0.91, 'f1': 0.76}."
                    ),
                },
                "model_name": {
                    "type": "string",
                    "description": "Display name for this MLflow run.",
                },
                "tags": {
                    "type": "object",
                    "description": (
                        "Optional tags for the MLflow run. E.g., "
                        "{'config': 'expansion_propensity', 'version': '1.0'}."
                    ),
                },
            },
            "required": [
                "model_type", "data", "target_column",
                "feature_columns", "metrics", "model_name",
            ],
        }

    def _execute(
        self,
        model_type: str = "",
        data: list[dict] | None = None,
        target_column: str = "",
        feature_columns: list[str] | None = None,
        parameters: dict | None = None,
        metrics: dict | None = None,
        model_name: str = "",
        tags: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if not model_type:
            return ToolResult(success=False, error="Missing 'model_type'.")
        if not data:
            return ToolResult(success=False, error="Missing 'data'.")
        if not target_column or not feature_columns:
            return ToolResult(
                success=False,
                error="'target_column' and 'feature_columns' are required.",
            )
        if not metrics:
            return ToolResult(success=False, error="Missing 'metrics'.")
        if not model_name:
            return ToolResult(success=False, error="Missing 'model_name'.")

        try:
            import mlflow
            import mlflow.sklearn
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "mlflow is not installed. Add 'mlflow>=2.10.0' to your "
                    "pip install to use model logging."
                ),
            )

        params = parameters or {}
        run_tags = tags or {}

        # Prepare data
        df = pd.DataFrame(data)
        clean = df[[target_column] + list(feature_columns)].dropna()
        y = clean[target_column].astype(float)
        X = clean[feature_columns].astype(float)

        # Build the model
        model = self._build_model(model_type, params, y)
        if model is None:
            return ToolResult(
                success=False,
                error=f"Unsupported model_type '{model_type}' for MLflow logging.",
            )

        # Fit the model
        model.fit(X, y)

        # Feature importance
        importance = {}
        if hasattr(model, "feature_importances_"):
            importance = {
                f: round(float(imp), 6)
                for f, imp in zip(feature_columns, model.feature_importances_)
            }
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            importance = {
                f: round(float(abs(c)), 6)
                for f, c in zip(feature_columns, coefs)
            }

        # Set MLflow experiment
        experiment_name = self._cfg.mlflow_experiment
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        # Log to MLflow
        with mlflow.start_run(run_name=model_name, tags=run_tags) as run:
            # Log parameters
            mlflow.log_params({
                "model_type": model_type,
                "n_features": len(feature_columns),
                "n_training_rows": len(clean),
                "target_column": target_column,
            })
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

            # Log the sklearn model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=(
                    self._cfg.mlflow_registry_name
                    if self._cfg.mlflow_registry_name
                    else None
                ),
            )

            # Log feature importance as artifact
            if importance:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False,
                ) as f:
                    json.dump(importance, f, indent=2)
                    f.flush()
                    mlflow.log_artifact(f.name, "feature_importance")
                os.unlink(f.name)

            # Log feature columns as artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as f:
                json.dump(list(feature_columns), f, indent=2)
                f.flush()
                mlflow.log_artifact(f.name, "schema")
            os.unlink(f.name)

            # Log source code as artifacts
            self._log_source_files(mlflow)

            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"

        result = {
            "mlflow_run_id": run_id,
            "model_uri": model_uri,
            "experiment": experiment_name,
            "registered_model": self._cfg.mlflow_registry_name or "(not registered)",
            "n_training_rows": len(clean),
            "n_features": len(feature_columns),
            "metrics_logged": metrics,
            "top_features": dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

        return ToolResult(
            success=True,
            data=result,
            summary=(
                f"Model logged to MLflow: run_id={run_id}, "
                f"model_uri={model_uri}"
            ),
        )

    def _build_model(self, model_type: str, params: dict, y: pd.Series):
        """Build an sklearn model object from type and parameters."""
        if model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            is_classification = y.nunique() <= 10
            cls = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
            return cls(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 3),
                random_state=42,
            )
        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                random_state=42,
                class_weight=params.get("class_weight"),
            )
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            is_classification = y.nunique() <= 10
            cls = RandomForestClassifier if is_classification else RandomForestRegressor
            return cls(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth"),
                random_state=42,
                n_jobs=-1,
            )
        return None

    def _log_source_files(self, mlflow) -> None:
        """Log relevant source code files as MLflow artifacts."""
        import analysis.scientist.configs as configs_pkg
        import analysis.scientist.tools as tools_pkg

        source_dirs = [
            os.path.dirname(configs_pkg.__file__),
            os.path.dirname(tools_pkg.__file__),
        ]

        for source_dir in source_dirs:
            if os.path.isdir(source_dir):
                for filename in os.listdir(source_dir):
                    if filename.endswith(".py"):
                        filepath = os.path.join(source_dir, filename)
                        try:
                            mlflow.log_artifact(filepath, "source_code")
                        except Exception:
                            pass
