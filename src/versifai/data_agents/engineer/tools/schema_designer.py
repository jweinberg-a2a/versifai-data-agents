"""
Tool: design_schema

Accepts a schema definition from the agent, validates it against naming
standards and join key requirements, and stores it in the agent's schema registry.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from versifai.data_agents.models.schema import ColumnDefinition, TargetSchema
from versifai.core.tools.base import BaseTool, ToolResult
from versifai._utils.naming import to_snake_case

if TYPE_CHECKING:
    from versifai.data_agents.engineer.config import ProjectConfig


def _build_metadata_columns(cfg: "ProjectConfig") -> list[ColumnDefinition]:
    """Build metadata ColumnDefinitions from ProjectConfig."""
    return [
        ColumnDefinition(
            target_name=col.name,
            data_type=col.data_type,
            source_name="_metadata",
            description=col.description,
            nullable=col.nullable,
            is_metadata=True,
        )
        for col in cfg.metadata_columns
    ]


class SchemaDesignerTool(BaseTool):
    """
    Validates and registers a target schema designed by the agent.

    The agent calls this tool with a schema definition after profiling
    a data source. The tool validates naming conventions, checks for a
    join key column, and stores the schema for later use in transformation.
    """

    def __init__(self, cfg: "ProjectConfig | None" = None) -> None:
        super().__init__()
        # In-memory schema registry — shared across the agent session
        self._schemas: dict[str, TargetSchema] = {}
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "design_schema"

    @property
    def description(self) -> str:
        meta_names = ", ".join(self._cfg.metadata_column_names)
        return (
            "Register a target schema for a data source. Two modes:\n\n"
            "**Mode 1 (recommended for large schemas):** Provide `column_names` — a simple "
            "list of original column names from the source file. The tool auto-converts "
            "names to snake_case and infers types from naming patterns. Optionally provide "
            "`join_key_source_column` to identify which source column maps to the join key.\n"
            "Example: column_names=['FIPS','STATE','E_TOTPOP','EP_POV150','RPL_THEMES']\n\n"
            "**Mode 2 (for precise control):** Provide `columns` — a list of full column "
            "definition dicts with target_name, data_type, source_name.\n\n"
            "The tool validates naming and automatically adds metadata columns "
            f"({meta_names}). Only include columns that actually exist in the source "
            "data — do NOT add columns that aren't in the source file."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source_name": {
                    "type": "string",
                    "description": "Identifier for this data source (e.g. 'cdc_svi', 'ma_enrollment').",
                },
                "table_name": {
                    "type": "string",
                    "description": (
                        "Target table name without catalog/schema prefix "
                        "(e.g. 'cdc_svi'). The tool will prepend the catalog.schema."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Description of the table.",
                },
                "column_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "SHORTHAND: List of original column names from the source file. "
                        "The tool auto-converts to snake_case and infers data types from "
                        "naming patterns. Use this instead of 'columns' for large schemas "
                        "with many columns. Example: ['FIPS', 'STATE', 'E_TOTPOP', 'EP_POV150']"
                    ),
                },
                "join_key_source_column": {
                    "type": "string",
                    "description": (
                        "When using column_names mode: the source column that is the "
                        "primary geographic join key (FIPS code). Only use this for "
                        "county-level tables that have a FIPS column. Do NOT use this "
                        "for contract-level or other non-county tables."
                    ),
                },
                "type_overrides": {
                    "type": "object",
                    "description": (
                        "When using column_names mode: override auto-detected types for "
                        "specific columns. Keys are source column names, values are types. "
                        "Example: {'FIPS': 'STRING', 'YEAR': 'INT'}"
                    ),
                },
                "name_overrides": {
                    "type": "object",
                    "description": (
                        "When using column_names mode: override auto-generated target names "
                        "with descriptive, human-readable names based on documentation. "
                        "Keys are source column names, values are target snake_case names. "
                        "You SHOULD provide name_overrides for cryptic/abbreviated columns. "
                        "Example: {'EP_POV150': 'pct_below_150_poverty', "
                        "'E_TOTPOP': 'total_population_estimate', "
                        "'RPL_THEMES': 'overall_vulnerability_percentile'}"
                    ),
                },
                "columns": {
                    "type": "array",
                    "description": "FULL MODE: List of column definitions (use column_names for simpler alternative).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_name": {
                                "type": "string",
                                "description": "Target column name in snake_case.",
                            },
                            "data_type": {
                                "type": "string",
                                "description": "Spark SQL data type: STRING, INT, BIGINT, DOUBLE, FLOAT, DATE, TIMESTAMP, BOOLEAN.",
                            },
                            "source_name": {
                                "type": "string",
                                "description": "Original column name in the source file.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Column description.",
                            },
                            "nullable": {
                                "type": "boolean",
                                "description": "Whether nulls are allowed. Default true.",
                                "default": True,
                            },
                            "is_join_key": {
                                "type": "boolean",
                                "description": "True if this is the primary join key column.",
                                "default": False,
                            },
                            "transform_expression": {
                                "type": "string",
                                "description": (
                                    "Optional Python expression to transform the value. "
                                    "Use 'value' as the placeholder for the source value. "
                                    "Example: 'str(value).zfill(5)' for zero-padding."
                                ),
                            },
                        },
                        "required": ["target_name", "data_type", "source_name"],
                    },
                },
                "partition_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of columns to partition by (e.g. ['source_year']).",
                },
                "historical_mappings": {
                    "type": "object",
                    "description": (
                        "Optional dict mapping year/file identifiers to column name overrides. "
                        "Example: {'2018': {'old_col_name': 'target_col_name'}}"
                    ),
                },
            },
            "required": ["source_name", "table_name"],
        }

    def _execute(
        self,
        source_name: str = "",
        table_name: str = "",
        columns: list[dict] | None = None,
        column_names: list[str] | None = None,
        join_key_source_column: str = "",
        type_overrides: dict | None = None,
        name_overrides: dict | None = None,
        description: str = "",
        partition_columns: list[str] | None = None,
        historical_mappings: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        cfg = self._cfg

        # Validate required parameters
        if not source_name:
            return ToolResult(
                success=False,
                error="Missing required parameter 'source_name'. Provide a name like 'cdc_svi' or 'ma_enrollment'.",
            )
        if not table_name:
            return ToolResult(
                success=False,
                error="Missing required parameter 'table_name'. Provide the target table name (e.g. 'cdc_svi').",
            )

        # Support column_names shorthand — auto-generate full column defs
        if column_names and not columns:
            columns = self._auto_map_columns(
                column_names, join_key_source_column,
                type_overrides or {}, name_overrides or {},
            )

        if not columns:
            return ToolResult(
                success=False,
                error=(
                    "Missing columns. Use one of:\n"
                    "  1. column_names=['FIPS','STATE','E_TOTPOP',...] — simple list of source column names (RECOMMENDED for large schemas)\n"
                    "  2. columns=[{\"target_name\":\"county_fips_code\",\"data_type\":\"STRING\",\"source_name\":\"FIPS\"},...] — full definitions\n"
                    "For column_names mode, also provide join_key_source_column to identify the join key."
                ),
            )

        # Build fully qualified table name
        full_table_name = f"{cfg.catalog}.{cfg.schema}.{table_name}"

        # Validate and normalize column definitions
        validated_cols: list[ColumnDefinition] = []
        warnings: list[str] = []
        errors: list[str] = []

        seen_targets: set[str] = set()
        has_join_key = False
        join_key_name = cfg.join_key.column_name

        for i, col_def in enumerate(columns):
            # Defensive: Claude may pass strings instead of dicts
            if isinstance(col_def, str):
                col_def = {"target_name": col_def, "data_type": "STRING", "source_name": col_def}
                warnings.append(f"Column {i} was a plain string '{col_def['target_name']}', converted to dict with STRING type")
            if not isinstance(col_def, dict):
                errors.append(f"Column {i} has unexpected type {type(col_def).__name__}, expected dict")
                continue

            target = col_def.get("target_name", "")
            if not target:
                errors.append(f"Column {i} is missing 'target_name'")
                continue
            # Auto-fix naming
            fixed_target = to_snake_case(target)
            if fixed_target != target:
                warnings.append(f"Column '{target}' renamed to '{fixed_target}' (snake_case)")
                target = fixed_target

            if target in seen_targets:
                errors.append(f"Duplicate column name: '{target}'")
                continue
            seen_targets.add(target)

            data_type = col_def.get("data_type", "STRING").upper()
            valid_types = {"STRING", "INT", "BIGINT", "DOUBLE", "FLOAT", "DATE", "TIMESTAMP", "BOOLEAN", "DECIMAL"}
            if data_type not in valid_types:
                warnings.append(f"Column '{target}' has unusual type '{data_type}', defaulting to STRING")
                data_type = "STRING"

            # Accept both is_fips (legacy) and is_join_key
            is_join_key = col_def.get("is_join_key", False) or col_def.get("is_fips", False)
            if target == join_key_name:
                is_join_key = True

            if is_join_key:
                has_join_key = True
                # Join key columns should match configured data type
                expected_type = cfg.join_key.data_type.upper()
                if data_type != expected_type:
                    warnings.append(
                        f"Join key column '{target}' type changed from {data_type} "
                        f"to {expected_type} per project config"
                    )
                    data_type = expected_type

            validated_cols.append(ColumnDefinition(
                target_name=target,
                data_type=data_type,
                source_name=col_def.get("source_name", target),
                description=col_def.get("description", ""),
                nullable=col_def.get("nullable", True),
                is_fips=is_join_key,
                transform_expression=col_def.get("transform_expression", ""),
            ))

        if not has_join_key:
            # Check if an alternative key (contract_id, plan_id, etc.) is present
            alt_key_names = {k.column_name for k in cfg.alternative_keys}
            found_alt_keys = alt_key_names & seen_targets
            if found_alt_keys:
                alt_names = ", ".join(f"`{k}`" for k in sorted(found_alt_keys))
                warnings.append(
                    f"NOTE: No county join key (`{join_key_name}`) found, but "
                    f"alternative key(s) detected: {alt_names}. This is correct "
                    f"for contract/plan-level tables — do NOT add `{join_key_name}`."
                )
            else:
                # Check if this table is known from source processing hints
                hint = cfg.get_source_hint(table_name) or cfg.get_source_hint(source_name)
                if hint:
                    # Known table — don't warn about missing keys, the hint
                    # defines what this table should contain
                    warnings.append(
                        f"NOTE: Matched source hint '{hint.source_pattern}'. "
                        f"Schema based on actual source data — no key injection needed."
                    )
                else:
                    warnings.append(
                        f"NOTE: No standard join key column found. This is fine if "
                        f"the table's grain doesn't use `{join_key_name}` or "
                        f"`contract_id`. Only include columns from the actual source data."
                    )

        if errors:
            return ToolResult(
                success=False,
                error=f"Schema validation errors: {errors}",
                data={"errors": errors, "warnings": warnings},
            )

        # Check for cryptic/abbreviated column names that should be renamed
        cryptic_cols = self._detect_cryptic_names(validated_cols)
        if cryptic_cols:
            cryptic_list = ", ".join(f"'{c}'" for c in cryptic_cols[:15])
            warnings.append(
                f"NAMING QUALITY: {len(cryptic_cols)} column(s) have cryptic or "
                f"abbreviated names that should be renamed using documentation: "
                f"{cryptic_list}. "
                f"Use name_overrides to map these to descriptive names. "
                f"Example: 'ep_pov150' → 'pct_below_150_poverty'"
            )

        # Add metadata columns from config
        metadata_cols = _build_metadata_columns(cfg)
        for meta_col in metadata_cols:
            if meta_col.target_name not in seen_targets:
                validated_cols.append(meta_col)

        # Build the schema
        schema = TargetSchema(
            source_name=source_name,
            table_name=full_table_name,
            columns=validated_cols,
            description=description,
            partition_columns=partition_columns or [],
            historical_mappings=historical_mappings or {},
        )

        # Store in registry
        self._schemas[source_name] = schema

        return ToolResult(
            success=True,
            data={
                "schema": schema.to_dict(),
                "create_table_sql": schema.to_create_table_sql(),
                "warnings": warnings,
                "column_count": len(validated_cols),
                "has_join_key": has_join_key,
            },
            summary=(
                f"Schema '{source_name}' registered with {len(validated_cols)} columns "
                f"(table: {full_table_name}, join_key: {has_join_key}). "
                f"Warnings: {len(warnings)}"
            ),
        )

    def _auto_map_columns(
        self,
        column_names: list[str],
        join_key_source_column: str,
        type_overrides: dict,
        name_overrides: dict | None = None,
    ) -> list[dict]:
        """
        Auto-generate full column definitions from a simple list of source
        column names. Infers types from naming patterns and applies the
        join key mapping. Applies name_overrides for documentation-driven
        friendly naming.
        """
        cfg = self._cfg
        name_overrides = name_overrides or {}
        columns = []
        for src_name in column_names:
            # Apply name override if provided, otherwise auto-convert
            target = name_overrides.get(src_name, to_snake_case(src_name))
            target = to_snake_case(target)  # Ensure override is also snake_case
            data_type = type_overrides.get(src_name, self._infer_type(src_name))

            col: dict[str, Any] = {
                "target_name": target,
                "data_type": data_type,
                "source_name": src_name,
            }

            # Map join key
            if join_key_source_column and src_name == join_key_source_column:
                col["target_name"] = cfg.join_key.column_name
                col["data_type"] = cfg.join_key.data_type
                col["is_join_key"] = True
                col["transform_expression"] = f"str(value).zfill({cfg.join_key.width})"

            columns.append(col)

        return columns

    @staticmethod
    def _infer_type(column_name: str) -> str:
        """Infer a likely data type from the column name."""
        upper = column_name.upper()

        # Common patterns that indicate numeric types
        numeric_prefixes = ("E_", "M_", "EP_", "MP_", "EPL_", "SPL_", "RPL_")
        if any(upper.startswith(p) for p in numeric_prefixes):
            return "DOUBLE"

        # Flag/binary columns
        if upper.startswith("F_") or upper.startswith("FLAG_"):
            return "INT"

        # Count, total, number patterns
        count_patterns = ("_COUNT", "_CNT", "_NUM", "_TOTAL", "_TOT", "_POP", "_RATE", "_PCT")
        if any(upper.endswith(p) for p in count_patterns):
            return "DOUBLE"

        # Year columns
        if upper in ("YEAR", "SOURCE_YEAR", "DATA_YEAR") or upper.endswith("_YEAR"):
            return "INT"

        # ID / code columns
        if upper in ("FIPS", "STCNTY", "STATE_FIPS", "COUNTY_FIPS", "TRACTCE", "GEOID"):
            return "STRING"

        # Latitude/longitude
        if upper in ("LAT", "LON", "LATITUDE", "LONGITUDE", "X", "Y"):
            return "DOUBLE"

        # Default to STRING for safety
        return "STRING"

    @staticmethod
    def _detect_cryptic_names(columns: list[ColumnDefinition]) -> list[str]:
        """
        Detect column names that look cryptic or abbreviated and should be
        renamed using documentation. Returns a list of target_names that
        need attention.

        Heuristics:
        - Name is very short (<=4 chars) and not a known abbreviation
        - Name contains number-heavy segments (e.g., 'ep_pov150')
        - Name is mostly consonants / looks like an abbreviation
        """
        known_short_names = {
            "id", "url", "lat", "lon", "year", "name", "type", "date",
            "code", "fips", "rank", "rate", "flag", "area", "pop",
            "state", "county",
        }
        # Skip metadata columns
        meta_names = {
            "source_file_name", "source_year", "source_period_start",
            "load_timestamp", "county_fips_code", "state_fips_code",
            "state_abbreviation", "county_name",
        }

        cryptic = []
        for col in columns:
            name = col.target_name
            if col.is_metadata or name in meta_names:
                continue

            parts = name.split("_")

            # Check 1: Very short name (<=4 chars total) not in known set
            if len(name) <= 4 and name not in known_short_names:
                cryptic.append(name)
                continue

            # Check 2: Any part has digits mixed with letters (e.g., 'pov150')
            has_mixed = any(
                any(c.isdigit() for c in p) and any(c.isalpha() for c in p)
                for p in parts if len(p) > 1
            )
            if has_mixed:
                cryptic.append(name)
                continue

            # Check 3: Most parts are very short abbreviations (<=3 chars)
            short_parts = [p for p in parts if 0 < len(p) <= 3 and p not in known_short_names]
            if len(parts) >= 2 and len(short_parts) >= len(parts) * 0.6:
                cryptic.append(name)
                continue

        return cryptic

    def get_schema(self, source_name: str) -> TargetSchema | None:
        """Retrieve a registered schema by source name."""
        return self._schemas.get(source_name)

    @property
    def registered_schemas(self) -> dict[str, TargetSchema]:
        return self._schemas
