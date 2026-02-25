"""
Tool: validate_silver

Silver dataset quality gate. The agent calls this after building or loading
any silver table, before using it for analysis.

Catches the exact class of issues found in production: cumulative enrollment
sums, suppressed string values leaking through, grain violations, year-lag
misalignment, broken joins producing all-zero columns, and out-of-range
values.

When issues are found, the tool suggests corrective SQL that derives from
the base bronze tables — never patching silver in place.
"""

from __future__ import annotations

import pandas as pd

from versifai.core.tools.base import BaseTool, ToolResult


class ValidateSilverTool(BaseTool):
    """
    Validate silver dataset quality before using it for analysis.

    WHEN TO USE:
    - After building any silver table (CTAS) — run 'grain', 'value_ranges',
      and 'zero_columns' at minimum.
    - After loading a silver table for analysis — run 'enrollment_sanity'
      on any table with enrollment columns.
    - When joining star_ratings to enrollment — run 'year_alignment'.
    - After any multi-table join — run 'join_completeness'.

    Each check returns structured results with issues_found, details, and
    a suggested_fix (corrective SQL or guidance derived from bronze tables).
    """

    @property
    def name(self) -> str:
        return "validate_silver"

    @property
    def description(self) -> str:
        return (
            "Silver dataset quality gate. Call after building or loading any "
            "silver table to catch data integrity issues before analysis.\n\n"
            "Check types:\n"
            "- 'grain': Detect duplicate rows on primary key columns.\n"
            "- 'enrollment_sanity': Detect cumulative sums (e.g., 259M for one "
            "county), suppressed '*' values still as strings, negatives/zeros.\n"
            "- 'year_alignment': Verify star ratings year lag offset (-1) between "
            "two year columns.\n"
            "- 'join_completeness': Compare left vs matched row counts to measure "
            "join coverage.\n"
            "- 'value_ranges': Validate columns against expected domain ranges "
            "(percentages 0-100, SVI 0-1, FIPS 5-digit, stars 1-5, etc.).\n"
            "- 'zero_columns': Detect columns where every value is 0 or NULL — "
            "the telltale sign of a failed CAST or broken JOIN.\n\n"
            "Each check returns issues_found, issue_count, details, and a "
            "suggested_fix with corrective SQL derived from bronze tables.\n\n"
            "MINIMUM after any CTAS: run 'grain', 'value_ranges', 'zero_columns'."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "check_type": {
                    "type": "string",
                    "description": (
                        "Type of validation: 'grain', 'enrollment_sanity', "
                        "'year_alignment', 'join_completeness', 'value_ranges', "
                        "'zero_columns'."
                    ),
                },
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Data as list of row dicts from SQL query results.",
                },
                "primary_key_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For 'grain': columns that should form a unique key "
                        "(e.g., ['contract_number', 'county_fips_code', 'source_year'])."
                    ),
                },
                "enrollment_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For 'enrollment_sanity': column names containing enrollment "
                        "counts to validate."
                    ),
                },
                "max_enrollment_per_row": {
                    "type": "number",
                    "description": (
                        "For 'enrollment_sanity': maximum plausible enrollment for a "
                        "single row. Default 1,000,000 (1M). Rows exceeding this are "
                        "flagged as likely cumulative sums."
                    ),
                },
                "year_column_left": {
                    "type": "string",
                    "description": (
                        "For 'year_alignment': the year column from the left table "
                        "(e.g., star ratings year)."
                    ),
                },
                "year_column_right": {
                    "type": "string",
                    "description": (
                        "For 'year_alignment': the year column from the right table "
                        "(e.g., enrollment year)."
                    ),
                },
                "expected_offset": {
                    "type": "integer",
                    "description": (
                        "For 'year_alignment': expected difference (left - right). "
                        "Default -1 for star ratings lag."
                    ),
                },
                "left_count": {
                    "type": "integer",
                    "description": "For 'join_completeness': row count of left table before join.",
                },
                "matched_count": {
                    "type": "integer",
                    "description": "For 'join_completeness': rows that matched in the join.",
                },
                "unmatched_sample": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "For 'join_completeness': sample rows from the left table that "
                        "did NOT match (helps diagnose key format issues)."
                    ),
                },
                "column_ranges": {
                    "type": "object",
                    "description": (
                        "For 'value_ranges': dict of column_name -> {min, max, type}. "
                        "type can be 'percentage' (0-100), 'proportion' (0-1), "
                        "'fips' (5-digit valid), 'stars' (1-5), 'positive' (>0), "
                        "'enrollment' (0-1M). If omitted, auto-detects from column names."
                    ),
                },
            },
            "required": ["check_type", "data"],
        }

    def _execute(
        self,
        check_type: str = "",
        data: list[dict] | None = None,
        primary_key_columns: list[str] | None = None,
        enrollment_columns: list[str] | None = None,
        max_enrollment_per_row: float = 1_000_000,
        year_column_left: str = "",
        year_column_right: str = "",
        expected_offset: int = -1,
        left_count: int = 0,
        matched_count: int = 0,
        unmatched_sample: list[dict] | None = None,
        column_ranges: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if not check_type:
            return ToolResult(success=False, error="Missing 'check_type'.")
        if not data:
            return ToolResult(success=False, error="Missing 'data'.")

        df = pd.DataFrame(data)
        if df.empty:
            return ToolResult(success=False, error="Data is empty.")

        dispatch = {
            "grain": self._check_grain,
            "enrollment_sanity": self._check_enrollment,
            "year_alignment": self._check_year_alignment,
            "join_completeness": self._check_join_completeness,
            "value_ranges": self._check_value_ranges,
            "zero_columns": self._check_zero_columns,
        }

        handler = dispatch.get(check_type)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown check_type '{check_type}'. Use: {list(dispatch.keys())}",
            )

        return handler(
            df=df,
            primary_key_columns=primary_key_columns,
            enrollment_columns=enrollment_columns,
            max_enrollment_per_row=max_enrollment_per_row,
            year_column_left=year_column_left,
            year_column_right=year_column_right,
            expected_offset=expected_offset,
            left_count=left_count,
            matched_count=matched_count,
            unmatched_sample=unmatched_sample,
            column_ranges=column_ranges,
        )

    # ------------------------------------------------------------------
    # Check: Grain integrity
    # ------------------------------------------------------------------

    def _check_grain(self, df: pd.DataFrame, primary_key_columns=None, **kw) -> ToolResult:
        if not primary_key_columns:
            return ToolResult(
                success=False,
                error="'primary_key_columns' required for grain check.",
            )
        missing = [c for c in primary_key_columns if c not in df.columns]
        if missing:
            return ToolResult(
                success=False,
                error=f"Key columns not found: {missing}. Available: {list(df.columns)}",
            )

        dupes = df[df.duplicated(subset=primary_key_columns, keep=False)]
        len(dupes) // 2 if len(dupes) > 0 else 0  # pairs
        n_unique = df.drop_duplicates(subset=primary_key_columns).shape[0]

        if len(dupes) == 0:
            return ToolResult(
                success=True,
                data={
                    "issues_found": False,
                    "issue_count": 0,
                    "total_rows": len(df),
                    "unique_keys": n_unique,
                    "details": [],
                    "suggested_fix": "",
                },
                summary=f"Grain OK: {len(df)} rows, {n_unique} unique keys on {primary_key_columns}.",
            )

        # Sample duplicates for diagnosis
        sample_dupes = (
            dupes.head(20)[primary_key_columns]
            .to_dict(orient="records")
        )
        pk_str = ", ".join(primary_key_columns)

        return ToolResult(
            success=True,
            data={
                "issues_found": True,
                "issue_count": len(dupes),
                "total_rows": len(df),
                "unique_keys": n_unique,
                "duplicate_rows": len(dupes),
                "details": sample_dupes,
                "suggested_fix": (
                    f"Duplicate rows on [{pk_str}]. Likely cause: missing GROUP BY "
                    f"or cartesian join. Fix: rebuild with SELECT DISTINCT or add "
                    f"ROW_NUMBER() OVER (PARTITION BY {pk_str} ORDER BY ...) = 1."
                ),
            },
            summary=(
                f"GRAIN VIOLATION: {len(dupes)} duplicate rows on {primary_key_columns} "
                f"({n_unique} unique of {len(df)} total)."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Enrollment sanity
    # ------------------------------------------------------------------

    def _check_enrollment(
        self, df: pd.DataFrame, enrollment_columns=None,
        max_enrollment_per_row=1_000_000, **kw,
    ) -> ToolResult:
        if not enrollment_columns:
            # Auto-detect enrollment-like columns
            enrollment_columns = [
                c for c in df.columns
                if any(kw in c.lower() for kw in ["enrollment", "enrolled", "members", "beneficiaries"])
            ]
        if not enrollment_columns:
            return ToolResult(
                success=False,
                error="No enrollment columns found or specified.",
            )

        issues = []
        total_issue_count = 0

        for col in enrollment_columns:
            if col not in df.columns:
                continue

            col_issues = {}

            # Check 1: String type with suppressed values
            if df[col].dtype == object:
                star_count = df[col].astype(str).str.strip().eq("*").sum()
                non_numeric = df[col].apply(
                    lambda x: not _is_numeric_str(x) if pd.notna(x) else False
                ).sum()
                if star_count > 0 or non_numeric > 0:
                    col_issues["suppressed_values"] = {
                        "star_count": int(star_count),
                        "non_numeric_count": int(non_numeric),
                        "fix": (
                            f"Column '{col}' is STRING with suppressed values. "
                            f"Apply: CAST(NULLIF(TRIM({col}), '*') AS DOUBLE)"
                        ),
                    }
                    total_issue_count += star_count + non_numeric

            # Convert to numeric for remaining checks
            numeric_vals = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(numeric_vals) == 0:
                col_issues["all_non_numeric"] = {
                    "fix": f"Column '{col}' has no numeric values. Likely needs CAST.",
                }
                total_issue_count += 1
                issues.append({"column": col, "issues": col_issues})
                continue

            # Check 2: Cumulative sums (implausibly large values)
            over_max = numeric_vals[numeric_vals > max_enrollment_per_row]
            if len(over_max) > 0:
                col_issues["cumulative_sums"] = {
                    "count": int(len(over_max)),
                    "max_value": float(numeric_vals.max()),
                    "mean_value": float(numeric_vals.mean()),
                    "example_values": [float(v) for v in over_max.head(5).values],
                    "fix": (
                        f"Column '{col}' has {len(over_max)} values > "
                        f"{max_enrollment_per_row:,.0f}. Likely summed across "
                        f"multiple time periods or contracts. Fix: rebuild from "
                        f"bronze with WHERE clause filtering to a single time "
                        f"period, or verify GROUP BY includes all grain columns."
                    ),
                }
                total_issue_count += len(over_max)

            # Check 3: Negative values
            neg_count = (numeric_vals < 0).sum()
            if neg_count > 0:
                col_issues["negative_values"] = {
                    "count": int(neg_count),
                    "min_value": float(numeric_vals.min()),
                    "fix": f"Column '{col}' has {neg_count} negative enrollment values.",
                }
                total_issue_count += neg_count

            # Check 4: All zeros
            zero_count = (numeric_vals == 0).sum()
            zero_pct = zero_count / len(numeric_vals) * 100
            if zero_pct > 50:
                col_issues["mostly_zeros"] = {
                    "zero_count": int(zero_count),
                    "zero_pct": round(float(zero_pct), 1),
                    "fix": (
                        f"Column '{col}' is {zero_pct:.0f}% zeros. May indicate "
                        f"failed CAST (suppressed '*' → NULL → 0 in COALESCE)."
                    ),
                }
                total_issue_count += 1

            if col_issues:
                issues.append({"column": col, "issues": col_issues})

        has_issues = len(issues) > 0
        return ToolResult(
            success=True,
            data={
                "issues_found": has_issues,
                "issue_count": total_issue_count,
                "columns_checked": enrollment_columns,
                "details": issues,
                "suggested_fix": (
                    "Rebuild silver table from bronze with proper CAST and "
                    "time-period filtering. See per-column fixes above."
                    if has_issues else ""
                ),
            },
            summary=(
                f"ENROLLMENT ISSUES: {total_issue_count} issues across "
                f"{len(issues)} columns."
                if has_issues
                else f"Enrollment OK: {len(enrollment_columns)} columns validated."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Year alignment
    # ------------------------------------------------------------------

    def _check_year_alignment(
        self, df: pd.DataFrame,
        year_column_left="", year_column_right="",
        expected_offset=-1, **kw,
    ) -> ToolResult:
        if not year_column_left or not year_column_right:
            return ToolResult(
                success=False,
                error="Both 'year_column_left' and 'year_column_right' required.",
            )
        for col in [year_column_left, year_column_right]:
            if col not in df.columns:
                return ToolResult(
                    success=False,
                    error=f"Column '{col}' not found. Available: {list(df.columns)}",
                )

        left = pd.to_numeric(df[year_column_left], errors="coerce")
        right = pd.to_numeric(df[year_column_right], errors="coerce")
        valid = left.notna() & right.notna()
        offsets = (left[valid] - right[valid]).astype(int)

        correct = (offsets == expected_offset).sum()
        incorrect = (offsets != expected_offset).sum()
        total = int(valid.sum())

        if incorrect == 0:
            return ToolResult(
                success=True,
                data={
                    "issues_found": False,
                    "issue_count": 0,
                    "total_rows": total,
                    "correct_offset": int(correct),
                    "expected_offset": expected_offset,
                    "details": [],
                    "suggested_fix": "",
                },
                summary=(
                    f"Year alignment OK: all {total} rows have "
                    f"{year_column_left} - {year_column_right} = {expected_offset}."
                ),
            )

        # Collect misaligned offset distribution
        offset_dist = offsets[offsets != expected_offset].value_counts().head(5).to_dict()

        return ToolResult(
            success=True,
            data={
                "issues_found": True,
                "issue_count": int(incorrect),
                "total_rows": total,
                "correct_offset": int(correct),
                "incorrect_offset": int(incorrect),
                "expected_offset": expected_offset,
                "actual_offset_distribution": {str(k): int(v) for k, v in offset_dist.items()},
                "details": [
                    {"offset": str(k), "count": int(v)} for k, v in offset_dist.items()
                ],
                "suggested_fix": (
                    f"{incorrect} rows have wrong year offset. Expected "
                    f"{year_column_left} - {year_column_right} = {expected_offset}. "
                    f"Fix JOIN clause: ON ... AND YEAR(star_ratings.source_period_start) - 1 "
                    f"= YEAR(enrollment.source_period_start)"
                ),
            },
            summary=(
                f"YEAR MISALIGNMENT: {incorrect}/{total} rows have wrong offset "
                f"(expected {expected_offset})."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Join completeness
    # ------------------------------------------------------------------

    def _check_join_completeness(
        self, df: pd.DataFrame,
        left_count=0, matched_count=0, unmatched_sample=None, **kw,
    ) -> ToolResult:
        if left_count <= 0:
            return ToolResult(
                success=False,
                error="'left_count' (row count of left table before join) is required.",
            )
        if matched_count < 0:
            matched_count = 0

        match_rate = matched_count / left_count * 100 if left_count > 0 else 0
        unmatched = left_count - matched_count
        acceptable = match_rate >= 80  # 80% is a reasonable threshold

        details = []
        if unmatched_sample:
            details = unmatched_sample[:10]

        fix = ""
        if not acceptable:
            fix = (
                f"Only {match_rate:.1f}% of left table rows matched. "
                f"Common causes: (1) join key format mismatch (STRING vs INT, "
                f"zero-padding, whitespace), (2) grain mismatch (one table has "
                f"more granular rows), (3) date/year filtering excludes valid rows. "
                f"Inspect unmatched sample rows to diagnose."
            )

        return ToolResult(
            success=True,
            data={
                "issues_found": not acceptable,
                "issue_count": unmatched if not acceptable else 0,
                "left_count": left_count,
                "matched_count": matched_count,
                "unmatched_count": unmatched,
                "match_rate_pct": round(match_rate, 1),
                "acceptable": acceptable,
                "details": details,
                "suggested_fix": fix,
            },
            summary=(
                f"Join {'OK' if acceptable else 'LOW MATCH'}: "
                f"{matched_count}/{left_count} rows matched "
                f"({match_rate:.1f}%)."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Value ranges
    # ------------------------------------------------------------------

    def _check_value_ranges(self, df: pd.DataFrame, column_ranges=None, **kw) -> ToolResult:
        # Auto-detect range expectations from column names if not provided
        if not column_ranges:
            column_ranges = self._auto_detect_ranges(df)

        if not column_ranges:
            return ToolResult(
                success=True,
                data={
                    "issues_found": False,
                    "issue_count": 0,
                    "details": [],
                    "suggested_fix": "",
                },
                summary="No range-checkable columns detected. Pass column_ranges explicitly.",
            )

        issues = []
        total_violations = 0

        for col, spec in column_ranges.items():
            if col not in df.columns:
                continue

            range_type = spec.get("type", "custom")

            if range_type == "fips":
                violations = self._check_fips_column(df, col)
            else:
                violations = self._check_numeric_range(
                    df, col,
                    min_val=spec.get("min"),
                    max_val=spec.get("max"),
                )

            if violations:
                issues.append({"column": col, "type": range_type, **violations})
                total_violations += violations.get("violation_count", 0)

        has_issues = total_violations > 0
        return ToolResult(
            success=True,
            data={
                "issues_found": has_issues,
                "issue_count": total_violations,
                "columns_checked": list(column_ranges.keys()),
                "details": issues,
                "suggested_fix": (
                    "Out-of-range values detected. Check source data and CAST "
                    "operations. See per-column details."
                    if has_issues else ""
                ),
            },
            summary=(
                f"VALUE RANGE ISSUES: {total_violations} violations across "
                f"{len(issues)} columns."
                if has_issues
                else f"Value ranges OK: {len(column_ranges)} columns validated."
            ),
        )

    # ------------------------------------------------------------------
    # Check: Zero columns
    # ------------------------------------------------------------------

    def _check_zero_columns(self, df: pd.DataFrame, **kw) -> ToolResult:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return ToolResult(
                success=True,
                data={"issues_found": False, "issue_count": 0, "details": []},
                summary="No numeric columns to check.",
            )

        suspect = []
        for col in numeric_cols:
            series = df[col]
            null_count = int(series.isna().sum())
            zero_count = int((series == 0).sum())
            total = len(series)

            if null_count + zero_count == total and total > 0:
                likely_cause = "failed CAST" if null_count > zero_count else "broken JOIN"
                suspect.append({
                    "column": col,
                    "null_count": null_count,
                    "zero_count": zero_count,
                    "total_rows": total,
                    "likely_cause": likely_cause,
                    "fix": (
                        f"Column '{col}' is entirely zeros/NULLs. "
                        f"Likely cause: {likely_cause}. "
                        f"If failed CAST: check source column type and apply "
                        f"CAST(NULLIF(TRIM(col), '*') AS DOUBLE). "
                        f"If broken JOIN: verify ON clause includes all required keys."
                    ),
                })

        has_issues = len(suspect) > 0
        return ToolResult(
            success=True,
            data={
                "issues_found": has_issues,
                "issue_count": len(suspect),
                "columns_checked": numeric_cols,
                "details": suspect,
                "suggested_fix": (
                    "Zero/NULL columns detected — likely failed CASTs or broken "
                    "JOINs. Rebuild from bronze with corrected SQL."
                    if has_issues else ""
                ),
            },
            summary=(
                f"ZERO COLUMNS: {len(suspect)} columns are entirely zeros/NULLs — "
                f"likely data pipeline failure."
                if has_issues
                else f"Zero column check OK: {len(numeric_cols)} numeric columns have data."
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auto_detect_ranges(self, df: pd.DataFrame) -> dict:
        """Infer expected value ranges from column names."""
        ranges = {}
        for col in df.columns:
            cl = col.lower()
            if "fips" in cl:
                ranges[col] = {"type": "fips"}
            elif any(k in cl for k in ["_pct", "percent", "rate", "penetration"]):
                ranges[col] = {"type": "percentage", "min": 0, "max": 100}
            elif "svi" in cl:
                ranges[col] = {"type": "proportion", "min": 0, "max": 1}
            elif "prevalence" in cl:
                ranges[col] = {"type": "percentage", "min": 0, "max": 100}
            elif any(k in cl for k in ["star", "rating", "overall_rating"]):
                ranges[col] = {"type": "stars", "min": 1, "max": 5}
            elif any(k in cl for k in ["enrollment", "members", "beneficiaries"]):
                ranges[col] = {"type": "enrollment", "min": 0, "max": 1_000_000}
        return ranges

    def _check_fips_column(self, df: pd.DataFrame, col: str) -> dict | None:
        """Validate a FIPS code column."""
        try:
            from versifai._utils.fips import validate_fips
        except ImportError:
            def validate_fips(x):
                return isinstance(x, str) and len(x) == 5 and x.isdigit()

        vals = df[col].dropna().astype(str)
        invalid = vals[~vals.apply(validate_fips)]
        if len(invalid) == 0:
            return None
        return {
            "violation_count": int(len(invalid)),
            "total": int(len(vals)),
            "violation_pct": round(len(invalid) / len(vals) * 100, 1),
            "examples": invalid.head(5).tolist(),
            "fix": (
                f"Column '{col}' has {len(invalid)} invalid FIPS codes. "
                f"Apply: LPAD(TRIM({col}), 5, '0') and filter state portion 01-78."
            ),
        }

    def _check_numeric_range(
        self, df: pd.DataFrame, col: str,
        min_val: float | None = None, max_val: float | None = None,
    ) -> dict | None:
        """Check a numeric column against expected range."""
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric) == 0:
            return None

        violations = pd.Series(False, index=numeric.index)
        if min_val is not None:
            violations |= numeric < min_val
        if max_val is not None:
            violations |= numeric > max_val

        bad_count = int(violations.sum())
        if bad_count == 0:
            return None

        bad_vals = numeric[violations]
        return {
            "violation_count": bad_count,
            "total": int(len(numeric)),
            "violation_pct": round(bad_count / len(numeric) * 100, 1),
            "expected_range": [min_val, max_val],
            "actual_range": [float(numeric.min()), float(numeric.max())],
            "examples": [float(v) for v in bad_vals.head(5).values],
            "fix": (
                f"Column '{col}' has {bad_count} values outside "
                f"[{min_val}, {max_val}]. Check source data and transformations."
            ),
        }


def _is_numeric_str(val) -> bool:
    """Check if a string value can be parsed as a number."""
    try:
        float(str(val).strip())
        return True
    except (ValueError, TypeError):
        return False
