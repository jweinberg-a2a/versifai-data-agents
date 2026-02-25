"""
Column naming conventions for data engineering standards.

All target columns use lowercase snake_case with descriptive names.
"""

from __future__ import annotations

import re


def to_snake_case(name: str) -> str:
    """
    Convert a column name to lowercase snake_case.

    Handles: CamelCase, Title Case, UPPER_CASE, spaces, hyphens,
    dots, parentheses, and other common patterns found in raw data.
    """
    if not name:
        return name

    # Remove leading/trailing whitespace
    s = name.strip()

    # Replace common separators with underscores
    s = re.sub(r"[\s\-\.\/\\]+", "_", s)

    # Remove parentheses and other special chars (keep underscores and alphanumeric)
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)

    # Insert underscore before uppercase runs: "HTTPSCode" → "HTTPS_Code"
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    # Insert underscore between lowercase/digit and uppercase: "countyName" → "county_Name"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)

    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)

    # Lowercase and strip leading/trailing underscores
    s = s.lower().strip("_")

    # Ensure it doesn't start with a digit (prefix with underscore if so)
    if s and s[0].isdigit():
        s = "_" + s

    return s


def standardize_column_names(columns: list[str]) -> dict[str, str]:
    """
    Given a list of original column names, return a mapping of
    original → snake_case standardized names.

    Handles duplicates by appending a numeric suffix.
    """
    mapping: dict[str, str] = {}
    seen: dict[str, int] = {}

    for original in columns:
        snake = to_snake_case(original)
        if not snake:
            snake = "unnamed_column"

        if snake in seen:
            seen[snake] += 1
            snake = f"{snake}_{seen[snake]}"
        else:
            seen[snake] = 0

        mapping[original] = snake

    return mapping
