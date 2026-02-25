"""
FIPS code normalization and validation utilities.

County FIPS codes are 5-digit strings (2-digit state + 3-digit county),
always zero-padded.  Example: "01001" = Autauga County, Alabama.
"""

from __future__ import annotations

import re

import pandas as pd


def pad_state_fips(value) -> str:
    """Normalize a state FIPS code to a 2-digit zero-padded string."""
    if value is None:
        return ""
    s = str(value).strip()
    # Remove any decimal (e.g. "1.0" → "1")
    s = re.sub(r"\.0+$", "", s)
    if not s or not s.isdigit():
        return ""
    return s.zfill(2)


def pad_county_fips(value) -> str:
    """Normalize a county FIPS code to a 5-digit zero-padded string."""
    if value is None:
        return ""
    s = str(value).strip()
    s = re.sub(r"\.0+$", "", s)
    if not s or not s.isdigit():
        return ""
    return s.zfill(5)


def pad_fips_series(series: pd.Series, width: int = 5) -> pd.Series:
    """Vectorized FIPS padding — ~50x faster than .apply(pad_county_fips).

    Converts the series to string, strips decimals (.0), strips whitespace,
    zero-pads to `width`, and masks non-numeric values as None.

    Args:
        series: A pandas Series of FIPS codes (int, float, or string).
        width: Target width (5 for county, 2 for state).
    """
    s = series.astype(str).str.strip()
    # Remove trailing .0 / .00 etc (from float columns: "1001.0" → "1001")
    s = s.str.replace(r"\.0+$", "", regex=True)
    # Mask non-numeric and empty/null-like values
    is_valid = s.str.match(r"^\d+$") & ~s.isin(["", "nan", "None", "NaN", "<NA>"])
    # Zero-pad
    s = s.str.zfill(width)
    # Replace invalid entries with None
    return s.where(is_valid, other=None)


def normalize_fips(state_fips, county_fips=None) -> str:
    """
    Build a 5-digit county FIPS from components.

    If county_fips is already 5 digits, return it zero-padded.
    If county_fips is 3 digits, prepend the state_fips.
    """
    if county_fips is not None:
        full = pad_county_fips(county_fips)
        if len(full) == 5:
            return full
        # 3-digit county portion
        county_part = str(county_fips).strip()
        county_part = re.sub(r"\.0+$", "", county_part).zfill(3)
        state_part = pad_state_fips(state_fips)
        if state_part and county_part:
            return state_part + county_part
    return pad_county_fips(state_fips)


def validate_fips(fips: str) -> bool:
    """Check that a FIPS code looks valid (5 digits, state 01-78)."""
    if not fips or len(fips) != 5 or not fips.isdigit():
        return False
    state = int(fips[:2])
    return 1 <= state <= 78
