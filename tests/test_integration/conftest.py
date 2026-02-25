"""Shared fixtures for agent-level integration tests."""

from __future__ import annotations

import json
import os

import pytest

from versifai.core.tools.save_note import SaveNoteTool
from versifai.story_agents.storyteller.tools.cite_source import CiteSourceTool
from versifai.story_agents.storyteller.tools.evaluate_evidence import EvaluateEvidenceTool

# ---------------------------------------------------------------------------
# Real, stable citation used across all tests
# ---------------------------------------------------------------------------
# Source: CMS (Centers for Medicare & Medicaid Services)
# Dataset: CMS Program Statistics — Medicare Total Enrollment
# URL: https://data.cms.gov/summary-statistics-on-beneficiary-enrollment/
#      medicare-and-medicaid-reports/cms-program-statistics-medicare-total-enrollment
# Key fact: ~65.7 million Americans enrolled in Medicare as of 2023.
# This is a long-term stable US government dataset published annually since 1966.

CMS_CITATION = {
    "title": "CMS Program Statistics — Medicare Total Enrollment",
    "author": "Centers for Medicare & Medicaid Services (CMS)",
    "year": "2023",
    "url": (
        "https://data.cms.gov/summary-statistics-on-beneficiary-enrollment/"
        "medicare-and-medicaid-reports/"
        "cms-program-statistics-medicare-total-enrollment"
    ),
    "key_stat": "65.7 million Medicare enrollees",
    "organization": "CMS",
}


# ---------------------------------------------------------------------------
# Skip marker — tests require an LLM API key
# ---------------------------------------------------------------------------

requires_llm = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="No LLM API key set (ANTHROPIC_API_KEY or OPENAI_API_KEY required)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def review_tools(tmp_path):
    """Tool set for citation/evidence review tests."""
    (tmp_path / "notes").mkdir(exist_ok=True)
    return [
        EvaluateEvidenceTool(),
        CiteSourceTool(),
        SaveNoteTool(notes_path=str(tmp_path / "notes")),
    ]


@pytest.fixture
def findings_cms_misattributed_to_cdc():
    """Finding where CMS Medicare data is incorrectly attributed to CDC."""
    return [
        {
            "title": "National Medicare enrollment reaches 65.7 million",
            "finding": (
                "According to CDC Program Statistics, total Medicare enrollment "
                "reached 65.7 million beneficiaries in 2023. This represents "
                "a 2.4% increase from the prior year."
            ),
            "source": "CDC (Centers for Disease Control and Prevention)",
            "source_url": "https://www.cdc.gov/",
            "p_value": None,
            "effect_size": None,
            "significance": "high",
        },
    ]


@pytest.fixture
def findings_wrong_enrollment_number():
    """Finding where the Medicare enrollment number is drastically wrong."""
    return [
        {
            "title": "Medicare enrollment at 45.2 million",
            "finding": (
                "CMS Program Statistics report that 45.2 million Americans "
                "were enrolled in Medicare as of 2023. This is roughly 14% "
                "of the US population."
            ),
            "source": "CMS (Centers for Medicare & Medicaid Services)",
            "source_url": CMS_CITATION["url"],
            "p_value": None,
            "effect_size": None,
            "significance": "high",
        },
    ]


@pytest.fixture
def findings_medicare_medicaid_confused():
    """Finding that confuses Medicare with Medicaid."""
    return [
        {
            "title": "Medicaid enrollment for seniors reaches 65.7 million",
            "finding": (
                "CMS reports that Medicaid enrollment among Americans aged 65+ "
                "reached 65.7 million in 2023. As a means-tested program for "
                "low-income individuals, this represents the growing healthcare "
                "needs of America's aging population."
            ),
            "source": "CMS (Centers for Medicare & Medicaid Services)",
            "source_url": CMS_CITATION["url"],
            "p_value": None,
            "effect_size": None,
            "significance": "high",
        },
    ]


@pytest.fixture
def findings_inflated_text_weak_stats():
    """Finding where narrative text wildly overclaims weak statistics."""
    return [
        {
            "title": "GROUNDBREAKING: Income perfectly predicts enrollment",
            "finding": (
                "We discovered a HIGHLY SIGNIFICANT and DEFINITIVE relationship "
                "between county income and Medicare Advantage enrollment rates. "
                "This is the most important finding in the entire analysis and "
                "PROVES a direct causal link. The evidence is OVERWHELMING and "
                "IRREFUTABLE."
            ),
            "p_value": 0.73,
            "effect_size": "negligible (d=0.02)",
            "significance": "low",
            "source": "Internal analysis",
        },
    ]
