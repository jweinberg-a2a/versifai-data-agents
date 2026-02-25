"""
Multi-source planner: groups discovered files into data sources
and creates a processing plan.

This module provides helper functions used by the orchestrator
to structure the agent's initial discovery phase.
"""

from __future__ import annotations

import os
import re
from typing import Any

from versifai.data_agents.models.source import DataSource, FileGroup, FileInfo


# Heuristic mapping of subfolder/filename patterns to data source names
SOURCE_PATTERNS = {
    "medicare": "ma_enrollment",
    "ma_enrollment": "ma_enrollment",
    "penetration": "ma_penetration",
    "service_area": "ma_service_area",
    "svi": "cdc_svi",
    "social_vulnerability": "cdc_svi",
    "places": "cdc_places",
    "ahrf": "hrsa_ahrf",
    "area_health": "hrsa_ahrf",
    "food_access": "food_access",
    "food_desert": "food_access",
    "acs": "acs",
    "census": "acs",
    "demographic": "acs_demographics",
    "economic": "acs_economic",
    "social": "acs_social",
    "dp02": "acs_social",
    "dp03": "acs_economic",
    "dp05": "acs_demographics",
}


def suggest_source_name(folder_name: str, file_names: list[str]) -> str:
    """
    Heuristically suggest a data source name based on folder and file names.

    Returns a best-guess identifier like 'cdc_svi' or 'ma_enrollment'.
    If no match, returns the folder name in snake_case.
    """
    combined = (folder_name + " " + " ".join(file_names)).lower()

    for pattern, source_name in SOURCE_PATTERNS.items():
        if pattern in combined:
            return source_name

    # Fallback: clean up folder name
    clean = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")
    return clean or "unknown_source"


def build_file_groups_from_exploration(exploration_data: dict) -> list[FileGroup]:
    """
    Convert the raw output from the explore_volume tool into FileGroup objects.

    Groups files by their immediate parent directory.
    """
    groups: dict[str, FileGroup] = {}

    entries = exploration_data.get("entries", [])

    for entry in entries:
        if entry.get("type") == "directory":
            dir_path = entry["path"]
            dir_name = entry["name"]
            group = FileGroup(name=dir_name, folder_path=dir_path)

            # If the exploration was recursive and has children
            children = entry.get("children", [])
            for child in children:
                if child.get("type") == "file":
                    try:
                        fi = FileInfo.from_path(child["path"])
                        group.files.append(fi)
                    except (OSError, KeyError):
                        pass
                elif child.get("type") == "directory":
                    # Nested directory — scan its children too
                    for nested in child.get("children", []):
                        if nested.get("type") == "file":
                            try:
                                fi = FileInfo.from_path(nested["path"])
                                group.files.append(fi)
                            except (OSError, KeyError):
                                pass

            if group.files:
                group.classify_files()
                groups[dir_name] = group

        elif entry.get("type") == "file":
            # Files at the root level — group them together
            root_group = groups.setdefault(
                "_root", FileGroup(name="_root", folder_path=exploration_data.get("path", ""))
            )
            try:
                fi = FileInfo.from_path(entry["path"])
                root_group.files.append(fi)
            except (OSError, KeyError):
                pass

    # Classify root group if it exists
    if "_root" in groups and groups["_root"].files:
        groups["_root"].classify_files()

    return list(groups.values())


def create_data_sources(file_groups: list[FileGroup]) -> list[DataSource]:
    """
    Convert FileGroups into DataSource objects with suggested names
    and processing metadata.
    """
    sources = []
    for group in file_groups:
        file_names = [f.name for f in group.files]
        suggested_name = suggest_source_name(group.name, file_names)

        source = DataSource(
            name=suggested_name,
            file_group=group,
            description=f"Data source from folder '{group.name}' with {len(group.files)} files",
        )
        sources.append(source)

    return sources


def prioritize_sources(sources: list[DataSource]) -> list[DataSource]:
    """
    Order sources for processing.

    Priority:
    1. Sources with documentation files (easier to profile)
    2. Sources with fewer files (faster to process)
    3. Alphabetical for consistency
    """
    def sort_key(s: DataSource) -> tuple:
        has_docs = -len(s.file_group.documentation_files)  # negative = more docs first
        file_count = len(s.file_group.files)
        return (has_docs, file_count, s.name)

    return sorted(sources, key=sort_key)
