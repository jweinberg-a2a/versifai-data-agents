"""
Tool: create_custom_tool

Allows the agent to dynamically create and register new tools at runtime.
This enables self-healing: when the agent encounters a problem it can't solve
with existing tools, it can author a custom tool to handle it.

Security: The dynamic tools execute in a restricted scope with only safe
imports (pandas, os, json, re, etc.). No shell execution or network access
beyond what's already available.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime

import pandas as pd

from versifai._utils.fips import pad_fips_series
from versifai.core.tools.base import BaseTool, ToolResult
from versifai.core.tools.registry import ToolRegistry

logger = logging.getLogger("agent.dynamic_tools")

# Safe modules available to dynamic tools — intentionally restrictive.
# Only in-memory data processing; no filesystem, no network, no shell.
# Note: os is included for os.path and os.walk (dangerous methods like
# os.remove, os.system etc. are blocked by security guardrails).
SAFE_IMPORTS = {
    "os": os,
    "re": re,
    "json": json,
    "pd": pd,
    "pandas": pd,
    "datetime": datetime,
    "math": __import__("math"),
    "pad_fips_series": pad_fips_series,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "abs": abs,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "type": type,
    "None": None,
    "True": True,
    "False": False,
}


class DynamicTool(BaseTool):
    """
    A tool created dynamically at runtime from agent-provided Python code.
    """

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict,
        execute_code: str,
        extra_scope: dict | None = None,
    ) -> None:
        self._name = tool_name
        self._description = tool_description
        self._parameters = tool_parameters
        self._execute_code = execute_code
        self._extra_scope = extra_scope or {}
        self._compiled_fn: Callable | None = None
        self._compile()

    def _compile(self) -> None:
        """Compile the execute code into a callable function."""
        # Wrap the user code in a function definition
        fn_code = "def _dynamic_execute(**kwargs):\n"

        # Auto-extract declared parameters into local variables so the agent's
        # code can reference them directly (e.g. `base_path`) instead of
        # requiring `kwargs["base_path"]` everywhere.
        if isinstance(self._parameters, dict):
            props = self._parameters.get("properties", {})
            if isinstance(props, dict):
                for param_name in props:
                    fn_code += f"    {param_name} = kwargs.get('{param_name}')\n"

        for line in self._execute_code.split("\n"):
            fn_code += f"    {line}\n"

        # Create execution namespace with safe imports + extra scope
        namespace = dict(SAFE_IMPORTS)
        namespace["ToolResult"] = ToolResult
        namespace.update(self._extra_scope)

        exec(fn_code, namespace)
        self._compiled_fn = namespace["_dynamic_execute"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict:
        return self._parameters

    def _execute(self, **kwargs) -> ToolResult:
        if self._compiled_fn is None:
            return ToolResult(success=False, error="Dynamic tool was not compiled successfully.")
        return self._compiled_fn(**kwargs)


class DynamicToolBuilderTool(BaseTool):
    """
    Meta-tool that allows the agent to create new tools at runtime.

    The agent provides:
    - Tool name, description, and parameter schema
    - Python code that implements the tool's execute logic

    The code must return a ToolResult object.
    """

    def __init__(self, registry: ToolRegistry, transformer_tool=None) -> None:
        super().__init__()
        self._registry = registry
        self._transformer = transformer_tool
        self._created_tools: list[str] = []

    @property
    def name(self) -> str:
        return "create_custom_tool"

    @property
    def description(self) -> str:
        return (
            "Create a new custom tool at runtime and register it for immediate use. "
            "Use this when you encounter a problem that existing tools cannot solve — "
            "for example, a custom data cleaning function, a specialized column parser, "
            "aggregation from tract-level to county-level, or a data transformation "
            "that doesn't fit existing tools. "
            "IMPORTANT CONSTRAINTS: Custom tools can ONLY process data in memory using "
            "pandas and return ToolResult objects. They CANNOT: write files, access the "
            "network, run shell commands, install packages, delete anything, or directly "
            "access Spark/dbutils. For file I/O use existing tools, for catalog writes "
            "use write_to_catalog. "
            "STAGING DATA: To save an in-memory DataFrame for catalog writing, call "
            "stage_dataframe(source_name, df) inside your code. This stages the "
            "DataFrame so you can then call write_to_catalog with that source_name. "
            "Available in code: re, json, pandas (as pd), datetime, math, ToolResult, "
            "pad_fips_series(series, width), stage_dataframe, and Python builtins. "
            "\n\n"
            "## PERFORMANCE BEST PRACTICES (MUST follow for data >10K rows)\n"
            "These rules are critical. Datasets can have millions of rows. A single "
            ".apply() call on 3M rows takes minutes; the vectorized equivalent takes seconds.\n\n"
            "**NEVER use row-by-row operations:**\n"
            "- NEVER .apply(lambda ...) — this is a Python for-loop in disguise\n"
            "- NEVER .iterrows() or .itertuples() for transformation\n"
            "- NEVER for row in df.iterrows(): ... — vectorize instead\n\n"
            "**USE vectorized pandas operations:**\n"
            "- String ops: df['col'].str.strip(), .str.replace(pat, rep, regex=True), "
            ".str.zfill(5), .str.upper(), .str.contains(), .str.extract()\n"
            "- Numeric: pd.to_numeric(df['col'], errors='coerce'), df['col'].astype('Int64')\n"
            "- Conditionals: df['col'].where(condition, other=default), "
            "df['col'].mask(condition, replacement), df.loc[mask, 'col'] = value\n"
            "- Null handling: df['col'].fillna(value), df['col'].isna(), df.dropna(subset=[...])\n"
            "- FIPS padding: pad_fips_series(df['col'], width=5) — use this, not apply()\n"
            "- Date parsing: pd.to_datetime(df['col'], errors='coerce') — not apply(dateutil.parse)\n\n"
            "**DataFrame memory best practices:**\n"
            "- NEVER pd.concat() inside a loop. Collect DataFrames in a list, concat ONCE at end.\n"
            "- After pd.concat(), call df = df.copy() to defragment memory.\n"
            "- Build DataFrames from a dict of columns: pd.DataFrame({'a': s1, 'b': s2}), "
            "not column-by-column assignment.\n"
            "- Use df.select_dtypes(include=['object']).columns to batch-process string columns.\n"
            "- Free large intermediates with del df when done.\n"
            "- For large CSV reads, use dtype={'col': 'str'} to avoid mixed-type inference.\n\n"
            "Example return: ToolResult(success=True, data={'key': 'value'}, summary='Done')"
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": (
                        "Unique name for the new tool (snake_case). "
                        "Example: 'parse_fixed_width_ahrf'"
                    ),
                },
                "tool_description": {
                    "type": "string",
                    "description": "Description of what the tool does (shown in tool list).",
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "JSON Schema for the tool's input parameters. "
                        "Example: {\"type\": \"object\", \"properties\": {\"file_path\": {\"type\": \"string\"}}, \"required\": [\"file_path\"]}"
                    ),
                },
                "code": {
                    "type": "string",
                    "description": (
                        "Python code for the tool's execute function. "
                        "Receives kwargs matching the parameters schema. "
                        "Must return a ToolResult(success=bool, data=any, summary=str). "
                        "Available: os, re, json, pd (pandas), datetime, ToolResult, "
                        "pad_fips_series(series, width) for vectorized FIPS zero-padding, "
                        "stage_dataframe(source_name, df) for staging DataFrames for catalog writing. "
                        "Code is the BODY of a function — no 'def' needed. "
                        "CRITICAL: Use ONLY vectorized pandas ops. NEVER .apply(lambda), "
                        ".iterrows(), or Python for-loops over rows. See the tool description "
                        "for the full list of best practices. "
                        "Example: 'df = pd.read_csv(kwargs[\"file_path\"])\n"
                        "df[\"fips\"] = pad_fips_series(df[\"fips\"], width=5)\n"
                        "agg = df.groupby(\"county_fips\").sum().reset_index()\n"
                        "stage_dataframe(\"my_source\", agg)\n"
                        "return ToolResult(success=True, data={\"rows\": len(agg)}, summary=\"Aggregated and staged\")'"
                    ),
                },
            },
            "required": ["tool_name", "tool_description", "parameters", "code"],
        }

    def _execute(
        self,
        tool_name: str,
        tool_description: str,
        parameters: dict,
        code: str,
        **kwargs,
    ) -> ToolResult:
        # Validate the tool name
        if not re.match(r"^[a-z][a-z0-9_]*$", tool_name):
            return ToolResult(
                success=False,
                error=f"Invalid tool name '{tool_name}'. Must be snake_case starting with a letter.",
            )

        # Check for name conflicts
        existing = self._registry.get(tool_name)
        if existing is not None:
            return ToolResult(
                success=False,
                error=(
                    f"Tool '{tool_name}' already exists. "
                    f"Use a different name or use modify_existing_tool to update it."
                ),
            )

        # Security guardrails: dynamic tools can ONLY process data in memory
        # and return results. No filesystem writes, no installs, no deletions,
        # no network access, no shell commands.
        dangerous_patterns = [
            (r"subprocess", "shell command execution"),
            (r"__import__", "dynamic imports"),
            (r"(?<!\w)eval\s*\(", "eval() execution"),
            (r"(?<!\w)exec\s*\(", "exec() execution"),
            (r"shutil", "file system operations"),
            (r"os\.remove", "file deletion"),
            (r"os\.unlink", "file deletion"),
            (r"os\.rmdir", "directory deletion"),
            (r"os\.makedirs", "directory creation"),
            (r"os\.mkdir", "directory creation"),
            (r"os\.rename", "file renaming"),
            (r"os\.system", "shell command execution"),
            (r"open\s*\(", "direct file I/O (use existing tools instead)"),
            (r"pip\s+install", "package installation"),
            (r"pip\.main", "package installation"),
            (r"importlib", "dynamic imports"),
            (r"pkgutil", "package utilities"),
            (r"requests\.", "network access (use web_search tool instead)"),
            (r"urllib", "network access"),
            (r"socket", "network access"),
            (r"pathlib.*write", "file writing"),
            (r"pathlib.*unlink", "file deletion"),
            (r"\.to_csv\(", "file writing (use write_to_catalog tool instead)"),
            (r"\.to_parquet\(", "file writing (use write_to_catalog tool instead)"),
            (r"\.to_excel\(", "file writing (use write_to_catalog tool instead)"),
            (r"dbutils", "direct dbutils access"),
            (r"spark\.", "direct Spark access (use execute_sql tool instead)"),
        ]
        for pattern, reason in dangerous_patterns:
            if re.search(pattern, code):
                return ToolResult(
                    success=False,
                    error=(
                        f"Security guardrail: code contains '{pattern}' ({reason}). "
                        f"Dynamic tools can ONLY process data in memory using pandas "
                        f"and return ToolResult objects. For file I/O use existing tools. "
                        f"For writing data use write_to_catalog. For web access use web_search."
                    ),
                )

        # Build extra scope — inject stage_dataframe if transformer is available
        extra_scope: dict = {}
        if self._transformer is not None:
            extra_scope["stage_dataframe"] = self._transformer.stage_dataframe
        else:
            # No-op fallback so code referencing stage_dataframe gets a clear error
            def _no_stage(source_name, df, append=False):
                raise RuntimeError(
                    "stage_dataframe is not available — no transformer tool configured."
                )
            extra_scope["stage_dataframe"] = _no_stage

        # Try to compile and create the tool
        try:
            dynamic_tool = DynamicTool(
                tool_name=tool_name,
                tool_description=tool_description,
                tool_parameters=parameters,
                execute_code=code,
                extra_scope=extra_scope,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to compile tool code: {e}\n\nCode:\n{code}",
                summary=f"Tool '{tool_name}' failed to compile.",
            )

        # Test the tool compiles correctly by checking the function exists
        if dynamic_tool._compiled_fn is None:
            return ToolResult(
                success=False,
                error="Tool compiled but no function was created. Check the code syntax.",
            )

        # Register the tool
        try:
            self._registry.register(dynamic_tool)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

        self._created_tools.append(tool_name)

        logger.info(f"Dynamic tool '{tool_name}' created and registered successfully.")

        return ToolResult(
            success=True,
            data={
                "tool_name": tool_name,
                "description": tool_description,
                "parameters": parameters,
                "registered": True,
                "total_custom_tools": len(self._created_tools),
            },
            summary=(
                f"Custom tool '{tool_name}' created and registered. "
                f"You can now use it like any other tool."
            ),
        )

    @property
    def created_tools(self) -> list[str]:
        return self._created_tools
