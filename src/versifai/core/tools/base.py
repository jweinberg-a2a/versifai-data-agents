"""
Base tool interface for the agentic framework.

Every tool the agent can invoke extends BaseTool.  Tools expose:
  - A name, description, and JSON Schema of parameters (sent to Claude).
  - An execute() method that performs the action and returns a ToolResult.
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Structured result returned by every tool invocation."""

    success: bool
    data: Any = None
    error: str = ""
    summary: str = ""  # short human-readable summary for display
    image_path: str = ""  # optional path to a chart image (PNG) to show Claude

    def to_content_str(self) -> str:
        """Serialize to a string suitable for returning to the LLM."""
        if self.success:
            if isinstance(self.data, (dict, list)):
                return json.dumps(self.data, indent=2, default=str)
            return str(self.data)
        else:
            return f"ERROR: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for agent tools.

    Subclasses must implement:
      - name (property)
      - description (property)
      - parameters_schema (property) — JSON Schema dict
      - _execute(**kwargs) -> ToolResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used in Claude tool_use calls."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description shown to Claude."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """
        JSON Schema describing the tool's input parameters.

        Example:
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Volume path to explore"}
                },
                "required": ["path"]
            }
        """
        ...

    @abstractmethod
    def _execute(self, **kwargs) -> ToolResult:
        """
        Core execution logic.  Override in subclasses.

        Receives validated keyword arguments and returns a ToolResult.
        Exceptions raised here are caught by execute() and converted to
        error ToolResults for self-healing.
        """
        ...

    def execute(self, **kwargs) -> ToolResult:
        """
        Public entry point.  Wraps _execute with error handling so that
        any exception is captured and returned as an error ToolResult
        instead of crashing the agent loop.

        Special handling for TypeError (missing/wrong parameters): instead
        of a raw traceback, returns the tool's parameter schema so the
        agent can self-correct in one retry.
        """
        try:
            return self._execute(**kwargs)
        except TypeError as exc:
            # Almost always means missing or mistyped parameters.
            # Give the agent the exact schema it needs to fix the call.
            schema = self.parameters_schema
            required = schema.get("required", [])
            props = schema.get("properties", {})
            param_hints = []
            for name, spec in props.items():
                req = " (REQUIRED)" if name in required else ""
                desc = spec.get("description", "")
                ptype = spec.get("type", "any")
                param_hints.append(f"  - {name}: {ptype}{req} — {desc}")
            param_block = "\n".join(param_hints) if param_hints else "  (see tool description)"

            return ToolResult(
                success=False,
                error=(
                    f"Parameter error calling '{self.name}': {exc}\n\n"
                    f"You provided: {list(kwargs.keys())}\n\n"
                    f"Expected parameters:\n{param_block}\n\n"
                    f"Fix: ensure all REQUIRED parameters are included with correct types."
                ),
                summary=f"Tool '{self.name}' called with wrong parameters: {exc}",
            )
        except Exception as exc:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}\n\nTraceback:\n{tb}",
                summary=f"Tool '{self.name}' failed: {exc}",
            )

    def to_claude_tool_definition(self) -> dict:
        """
        Return the tool definition in the format expected by
        the Anthropic API's tool_use feature.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }
