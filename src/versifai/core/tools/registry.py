"""
Tool registry: maps tool names to instances and provides Claude-formatted definitions.
"""

from __future__ import annotations

from versifai.core.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """Central registry of all tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def execute(self, tool_name: str, /, **kwargs) -> ToolResult:
        """
        Look up a tool by name and execute it.

        The ``/`` makes ``tool_name`` positional-only so it can never
        collide with a keyword argument in ``**kwargs`` — even if a
        tool's own input schema includes a ``tool_name`` parameter
        (e.g. ``create_custom_tool``).
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: '{tool_name}'. Available tools: {list(self._tools.keys())}",
            )
        return tool.execute(**kwargs)

    def to_claude_tools(self) -> list[dict]:
        """Return all tool definitions in Anthropic API format."""
        return [t.to_claude_tool_definition() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
