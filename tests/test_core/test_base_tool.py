"""Tests for BaseTool, ToolResult, and ToolRegistry."""

from __future__ import annotations

import pytest

from versifai.core.tools.base import BaseTool, ToolResult
from versifai.core.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Concrete tool for testing
# ---------------------------------------------------------------------------

class _EchoTool(BaseTool):
    """Minimal concrete tool that echoes its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the input back."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Text to echo."},
            },
            "required": ["message"],
        }

    def _execute(self, message: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, data={"echoed": message}, summary=f"Echoed: {message}")


class _FailingTool(BaseTool):
    """Tool that always raises an exception."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def _execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("intentional error")


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_success_result(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error == ""

    def test_error_result(self):
        result = ToolResult(success=False, error="something went wrong")
        assert result.success is False
        assert result.error == "something went wrong"

    def test_to_content_str_success_dict(self):
        result = ToolResult(success=True, data={"a": 1})
        content = result.to_content_str()
        assert '"a": 1' in content

    def test_to_content_str_success_list(self):
        result = ToolResult(success=True, data=[1, 2, 3])
        content = result.to_content_str()
        assert "1" in content

    def test_to_content_str_success_string(self):
        result = ToolResult(success=True, data="plain text")
        assert result.to_content_str() == "plain text"

    def test_to_content_str_error(self):
        result = ToolResult(success=False, error="bad input")
        assert result.to_content_str() == "ERROR: bad input"

    def test_summary_and_image_path(self):
        result = ToolResult(success=True, data="x", summary="done", image_path="/tmp/chart.png")
        assert result.summary == "done"
        assert result.image_path == "/tmp/chart.png"


# ---------------------------------------------------------------------------
# BaseTool tests
# ---------------------------------------------------------------------------

class TestBaseTool:
    def test_tool_properties(self):
        tool = _EchoTool()
        assert tool.name == "echo"
        assert "Echoes" in tool.description
        assert "message" in tool.parameters_schema["properties"]

    def test_execute_returns_result(self):
        tool = _EchoTool()
        result = tool.execute(message="hello")
        assert result.success is True
        assert result.data["echoed"] == "hello"

    def test_execute_catches_general_exception(self):
        tool = _FailingTool()
        result = tool.execute()
        assert result.success is False
        assert "intentional error" in result.error
        assert "RuntimeError" in result.error

    def test_execute_catches_type_error(self):
        """Missing required parameter → TypeError → schema hint returned."""
        tool = _EchoTool()
        # Call without the required 'message' parameter
        result = tool.execute()
        assert result.success is False
        assert "Parameter error" in result.error
        assert "message" in result.error  # schema hints

    def test_to_claude_tool_definition(self):
        tool = _EchoTool()
        defn = tool.to_claude_tool_definition()
        assert defn["name"] == "echo"
        assert "description" in defn
        assert defn["input_schema"]["type"] == "object"
        assert "message" in defn["input_schema"]["properties"]


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = _EchoTool()
        registry.register(tool)
        assert registry.get("echo") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_EchoTool())

    def test_execute_by_name(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        result = registry.execute("echo", message="test")
        assert result.success is True
        assert result.data["echoed"] == "test"

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent")
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_tool_names(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        registry.register(_FailingTool())
        assert set(registry.tool_names) == {"echo", "fail"}

    def test_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(_EchoTool())
        assert len(registry) == 1

    def test_to_claude_tools(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        tools = registry.to_claude_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
