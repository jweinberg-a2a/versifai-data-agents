"""Tests for DynamicToolBuilderTool."""

from __future__ import annotations

from versifai.core.tools.dynamic_tool_builder import DynamicToolBuilderTool
from versifai.core.tools.registry import ToolRegistry


def _make_builder() -> tuple[DynamicToolBuilderTool, ToolRegistry]:
    registry = ToolRegistry()
    builder = DynamicToolBuilderTool(registry=registry)
    registry.register(builder)
    return builder, registry


class TestDynamicToolBuilder:
    def test_create_simple_tool(self):
        builder, registry = _make_builder()
        result = builder.execute(
            tool_name="add_numbers",
            tool_description="Add two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            code='return ToolResult(success=True, data={"sum": a + b})',
        )
        assert result.success is True
        assert "add_numbers" in registry.tool_names

    def test_execute_created_tool(self):
        builder, registry = _make_builder()
        builder.execute(
            tool_name="multiply",
            tool_description="Multiply two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
            },
            code='return ToolResult(success=True, data={"product": x * y})',
        )
        result = registry.execute("multiply", x=3, y=7)
        assert result.success is True
        assert result.data["product"] == 21

    def test_security_rejects_subprocess(self):
        builder, _registry = _make_builder()
        result = builder.execute(
            tool_name="bad_tool",
            tool_description="Should be blocked.",
            parameters={"type": "object", "properties": {}},
            code='import subprocess; subprocess.run(["ls"])',
        )
        assert result.success is False
        assert (
            "security" in result.error.lower()
            or "dangerous" in result.error.lower()
            or "blocked" in result.error.lower()
            or "not allowed" in result.error.lower()
        )

    def test_security_rejects_open(self):
        builder, _registry = _make_builder()
        result = builder.execute(
            tool_name="file_tool",
            tool_description="Should be blocked.",
            parameters={"type": "object", "properties": {}},
            code='f = open("/etc/passwd"); data = f.read()',
        )
        assert result.success is False

    def test_security_rejects_os_operations(self):
        builder, _registry = _make_builder()
        result = builder.execute(
            tool_name="os_tool",
            tool_description="Should be blocked.",
            parameters={"type": "object", "properties": {}},
            code='os.remove("/tmp/important")',
        )
        assert result.success is False

    def test_name_must_be_snake_case(self):
        builder, _registry = _make_builder()
        result = builder.execute(
            tool_name="BadName",
            tool_description="Should fail naming check.",
            parameters={"type": "object", "properties": {}},
            code='return ToolResult(success=True, data="ok")',
        )
        assert result.success is False

    def test_duplicate_name_rejected(self):
        builder, _registry = _make_builder()
        builder.execute(
            tool_name="my_tool",
            tool_description="First.",
            parameters={"type": "object", "properties": {}},
            code='return ToolResult(success=True, data="ok")',
        )
        result = builder.execute(
            tool_name="my_tool",
            tool_description="Duplicate.",
            parameters={"type": "object", "properties": {}},
            code='return ToolResult(success=True, data="ok")',
        )
        assert result.success is False

    def test_created_tools_tracked(self):
        builder, _registry = _make_builder()
        builder.execute(
            tool_name="tracked_tool",
            tool_description="Track me.",
            parameters={"type": "object", "properties": {}},
            code='return ToolResult(success=True, data="ok")',
        )
        assert "tracked_tool" in builder.created_tools
