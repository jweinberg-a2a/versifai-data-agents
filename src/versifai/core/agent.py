"""
BaseAgent — shared ReAct loop infrastructure for all agentic orchestrators.

Extracts the common run-phase logic (send → extract_actions → execute tools →
add to memory → check stop) so that DataScientistAgent, StoryTellerAgent, and
future agents all share the same loop without code duplication.
"""

from __future__ import annotations

import logging

from versifai.core.display import AgentDisplay
from versifai.core.llm import LLMClient
from versifai.core.memory import AgentMemory
from versifai.core.tools.registry import ToolRegistry

logger = logging.getLogger("agent.base")


class BaseAgent:
    """
    Base class providing the ReAct loop and shared infrastructure.

    Subclasses must:
      - Call ``super().__init__(display, memory, llm, registry)``
      - Implement ``_register_tools()``
      - Set ``self._system_prompt``
    """

    # Dump the display log to the progress file every N turns.
    _PROGRESS_DUMP_INTERVAL = 20

    def __init__(
        self,
        display: AgentDisplay,
        memory: AgentMemory,
        llm: LLMClient,
        registry: ToolRegistry,
    ) -> None:
        self._display = display
        self._memory = memory
        self._llm = llm
        self._registry = registry

        # Error tracking
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # Track repeated missing-parameter patterns (LLM output size limits)
        self._missing_param_tracker: dict[str, dict[str, int]] = {}

        # Subclasses must set this
        self._system_prompt: str = ""

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _get_tool_definitions(self) -> list[dict]:
        """Get tool definitions including ask_human pseudo-tool."""
        tools = self._registry.to_claude_tools()
        tools.append(
            {
                "name": "ask_human",
                "description": (
                    "Pause execution and ask the human operator a question. "
                    "Use when you need clarification about research direction, "
                    "metric interpretation, or domain context."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the operator.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context to help the operator answer.",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of suggested answers.",
                        },
                    },
                    "required": ["question"],
                },
            }
        )
        return tools

    # ------------------------------------------------------------------
    # Instruction injection
    # ------------------------------------------------------------------

    def _inject_instructions(self, prompt: str) -> str:
        """Prepend user instructions to a phase prompt if set."""
        instructions = getattr(self, "_instructions", "")
        if instructions:
            return f"## Operator Instructions\n{instructions}\n\n---\n\n{prompt}"
        return prompt

    # ------------------------------------------------------------------
    # Phase runner — ReAct loop
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        prompt: str,
        max_turns: int,
        progress_path: str = "",
    ) -> str:
        """Run a single agent phase with the ReAct loop.

        Parameters
        ----------
        prompt : str
            The user-role message that kicks off this phase.
        max_turns : int
            Maximum LLM round-trips before forcing a stop.
        progress_path : str, optional
            File path to periodically dump display logs. If empty, progress
            dumping is skipped.
        """
        self._memory.add_user_message(prompt)

        last_text = ""
        turns = 0

        while turns < max_turns:
            turns += 1
            context_recovered = False
            self._display.step(f"Turn {turns}/{max_turns}")

            # Periodic progress dump so operators can tail the file
            if progress_path and turns % self._PROGRESS_DUMP_INTERVAL == 0:
                self._display.dump_progress(progress_path)

            try:
                response = self._llm.send(
                    messages=self._memory.messages,
                    system=self._system_prompt,
                    tools=self._get_tool_definitions(),
                )
            except Exception as e:
                err_str = str(e).lower()
                is_context_overflow = any(
                    kw in err_str for kw in ("context window", "prompt is too long", "tokens >")
                )
                if is_context_overflow and not context_recovered:
                    self._display.warning(
                        "Context window exceeded — recalibrating inputs and retrying..."
                    )
                    self._memory.recalibrate()
                    context_recovered = True
                    continue  # retry this turn
                self._display.error(f"LLM call failed: {e}")
                break

            actions = LLMClient.extract_actions(response)
            self._memory.add_assistant_message(response.content)

            has_tool_use = False
            for action in actions:
                if action["type"] == "text":
                    last_text = action["text"]
                    self._display.thinking(action["text"])

                elif action["type"] == "tool_use":
                    has_tool_use = True
                    tool_name = action["name"]
                    tool_input = action["input"]
                    tool_use_id = action["id"]

                    self._display.tool_call(tool_name, tool_input)

                    # Handle ask_human
                    if tool_name == "ask_human":
                        result_str = self._display.ask_human(
                            question=tool_input.get("question", ""),
                            context=tool_input.get("context", ""),
                            options=tool_input.get("options"),
                        )
                        self._memory.add_tool_result(tool_use_id, result_str or "(no response)")
                        self._display.tool_result(tool_name, result_str or "(no response)")
                        continue

                    # PRE-CHECK: Detect repeated missing-parameter failures
                    intercept = self._check_missing_param_pattern(tool_name, tool_input)
                    if intercept:
                        result_str = intercept
                        self._consecutive_errors += 1
                        self._display.tool_result(
                            tool_name,
                            intercept[:400],
                            is_error=True,
                        )
                        self._memory.add_tool_result(
                            tool_use_id,
                            result_str,
                            is_error=True,
                        )
                        continue

                    # Execute tool
                    result = self._registry.execute(tool_name, **tool_input)
                    result_str = result.to_content_str()

                    # Error tracking
                    if not result.success:
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= self._max_consecutive_errors:
                            result_str += (
                                f"\n\n[CRITICAL — {self._consecutive_errors} CONSECUTIVE ERRORS]\n"
                                f"You MUST change your approach NOW. DO NOT retry the same call.\n\n"
                                f"You may be hitting LLM output size limits — the parameter "
                                f"you're trying to produce is too large to fit in a single response.\n\n"
                                f"SOLUTION: Use `create_custom_tool` to build a helper function that "
                                f"generates the large parameter programmatically in Python code. "
                                f"Or use `ask_human` for guidance.\n\n"
                                f"DO NOT retry the same call. Change your approach."
                            )
                    else:
                        self._consecutive_errors = 0

                    # Truncate large results
                    if len(result_str) > 15000:
                        result_str = result_str[:15000] + "\n\n[... TRUNCATED ...]"

                    self._display.tool_result(
                        tool_name,
                        result.summary or result_str[:400],
                        is_error=not result.success,
                    )

                    # If the tool returned a chart image, encode it for
                    # Claude vision so the agent can review the output
                    image_b64 = ""
                    if result.image_path and result.success:
                        image_b64 = self._read_chart_image(result.image_path)

                    self._memory.add_tool_result(
                        tool_use_id,
                        result_str,
                        is_error=not result.success,
                        image_base64=image_b64,
                    )

            # Stop if Claude ended its turn without tool calls
            if response.stop_reason == "end_turn" and not has_tool_use:
                self._display.step("Phase complete.")
                break

        if turns >= max_turns:
            self._display.warning(f"Reached max turns ({max_turns}). Moving on.")

        # Always dump progress at end of phase
        if progress_path:
            self._display.dump_progress(progress_path)

        return last_text

    # ------------------------------------------------------------------
    # Output-limit detection — catch repeated missing parameter patterns
    # ------------------------------------------------------------------

    def _check_missing_param_pattern(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> str | None:
        """
        Detect when the LLM keeps calling a tool without a required parameter.

        This happens when the parameter is too large for the LLM to produce
        in a single response — the parameter silently gets dropped. After 2
        occurrences of the same missing required parameter, intercept the call
        and return guidance to use create_custom_tool instead.

        Returns None if no problem detected, or a guidance string to return
        to the agent instead of executing the tool.
        """
        tool_obj = self._registry.get(tool_name)
        if tool_obj is None:
            return None

        schema = tool_obj.parameters_schema
        required = set(schema.get("required", []))
        provided = set(tool_input.keys())
        missing = required - provided

        if not missing:
            self._missing_param_tracker.pop(tool_name, None)
            return None

        if tool_name not in self._missing_param_tracker:
            self._missing_param_tracker[tool_name] = {}

        for param in missing:
            self._missing_param_tracker[tool_name][param] = (
                self._missing_param_tracker[tool_name].get(param, 0) + 1
            )

        repeated = {
            p: count for p, count in self._missing_param_tracker[tool_name].items() if count >= 2
        }

        if not repeated:
            return None

        param_names = ", ".join(f"'{p}'" for p in repeated)
        counts = ", ".join(f"{p}: {c}x" for p, c in repeated.items())

        props = schema.get("properties", {})
        param_details = []
        for p in repeated:
            ptype = props.get(p, {}).get("type", "unknown")
            pdesc = props.get(p, {}).get("description", "")
            param_details.append(f"  - {p} ({ptype}): {pdesc[:100]}")
        param_block = "\n".join(param_details)

        return (
            f"ERROR: Required parameter(s) {param_names} missing from '{tool_name}' "
            f"call — this has happened {counts} times.\n\n"
            f"THIS IS LIKELY AN LLM OUTPUT SIZE LIMIT. You are trying to produce a "
            f"parameter that is too large to fit in your response. The parameter keeps "
            f"getting silently dropped.\n\n"
            f"Missing parameter details:\n{param_block}\n\n"
            f"YOU MUST USE A DIFFERENT APPROACH:\n\n"
            f"**Use `create_custom_tool` to build a helper function** that generates "
            f"the large parameter in Python code and calls the target tool programmatically.\n\n"
            f"DO NOT attempt to call '{tool_name}' directly again with this pattern. "
            f"Delegate to code."
        )

    # ------------------------------------------------------------------
    # Image reading for Claude vision
    # ------------------------------------------------------------------

    @staticmethod
    def _read_chart_image(image_path: str, max_width: int = 400) -> str:
        """Read a chart PNG and return a resized base64 string for Claude vision.

        Resizes to max_width to keep token costs reasonable. Returns empty
        string if the file can't be read (e.g., HTML plotly charts).
        """
        if not image_path or not image_path.lower().endswith(".png"):
            return ""
        try:
            import base64
            import io

            from PIL import Image

            img = Image.open(image_path)
            # Resize if wider than max_width, maintaining aspect ratio
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)  # type: ignore[assignment, attr-defined]

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            # PIL not available or file unreadable — skip image
            return ""
