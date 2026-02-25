"""
Notebook display helpers for verbose agent output and human-in-the-loop interaction.

Works inside Databricks notebooks via displayHTML / dbutils.widgets.
Falls back to clean print() when running outside a notebook.
"""

from __future__ import annotations

import html
import logging
import os
import re
import textwrap
from datetime import datetime
from typing import Optional

logger = logging.getLogger("agent.display")

# ---------------------------------------------------------------------------
# Shared CSS injected once into the notebook
# ---------------------------------------------------------------------------
_CHAT_CSS = """
<style>
.agent-container { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; }
.agent-phase {
    margin: 20px 0 12px 0; padding: 14px 20px;
    background: linear-gradient(135deg, #1e1e2e 0%, #2d2b55 100%);
    border-radius: 12px; border-left: 5px solid #7c3aed;
    color: #e2e8f0; font-size: 17px; font-weight: 700; letter-spacing: 0.5px;
}
.agent-bubble {
    margin: 6px 0 6px 0; padding: 12px 18px;
    background: #1e293b; border-radius: 12px;
    color: #e2e8f0; font-size: 14px; line-height: 1.6;
    border: 1px solid #334155; position: relative;
}
.agent-bubble .label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 4px;
}
.agent-bubble.thinking .label { color: #fbbf24; }
.agent-bubble.thinking { border-left: 3px solid #fbbf24; background: #1c1917; }
.agent-bubble.tool { border-left: 3px solid #3b82f6; background: #0f172a; }
.agent-bubble.tool .label { color: #60a5fa; }
.agent-bubble.result { border-left: 3px solid #10b981; background: #022c22; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }
.agent-bubble.result .label { color: #34d399; }
.agent-bubble.result.error { border-left: 3px solid #ef4444; background: #1c0a0a; }
.agent-bubble.result.error .label { color: #f87171; }
.agent-bubble.success { border-left: 3px solid #10b981; background: #052e16; }
.agent-bubble.success .label { color: #34d399; }
.agent-bubble.warn { border-left: 3px solid #f59e0b; background: #1c1305; }
.agent-bubble.warn .label { color: #fbbf24; }
.agent-bubble.err { border-left: 3px solid #ef4444; background: #1c0a0a; }
.agent-bubble.err .label { color: #f87171; }
.agent-bubble.human {
    border: 2px solid #8b5cf6; background: #1e1b4b; border-radius: 12px;
}
.agent-bubble.human .label { color: #a78bfa; }
.agent-step {
    margin: 2px 0 2px 12px; padding: 4px 12px;
    color: #64748b; font-size: 12px; font-family: monospace;
}
.agent-table {
    border-collapse: collapse; width: 100%; margin-top: 8px;
    font-size: 13px; color: #e2e8f0;
}
.agent-table th { background: #1e293b; padding: 8px 14px; text-align: left; border-bottom: 2px solid #334155; color: #94a3b8; font-weight: 600; }
.agent-table td { padding: 6px 14px; border-bottom: 1px solid #1e293b; }
.agent-table tr:hover td { background: #0f172a; }
</style>
"""

def _in_databricks() -> bool:
    """Detect if we're running inside a Databricks notebook."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        return spark is not None
    except Exception:
        return False


class AgentDisplay:
    """
    Clean chat-style display for the agent.

    In Databricks: renders styled HTML bubbles via displayHTML.
    Outside: prints clean formatted text with unicode icons.
    """

    # After this many displayHTML calls, clear notebook output and show
    # only recent entries.  Keeps the notebook responsive.
    MAX_NOTEBOOK_DISPLAYS = 60

    def __init__(self, dbutils=None):
        self._dbutils = dbutils
        self._is_databricks = dbutils is not None or _in_databricks()
        # Rolling plain-text log of everything displayed
        self._log: list[str] = []
        # Counter for notebook display calls (for periodic truncation)
        self._display_count = 0

    # ------------------------------------------------------------------
    # Progress log
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        """Append a plain-text line to the in-memory log."""
        self._log.append(text)

    def dump_progress(self, path: str) -> None:
        """Overwrite *path* with the full plain-text log so far.

        Builds the full string first and writes in a single call —
        Databricks FUSE mounts don't support append and can behave
        unpredictably with multiple writes within one open().
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = (
            f"Progress dump — {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            + "=" * 72
            + "\n\n"
            + "\n".join(self._log)
            + "\n"
        )
        with open(path, "w") as f:
            f.write(content)

    # ------------------------------------------------------------------
    # HTML rendering with automatic truncation
    # ------------------------------------------------------------------

    def _display_html(self, html_content: str) -> None:
        """Render HTML in the notebook or fall back to print.

        CSS is included with EVERY call because Databricks isolates
        each displayHTML() output into its own block — styles from a
        previous block don't carry over.

        After ``MAX_NOTEBOOK_DISPLAYS`` calls the cell output is cleared
        so the notebook stays responsive.  The full log is always
        available in the progress file.
        """
        if self._is_databricks:
            self._display_count += 1
            if self._display_count % self.MAX_NOTEBOOK_DISPLAYS == 0:
                self._clear_notebook_output()
                self._raw_html(
                    f'{_CHAT_CSS}<div class="agent-container">'
                    f'<div class="agent-step" style="color:#94a3b8;">'
                    f'--- Earlier output truncated (see progress.txt in notes/) ---'
                    f'</div></div>'
                )
            self._raw_html(
                f'{_CHAT_CSS}<div class="agent-container">{html_content}</div>'
            )
        else:
            plain = re.sub(r"<[^>]+>", "", html_content)
            print(plain.strip())

    def _clear_notebook_output(self) -> None:
        """Clear the current cell's accumulated output."""
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except Exception:
            pass

    def _raw_html(self, content: str) -> None:
        try:
            if self._dbutils:
                self._dbutils.notebook.displayHTML(content)
            else:
                from IPython.display import display, HTML
                display(HTML(content))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def phase(self, title: str) -> None:
        """Display a major phase header."""
        self._append_log(f"\n{'='*72}\n  {title}\n{'='*72}")
        if self._is_databricks:
            self._display_html(
                f'<div class="agent-phase">{html.escape(title)}</div>'
            )
        else:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
        logger.debug(f"PHASE: {title}")

    def step(self, message: str) -> None:
        """Display a minor step (turn counter, etc). Kept minimal."""
        self._append_log(f"  {message}")
        if self._is_databricks:
            self._display_html(f'<div class="agent-step">{html.escape(message)}</div>')
        else:
            print(f"  {message}")
        logger.debug(f"STEP: {message}")

    def thinking(self, thought: str) -> None:
        """Display the agent's reasoning as a chat bubble."""
        self._append_log(f"\n  [Reasoning]\n  {thought[:2000]}")
        # Truncate very long thoughts for display (full text stays in memory)
        display_text = thought[:1200] + ("..." if len(thought) > 1200 else "")
        escaped = html.escape(display_text).replace("\n", "<br>")
        if self._is_databricks:
            self._display_html(
                f'<div class="agent-bubble thinking">'
                f'<div class="label">Agent Reasoning</div>'
                f'{escaped}</div>'
            )
        else:
            wrapped = textwrap.fill(display_text, width=88, initial_indent="  ", subsequent_indent="  ")
            print(f"\n  [Reasoning]")
            print(wrapped)
        logger.debug(f"THINKING: {thought[:200]}")

    def tool_call(self, tool_name: str, params: dict) -> None:
        """Display a tool invocation."""
        params_plain = ", ".join(f"{k}={str(v)[:80]}" for k, v in params.items())
        self._append_log(f"\n  > {tool_name}({params_plain})")

        # Show params concisely — truncate long values
        param_parts = []
        for k, v in params.items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            param_parts.append(f"<b>{html.escape(k)}</b>={html.escape(val_str)}")
        params_html = ", ".join(param_parts)

        if self._is_databricks:
            self._display_html(
                f'<div class="agent-bubble tool">'
                f'<div class="label">Tool Call</div>'
                f'<code style="color:#93c5fd;">{html.escape(tool_name)}</code>'
                f'<span style="color:#64748b;font-size:12px;margin-left:8px;">'
                f'({params_html})</span></div>'
            )
        else:
            print(f"\n  > {tool_name}({params_plain})")
        logger.debug(f"TOOL: {tool_name}")

    def tool_result(self, tool_name: str, result_preview: str, is_error: bool = False) -> None:
        """Display a tool result."""
        preview = result_preview[:600]
        icon = "x" if is_error else "+"
        self._append_log(f"  [{icon}] {tool_name}: {preview[:300]}")

        escaped = html.escape(preview).replace("\n", "<br>")
        err_class = " error" if is_error else ""
        label = "Error" if is_error else "Result"

        if self._is_databricks:
            self._display_html(
                f'<div class="agent-bubble result{err_class}">'
                f'<div class="label">{html.escape(tool_name)} - {label}</div>'
                f'{escaped}</div>'
            )
        else:
            # Keep result output compact
            lines = preview.split("\n")
            compact = "\n    ".join(lines[:8])
            if len(lines) > 8:
                compact += f"\n    ... ({len(lines) - 8} more lines)"
            print(f"  [{icon}] {tool_name}: {compact}")

        if is_error:
            logger.debug(f"TOOL ERROR [{tool_name}]: {preview[:120]}")
        else:
            logger.debug(f"TOOL OK [{tool_name}]")

    def success(self, message: str) -> None:
        """Display a success message."""
        self._append_log(f"\n  [OK] {message}")
        if self._is_databricks:
            self._display_html(
                f'<div class="agent-bubble success">'
                f'<div class="label">Success</div>'
                f'{html.escape(message)}</div>'
            )
        else:
            print(f"\n  [OK] {message}")
        logger.info(f"SUCCESS: {message}")

    def warning(self, message: str) -> None:
        """Display a warning."""
        self._append_log(f"\n  [!] {message}")
        if self._is_databricks:
            self._display_html(
                f'<div class="agent-bubble warn">'
                f'<div class="label">Warning</div>'
                f'{html.escape(message)}</div>'
            )
        else:
            print(f"\n  [!] {message}")
        logger.warning(f"WARNING: {message}")

    def error(self, message: str) -> None:
        """Display an error."""
        self._append_log(f"\n  [X] {message}")
        if self._is_databricks:
            self._display_html(
                f'<div class="agent-bubble err">'
                f'<div class="label">Error</div>'
                f'{html.escape(message)}</div>'
            )
        else:
            print(f"\n  [X] {message}")
        logger.error(f"ERROR: {message}")

    def summary_table(self, title: str, rows: list[dict]) -> None:
        """Display a summary table."""
        if not rows:
            return
        headers = list(rows[0].keys())

        if self._is_databricks:
            header_row = "".join(
                f"<th>{html.escape(h)}</th>" for h in headers
            )
            body_rows = ""
            for row in rows:
                cells = "".join(
                    f"<td>{html.escape(str(row.get(h, '')))}</td>"
                    for h in headers
                )
                body_rows += f"<tr>{cells}</tr>"

            self._display_html(
                f'<div class="agent-bubble">'
                f'<div class="label">{html.escape(title)}</div>'
                f'<table class="agent-table">'
                f'<thead><tr>{header_row}</tr></thead>'
                f'<tbody>{body_rows}</tbody></table></div>'
            )
        else:
            print(f"\n  {title}")
            print(f"  {'-' * 60}")
            for row in rows:
                line = " | ".join(f"{k}: {v}" for k, v in row.items())
                print(f"    {line}")

    # ------------------------------------------------------------------
    # Human-in-the-loop
    # ------------------------------------------------------------------

    def ask_human(
        self,
        question: str,
        context: str = "",
        options: Optional[list[str]] = None,
    ) -> str:
        """
        Pause the agent and ask the human operator a question.

        Always uses Python ``input()`` for reliability — Databricks
        widgets don't work well in all notebook execution contexts.
        Displays the question with HTML formatting if in Databricks,
        then collects the answer via stdin.
        """
        # Show formatted question in notebook if possible
        if self._is_databricks:
            options_html = ""
            if options:
                items = "".join(
                    f'<li style="margin:4px 0;color:#c4b5fd;">{html.escape(o)}</li>'
                    for o in options
                )
                options_html = f'<ul style="margin:8px 0;padding-left:20px;">{items}</ul>'

            context_html = ""
            if context:
                context_html = (
                    f'<div style="color:#94a3b8;font-size:12px;margin-top:8px;'
                    f'padding:8px;background:#0f172a;border-radius:6px;'
                    f'font-family:monospace;white-space:pre-wrap;">'
                    f'{html.escape(context)}</div>'
                )

            self._display_html(
                f'<div class="agent-bubble human">'
                f'<div class="label">Agent Needs Your Input</div>'
                f'<div style="font-size:15px;margin:8px 0;">{html.escape(question)}</div>'
                f'{context_html}{options_html}'
                f'<div style="color:#64748b;font-size:11px;margin-top:8px;">'
                f'Type your answer below.</div></div>'
            )

        # Always print to stdout so the question is visible in all contexts
        print(f"\n{'='*60}")
        print(f"  AGENT NEEDS YOUR INPUT")
        print(f"{'='*60}")
        print(f"  {question}")
        if context:
            print(f"\n  Context: {context[:500]}")
        if options:
            print()
            for i, opt in enumerate(options, 1):
                print(f"    {i}. {opt}")
        print()

        # Collect input via Python input() — blocks until user responds
        answer = input("  Your answer> ")

        # Show the response as a styled bubble so it's visible in the output
        if self._is_databricks and answer:
            self._display_html(
                f'<div class="agent-bubble" style="border-left:3px solid #8b5cf6;'
                f'background:#1e1b4b;">'
                f'<div class="label" style="color:#a78bfa;">Human Response</div>'
                f'{html.escape(answer)}</div>'
            )
        print(f"  Human responded: {answer}")
        return answer
