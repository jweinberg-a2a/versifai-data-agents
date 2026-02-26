"""
Agent memory: conversation history management and context windowing.

Keeps the conversation history within Claude's context limits while
preserving the most important context (system prompt, recent turns,
key decisions).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("agent.memory")


def _strip_images_from_blocks(blocks: list) -> list:
    """Replace image content blocks with a lightweight text placeholder."""
    result = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "image":
            result.append({"type": "text", "text": "[image removed to save context]"})
        else:
            result.append(b)
    return result


# Approximate token budget — leave room for system prompt + tool definitions
SUMMARY_TRIGGER = 30  # Summarize old messages when we exceed this count
KEEP_RECENT = 12  # Messages to keep after summarization

# Tool results beyond this age (in messages) get compressed to their summary
TOOL_RESULT_COMPRESS_AGE = 10  # Compress tool results older than this many messages


@dataclass
class DecisionRecord:
    """A logged decision the agent made during processing."""

    timestamp: str
    source_name: str
    decision: str
    reasoning: str
    tool_used: str | None = None


class AgentMemory:
    """
    Manages conversation history and decision logging for the agent.

    Handles context window management by summarizing old messages
    when the history gets too long. Supports per-source resets
    to keep the context clean across data sources.
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []
        self._decisions: list[DecisionRecord] = []
        self._source_summaries: dict[str, str] = {}
        self._context_notes: list[str] = []

    # ------------------------------------------------------------------
    # Message history management
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict]:
        """Return the current conversation history."""
        return self._messages

    def add_user_message(self, content: str) -> None:
        """Add a user message (or initial prompt) to the conversation."""
        self._messages.append({"role": "user", "content": content})
        self._maybe_summarize()

    def add_assistant_message(self, content: list[dict]) -> None:
        """
        Add an assistant message to the conversation.

        Content is the raw content blocks from Claude's response.
        """
        self._messages.append({"role": "assistant", "content": content})
        self._maybe_summarize()

    def add_tool_result(
        self,
        tool_use_id: str,
        result: str,
        is_error: bool = False,
        image_base64: str = "",
        image_media_type: str = "image/png",
    ) -> None:
        """Add a tool result message, optionally with an image.

        When image_base64 is provided, the tool result content becomes a list
        of content blocks (text + image) so Claude can see chart output.
        """
        if image_base64:
            # Multi-block content: text + image for Claude vision
            content = [
                {"type": "text", "text": result},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_base64,
                    },
                },
            ]
        else:
            content = result  # type: ignore[assignment]

        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                        "is_error": is_error,
                    }
                ],
            }
        )
        self._compress_old_tool_results()
        self._maybe_summarize()

    def add_context_note(self, note: str) -> None:
        """Add a persistent context note that survives summarization."""
        self._context_notes.append(note)

    def reset_for_new_source(self, source_summary: str = "") -> None:
        """
        Reset the conversation for a new data source.

        Clears all messages but preserves decisions, source summaries,
        and context notes. Optionally logs a summary of the previous source.
        """
        if source_summary:
            # Auto-detect the source name from the summary or use a generic key
            self._context_notes.append(source_summary)

        self._messages.clear()

        logger.debug(
            f"Memory reset for new source. "
            f"Preserved: {len(self._decisions)} decisions, "
            f"{len(self._source_summaries)} source summaries, "
            f"{len(self._context_notes)} context notes."
        )

    def get_carryover_context(self) -> str:
        """
        Build a context string of everything learned so far.
        Injected into the prompt for a new source so the agent retains
        cross-source knowledge (schema decisions, FIPS patterns, etc).
        """
        parts = []

        if self._source_summaries:
            parts.append("## Sources Already Processed")
            for name, summ in self._source_summaries.items():
                parts.append(f"- **{name}**: {summ}")

        if self._context_notes:
            parts.append("\n## Key Notes From Prior Work")
            for note in self._context_notes[-10:]:  # Last 10 notes
                parts.append(f"- {note}")

        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Tool result compression — trim old, large results in-place
    # ------------------------------------------------------------------

    def _compress_old_tool_results(self) -> None:
        """Compress tool results older than TOOL_RESULT_COMPRESS_AGE messages.

        Large tool results (explore_volume, profile_data, etc.) consume
        thousands of tokens per turn.  Once the agent has acted on them,
        the detailed JSON is no longer needed — a short summary suffices.
        This replaces old tool_result content with a truncated version,
        keeping tokens under control without losing critical context.

        Also strips base64 images from old tool results — images are the
        single biggest token sink (15K-60K+ tokens each) and the agent has
        already seen and reasoned about them.
        """
        if len(self._messages) <= TOOL_RESULT_COMPRESS_AGE:
            return

        cutoff = len(self._messages) - TOOL_RESULT_COMPRESS_AGE
        self._compress_tool_results_in_range(0, cutoff)

    def _compress_tool_results_in_range(
        self, start: int, end: int, char_threshold: int = 500
    ) -> None:
        """Compress tool results in the given message index range.

        Handles both string content (truncation) and list content (image stripping).
        """
        for i in range(start, min(end, len(self._messages))):
            msg = self._messages[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue

                result_content = block.get("content", "")

                # String content — truncate large results
                if isinstance(result_content, str):
                    if len(result_content) <= char_threshold:
                        continue
                    summary_idx = result_content.find("SUMMARY (complete):")
                    if summary_idx >= 0:
                        compressed = result_content[:300] + "\n...\n" + result_content[summary_idx:]
                    else:
                        compressed = result_content[:300] + "\n[... compressed — old result ...]"
                    block["content"] = compressed

                # List content (text + image blocks) — strip images
                elif isinstance(result_content, list):
                    block["content"] = _strip_images_from_blocks(result_content)

    # ------------------------------------------------------------------
    # Emergency context reduction — called on ContextWindowExceeded
    # ------------------------------------------------------------------

    def strip_images(self) -> int:
        """Remove ALL base64 images from conversation history.

        Returns the number of images stripped. Images are replaced with a
        text placeholder so the agent knows a chart was viewed.
        """
        count = 0
        for msg in self._messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                result_content = block.get("content")
                if isinstance(result_content, list):
                    new_blocks = _strip_images_from_blocks(result_content)
                    if len(new_blocks) != len(result_content):
                        count += len(result_content) - len(new_blocks)
                    block["content"] = new_blocks
        return count

    def compress_all_tool_results(self) -> None:
        """Force-compress ALL tool results regardless of age.

        Uses a lower threshold (300 chars) than the normal compression.
        Called during emergency recalibration after context window overflow.
        """
        self._compress_tool_results_in_range(0, len(self._messages), char_threshold=300)

    def force_summarize(self) -> None:
        """Summarize the conversation regardless of message count.

        Used during emergency recalibration to reduce context size.
        """
        if len(self._messages) <= 3:
            return
        # Temporarily lower the trigger and run summarization
        original_trigger = SUMMARY_TRIGGER
        try:
            self._force_summarize_impl()
        finally:
            # SUMMARY_TRIGGER is a module constant, not mutated — we just
            # call the summarization logic directly instead
            _ = original_trigger

    def _force_summarize_impl(self) -> None:
        """Internal: run summarization unconditionally."""
        target_split = max(1, len(self._messages) - KEEP_RECENT)
        split_idx = self._find_safe_split_index(target_split)
        if split_idx <= 1:
            return

        old_messages = self._messages[1:split_idx]
        recent_msgs = self._messages[split_idx:]
        if not old_messages:
            return

        summary_parts = ["[CONVERSATION HISTORY SUMMARY — compressed to reduce context]"]
        for msg in old_messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                summary_parts.append(f"{role}: {content[:150]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            summary_parts.append(f"{role}: {block['text'][:150]}")
                        elif block.get("type") == "tool_use":
                            summary_parts.append(f"{role}: called '{block['name']}'")
                        elif block.get("type") == "tool_result":
                            content_str = str(block.get("content", ""))[:80]
                            summary_parts.append(f"tool_result: {content_str}")

        summary_text = "\n".join(summary_parts[:40])

        first_msg = self._messages[0]
        self._messages = [
            first_msg,
            {"role": "user", "content": summary_text},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Understood. Context compressed. Continuing.",
                    }
                ],
            },
        ] + recent_msgs

        logger.info(
            f"Force-summarized: compressed {len(old_messages)} messages, "
            f"history now has {len(self._messages)} messages."
        )

    def recalibrate(self) -> None:
        """Emergency context reduction pipeline.

        Called when the LLM rejects a request due to context window overflow.
        Applies three progressive reduction steps:
          1. Strip all base64 images (biggest per-item savings)
          2. Compress all tool results aggressively (300 char threshold)
          3. Force-summarize old conversation turns
        """
        images_stripped = self.strip_images()
        logger.info(f"Recalibrate step 1: stripped {images_stripped} images from memory.")

        self.compress_all_tool_results()
        logger.info("Recalibrate step 2: compressed all tool results.")

        self.force_summarize()
        logger.info(
            f"Recalibrate step 3: force-summarized. History now has {len(self._messages)} messages."
        )

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _find_safe_split_index(self, target_idx: int) -> int:
        """
        Find a safe index to split messages at, ensuring we never cut
        between a tool_use (in an assistant message) and its tool_result
        (in the following user message).

        Walks backward from target_idx until we find a position where
        the message at that index is NOT a tool_result referencing a
        tool_use from before the split.
        """
        idx = target_idx

        while idx > 1 and idx < len(self._messages):
            msg = self._messages[idx]

            # Check if this message is a tool_result
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                has_tool_result = any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in msg["content"]
                )
                if has_tool_result:
                    # This is a tool_result — we can't split here.
                    # Move back to include the preceding assistant message too.
                    idx -= 1
                    continue

            # Check if this is an assistant message with tool_use
            # (the next message might be its tool_result)
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                has_tool_use = any(
                    (hasattr(block, "type") and block.type == "tool_use")
                    or (isinstance(block, dict) and block.get("type") == "tool_use")
                    for block in msg["content"]
                )
                if has_tool_use:
                    # Don't split right after a tool_use — its result follows
                    idx -= 1
                    continue

            # Safe to split here
            break

        return max(1, idx)  # Never go before the first message

    def _maybe_summarize(self) -> None:
        """
        If the message history exceeds the trigger threshold,
        compress older messages into a summary.

        Carefully avoids splitting tool_use / tool_result pairs.
        """
        if len(self._messages) < SUMMARY_TRIGGER:
            return

        # Find a safe split point
        target_split = len(self._messages) - KEEP_RECENT
        split_idx = self._find_safe_split_index(target_split)

        # Don't summarize if we can't find a good split point
        if split_idx <= 1:
            return

        old_messages = self._messages[1:split_idx]  # Skip first message
        recent_msgs = self._messages[split_idx:]

        if not old_messages:
            return

        # Build a summary of old messages
        summary_parts = ["[CONVERSATION HISTORY SUMMARY]"]
        for msg in old_messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                summary_parts.append(f"{role}: {content[:200]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            summary_parts.append(f"{role}: {block['text'][:200]}")
                        elif block.get("type") == "tool_use":
                            summary_parts.append(
                                f"{role}: called '{block['name']}' → {json.dumps(block.get('input', {}))[:80]}"
                            )
                        elif block.get("type") == "tool_result":
                            content_str = str(block.get("content", ""))[:100]
                            is_err = block.get("is_error", False)
                            summary_parts.append(
                                f"tool_result ({'ERROR' if is_err else 'ok'}): {content_str}"
                            )
                    elif hasattr(block, "type"):
                        # Anthropic SDK content block objects
                        if block.type == "text":
                            summary_parts.append(f"{role}: {block.text[:200]}")
                        elif block.type == "tool_use":
                            summary_parts.append(
                                f"{role}: called '{block.name}' → {json.dumps(block.input)[:80]}"
                            )

        summary_text = "\n".join(summary_parts[:60])

        # Add persistent context
        if self._context_notes:
            summary_text += "\n\n[CONTEXT NOTES]\n" + "\n".join(self._context_notes[-10:])
        if self._source_summaries:
            summary_text += "\n\n[SOURCES PROCESSED]"
            for name, summ in self._source_summaries.items():
                summary_text += f"\n- {name}: {summ}"

        # Rebuild: first msg + summary + "understood" + recent
        first_msg = self._messages[0]
        self._messages = [
            first_msg,
            {"role": "user", "content": summary_text},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Understood. I have the context from previous work. Continuing with the task.",
                    }
                ],
            },
        ] + recent_msgs

        logger.debug(
            f"Memory summarized: compressed {len(old_messages)} messages, "
            f"history now has {len(self._messages)} messages."
        )

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def log_decision(
        self, source_name: str, decision: str, reasoning: str, tool_used: str | None = None
    ) -> None:
        """Log a decision made by the agent."""
        self._decisions.append(
            DecisionRecord(
                timestamp=datetime.now().isoformat(),
                source_name=source_name,
                decision=decision,
                reasoning=reasoning,
                tool_used=tool_used,
            )
        )

    def log_source_summary(self, source_name: str, summary: str) -> None:
        """Store a summary for a completed source (survives context compression)."""
        self._source_summaries[source_name] = summary

    @property
    def decisions(self) -> list[DecisionRecord]:
        return self._decisions

    @property
    def source_summaries(self) -> dict[str, str]:
        return self._source_summaries

    def get_decisions_for_source(self, source_name: str) -> list[DecisionRecord]:
        return [d for d in self._decisions if d.source_name == source_name]

    # ------------------------------------------------------------------
    # Serialization (for checkpointing)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "message_count": len(self._messages),
            "decisions": [
                {
                    "timestamp": d.timestamp,
                    "source_name": d.source_name,
                    "decision": d.decision,
                    "reasoning": d.reasoning,
                    "tool_used": d.tool_used,
                }
                for d in self._decisions
            ],
            "source_summaries": self._source_summaries,
            "context_notes": self._context_notes,
        }
