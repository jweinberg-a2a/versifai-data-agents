"""Tests for SaveNoteTool."""

from __future__ import annotations

import os

from versifai.core.tools.save_note import SaveNoteTool


class TestSaveNote:
    def test_save_creates_file(self, tmp_path):
        notes_dir = str(tmp_path / "notes")
        tool = SaveNoteTool(notes_path=notes_dir)
        result = tool.execute(theme_id="theme_0", note="First observation")
        assert result.success is True
        assert os.path.isfile(os.path.join(notes_dir, "theme_0_notes.txt"))

    def test_save_appends_via_read_write(self, tmp_path):
        """Volumes use FUSE — no file append. Tool must read-then-write."""
        notes_dir = str(tmp_path / "notes")
        tool = SaveNoteTool(notes_path=notes_dir)
        tool.execute(theme_id="theme_0", note="First note")
        tool.execute(theme_id="theme_0", note="Second note")

        content = (tmp_path / "notes" / "theme_0_notes.txt").read_text()
        assert "First note" in content
        assert "Second note" in content

    def test_returns_file_path(self, tmp_path):
        notes_dir = str(tmp_path / "notes")
        tool = SaveNoteTool(notes_path=notes_dir)
        result = tool.execute(theme_id="theme_1", note="A note")
        assert result.success is True
        assert "theme_1" in str(result.data)

    def test_in_memory_accumulation(self):
        tool = SaveNoteTool()
        tool.execute(theme_id="theme_0", note="mem note 1")
        tool.execute(theme_id="theme_0", note="mem note 2")
        tool.execute(theme_id="theme_1", note="different theme")

        assert len(tool.notes["theme_0"]) == 2
        assert len(tool.notes["theme_1"]) == 1

    def test_missing_theme_id_error(self):
        tool = SaveNoteTool()
        result = tool.execute(theme_id="", note="some note")
        assert result.success is False

    def test_missing_note_error(self):
        tool = SaveNoteTool()
        result = tool.execute(theme_id="theme_0", note="")
        assert result.success is False

    def test_get_notes_for_theme(self):
        tool = SaveNoteTool()
        tool.execute(theme_id="theme_0", note="note A")
        notes = tool.get_notes_for_theme("theme_0")
        assert "note A" in notes

    def test_load_notes_from_disk(self, tmp_path):
        notes_dir = str(tmp_path / "notes")
        tool = SaveNoteTool(notes_path=notes_dir)
        tool.execute(theme_id="theme_0", note="disk note")

        # Create fresh tool to load from disk
        tool2 = SaveNoteTool(notes_path=notes_dir)
        loaded = tool2.load_notes_from_disk()
        assert "theme_0" in loaded
        assert "disk note" in loaded["theme_0"]
