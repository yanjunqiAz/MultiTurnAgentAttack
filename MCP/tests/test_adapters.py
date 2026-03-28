"""Tests for MCP/core/adapters.py — state adapters.

Tests use a mock call_tool function (no real MCP servers needed).
"""

import json
import pytest

from MCP.core.adapters import (
    ADAPTER_REGISTRY,
    BrowserAdapter,
    DatabaseAdapter,
    FilesystemAdapter,
    MCPStateAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CallLog:
    """Records tool calls for assertion."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.responses = responses or {}

    def __call__(self, tool_name: str, arguments: dict) -> str:
        self.calls.append((tool_name, arguments))
        return self.responses.get(tool_name, "")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_all_keys_present(self):
        assert "database" in ADAPTER_REGISTRY
        assert "filesystem" in ADAPTER_REGISTRY
        assert "browser" in ADAPTER_REGISTRY

    def test_values_are_adapter_subclasses(self):
        for cls in ADAPTER_REGISTRY.values():
            assert issubclass(cls, MCPStateAdapter)


# ---------------------------------------------------------------------------
# DatabaseAdapter
# ---------------------------------------------------------------------------

class TestDatabaseAdapter:

    def test_seed_executes_statements(self):
        log = CallLog()
        adapter = DatabaseAdapter()
        adapter.seed_state(log, {
            "statements": [
                "CREATE TABLE users (id INT)",
                "INSERT INTO users VALUES (1)",
            ],
        })
        assert len(log.calls) == 2
        assert all(name == "execute_sql" for name, _ in log.calls)
        assert log.calls[0][1]["query"] == "CREATE TABLE users (id INT)"

    def test_seed_empty_config(self):
        log = CallLog()
        DatabaseAdapter().seed_state(log, {})
        assert log.calls == []

    def test_reset_drops_tables(self):
        log = CallLog(responses={
            "execute_sql": json.dumps([{"tablename": "users"}, {"tablename": "logs"}]),
        })
        DatabaseAdapter().reset_state(log)
        # First call: list tables, then one DROP per table
        assert len(log.calls) == 3
        assert "DROP TABLE" in log.calls[1][1]["query"]
        assert "DROP TABLE" in log.calls[2][1]["query"]

    def test_snapshot_returns_dict(self):
        log = CallLog(responses={"execute_sql": "users, logs"})
        result = DatabaseAdapter().snapshot_state(log)
        assert "tables" in result

    def test_parse_tables_json(self):
        raw = json.dumps([{"tablename": "a"}, {"tablename": "b"}])
        assert DatabaseAdapter._parse_tables(raw) == ["a", "b"]

    def test_parse_tables_text_fallback(self):
        raw = "tablename\n---------\nusers\nlogs"
        tables = DatabaseAdapter._parse_tables(raw)
        assert "users" in tables
        assert "logs" in tables

    def test_seed_handles_error_gracefully(self):
        """Adapter should log warning, not crash, on tool failure."""
        def failing_call(name, args):
            raise RuntimeError("connection refused")
        DatabaseAdapter().seed_state(failing_call, {"statements": ["SELECT 1"]})


# ---------------------------------------------------------------------------
# FilesystemAdapter
# ---------------------------------------------------------------------------

class TestFilesystemAdapter:

    def test_seed_writes_files(self):
        log = CallLog()
        FilesystemAdapter().seed_state(log, {
            "files": [
                {"path": "/sandbox/a.txt", "content": "hello"},
                {"path": "/sandbox/b.txt", "content": "world"},
            ],
        })
        assert len(log.calls) == 2
        assert all(name == "write_file" for name, _ in log.calls)
        assert log.calls[0][1]["path"] == "/sandbox/a.txt"

    def test_reset_clears_sandbox(self):
        log = CallLog()
        FilesystemAdapter().reset_state(log)
        assert len(log.calls) == 1
        assert log.calls[0][0] == "list_directory"

    def test_snapshot_lists_directory(self):
        log = CallLog(responses={"list_directory": "a.txt\nb.txt"})
        result = FilesystemAdapter().snapshot_state(log)
        assert "files" in result


# ---------------------------------------------------------------------------
# BrowserAdapter
# ---------------------------------------------------------------------------

class TestBrowserAdapter:

    def test_seed_navigates(self):
        log = CallLog()
        BrowserAdapter().seed_state(log, {"navigate_to": "http://example.com"})
        assert len(log.calls) == 1
        assert log.calls[0][0] == "navigate"
        assert log.calls[0][1]["url"] == "http://example.com"

    def test_seed_no_navigate(self):
        log = CallLog()
        BrowserAdapter().seed_state(log, {})
        assert log.calls == []

    def test_reset_navigates_blank(self):
        log = CallLog()
        BrowserAdapter().reset_state(log)
        assert log.calls[0][1]["url"] == "about:blank"

    def test_snapshot_captures_url_and_screenshot(self):
        log = CallLog(responses={
            "get_url": "http://example.com/page",
            "screenshot": "<base64data>",
        })
        result = BrowserAdapter().snapshot_state(log)
        assert result["url"] == "http://example.com/page"
        assert result["screenshot"] == "<base64data>"

    def test_snapshot_handles_missing_tools(self):
        """If get_url or screenshot tools don't exist, snapshot still works."""
        def partial_call(name, args):
            if name == "get_url":
                raise Exception("tool not found")
            return "screenshot_data"
        result = BrowserAdapter().snapshot_state(partial_call)
        assert "url" not in result
        assert result["screenshot"] == "screenshot_data"
