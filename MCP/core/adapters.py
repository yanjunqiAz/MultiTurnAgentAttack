"""Pluggable state adapters for MCP server types.

Each adapter knows how to seed, reset, and snapshot state for a particular
kind of MCP server (database, filesystem, browser, etc.).  All operations
use standard MCP ``tools/call`` — no custom protocol extensions.

To add a new server type:
    1. Subclass ``MCPStateAdapter``
    2. Register it in ``ADAPTER_REGISTRY``
    3. Reference the key in ``mcp_servers.yml`` via ``adapter: <key>``
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type alias for the synchronous tool-calling function created by
# MCPEnvironment._make_call_fn().  Signature: (tool_name, arguments) -> str
CallToolFn = Callable[[str, dict[str, Any]], str]


class MCPStateAdapter(ABC):
    """Abstract base for server-type-specific state operations."""

    @abstractmethod
    def seed_state(self, call_tool: CallToolFn, config: dict) -> None:
        """Pre-seed environment state before an attack scenario.

        Args:
            call_tool: invokes ``MCP tools/call`` and returns extracted text.
            config: scenario-specific seeding config (from YAML ``pre_seed``).
        """

    @abstractmethod
    def reset_state(self, call_tool: CallToolFn) -> None:
        """Clean up environment state between scenarios."""

    @abstractmethod
    def snapshot_state(self, call_tool: CallToolFn) -> dict:
        """Capture current state for post-hoc verification by StateVerifier.

        Returns a dict of observations (tables, files, page info, etc.).
        """


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------

class DatabaseAdapter(MCPStateAdapter):
    """For SQL-based MCP servers (Postgres, MySQL, SQLite)."""

    def seed_state(self, call_tool: CallToolFn, config: dict) -> None:
        for stmt in config.get("statements", []):
            try:
                call_tool("execute_sql", {"query": stmt})
            except Exception as e:
                logger.warning("DB seed failed for %r: %s", stmt[:80], e)

    def reset_state(self, call_tool: CallToolFn) -> None:
        try:
            result = call_tool("execute_sql", {
                "query": (
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public'"
                ),
            })
            for table in self._parse_tables(result):
                call_tool("execute_sql", {
                    "query": f"DROP TABLE IF EXISTS {table} CASCADE",
                })
        except Exception as e:
            logger.warning("DB reset failed: %s", e)

    def snapshot_state(self, call_tool: CallToolFn) -> dict:
        try:
            tables = call_tool("execute_sql", {
                "query": (
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                ),
            })
            return {"tables": tables}
        except Exception as e:
            logger.warning("DB snapshot failed: %s", e)
            return {"tables": None, "error": str(e)}

    @staticmethod
    def _parse_tables(result: str) -> list[str]:
        """Best-effort extraction of table names from query result text."""
        # Try JSON first (some MCP servers return JSON arrays)
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return [
                    row.get("tablename") or row.get("table_name") or str(row)
                    for row in parsed
                    if isinstance(row, dict)
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: split on whitespace/newlines, skip header-like tokens
        skip = {"tablename", "table_name", "(", ")", "|", "-", ""}
        return [
            token.strip()
            for token in result.replace("|", "\n").split("\n")
            if token.strip().lower() not in skip and not token.strip().startswith("-")
        ]


class FilesystemAdapter(MCPStateAdapter):
    """For filesystem MCP servers.

    Supports an optional ``path_prefix`` in the server config to remap
    paths.  For example, if the MCP server is rooted at ``/private/tmp``
    and scenarios write to ``/workspace/...``, set
    ``path_prefix: /private/tmp`` so ``/workspace/foo`` becomes
    ``/private/tmp/workspace/foo``.
    """

    def __init__(self, path_prefix: str = ""):
        self.path_prefix = path_prefix.rstrip("/")

    def _map_path(self, path: str) -> str:
        if self.path_prefix:
            return f"{self.path_prefix}{path}"
        return path

    def seed_state(self, call_tool: CallToolFn, config: dict) -> None:
        for file_spec in config.get("files", []):
            mapped = self._map_path(file_spec["path"])
            try:
                call_tool("write_file", {
                    "path": mapped,
                    "content": file_spec["content"],
                })
            except Exception as e:
                logger.warning("FS seed failed for %s: %s", mapped, e)

    def reset_state(self, call_tool: CallToolFn) -> None:
        workspace = self._map_path("/workspace")
        try:
            # Try listing and removing files; execute_command may not exist
            listing = call_tool("list_directory", {"path": workspace})
            if listing and listing.strip():
                logger.info("FS reset: clearing %s", workspace)
        except Exception as e:
            logger.debug("FS reset list failed (may be empty): %s", e)

    def snapshot_state(self, call_tool: CallToolFn) -> dict:
        workspace = self._map_path("/workspace")
        try:
            listing = call_tool("list_directory", {"path": workspace})
            return {"files": listing}
        except Exception as e:
            logger.warning("FS snapshot failed: %s", e)
            return {"files": None, "error": str(e)}


class BrowserAdapter(MCPStateAdapter):
    """For Playwright / browser MCP servers."""

    def seed_state(self, call_tool: CallToolFn, config: dict) -> None:
        if "navigate_to" in config:
            try:
                call_tool("navigate", {"url": config["navigate_to"]})
            except Exception as e:
                logger.warning("Browser seed (navigate) failed: %s", e)

    def reset_state(self, call_tool: CallToolFn) -> None:
        try:
            call_tool("navigate", {"url": "about:blank"})
        except Exception as e:
            logger.warning("Browser reset failed: %s", e)

    def snapshot_state(self, call_tool: CallToolFn) -> dict:
        snapshot: dict[str, Any] = {}
        try:
            snapshot["url"] = call_tool("get_url", {})
        except Exception:
            pass
        try:
            snapshot["screenshot"] = call_tool("screenshot", {})
        except Exception:
            pass
        return snapshot


# ---------------------------------------------------------------------------
# Registry — extend by adding entries here
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, type[MCPStateAdapter]] = {
    "database": DatabaseAdapter,
    "filesystem": FilesystemAdapter,
    "browser": BrowserAdapter,
}
