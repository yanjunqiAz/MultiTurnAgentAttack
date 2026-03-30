"""MCPEnvironment — BaseEnvironment backed by real MCP servers.

Connects to one or more MCP-compatible servers (over stdio, SSE, or
Streamable HTTP), discovers tools dynamically via ``tools/list``, routes
tool calls to the correct server, and manages state through pluggable
adapters.  The default transport is **stdio** via ``docker exec -i``.

All async MCP SDK calls are wrapped behind the synchronous
``BaseEnvironment`` interface using a background thread running an
``asyncio`` event loop.  This keeps MCP transport task-groups alive
between calls (required by the ``anyio``-based MCP SDK).

Usage::

    from MCP.core.mcp_environment import MCPEnvironment

    env = MCPEnvironment(
        server_configs={"postgres": {"endpoint": "http://localhost:9091/mcp", ...}},
        model_id="gpt-4.1",
        scenario_config={"pre_seed": {"postgres": {"statements": ["..."]}}},
    )
    # env.tool_config is set — pass to Agent via Agent.reset([env])
    result_msgs = env.step({"tool_call_id": "x", "tool_name": "execute_sql",
                            "arguments": {"query": "SELECT 1"}})
    env.close()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
import time
from typing import Any

from mcp import ClientSession

from src.Environments import BaseEnvironment
from MCP.core.adapters import ADAPTER_REGISTRY, MCPStateAdapter
from MCP.core.utils import format_tool_result_for_model, format_tools_for_model

logger = logging.getLogger(__name__)


class MCPEnvironment(BaseEnvironment):
    """Environment backed by one or more MCP servers.

    Implements the ``BaseEnvironment`` ABC so it can be used with
    ``Agent.reset([env])`` and ``AdaptivePlanningSystem`` without changes.

    Key contracts fulfilled:
    * ``self.tool_config`` attribute — set during ``_discover_tools()``,
      read by ``Agent.reset()`` at ``Agents.py:172``.
    * ``step()`` returns model-specific message list via
      ``format_tool_result_for_model()``.
    * ``get_env_info()`` / ``get_tool_info()`` return strings.

    Async handling:
        The MCP Python SDK is async and uses ``anyio`` task groups that
        must stay alive for the duration of the connection.  This class
        runs a dedicated ``asyncio`` event loop in a **background thread**
        so transport tasks persist between synchronous ``_run()`` calls.
    """

    def __init__(
        self,
        server_configs: dict[str, dict],
        model_id: str,
        scenario_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.tool_config: Any = None  # set by _discover_tools()

        # Background event loop — keeps anyio task groups alive
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
        )
        self._thread.start()

        # Per-server bookkeeping
        self.sessions: dict[str, ClientSession] = {}
        self.adapters: dict[str, MCPStateAdapter] = {}
        self.tool_routing: dict[str, str] = {}  # tool_name -> server_name
        self._context_managers: list[tuple] = []  # for cleanup

        # Connect to each MCP server (with retry)
        for name, cfg in server_configs.items():
            max_retries = 3
            retry_delay = 2
            for attempt in range(1, max_retries + 1):
                try:
                    session = self._run(self._connect(cfg), timeout=30)
                    self.sessions[name] = session
                    # Verify the connection works by listing tools
                    self._run(session.list_tools(), timeout=15)
                    logger.info("Connected to MCP server %r (attempt %d)", name, attempt)
                    break
                except Exception as exc:
                    logger.warning(
                        "MCP server %r connection attempt %d/%d failed: %s",
                        name, attempt, max_retries, exc,
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay * attempt)
                    else:
                        logger.error("Failed to connect to MCP server %r after %d attempts", name, max_retries)
                        self.close()
                        raise

            adapter_key = cfg.get("adapter")
            adapter_cls = ADAPTER_REGISTRY.get(adapter_key) if adapter_key else None
            if adapter_cls:
                adapter_kwargs = {}
                if cfg.get("path_prefix"):
                    adapter_kwargs["path_prefix"] = cfg["path_prefix"]
                try:
                    self.adapters[name] = adapter_cls(**adapter_kwargs)
                except TypeError:
                    self.adapters[name] = adapter_cls()

        # Discover tools from all connected servers
        self._discover_tools()

        # Seed initial state if scenario provides pre_seed
        if scenario_config:
            self._seed_state(scenario_config)

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    def _run(self, coro, timeout: float | None = 60):
        """Submit a coroutine to the background loop and block for the result.

        The background thread keeps the event loop (and anyio task groups)
        alive between calls.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _connect(self, cfg: dict) -> ClientSession:
        """Establish an MCP session.

        Supports three transports:

        * **stdio** (default) — spawns a subprocess (typically
          ``docker exec -i <container> <mcp-server>``).
        * **sse** — connects via Server-Sent Events.
        * **streamable_http** — connects to a running HTTP server.

        Context managers are kept alive via manual ``__aenter__`` and
        cleaned up in ``close()``.
        """
        transport_type = cfg.get("transport", "stdio")

        if transport_type == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            transport_cm = stdio_client(StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
            ))
        elif transport_type == "sse":
            from mcp.client.sse import sse_client

            transport_cm = sse_client(url=cfg["endpoint"])
        else:
            from mcp.client.streamable_http import streamable_http_client

            transport_cm = streamable_http_client(url=cfg["endpoint"])

        # Manually enter transport context manager (must stay alive)
        read_stream, write_stream, *_ = await transport_cm.__aenter__()

        # Manually enter session context manager
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        await session.initialize()

        self._context_managers.append((transport_cm, session_cm))
        return session

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def _discover_tools(self) -> None:
        """Query ``tools/list`` on every server; build routing table and
        ``self.tool_config``."""
        raw_schemas: list[dict] = []

        for server_name, session in self.sessions.items():
            result = self._run(session.list_tools())
            for tool in result.tools:
                if tool.name in self.tool_routing:
                    logger.warning(
                        "Tool name collision: '%s' exists in both '%s' and "
                        "'%s' — routing to '%s'.",
                        tool.name,
                        self.tool_routing[tool.name],
                        server_name,
                        server_name,
                    )
                self.tool_routing[tool.name] = server_name
                raw_schemas.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema or {
                        "type": "object",
                        "properties": {},
                    },
                })

        self.tool_config = format_tools_for_model(raw_schemas, self.model_id)
        logger.info(
            "Discovered %d tools across %d servers",
            len(raw_schemas),
            len(self.sessions),
        )

    # ------------------------------------------------------------------
    # State management (via adapters)
    # ------------------------------------------------------------------

    def _make_call_fn(self, server_name: str):
        """Return a *synchronous* ``(tool_name, arguments) -> str`` callable
        bound to *server_name*."""
        def call_tool(tool_name: str, arguments: dict) -> str:
            result = self._run(
                self.sessions[server_name].call_tool(tool_name, arguments),
            )
            return "\n".join(
                c.text for c in result.content if hasattr(c, "text")
            ) or str(result)
        return call_tool

    def _seed_state(self, scenario_config: dict) -> None:
        """Seed environment state from scenario ``pre_seed`` section."""
        pre_seed = scenario_config.get("pre_seed", {})
        for server_name, seed_cfg in pre_seed.items():
            adapter = self.adapters.get(server_name)
            if adapter:
                adapter.seed_state(self._make_call_fn(server_name), seed_cfg)
            elif server_name in self.sessions:
                logger.debug(
                    "No adapter for server '%s' — skipping seed", server_name,
                )

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def step(self, completion: dict) -> list[dict]:
        """Execute a tool call via the appropriate MCP server."""
        tool_name = completion["tool_name"]
        tool_call_id = completion["tool_call_id"]
        server_name = self.tool_routing.get(tool_name)

        if server_name is None:
            result_text = f"Tool '{tool_name}' does not exist."
        else:
            try:
                result = self._run(
                    self.sessions[server_name].call_tool(
                        tool_name, completion.get("arguments", {}),
                    ),
                )
                result_text = "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                ) or str(result)
            except Exception as exc:
                logger.warning(
                    "Tool call '%s' failed on server '%s': %s",
                    tool_name, server_name, exc,
                )
                result_text = f"Error: tool execution failed ({type(exc).__name__}: {exc})"

        return format_tool_result_for_model(
            tool_call_id, tool_name, result_text, self.model_id,
        )

    def get_env_info(self) -> str:
        """Return a JSON string summarising connected servers and tool counts."""
        info = {}
        for server_name in self.sessions:
            tools = [t for t, s in self.tool_routing.items() if s == server_name]
            info[server_name] = {
                "num_tools": len(tools),
                "tools": tools,
            }
        return json.dumps(info)

    def get_tool_info(self) -> str | None:
        """Return tool config as a string (for Planner/Judge context)."""
        if self.tool_config is None:
            return None
        if isinstance(self.tool_config, str):
            return self.tool_config
        if isinstance(self.tool_config, dict) and "tools" in self.tool_config:
            return json.dumps(self.tool_config["tools"])
        return json.dumps(self.tool_config)

    def reset(self, scenario_config: dict | None = None) -> None:
        """Reset all server state via adapters, optionally re-seed."""
        for server_name, adapter in self.adapters.items():
            adapter.reset_state(self._make_call_fn(server_name))
        if scenario_config:
            self._seed_state(scenario_config)

    # ------------------------------------------------------------------
    # State snapshots (for StateVerifier)
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> dict:
        """Capture current state from all servers that have adapters."""
        snapshots: dict[str, Any] = {}
        for server_name, adapter in self.adapters.items():
            try:
                snapshots[server_name] = adapter.snapshot_state(
                    self._make_call_fn(server_name),
                )
            except Exception as exc:
                logger.warning(
                    "Snapshot failed for server '%s': %s", server_name, exc,
                )
                snapshots[server_name] = {"error": str(exc)}
        return snapshots

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Tear down MCP sessions, context managers, and event loop."""
        if hasattr(self, "_loop") and self._loop.is_closed():
            return
        for transport_cm, session_cm in self._context_managers:
            try:
                self._run(session_cm.__aexit__(None, None, None), timeout=5)
            except Exception:
                pass
            try:
                self._run(transport_cm.__aexit__(None, None, None), timeout=5)
            except Exception:
                pass
        self._context_managers.clear()
        self.sessions.clear()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.close()

    def __del__(self):
        """Best-effort cleanup if close() was not called explicitly."""
        if hasattr(self, "_loop") and not self._loop.is_closed():
            try:
                self.close()
            except Exception:
                pass
