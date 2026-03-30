"""Tests for MCP/core/mcp_environment.py.

Uses mock MCP sessions (no real servers needed).  Tests verify:
- Tool discovery and routing
- Tool config formatting for each model type
- step() returns correctly formatted messages
- State seeding/reset/snapshot via adapters
- Error handling (unknown tools, connection failures)
- Cleanup via close()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from MCP.core.mcp_environment import MCPEnvironment


# ---------------------------------------------------------------------------
# MCP SDK type stubs (mirror the real SDK types used by MCPEnvironment)
# ---------------------------------------------------------------------------

@dataclass
class FakeTool:
    name: str
    description: str = ""
    inputSchema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class FakeListToolsResult:
    tools: list[FakeTool] = field(default_factory=list)


@dataclass
class FakeTextContent:
    text: str
    type: str = "text"


@dataclass
class FakeCallToolResult:
    content: list[FakeTextContent] = field(default_factory=list)
    isError: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_session(tools: list[FakeTool], call_results: dict[str, str] | None = None):
    """Create a mock ClientSession that responds to list_tools/call_tool."""
    session = AsyncMock()
    session.list_tools.return_value = FakeListToolsResult(tools=tools)
    session.initialize = AsyncMock()

    call_results = call_results or {}

    async def mock_call_tool(name, arguments=None):
        text = call_results.get(name, f"result from {name}")
        return FakeCallToolResult(content=[FakeTextContent(text=text)])

    session.call_tool = AsyncMock(side_effect=mock_call_tool)
    return session


def build_env(
    server_tools: dict[str, list[FakeTool]],
    model_id: str = "gpt-4.1",
    scenario_config: dict | None = None,
    call_results: dict[str, str] | None = None,
) -> MCPEnvironment:
    """Build an MCPEnvironment with mocked MCP sessions.

    Bypasses actual network connections by patching _connect().
    """
    sessions = {
        name: make_mock_session(tools, call_results)
        for name, tools in server_tools.items()
    }
    server_configs = {
        name: {"command": "echo", "args": ["test"], "transport": "stdio"}
        for name in server_tools
    }

    with patch.object(MCPEnvironment, "_run") as mock_run, \
         patch.object(MCPEnvironment, "__init__", lambda self, *a, **kw: None):

        env = MCPEnvironment.__new__(MCPEnvironment)
        # Manually init fields (mirrors __init__)
        env.model_id = model_id
        env.tool_config = None
        env._loop = asyncio.new_event_loop()
        env.sessions = sessions
        env.adapters = {}
        env.tool_routing = {}
        env._context_managers = []
        env._thread = MagicMock()  # stub for close()

        # Wire up _run to actually run coroutines on the real loop
        def real_run(coro, timeout=60):
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout)
            return env._loop.run_until_complete(coro)

        env._run = real_run

        # Run discovery
        env._discover_tools()

        # Seed if provided
        if scenario_config:
            env._seed_state(scenario_config)

    return env


# Convenience: common tool lists
SQL_TOOLS = [
    FakeTool("execute_sql", "Run SQL", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    FakeTool("list_tables", "List tables"),
]

FS_TOOLS = [
    FakeTool("read_file", "Read file contents", {"type": "object", "properties": {"path": {"type": "string"}}}),
    FakeTool("write_file", "Write file"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolDiscovery:

    def test_single_server_tools_discovered(self):
        env = build_env({"db": SQL_TOOLS})
        assert "execute_sql" in env.tool_routing
        assert "list_tables" in env.tool_routing
        assert env.tool_routing["execute_sql"] == "db"
        env._loop.close()

    def test_multi_server_routing(self):
        env = build_env({"db": SQL_TOOLS, "fs": FS_TOOLS})
        assert env.tool_routing["execute_sql"] == "db"
        assert env.tool_routing["read_file"] == "fs"
        env._loop.close()

    def test_tool_name_collision_last_wins(self):
        """If two servers expose the same tool name, the last one wins."""
        dup_tools = [FakeTool("execute_sql", "Duplicate")]
        env = build_env({"db1": SQL_TOOLS, "db2": dup_tools})
        # db2 was iterated second, so it should win
        assert env.tool_routing["execute_sql"] == "db2"
        env._loop.close()

    def test_empty_server(self):
        env = build_env({"empty": []})
        assert env.tool_routing == {}
        env._loop.close()


class TestToolConfig:

    def test_bedrock_format(self):
        env = build_env({"db": SQL_TOOLS}, model_id="us.anthropic.claude-sonnet-4-5-v1")
        assert isinstance(env.tool_config, dict)
        assert "tools" in env.tool_config
        assert len(env.tool_config["tools"]) == 2
        spec = env.tool_config["tools"][0]["toolSpec"]
        assert spec["name"] == "execute_sql"
        assert "json" in spec["inputSchema"]
        env._loop.close()

    def test_openai_format(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        assert isinstance(env.tool_config, str)
        parsed = json.loads(env.tool_config)
        assert isinstance(parsed, list)
        assert parsed[0]["type"] == "function"
        env._loop.close()

    def test_vllm_format(self):
        env = build_env({"db": SQL_TOOLS}, model_id="Qwen/Qwen3-32B")
        assert isinstance(env.tool_config, str)
        parsed = json.loads(env.tool_config)
        assert parsed[0]["name"] == "execute_sql"
        assert "type" not in parsed[0]  # vLLM doesn't wrap in {"type": "function"}
        env._loop.close()


class TestStep:

    def test_step_returns_openai_format(self):
        env = build_env(
            {"db": SQL_TOOLS},
            model_id="gpt-4.1",
            call_results={"execute_sql": "row1\nrow2"},
        )
        result = env.step({
            "tool_call_id": "call-1",
            "tool_name": "execute_sql",
            "arguments": {"query": "SELECT 1"},
        })
        assert isinstance(result, list)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-1"
        assert msg["name"] == "execute_sql"
        assert "row1" in msg["content"]
        env._loop.close()

    def test_step_returns_bedrock_format(self):
        env = build_env(
            {"db": SQL_TOOLS},
            model_id="claude-sonnet-4-5",
            call_results={"execute_sql": "OK"},
        )
        result = env.step({
            "tool_call_id": "call-2",
            "tool_name": "execute_sql",
            "arguments": {"query": "SELECT 1"},
        })
        msg = result[0]
        assert msg["role"] == "tool"
        tr = msg["content"][0]["toolResult"]
        assert tr["toolUseId"] == "call-2"
        assert tr["content"][0]["text"] == "OK"
        env._loop.close()

    def test_step_unknown_tool(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        result = env.step({
            "tool_call_id": "call-3",
            "tool_name": "nonexistent_tool",
            "arguments": {},
        })
        msg = result[0]
        assert "does not exist" in msg["content"]
        env._loop.close()

    def test_step_tool_execution_error(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        # Make call_tool raise
        env.sessions["db"].call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        result = env.step({
            "tool_call_id": "call-4",
            "tool_name": "execute_sql",
            "arguments": {"query": "BAD"},
        })
        msg = result[0]
        assert "Error" in msg["content"]
        assert "boom" in msg["content"]
        env._loop.close()

    def test_step_routes_to_correct_server(self):
        env = build_env({"db": SQL_TOOLS, "fs": FS_TOOLS}, model_id="gpt-4.1")
        env.step({
            "tool_call_id": "c1",
            "tool_name": "read_file",
            "arguments": {"path": "/a"},
        })
        # Verify call went to fs session, not db
        env.sessions["fs"].call_tool.assert_called_once()
        env.sessions["db"].call_tool.assert_not_called()
        env._loop.close()


class TestEnvAndToolInfo:

    def test_get_env_info_returns_json_string(self):
        env = build_env({"db": SQL_TOOLS, "fs": FS_TOOLS})
        info = env.get_env_info()
        parsed = json.loads(info)
        assert parsed["db"]["num_tools"] == 2
        assert parsed["fs"]["num_tools"] == 2
        assert "execute_sql" in parsed["db"]["tools"]
        env._loop.close()

    def test_get_tool_info_openai(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        info = env.get_tool_info()
        assert isinstance(info, str)
        parsed = json.loads(info)
        assert isinstance(parsed, list)
        env._loop.close()

    def test_get_tool_info_bedrock(self):
        env = build_env({"db": SQL_TOOLS}, model_id="claude-sonnet-4-5")
        info = env.get_tool_info()
        assert isinstance(info, str)
        parsed = json.loads(info)
        # Bedrock returns tools list (unwrapped from {"tools": ...})
        assert isinstance(parsed, list)
        assert "toolSpec" in parsed[0]
        env._loop.close()

    def test_get_tool_info_none_when_empty(self):
        env = build_env({"empty": []}, model_id="gpt-4.1")
        # tool_config is '[]' (empty JSON list), which is truthy
        info = env.get_tool_info()
        assert info == "[]"
        env._loop.close()


class TestStateManagement:

    def test_seed_state_calls_adapter(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        # Attach a mock adapter
        mock_adapter = MagicMock()
        env.adapters["db"] = mock_adapter

        env._seed_state({"pre_seed": {"db": {"statements": ["CREATE TABLE x"]}}})
        mock_adapter.seed_state.assert_called_once()
        call_fn, cfg = mock_adapter.seed_state.call_args[0]
        assert cfg == {"statements": ["CREATE TABLE x"]}
        env._loop.close()

    def test_reset_calls_adapter(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        mock_adapter = MagicMock()
        env.adapters["db"] = mock_adapter

        env.reset()
        mock_adapter.reset_state.assert_called_once()
        env._loop.close()

    def test_reset_with_reseed(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        mock_adapter = MagicMock()
        env.adapters["db"] = mock_adapter

        env.reset(scenario_config={"pre_seed": {"db": {"statements": ["INSERT 1"]}}})
        mock_adapter.reset_state.assert_called_once()
        mock_adapter.seed_state.assert_called_once()
        env._loop.close()

    def test_get_state_snapshot(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        mock_adapter = MagicMock()
        mock_adapter.snapshot_state.return_value = {"tables": "users, logs"}
        env.adapters["db"] = mock_adapter

        snap = env.get_state_snapshot()
        assert snap["db"] == {"tables": "users, logs"}
        env._loop.close()

    def test_snapshot_handles_adapter_error(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        mock_adapter = MagicMock()
        mock_adapter.snapshot_state.side_effect = RuntimeError("fail")
        env.adapters["db"] = mock_adapter

        snap = env.get_state_snapshot()
        assert "error" in snap["db"]
        env._loop.close()


class TestCleanup:

    def test_close_clears_sessions(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        assert len(env.sessions) == 1
        env.close()
        assert len(env.sessions) == 0
        assert env._loop.is_closed()

    def test_double_close_safe(self):
        env = build_env({"db": SQL_TOOLS}, model_id="gpt-4.1")
        env.close()
        env.close()  # should not raise
