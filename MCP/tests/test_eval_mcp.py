"""Tests for MCP/eval/eval_mcp.py — scenario loading, criteria evaluation, helpers.

No API keys or MCP servers needed; tests cover the non-LLM logic.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from MCP.eval.eval_mcp import (
    _error_result,
    _extract_final_judge_scores,
    _truncate,
    check_criteria,
    evaluate_expectation,
    load_scenarios_from_yaml,
    load_server_registry,
)


# ---------------------------------------------------------------------------
# evaluate_expectation
# ---------------------------------------------------------------------------

class TestEvaluateExpectation:

    def test_non_empty_with_content(self):
        assert evaluate_expectation("some text", "non_empty") is True

    def test_non_empty_with_empty(self):
        assert evaluate_expectation("", "non_empty") is False

    def test_non_empty_with_whitespace(self):
        assert evaluate_expectation("   ", "non_empty") is False

    def test_match_found(self):
        assert evaluate_expectation("data here", "match_found") is True

    def test_match_found_empty(self):
        assert evaluate_expectation("", "match_found") is False

    def test_contains_present(self):
        assert evaluate_expectation("alice@corp.com in results", "contains:alice@corp.com") is True

    def test_contains_absent(self):
        assert evaluate_expectation("no email here", "contains:alice@corp.com") is False

    def test_contains_empty_result(self):
        assert evaluate_expectation("", "contains:foo") is False

    def test_numeric_greater_than_pass(self):
        assert evaluate_expectation("n = 5", "n > 0") is True

    def test_numeric_greater_than_fail(self):
        assert evaluate_expectation("n = 0", "n > 0") is False

    def test_numeric_equals(self):
        assert evaluate_expectation("count is 3", "count == 3") is True

    def test_numeric_not_equals(self):
        assert evaluate_expectation("count is 3", "count != 5") is True

    def test_numeric_less_than(self):
        assert evaluate_expectation("x: 2", "x < 5") is True

    def test_numeric_no_match_in_result(self):
        assert evaluate_expectation("no numbers here", "n > 0") is False

    def test_unknown_expect_format(self):
        assert evaluate_expectation("anything", "some_unknown_format") is False

    def test_none_result(self):
        assert evaluate_expectation(None, "non_empty") is False

    def test_none_result_unknown_expect(self):
        # When result is falsy and expect isn't non_empty/match_found
        assert evaluate_expectation("", "contains:x") is False


# ---------------------------------------------------------------------------
# check_criteria
# ---------------------------------------------------------------------------

class TestCheckCriteria:

    def _make_env(self, call_results: dict[str, str]):
        env = MagicMock()

        def make_call_fn(server_name):
            def call_tool(tool_name, arguments):
                return call_results.get(tool_name, "")
            return call_tool

        env._make_call_fn = make_call_fn
        return env

    def test_single_criterion_pass(self):
        env = self._make_env({"search_files": "alice@corp.com found"})
        results = check_criteria(env, [{
            "server": "filesystem",
            "tool": "search_files",
            "arguments": {"path": "/sandbox"},
            "expect": "contains:alice@corp.com",
        }])
        assert len(results) == 1
        assert results[0]["passed"] is True

    def test_single_criterion_fail(self):
        env = self._make_env({"search_files": ""})
        results = check_criteria(env, [{
            "server": "filesystem",
            "tool": "search_files",
            "arguments": {},
            "expect": "non_empty",
        }])
        assert results[0]["passed"] is False

    def test_multiple_criteria(self):
        env = self._make_env({
            "search_files": "data",
            "execute_sql": "n = 0",
        })
        results = check_criteria(env, [
            {"server": "fs", "tool": "search_files", "arguments": {},
             "expect": "non_empty"},
            {"server": "db", "tool": "execute_sql", "arguments": {},
             "expect": "n > 0"},
        ])
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False

    def test_tool_error_returns_failed(self):
        env = MagicMock()

        def make_call_fn(server_name):
            def call_tool(tool_name, arguments):
                raise RuntimeError("connection refused")
            return call_tool

        env._make_call_fn = make_call_fn
        results = check_criteria(env, [{
            "server": "db",
            "tool": "execute_sql",
            "arguments": {},
            "expect": "non_empty",
        }])
        assert results[0]["passed"] is False

    def test_empty_criteria(self):
        env = MagicMock()
        assert check_criteria(env, []) == []

    def test_result_preview_truncated(self):
        env = self._make_env({"big_tool": "x" * 500})
        results = check_criteria(env, [{
            "server": "s", "tool": "big_tool",
            "arguments": {}, "expect": "non_empty",
        }])
        assert len(results[0]["result_preview"]) <= 200


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

class TestLoadScenariosFromYaml:

    def test_loads_yaml_files(self, tmp_path):
        (tmp_path / "test1.yml").write_text(textwrap.dedent("""\
            id: test1
            attack_goal: "Do something"
            mcp_servers: [db]
        """))
        (tmp_path / "test2.yaml").write_text(textwrap.dedent("""\
            id: test2
            attack_goal: "Do something else"
            mcp_servers: [fs]
        """))
        scenarios = load_scenarios_from_yaml(tmp_path)
        assert len(scenarios) == 2
        ids = {s["id"] for s in scenarios}
        assert ids == {"test1", "test2"}

    def test_skips_schema_file(self, tmp_path):
        (tmp_path / "schema.yml").write_text("type: object\n")
        (tmp_path / "real.yml").write_text("id: r\nattack_goal: x\n")
        scenarios = load_scenarios_from_yaml(tmp_path)
        assert len(scenarios) == 1

    def test_skips_non_scenario_yaml(self, tmp_path):
        (tmp_path / "random.yml").write_text("key: value\n")
        scenarios = load_scenarios_from_yaml(tmp_path)
        assert len(scenarios) == 0  # no attack_goal

    def test_assigns_id_from_filename(self, tmp_path):
        (tmp_path / "my_scenario.yml").write_text("attack_goal: test\n")
        scenarios = load_scenarios_from_yaml(tmp_path)
        assert scenarios[0]["id"] == "my_scenario"

    def test_empty_directory(self, tmp_path):
        assert load_scenarios_from_yaml(tmp_path) == []


class TestLoadServerRegistry:

    def test_loads_servers(self, tmp_path):
        cfg = tmp_path / "servers.yml"
        cfg.write_text(textwrap.dedent("""\
            servers:
              postgres:
                endpoint: "http://localhost:9091/mcp"
                adapter: database
              filesystem:
                endpoint: "http://localhost:9090/mcp"
                adapter: filesystem
        """))
        registry = load_server_registry(cfg)
        assert "postgres" in registry
        assert "filesystem" in registry
        assert registry["postgres"]["endpoint"] == "http://localhost:9091/mcp"

    def test_missing_file(self, tmp_path):
        assert load_server_registry(tmp_path / "missing.yml") == {}

    def test_empty_file(self, tmp_path):
        cfg = tmp_path / "empty.yml"
        cfg.write_text("")
        assert load_server_registry(cfg) == {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_extract_final_judge_scores_normal(self):
        history = [
            {"goal_progress": {"score": 1}, "agent_helpfulness": {"score": 2},
             "prompt_harmlessness": {"score": 0}},
            {"goal_progress": {"score": 3}, "agent_helpfulness": {"score": 3},
             "prompt_harmlessness": {"score": 1}},
        ]
        scores = _extract_final_judge_scores(history)
        assert scores["goal_progress"]["score"] == 3

    def test_extract_final_judge_scores_empty(self):
        assert _extract_final_judge_scores(None) == {}
        assert _extract_final_judge_scores([]) == {}

    def test_truncate_short(self):
        assert _truncate({"a": 1}) == '{"a": 1}'

    def test_truncate_long(self):
        result = _truncate({"data": "x" * 1000}, max_len=50)
        assert len(result) < 100
        assert "truncated" in result

    def test_error_result(self):
        r = _error_result("sc1", "boom")
        assert r["id"] == "sc1"
        assert r["error"] == "boom"
