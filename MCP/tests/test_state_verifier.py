"""Tests for MCP/eval/state_verifier.py.

Tests verify:
- JSON validation (outcome values, evidence type, missing fields)
- verify() builds correct prompts and calls BaseLM.step()
- Running array is numpy bool (required by BaseLM.step)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from MCP.eval.state_verifier import StateVerifier, VALID_OUTCOMES


# ---------------------------------------------------------------------------
# Helpers — we mock BaseLM.__init__ to avoid creating real LLM backends
# ---------------------------------------------------------------------------

def make_verifier() -> StateVerifier:
    """Create a StateVerifier with mocked LLM backend."""
    with patch("src.STAC.BaseLM.__init__", return_value=None):
        v = StateVerifier.__new__(StateVerifier)
        # Set the attributes BaseLM.__init__ would set
        v.role = "StateVerifier"
        v.model_id = "gpt-4.1"
        v.temperature = 0
        v.top_p = 0.95
        v.max_tokens = 2048
        v.attack_goals = None
        v.explanations = None
        v.env_infos = None
        v.tool_infos = None
        v.output_history = None
        v.output_json_fields = ["outcome", "evidence"]
        v.sys_prompt_path = "MCP/prompts/state_verifier.md"
        v.sys_prompt = "You are a state verification judge."
        v.model = MagicMock()
    return v


# ---------------------------------------------------------------------------
# check_valid_json
# ---------------------------------------------------------------------------

class TestCheckValidJson:

    def test_valid_complete(self):
        v = make_verifier()
        assert v.check_valid_json({
            "outcome": "COMPLETE",
            "evidence": ["File created at /sandbox/dump.csv"],
        })

    def test_valid_partial(self):
        v = make_verifier()
        assert v.check_valid_json({
            "outcome": "PARTIAL",
            "evidence": ["Query executed but file not written"],
        })

    def test_valid_failed(self):
        v = make_verifier()
        assert v.check_valid_json({
            "outcome": "FAILED",
            "evidence": [],
        })

    def test_invalid_outcome_value(self):
        v = make_verifier()
        assert not v.check_valid_json({
            "outcome": "SUCCESS",  # not in VALID_OUTCOMES
            "evidence": [],
        })

    def test_missing_outcome(self):
        v = make_verifier()
        assert not v.check_valid_json({
            "evidence": ["something"],
        })

    def test_missing_evidence(self):
        v = make_verifier()
        assert not v.check_valid_json({
            "outcome": "COMPLETE",
        })

    def test_evidence_not_list(self):
        v = make_verifier()
        assert not v.check_valid_json({
            "outcome": "COMPLETE",
            "evidence": "should be a list",
        })

    def test_none_input(self):
        v = make_verifier()
        assert not v.check_valid_json(None)

    @pytest.mark.parametrize("outcome", VALID_OUTCOMES)
    def test_all_valid_outcomes_accepted(self, outcome):
        v = make_verifier()
        assert v.check_valid_json({"outcome": outcome, "evidence": []})


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------

class TestVerify:

    def test_verify_calls_step_with_numpy_running(self):
        v = make_verifier()
        called_args = {}

        def mock_step(running, user_prompts, batch_size=32):
            called_args["running"] = running
            called_args["user_prompts"] = user_prompts
            called_args["batch_size"] = batch_size
            return [{"outcome": "COMPLETE", "evidence": ["test"]}]

        v.step = mock_step

        result = v.verify(
            attack_goals=["Exfiltrate data"],
            pre_states=[{"tables": "users"}],
            post_states=[{"tables": "users", "files": "dump.csv"}],
        )

        # running must be numpy array (BaseLM.step uses it for indexing)
        assert isinstance(called_args["running"], np.ndarray)
        assert called_args["running"].dtype == bool
        assert np.all(called_args["running"])
        assert len(called_args["running"]) == 1

        # batch_size defaults to 1
        assert called_args["batch_size"] == 1

        # result passed through
        assert result[0]["outcome"] == "COMPLETE"

    def test_verify_prompt_contains_goal_and_states(self):
        v = make_verifier()
        prompts_captured = []

        def mock_step(running, user_prompts, batch_size=32):
            prompts_captured.extend(user_prompts)
            return [{"outcome": "FAILED", "evidence": []}]

        v.step = mock_step

        goal = "Extract all emails from the users table"
        pre = {"db": {"tables": "users(id, email)"}}
        post = {"db": {"tables": "users(id, email)"}}

        v.verify([goal], [pre], [post])

        prompt = prompts_captured[0]
        assert "Extract all emails" in prompt
        assert "BEFORE" in prompt
        assert "AFTER" in prompt
        assert "users" in prompt

    def test_verify_multiple_instances(self):
        v = make_verifier()
        captured = {}

        def mock_step(running, user_prompts, batch_size=32):
            captured["n"] = len(user_prompts)
            captured["running_len"] = len(running)
            return [
                {"outcome": "COMPLETE", "evidence": ["ok"]},
                {"outcome": "FAILED", "evidence": []},
                {"outcome": "PARTIAL", "evidence": ["half"]},
            ]

        v.step = mock_step

        results = v.verify(
            attack_goals=["goal1", "goal2", "goal3"],
            pre_states=[{}, {}, {}],
            post_states=[{"changed": True}, {}, {"partial": True}],
        )

        assert captured["n"] == 3
        assert captured["running_len"] == 3
        assert len(results) == 3
        assert results[0]["outcome"] == "COMPLETE"
        assert results[1]["outcome"] == "FAILED"
        assert results[2]["outcome"] == "PARTIAL"

    def test_verify_serialises_complex_states(self):
        """States with non-JSON-native types should be serialised via default=str."""
        v = make_verifier()
        prompts_captured = []

        def mock_step(running, user_prompts, batch_size=32):
            prompts_captured.extend(user_prompts)
            return [{"outcome": "FAILED", "evidence": []}]

        v.step = mock_step

        # Include a set and bytes which aren't JSON-native
        pre = {"data": {1, 2, 3}}
        post = {"data": b"binary"}

        # Should not raise
        v.verify(["goal"], [pre], [post])
        assert len(prompts_captured) == 1

    def test_verify_custom_batch_size(self):
        v = make_verifier()
        captured = {}

        def mock_step(running, user_prompts, batch_size=32):
            captured["batch_size"] = batch_size
            return [{"outcome": "FAILED", "evidence": []}]

        v.step = mock_step

        v.verify(["goal"], [{}], [{}], batch_size=4)
        assert captured["batch_size"] == 4
