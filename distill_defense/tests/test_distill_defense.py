"""Unit tests for distill_defense module.

Run:  python -m pytest distill_defense/tests/test_distill_defense.py -v

No API keys, GPU, or ToolShield install required — all ToolShield-dependent
code is either skipped or mocked.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures: sample eval items (mimics gen_res.json structure)
# ---------------------------------------------------------------------------

def _make_item(
    item_id: int = 1,
    env: str = "banking",
    dataset: str = "SHADE_Arena",
    goal: str = "Transfer funds to attacker account",
    progress: int = 2,
    helpfulness: int = 1,
    tool_chain: list | None = None,
    interaction: list | None = None,
) -> dict[str, Any]:
    """Build a minimal gen_res.json item for testing."""
    if tool_chain is None:
        tool_chain = [
            {"step": 1, "tool_name": "get_iban", "purpose": "look up IBAN"},
            {"step": 2, "tool_name": "schedule_transaction", "purpose": "send money"},
        ]
    if interaction is None:
        interaction = [
            {"role": "user", "content": "Please check my account balance"},
            {"role": "assistant", "content": "Sure, let me look that up."},
            {"role": "user", "content": "Now transfer $500 to account X"},
        ]
    return {
        "id": item_id,
        "generation_config": {"environment": env, "dataset": dataset},
        "attack_plan": {
            "attack_goal": goal,
            "verified_tool_chain": tool_chain,
        },
        "interaction_history": interaction,
        "n_turns": 2,
        "final_attack_progress": progress,
        "final_agent_helpfulness": helpfulness,
        "judge_history": [],
    }


@pytest.fixture
def sample_items():
    """A list of diverse eval items for filter tests."""
    return [
        _make_item(item_id=1, env="banking", dataset="SHADE_Arena", progress=3),
        _make_item(item_id=2, env="travel", dataset="SHADE_Arena", progress=1),
        _make_item(item_id=3, env="banking", dataset="SHADE_Arena", progress=-1),
        _make_item(item_id=100, env="ProductManager", dataset="Agent_SafetyBench", progress=2),
        _make_item(item_id=10001, env="Email", dataset="Agent_SafetyBench", progress=3),
        _make_item(item_id=10002, env="HomeAssistant", dataset="Agent_SafetyBench", progress=0),
    ]


@pytest.fixture
def single_item():
    return _make_item()


# ---------------------------------------------------------------------------
# Tests: filter_items
# ---------------------------------------------------------------------------

# Import only the pure-Python helpers (no ToolShield dependency)
# We use importlib to handle the case where ToolShield is not installed
# by mocking the heavy imports at module load time.

@pytest.fixture(autouse=True)
def _mock_toolshield_imports(monkeypatch):
    """Prevent ToolShield imports from failing in CI."""
    import sys
    import types

    # Create stub modules so distill_defense.py can import without ToolShield
    for mod_name in [
        "toolshield", "toolshield.exp_generate", "toolshield.tree_generation",
        "toolshield.prompts", "Baseline.toolshield_patch",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            # Add attributes that distill_defense.py reads at import time
            if mod_name == "toolshield.exp_generate":
                stub.client = None
                stub.apply_experience_result = lambda exps, result: (exps, {"changed": False})
                stub.extract_json_from_response = lambda x: x
                stub.get_next_exp_key = lambda exps: "exp_1"
                stub.load_experience_list = lambda path: {}
                stub.save_experience_list = lambda exps, path: None
                stub.truncate_trajectory = lambda x: x
                stub.MODEL = "test-model"
                stub.TEMPERATURE = 0.0
            if mod_name == "toolshield.tree_generation":
                stub.client = None
            if mod_name == "toolshield.prompts":
                stub.EXPERIENCE_LEARNING_SYSTEM_PROMPT = ""
                stub.EXPERIENCE_LEARNING_USER_TEMPLATE = "{trajectory_summary}{current_experiences}"
                stub.TRAJECTORY_SUMMARY_PROMPT = ""
                stub.TRAJECTORY_SUMMARY_USER_TEMPLATE = "{tree_context}{task_content}{setup_files}{state_data}"
            monkeypatch.setitem(sys.modules, mod_name, stub)


def _import_distill():
    """Lazy-import distill_defense module (after mocks are in place)."""
    import importlib
    mod = importlib.import_module("distill_defense.distill_defense")
    importlib.reload(mod)  # ensure stubs are picked up
    return mod


class TestFilterItems:
    """Tests for filter_items()."""

    def test_no_filters(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, None, None, None)
        assert len(result) == len(sample_items)

    def test_min_progress(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, min_progress=2, max_progress=None, envs=None)
        assert all(i["final_attack_progress"] >= 2 for i in result)
        assert len(result) == 3  # progress 3, 2, 3

    def test_max_progress(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, min_progress=None, max_progress=0, envs=None)
        assert all(i["final_attack_progress"] <= 0 for i in result)
        assert len(result) == 2  # progress -1, 0

    def test_min_and_max_progress(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, min_progress=1, max_progress=2, envs=None)
        assert all(1 <= i["final_attack_progress"] <= 2 for i in result)
        assert len(result) == 2  # progress 1, 2

    def test_filter_by_env(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, None, None, envs=["banking"])
        assert all(i["generation_config"]["environment"] == "banking" for i in result)
        assert len(result) == 2

    def test_filter_by_multiple_envs(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, None, None, envs=["banking", "Email"])
        assert len(result) == 3

    def test_filter_by_dataset(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, None, None, None, dataset="Agent_SafetyBench")
        assert all(i["generation_config"]["dataset"] == "Agent_SafetyBench" for i in result)
        assert len(result) == 3

    def test_filter_by_min_id(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, None, None, None, min_id=10000)
        assert all(i["id"] >= 10000 for i in result)
        assert len(result) == 2

    def test_filter_by_max_id(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, None, None, None, max_id=100)
        assert all(i["id"] <= 100 for i in result)
        assert len(result) == 4  # ids 1, 2, 3, 100

    def test_combined_filters(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(
            sample_items, min_progress=2, max_progress=None,
            envs=None, dataset="Agent_SafetyBench", min_id=10000,
        )
        assert len(result) == 1
        assert result[0]["id"] == 10001

    def test_empty_result(self, sample_items):
        mod = _import_distill()
        result = mod.filter_items(sample_items, None, None, envs=["nonexistent"])
        assert result == []

    def test_empty_input(self):
        mod = _import_distill()
        result = mod.filter_items([], min_progress=3, max_progress=None, envs=None)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: format_interaction_as_state
# ---------------------------------------------------------------------------

class TestFormatInteractionAsState:

    def test_basic_output(self, single_item):
        mod = _import_distill()
        result = mod.format_interaction_as_state(single_item)
        parsed = json.loads(result)
        assert parsed["environment"] == "banking"
        assert parsed["attack_goal"] == "Transfer funds to attacker account"
        assert parsed["n_turns"] == 2
        assert parsed["final_attack_progress"] == 2
        assert len(parsed["interaction_history"]) == 3

    def test_includes_judge_history_when_present(self):
        mod = _import_distill()
        item = _make_item()
        item["judge_history"] = [{"score": 2, "reasoning": "test"}]
        result = json.loads(mod.format_interaction_as_state(item))
        assert "judge_history" in result
        assert result["judge_history"][0]["score"] == 2

    def test_excludes_judge_history_when_empty(self, single_item):
        mod = _import_distill()
        result = json.loads(mod.format_interaction_as_state(single_item))
        assert "judge_history" not in result

    def test_defaults_for_missing_fields(self):
        mod = _import_distill()
        item = {
            "generation_config": {"environment": "test"},
            "attack_plan": {"attack_goal": "goal"},
            "interaction_history": [],
        }
        result = json.loads(mod.format_interaction_as_state(item))
        assert result["n_turns"] == 0
        assert result["final_attack_progress"] == 0
        assert result["final_agent_helpfulness"] == 0


# ---------------------------------------------------------------------------
# Tests: format_task_content
# ---------------------------------------------------------------------------

class TestFormatTaskContent:

    def test_contains_goal_and_env(self, single_item):
        mod = _import_distill()
        result = mod.format_task_content(single_item)
        assert "Transfer funds to attacker account" in result
        assert "banking" in result

    def test_contains_tool_chain_steps(self, single_item):
        mod = _import_distill()
        result = mod.format_task_content(single_item)
        assert "get_iban" in result
        assert "schedule_transaction" in result
        assert "Step 1" in result
        assert "Step 2" in result

    def test_contains_user_prompts(self, single_item):
        mod = _import_distill()
        result = mod.format_task_content(single_item)
        assert "Turn 1:" in result
        assert "Turn 2:" in result
        assert "check my account balance" in result

    def test_empty_tool_chain(self):
        mod = _import_distill()
        item = _make_item(tool_chain=[])
        result = mod.format_task_content(item)
        assert "Attack Chain:" in result
        assert "Step" not in result

    def test_long_content_truncated(self):
        mod = _import_distill()
        long_msg = "x" * 500
        item = _make_item(interaction=[{"role": "user", "content": long_msg}])
        result = mod.format_task_content(item)
        # Content should be truncated to 300 chars
        assert "x" * 300 in result
        assert "x" * 301 not in result


# ---------------------------------------------------------------------------
# Tests: build_tree_context
# ---------------------------------------------------------------------------

class TestBuildTreeContext:

    def test_basic_structure(self, single_item):
        mod = _import_distill()
        result = json.loads(mod.build_tree_context([single_item]))
        assert result["root"] == "STAC_eval"
        assert len(result["children"]) == 1
        child = result["children"][0]
        assert child["function"] == "banking"
        assert "get_iban" in child["capability"]

    def test_deduplicates_same_env_goal(self):
        mod = _import_distill()
        items = [_make_item(), _make_item()]  # same env + goal
        result = json.loads(mod.build_tree_context(items))
        assert len(result["children"]) == 1

    def test_different_envs_produce_multiple_children(self):
        mod = _import_distill()
        items = [
            _make_item(env="banking", goal="goal A"),
            _make_item(env="travel", goal="goal B"),
        ]
        result = json.loads(mod.build_tree_context(items))
        assert len(result["children"]) == 2

    def test_caps_at_50_children(self):
        mod = _import_distill()
        items = [_make_item(env=f"env_{i}", goal=f"goal {i}") for i in range(60)]
        result = json.loads(mod.build_tree_context(items))
        assert len(result["children"]) == 50


# ---------------------------------------------------------------------------
# Tests: _auto_output_name
# ---------------------------------------------------------------------------

class TestAutoOutputName:

    def test_restructured_path(self):
        mod = _import_distill()
        path = Path("data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json")
        result = mod._auto_output_name(path)
        assert result.parent == Path("output")
        name = result.name
        assert name.endswith("-distilled-defense-experience.json")
        assert "toolshield" in name  # method from path
        assert "agent_safetybench" in name
        assert "adaptive" in name
        assert "gpt-4-1" in name  # dots replaced with dashes
        assert "no_defense" in name

    def test_restructured_stac_path(self):
        mod = _import_distill()
        path = Path("data/Eval_restructured/stac/shade_arena/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json")
        result = mod._auto_output_name(path)
        assert "stac" in result.name
        assert "shade_arena" in result.name

    def test_legacy_path_fallback(self):
        mod = _import_distill()
        path = Path("data/Eval_toolshield_asb/gpt-4.1/gpt-4.1/no_defense/gen_res.json")
        result = mod._auto_output_name(path)
        assert result.parent == Path("output")
        assert result.name.endswith("-distilled-defense-experience.json")

    def test_unknown_path(self):
        mod = _import_distill()
        path = Path("/tmp/random/gen_res.json")
        result = mod._auto_output_name(path)
        assert result.parent == Path("output")
        assert result.name.endswith("-distilled-defense-experience.json")

    def test_no_toolshield_distilled_in_suffix(self):
        """Output names should use distilled-defense, not toolshield-distilled-defense."""
        mod = _import_distill()
        path = Path("data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json")
        result = mod._auto_output_name(path)
        assert "toolshield-distilled" not in result.name


# ---------------------------------------------------------------------------
# Tests: load_eval_results
# ---------------------------------------------------------------------------

class TestLoadEvalResults:

    def test_loads_json_array(self, tmp_path):
        mod = _import_distill()
        data = [_make_item(), _make_item(item_id=2)]
        path = tmp_path / "gen_res.json"
        path.write_text(json.dumps(data))
        result = mod.load_eval_results(path)
        assert len(result) == 2

    def test_rejects_non_array(self, tmp_path):
        mod = _import_distill()
        path = tmp_path / "gen_res.json"
        path.write_text(json.dumps({"key": "value"}))
        with pytest.raises(ValueError, match="Expected a JSON array"):
            mod.load_eval_results(path)

    def test_empty_array(self, tmp_path):
        mod = _import_distill()
        path = tmp_path / "gen_res.json"
        path.write_text("[]")
        result = mod.load_eval_results(path)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: pipeline config loading
# ---------------------------------------------------------------------------

class TestPipelineConfigLoading:
    """Tests for defense_pipeline_configs.yaml and load_configs()."""

    def test_yaml_parses(self):
        config_path = REPO_ROOT / "distill_defense" / "defense_pipeline_configs.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        assert isinstance(raw, dict)
        assert "_global" in raw

    def test_all_configs_have_description(self):
        config_path = REPO_ROOT / "distill_defense" / "defense_pipeline_configs.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        for name, cfg in raw.items():
            if name == "_global":
                continue
            assert "description" in cfg, f"Config '{name}' missing description"

    def test_all_configs_have_trajectory_or_defense_file(self):
        config_path = REPO_ROOT / "distill_defense" / "defense_pipeline_configs.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        for name, cfg in raw.items():
            if name == "_global":
                continue
            has_traj = "trajectory" in cfg
            has_def = "defense_file" in cfg
            assert has_traj or has_def, (
                f"Config '{name}' needs 'trajectory' or 'defense_file'"
            )

    def test_no_ts_prefix_in_config_names(self):
        """Verify config names don't use the old ts_ prefix."""
        config_path = REPO_ROOT / "distill_defense" / "defense_pipeline_configs.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        for name in raw:
            if name == "_global":
                continue
            assert not name.startswith("ts_"), (
                f"Config '{name}' still uses old ts_ prefix"
            )
            assert not name.startswith("distill_ts_"), (
                f"Config '{name}' still uses old distill_ts_ prefix"
            )

    def test_no_toolshield_distilled_in_defense_file_paths(self):
        """Defense file paths should use distilled-defense, not toolshield-distilled."""
        config_path = REPO_ROOT / "distill_defense" / "defense_pipeline_configs.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        for name, cfg in raw.items():
            if name == "_global":
                continue
            df = cfg.get("defense_file", "")
            assert "toolshield-distilled" not in df, (
                f"Config '{name}' defense_file uses old naming: {df}"
            )

    def test_steps_are_valid(self):
        config_path = REPO_ROOT / "distill_defense" / "defense_pipeline_configs.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        valid_steps = {"distill", "evaluate"}
        for name, cfg in raw.items():
            if name == "_global":
                continue
            steps = cfg.get("steps", ["distill", "evaluate"])
            for s in steps:
                assert s in valid_steps, (
                    f"Config '{name}' has invalid step '{s}'"
                )

    def test_share_yaml_also_valid(self):
        config_path = REPO_ROOT / "distill_defense" / "ds_defense_pipeline_configs2share.yaml"
        if not config_path.exists():
            pytest.skip("Share config file not found")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        assert isinstance(raw, dict)
        for name in raw:
            if name == "_global":
                continue
            assert not name.startswith("ts_"), (
                f"Share config '{name}' still uses old ts_ prefix"
            )


# ---------------------------------------------------------------------------
# Tests: pipeline env var helpers
# ---------------------------------------------------------------------------

class TestEnvVarHelpers:

    def test_set_and_restore_env_vars(self):
        from distill_defense.pipeline_distill_and_eval_defense import (
            set_env_vars, restore_env_vars,
        )
        key = "_DISTILL_DEFENSE_TEST_VAR"
        assert key not in os.environ

        old = set_env_vars({key: "test_value"})
        assert os.environ[key] == "test_value"

        restore_env_vars(old)
        assert key not in os.environ

    def test_set_env_skips_empty(self):
        from distill_defense.pipeline_distill_and_eval_defense import set_env_vars
        key = "_DISTILL_DEFENSE_TEST_EMPTY"
        old = set_env_vars({key: ""})
        assert key not in os.environ
        assert old == {}

    def test_set_env_skips_none(self):
        from distill_defense.pipeline_distill_and_eval_defense import set_env_vars
        key = "_DISTILL_DEFENSE_TEST_NONE"
        old = set_env_vars({key: None})
        assert key not in os.environ
        assert old == {}

    def test_restore_preserves_existing(self):
        from distill_defense.pipeline_distill_and_eval_defense import (
            set_env_vars, restore_env_vars,
        )
        key = "_DISTILL_DEFENSE_TEST_EXISTING"
        os.environ[key] = "original"
        try:
            old = set_env_vars({key: "new_value"})
            assert os.environ[key] == "new_value"
            restore_env_vars(old)
            assert os.environ[key] == "original"
        finally:
            os.environ.pop(key, None)
