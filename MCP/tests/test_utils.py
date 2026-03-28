"""Tests for MCP/core/utils.py — model-specific formatting utilities.

Verifies that format_tools_for_model() and format_tool_result_for_model()
produce output matching the patterns in src/Environments.py for each model
type (Bedrock, OpenAI, vLLM).
"""

import json
import pytest

from MCP.core.utils import format_tools_for_model, format_tool_result_for_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SCHEMAS = [
    {
        "name": "execute_sql",
        "description": "Run a SQL query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL statement"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_tables",
        "description": "List database tables",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

BEDROCK_MODELS = [
    "us.anthropic.claude-sonnet-4-5-v1",
    "us.meta.llama3-1-405b-instruct-v1:0",
    "us.deepseek.r1-v1:0",
]

OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4o",
    "o3-mini",
    "o4-mini",
]

VLLM_MODELS = [
    "Qwen/Qwen3-32B",
    "mistralai/Mistral-7B",
]


# ---------------------------------------------------------------------------
# format_tools_for_model
# ---------------------------------------------------------------------------

class TestFormatToolsForModel:

    @pytest.mark.parametrize("model_id", BEDROCK_MODELS)
    def test_bedrock_returns_dict(self, model_id):
        result = format_tools_for_model(SAMPLE_SCHEMAS, model_id)
        assert isinstance(result, dict)
        assert "tools" in result

    @pytest.mark.parametrize("model_id", BEDROCK_MODELS)
    def test_bedrock_toolspec_structure(self, model_id):
        result = format_tools_for_model(SAMPLE_SCHEMAS, model_id)
        for tool in result["tools"]:
            assert "toolSpec" in tool
            spec = tool["toolSpec"]
            assert "name" in spec
            assert "description" in spec
            assert "inputSchema" in spec
            # Bedrock wraps under {"json": ...}
            assert "json" in spec["inputSchema"]

    @pytest.mark.parametrize("model_id", BEDROCK_MODELS)
    def test_bedrock_preserves_schema(self, model_id):
        result = format_tools_for_model(SAMPLE_SCHEMAS, model_id)
        spec = result["tools"][0]["toolSpec"]
        assert spec["name"] == "execute_sql"
        schema = spec["inputSchema"]["json"]
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    @pytest.mark.parametrize("model_id", OPENAI_MODELS)
    def test_openai_returns_json_string(self, model_id):
        result = format_tools_for_model(SAMPLE_SCHEMAS, model_id)
        assert isinstance(result, str)
        # Must be parseable by json.loads (as OpenAILM.generate does)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @pytest.mark.parametrize("model_id", OPENAI_MODELS)
    def test_openai_function_structure(self, model_id):
        parsed = json.loads(format_tools_for_model(SAMPLE_SCHEMAS, model_id))
        for tool in parsed:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"

    @pytest.mark.parametrize("model_id", VLLM_MODELS)
    def test_vllm_returns_json_string(self, model_id):
        result = format_tools_for_model(SAMPLE_SCHEMAS, model_id)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @pytest.mark.parametrize("model_id", VLLM_MODELS)
    def test_vllm_flat_structure(self, model_id):
        parsed = json.loads(format_tools_for_model(SAMPLE_SCHEMAS, model_id))
        for tool in parsed:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            # vLLM gets only properties, not full JSON Schema
            assert "type" not in tool.get("parameters", {})

    def test_empty_schemas(self):
        for model_id in ["gpt-4.1", "claude-sonnet-4-5", "Qwen/Qwen3-32B"]:
            result = format_tools_for_model([], model_id)
            if isinstance(result, dict):
                assert result["tools"] == []
            else:
                assert json.loads(result) == []

    def test_missing_inputschema_defaults(self):
        schema = [{"name": "foo", "description": "bar"}]
        result = format_tools_for_model(schema, "claude-sonnet-4-5")
        spec = result["tools"][0]["toolSpec"]
        assert spec["inputSchema"]["json"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# format_tool_result_for_model
# ---------------------------------------------------------------------------

class TestFormatToolResultForModel:

    @pytest.mark.parametrize("model_id", BEDROCK_MODELS)
    def test_bedrock_result_format(self, model_id):
        result = format_tool_result_for_model("call-1", "execute_sql", "OK", model_id)
        assert isinstance(result, list)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert isinstance(msg["content"], list)
        tr = msg["content"][0]["toolResult"]
        assert tr["toolUseId"] == "call-1"
        assert tr["content"][0]["text"] == "OK"

    @pytest.mark.parametrize("model_id", OPENAI_MODELS + VLLM_MODELS)
    def test_openai_vllm_result_format(self, model_id):
        result = format_tool_result_for_model("call-2", "list_tables", "t1,t2", model_id)
        assert isinstance(result, list)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-2"
        assert msg["name"] == "list_tables"
        assert msg["content"] == "t1,t2"

    def test_empty_result_text(self):
        result = format_tool_result_for_model("id", "fn", "", "gpt-4.1")
        assert result[0]["content"] == ""

        result = format_tool_result_for_model("id", "fn", "", "claude-sonnet-4-5")
        assert result[0]["content"][0]["toolResult"]["content"][0]["text"] == ""
