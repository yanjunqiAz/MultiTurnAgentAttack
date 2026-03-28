"""Model-specific formatting utilities for MCP tool schemas and results.

Replicates the formatting patterns from src/Environments.py (lines 437-474
for tool configs, lines 302-330 for tool results) without modifying any
existing file. All model routing uses the same string-matching logic as
the rest of the codebase.
"""

from __future__ import annotations

import json


def _is_bedrock_model(model_id: str) -> bool:
    """Check if model_id routes to AWS Bedrock (Claude, Llama, DeepSeek)."""
    mid = model_id.lower()
    return "claude" in mid or "llama" in mid or "deepseek" in mid


def _is_openai_model(model_id: str) -> bool:
    """Check if model_id routes to OpenAI (GPT, o3, o4)."""
    mid = model_id.lower()
    return "gpt" in mid or "o3" in mid or "o4" in mid


def format_tools_for_model(tool_schemas: list[dict], model_id: str):
    """Convert MCP tool schemas to model-specific tool_config format.

    Args:
        tool_schemas: list of dicts with keys "name", "description",
            "inputSchema" (JSON Schema object with "type", "properties", etc.)
        model_id: determines output format via string matching.

    Returns:
        Bedrock (Claude/Llama/DeepSeek): dict  ``{"tools": [{"toolSpec": ...}]}``
        OpenAI / vLLM:                   str   JSON-encoded list (OpenAILM.generate
                                               calls json.loads() on it at
                                               LanguageModels.py:1040)
    """
    if _is_bedrock_model(model_id):
        tool_list = []
        for s in tool_schemas:
            input_schema = s.get("inputSchema", {"type": "object", "properties": {}})
            # Bedrock's toolSpec wraps inputSchema under an extra {"json": ...} key
            tool_list.append({
                "toolSpec": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "inputSchema": {
                        "json": input_schema
                    },
                }
            })
        return {"tools": tool_list}

    # Both OpenAI and vLLM store tool_config as a JSON string.
    tool_list = []
    for s in tool_schemas:
        input_schema = s.get("inputSchema", {"type": "object", "properties": {}})
        if _is_openai_model(model_id):
            tool_list.append({
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": input_schema,
                    "type": "object",
                },
            })
        else:
            # vLLM format — flat dict, only properties (not full JSON Schema)
            tool_list.append({
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": input_schema.get("properties", {}),
            })
    return json.dumps(tool_list)


def format_tool_result_for_model(
    tool_call_id: str,
    tool_name: str,
    result_text: str,
    model_id: str,
) -> list[dict]:
    """Format a tool execution result into model-specific message format.

    Replicates the branching at Environments.py:317-328.

    Returns:
        Bedrock: ``[{"role": "tool", "content": [{"toolResult": ...}]}]``
        OpenAI:  ``[{"role": "tool", "tool_call_id": ..., "content": "..."}]``
    """
    tool_result = {
        "toolResult": {
            "toolUseId": tool_call_id,
            "content": [{"text": result_text}],
        }
    }

    if _is_bedrock_model(model_id):
        return [{"role": "tool", "content": [tool_result]}]

    # OpenAI / vLLM — flat format
    return [{
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": result_text,
    }]
