#!/usr/bin/env python3
"""Wrapper to run STAC attacks against MCP-backed benchmarks.

Handles the vllm import issue (not available on macOS/non-GPU machines)
and provides a clean entry point for OAS and SafeArena evaluation.

Usage::

    # OpenAgentSafety (all 356 tasks)
    python MCP/run_eval.py --benchmark oas --model_agent gpt-4.1

    # OAS benign-intent only (245 tasks)
    python MCP/run_eval.py --benchmark oas --benchmark_split benign --model_agent gpt-4.1

    # OAS with defense
    python MCP/run_eval.py --benchmark oas --model_agent gpt-4.1 --defense reasoning

    # SafeArena
    python MCP/run_eval.py --benchmark safearena --model_agent gpt-4.1

    # Hand-crafted scenarios
    python MCP/run_eval.py --scenarios MCP/benchmarks_data/quick_scenarios/ --model_agent gpt-4.1

    # Smoke test (2 tasks, no state verification)
    python MCP/run_eval.py --benchmark oas --max_tasks 2 --model_agent gpt-4.1 --no_state_verify
"""

from __future__ import annotations

import importlib
import sys
import types


def _ensure_vllm_importable():
    """Install a stub ``vllm`` module if the real one is not available.

    ``src/LanguageModels.py`` has ``import vllm`` at the top level.
    On macOS / non-GPU machines vllm is not installable, which blocks
    the entire import chain (LanguageModels → Agents → STAC → eval_mcp).

    The stub is only used so that imports succeed — VllmLM will raise at
    runtime if actually instantiated, which is fine because MCP eval uses
    OpenAI or Bedrock models.
    """
    if importlib.util.find_spec("vllm") is not None:
        return  # real vllm available, nothing to do

    stub = types.ModuleType("vllm")
    stub.__file__ = "<vllm-stub>"
    stub.__path__ = []

    # vllm.LLM and vllm.SamplingParams are referenced in VllmLM
    class _StubLLM:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "vllm is not installed. Use an OpenAI or Bedrock model "
                "(e.g. --model_agent gpt-4.1) for MCP evaluation."
            )

    class _StubSamplingParams:
        def __init__(self, *a, **kw):
            raise RuntimeError("vllm is not installed.")

    stub.LLM = _StubLLM
    stub.SamplingParams = _StubSamplingParams
    sys.modules["vllm"] = stub


def main():
    import os

    # Ensure project root is CWD and on sys.path — required because
    # src/Environments.py uses relative sys.path.append('SHADE_Arena') etc.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Remove script directory (MCP/) from sys.path if present — MCP/core/utils.py
    # shadows SHADE_Arena/utils/ (a package) and breaks SHADE_Arena imports.
    mcp_dir = os.path.join(project_root, "MCP")
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != mcp_dir]

    _ensure_vllm_importable()

    # Now safe to import the real eval entry point
    from MCP.eval.eval_mcp import main as eval_main
    eval_main()


if __name__ == "__main__":
    main()
