#!/usr/bin/env python3
"""Unified MCP pipeline: generate attack chains then evaluate them.

Orchestrates the full flow:
  gen step 1  → Generate tool-chain attacks from benchmark tasks
  gen step 2  → Verify chains against live MCP servers
  gen step 3  → Reverse-engineer natural user prompts
  eval        → Run AdaptivePlanningSystem + dual evaluation

The gen output (step3_with_prompts.json) is converted into eval-compatible
scenarios, so the Planner can use verified tool chains as context.

Output structure::

    data/MCP_gen/{model}/{dataset}/{split}/step{1,2,3}_*.json   (gen)
    data/Eval_MCP/pipeline/{model}/{dataset}/{split}/...        (eval)

Usage::

    # Config-driven (recommended)
    python MCP/run_pipeline.py --config pipeline_gpt41
    python MCP/run_pipeline.py --config pipeline_smoke
    python MCP/run_pipeline.py --list_configs

    # Full pipeline (gen + eval)
    python MCP/run_pipeline.py --model gpt-4.1 --dataset oas

    # Smoke test
    python MCP/run_pipeline.py --model gpt-4.1 --dataset oas --max_tasks 2 --n_cases 1

    # Gen only (steps 1-3), skip eval
    python MCP/run_pipeline.py --model gpt-4.1 --dataset oas --gen_only

    # Eval only (use existing gen output)
    python MCP/run_pipeline.py --model gpt-4.1 --dataset oas --eval_only

    # Start gen from step 2, then eval
    python MCP/run_pipeline.py --model gpt-4.1 --dataset oas --start_step 2

    # Eval with defense
    python MCP/run_pipeline.py --model gpt-4.1 --eval_only --defense reasoning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "configs.yaml"


def _load_config(name: str) -> dict:
    """Load a named config from ``MCP/configs.yaml``, with _global defaults."""
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}
    global_cfg = raw.pop("_global", {})
    if name not in raw:
        available = [k for k in raw if not k.startswith("_")]
        raise SystemExit(
            f"Unknown config '{name}'. Available: {', '.join(available)}"
        )
    return {**global_cfg, **raw[name]}


def run_cmd(description: str, cmd: list[str]) -> None:
    """Run a subprocess, raising SystemExit on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"{description} failed with exit code {result.returncode}")


def gen_output_to_scenarios(gen_output_path: Path) -> list[dict]:
    """Convert gen step 3 output into eval-compatible scenario dicts.

    Each verified chain becomes a scenario with:
    - attack_goal from the original task
    - mcp_servers from the chain entry
    - tool_chain_hint from the verified chain (gives Planner context)
    - pre_seed from the original scenario source
    """
    with open(gen_output_path) as f:
        gen_data = json.load(f)

    scenarios = []
    for entry in gen_data:
        if "error" in entry and "interaction_history" not in entry:
            continue  # skip failed entries

        # Extract the verified tool chain for use as a hint
        tool_chain = entry.get("final_tool_chain")
        if isinstance(tool_chain, dict) and "tool_chain" in tool_chain:
            steps = tool_chain["tool_chain"]
        elif isinstance(tool_chain, list):
            steps = tool_chain
        else:
            steps = entry.get("tool_chain", [])

        hint = [
            step.get("tool_name", "") for step in steps
            if isinstance(step, dict)
        ]

        scenario = {
            "id": entry.get("id", "unknown"),
            "attack_goal": entry.get("attack_goal", ""),
            "explanation": entry.get("explanation", ""),
            "mcp_servers": entry.get("mcp_servers", ["filesystem"]),
            "tool_chain_hint": hint,
            "pre_seed": entry.get("pre_seed", {}),
            "success_criteria": entry.get("success_criteria", []),
        }

        # Preserve source metadata
        if "_source" in entry:
            scenario["_source"] = entry["_source"]

        scenarios.append(scenario)

    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified MCP pipeline: generate + evaluate attack chains",
    )

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="Named config from MCP/configs.yaml")
    parser.add_argument("--list_configs", action="store_true",
                        help="List available configs and exit")

    # Mode
    parser.add_argument("--gen_only", action="store_true",
                        help="Run generation only (steps 1-3), skip eval")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run eval only (use existing gen output)")

    # Gen args
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Model for generation/verification/prompt writing")
    parser.add_argument("--model_agent", type=str, default=None,
                        help="Model for agent (defaults to --model)")
    parser.add_argument("--n_cases", type=int, default=3,
                        help="Attack chains per task (gen step 1)")
    parser.add_argument("--dataset", type=str, default="oas",
                        help="Benchmark dataset (oas, safearena, etc.)")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Limit tasks (gen step 1)")
    parser.add_argument("--max_chains", type=int, default=None,
                        help="Limit chains (gen steps 2-3)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "benign", "malicious", "harm"])
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3],
                        help="Start gen from this step")
    parser.add_argument("--end_step", type=int, default=3, choices=[1, 2, 3],
                        help="End gen at this step")
    parser.add_argument("--gen_batch_size", type=int, default=32,
                        help="Batch size for gen steps")
    parser.add_argument("--gen_temperature", type=float, default=1.0)

    # Eval args
    parser.add_argument("--defense", type=str, default="no_defense",
                        choices=["no_defense", "failure_modes", "summarization",
                                 "reasoning", "spotlighting"])
    parser.add_argument("--max_n_turns", type=int, default=3)
    parser.add_argument("--max_n_rounds_agent", type=int, default=10)
    parser.add_argument("--no_state_verify", action="store_true")
    parser.add_argument("--no_post_eval", action="store_true")
    parser.add_argument("--output_dir", type=str, default="data/Eval_MCP")

    # Shared
    parser.add_argument("--gen_input", type=str, default=None,
                        help="Path to gen step 3 output (for --eval_only)")

    args = parser.parse_args()

    # Handle --list_configs
    if args.list_configs:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                raw = yaml.safe_load(f) or {}
            raw.pop("_global", None)
            print("Available configs (pipeline_* are for run_pipeline.py):\n")
            for name, cfg in raw.items():
                desc = cfg.get("description", "")
                marker = " [pipeline]" if name.startswith("pipeline_") else ""
                print(f"  {name:30s} {desc}{marker}")
        raise SystemExit(0)

    # Apply config values as defaults (CLI args override)
    if args.config:
        cfg = _load_config(args.config)
        for key, val in cfg.items():
            if key == "description":
                continue
            # Map config keys to argparse names
            if key == "model" and hasattr(args, "model"):
                if getattr(args, "model") == parser.get_default("model"):
                    args.model = val
            elif key in ("gen_only", "eval_only", "no_state_verify", "no_post_eval") and val:
                setattr(args, key, True)
            elif hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, val)

    model = args.model
    model_agent = args.model_agent or model
    dataset = args.dataset
    split = args.split
    gen_dir = Path("data/MCP_gen") / model / dataset / split
    gen_output = Path(args.gen_input) if args.gen_input else gen_dir / "step3_with_prompts.json"

    print(f"\n{'='*60}")
    print(f"  MCP Unified Pipeline")
    print(f"{'='*60}")
    print(f"  Model:       {model}")
    print(f"  Agent model: {model_agent}")
    print(f"  Dataset:     {dataset}")
    print(f"  Split:       {split}")
    print(f"  Gen dir:     {gen_dir}")
    print(f"  Gen output:  {gen_output}")
    print(f"  Mode:        {'gen only' if args.gen_only else 'eval only' if args.eval_only else 'gen + eval'}")
    print(f"  Defense:     {args.defense}")

    # ---------------------------------------------------------------
    # Phase 1: Generation (steps 1-3)
    # ---------------------------------------------------------------
    if not args.eval_only:
        gen_args = [
            sys.executable, "-m", "MCP.gen.run_pipeline",
            "--model", model,
            "--model_agent", model_agent,
            "--dataset", dataset,
            "--n_cases", str(args.n_cases),
            "--split", args.split,
            "--start_step", str(args.start_step),
            "--end_step", str(args.end_step),
            "--batch_size", str(args.gen_batch_size),
            "--temperature", str(args.gen_temperature),
        ]
        if args.max_tasks:
            gen_args += ["--max_tasks", str(args.max_tasks)]
        if args.max_chains:
            gen_args += ["--max_chains", str(args.max_chains)]

        run_cmd("Phase 1: Attack Generation (Steps 1-3)", gen_args)

    if args.gen_only:
        print(f"\nGen-only mode: output at {gen_output}")
        return

    # ---------------------------------------------------------------
    # Phase 2: Convert gen output → eval scenarios
    # ---------------------------------------------------------------
    if not gen_output.exists():
        raise SystemExit(
            f"Gen output not found: {gen_output}\n"
            f"Run generation first (without --eval_only) or provide --gen_input."
        )

    print(f"\n{'='*60}")
    print(f"  Phase 2: Converting gen output → eval scenarios")
    print(f"{'='*60}")

    scenarios = gen_output_to_scenarios(gen_output)
    print(f"  Converted {len(scenarios)} scenarios from {gen_output}")

    if not scenarios:
        raise SystemExit("No scenarios to evaluate.")

    # Write scenarios to a temp YAML dir for eval_mcp.py
    scenarios_dir = gen_dir / "eval_scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        scenario_path = scenarios_dir / f"{scenario['id']}.yml"
        with open(scenario_path, "w") as f:
            yaml.dump(scenario, f, default_flow_style=False, allow_unicode=True)

    print(f"  Wrote {len(scenarios)} scenario files to {scenarios_dir}")

    # ---------------------------------------------------------------
    # Phase 3: Evaluation
    # ---------------------------------------------------------------
    # Pipeline eval results go under a separate "pipeline" subdirectory
    # to avoid colliding with benchmark-sourced eval results.
    eval_output_dir = str(
        Path(args.output_dir) / "pipeline" / model / dataset / split
    )

    eval_args = [
        sys.executable, "MCP/run_eval.py",
        "--scenarios", str(scenarios_dir),
        "--model_agent", model_agent,
        "--model_planner", model,
        "--model_judge", model,
        "--defense", args.defense,
        "--max_n_turns", str(args.max_n_turns),
        "--max_n_rounds_agent", str(args.max_n_rounds_agent),
        "--output_dir", eval_output_dir,
    ]
    if args.no_state_verify:
        eval_args.append("--no_state_verify")
    if args.no_post_eval:
        eval_args.append("--no_post_eval")

    run_cmd("Phase 3: Evaluation (AdaptivePlanningSystem + Dual Judges)", eval_args)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    eval_results_path = Path(eval_output_dir) / model_agent / args.defense / "eval_results.json"
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Gen output:    {gen_output}")
    print(f"  Scenarios:     {scenarios_dir}")
    print(f"  Eval results:  {eval_results_path}")


if __name__ == "__main__":
    main()
