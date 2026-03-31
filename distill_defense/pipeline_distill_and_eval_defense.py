#!/usr/bin/env python3
"""
Closed-loop defense pipeline: distill defense from attack trajectories, then
evaluate with the distilled defense applied.

Stages:
  1. distill  -- Run distill_defense.py on a gen_res.json to produce a defense experience JSON
  2. evaluate -- Run eval_baseline.py (or eval_STAC_benchmark.py) with --defense toolshield_experience

This enables experiments like:
  - Distill defense from ASB trajectories, evaluate on SHADE attacks
  - Distill defense from no_defense trajectories, evaluate on the same data
  - Compare distilled defense against reasoning/summarization baselines

Usage:
    # Run a named config
    python -m distill_defense.pipeline_distill_and_eval_defense --config asb_distill_eval_stac

    # List all configs
    python -m distill_defense.pipeline_distill_and_eval_defense --list-configs

    # Full pipeline: distill from ASB eval, then evaluate on STAC benchmark
    python -m distill_defense.pipeline_distill_and_eval_defense \
        --trajectory data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
        --eval-input data/STAC_benchmark_data.json \
        --model_agent gpt-4.1

    # Distill only (skip evaluation)
    python -m distill_defense.pipeline_distill_and_eval_defense \
        --trajectory data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
        --steps distill

    # Evaluate with a pre-distilled defense file (skip distillation)
    python -m distill_defense.pipeline_distill_and_eval_defense \
        --defense-file output/toolshield_asb-gpt-4-1-no_defense-distilled-defense-experience.json \
        --eval-input data/STAC_benchmark_data.json \
        --model_agent gpt-4.1 \
        --steps evaluate

    # Cross-dataset: distill from ASB, evaluate on SHADE
    python -m distill_defense.pipeline_distill_and_eval_defense \
        --trajectory data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
        --eval-input data/toolshield_shade_stac.json \
        --model_agent gpt-4.1

    # Use STAC adaptive planner for evaluation instead of eval_baseline
    python -m distill_defense.pipeline_distill_and_eval_defense \
        --trajectory data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
        --eval-input data/STAC_benchmark_data.json \
        --model_agent gpt-4.1 \
        --evaluator stac
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "defense_pipeline_configs.yaml"
ALL_STEPS = ["distill", "evaluate"]


# -------------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------------

def load_configs() -> dict[str, Any]:
    """Load named configurations from defense_pipeline_configs.yaml."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    global_cfg: dict[str, Any] = {}
    if "_global" in raw:
        global_cfg = raw.pop("_global")

    global_env = global_cfg.pop("env", {}) or {}

    for cfg in raw.values():
        for key, val in global_cfg.items():
            cfg.setdefault(key, val)
        merged_env = dict(global_env)
        merged_env.update(cfg.get("env", {}) or {})
        cfg["env"] = merged_env

    return raw


def list_configs() -> None:
    """Print all available configurations and exit."""
    configs = load_configs()
    if not configs:
        print("No configurations found in defense_pipeline_configs.yaml")
        sys.exit(0)

    print(f"\nAvailable configurations ({CONFIG_PATH.name}):\n")
    print(f"  {'Name':<30} Description")
    print(f"  {'-'*30} {'-'*50}")
    for name, cfg in configs.items():
        desc = cfg.get("description", "")
        steps = cfg.get("steps", ALL_STEPS)
        print(f"  {name:<30} {desc}")
        print(f"  {'':30} steps={steps}")
        print()
    print(f"Usage: python -m distill_defense.pipeline_distill_and_eval_defense --config <name>")


def set_env_vars(env_vars: dict[str, str]) -> dict[str, str | None]:
    """Set environment variables, returning old values for restoration."""
    old = {}
    for key, val in env_vars.items():
        if val is None or str(val).strip() == "":
            continue
        old[key] = os.environ.get(key)
        os.environ[key] = str(val)
        display = str(val)[:8] + "..." if len(str(val)) > 12 else str(val)
        print(f"  export {key}={display}")
    return old


def restore_env_vars(old: dict[str, str | None]) -> None:
    """Restore environment variables to previous values."""
    for key, val in old.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def run_cmd(cmd: list[str], description: str) -> None:
    """Run a command, streaming output. Exit on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def auto_defense_path(trajectory_path: Path) -> Path:
    """Derive the defense output path using distill_defense's naming convention."""
    # Import the auto-naming function from distill_defense
    # We do a lazy import to avoid the heavy ToolShield library patching at CLI parse time
    import importlib
    mod = importlib.import_module("distill_defense.distill_defense")
    return mod._auto_output_name(trajectory_path)


def step_distill(args: argparse.Namespace) -> Path:
    """Stage 1: Distill defense from attack trajectories."""
    if args.defense_file:
        path = Path(args.defense_file)
        if not path.exists():
            print(f"ERROR: Defense file not found: {path}")
            sys.exit(1)
        print(f"Using pre-existing defense file: {path}")
        return path

    if not args.trajectory:
        print("ERROR: --trajectory is required for the distill step")
        sys.exit(1)

    trajectory = Path(args.trajectory)
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        sys.exit(1)

    defense_path = auto_defense_path(trajectory)

    cmd = [
        sys.executable, "-m", "distill_defense.distill_defense",
        "--input", str(trajectory),
        "--output", str(defense_path),
    ]
    if args.min_progress is not None:
        cmd += ["--min-progress", str(args.min_progress)]
    if args.max_progress is not None:
        cmd += ["--max-progress", str(args.max_progress)]
    if args.distill_envs:
        cmd += ["--envs"] + args.distill_envs
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.min_id is not None:
        cmd += ["--min-id", str(args.min_id)]
    if args.max_id is not None:
        cmd += ["--max-id", str(args.max_id)]
    if args.no_resume:
        cmd += ["--no-resume"]

    run_cmd(cmd, f"Distill defense from {trajectory.name}")
    return defense_path


def step_evaluate(args: argparse.Namespace, defense_file: Path) -> None:
    """Stage 2: Evaluate with distilled defense."""
    if not args.eval_input:
        print("ERROR: --eval-input is required for the evaluate step")
        sys.exit(1)

    eval_input = Path(args.eval_input)
    if not eval_input.exists():
        print(f"ERROR: Eval input not found: {eval_input}")
        sys.exit(1)

    if not defense_file.exists():
        print(f"ERROR: Defense file not found: {defense_file}")
        sys.exit(1)

    if args.evaluator == "baseline":
        cmd = [
            sys.executable, "-m", "Baseline.eval_baseline",
            "--input_path", str(eval_input),
            "--model_agent", args.model_agent,
            "--model_judge", args.model_judge,
            "--defense", "toolshield_experience",
            "--experience-file", str(defense_file),
            "--batch_size", str(args.batch_size),
        ]
    elif args.evaluator == "stac":
        cmd = [
            sys.executable, "-m", "STAC_eval.eval_STAC_benchmark",
            "--input_path", str(eval_input),
            "--model_agent", args.model_agent,
            "--model_judge", args.model_judge,
            "--defense", "toolshield_experience",
            "--experience-file", str(defense_file),
            "--batch_size", str(args.batch_size),
        ]
    else:
        print(f"ERROR: Unknown evaluator: {args.evaluator}")
        sys.exit(1)

    if args.region:
        cmd += ["--region", args.region]

    run_cmd(cmd, f"Evaluate with defense: {defense_file.name}  evaluator={args.evaluator}")


DEFAULTS = dict(
    steps=ALL_STEPS,
    trajectory=None,
    defense_file=None,
    min_progress=None,
    max_progress=None,
    distill_envs=None,
    dataset=None,
    min_id=None,
    max_id=None,
    no_resume=False,
    eval_input=None,
    evaluator="baseline",
    model_agent="gpt-4.1",
    model_judge="gpt-4.1",
    batch_size=1,
    region=None,
)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the distill + evaluate pipeline."""
    # Validate
    if "distill" in args.steps and not args.trajectory and not args.defense_file:
        print("ERROR: --trajectory or --defense-file is required for the distill step")
        sys.exit(1)
    if "evaluate" in args.steps and not args.eval_input:
        print("ERROR: --eval-input is required for the evaluate step")
        sys.exit(1)

    print(f"Pipeline: {' -> '.join(args.steps)}")

    # Stage 1: Distill
    defense_file = None
    if "distill" in args.steps:
        defense_file = step_distill(args)
    elif args.defense_file:
        defense_file = Path(args.defense_file)
    elif args.trajectory:
        defense_file = auto_defense_path(Path(args.trajectory))

    # Stage 2: Evaluate
    if "evaluate" in args.steps:
        if defense_file is None:
            print("ERROR: No defense file available for evaluation. "
                  "Run the distill step or pass --defense-file.")
            sys.exit(1)
        step_evaluate(args, defense_file)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    if defense_file:
        print(f"  Defense file: {defense_file}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop defense pipeline: distill + evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument("--config", nargs="+", default=None, metavar="NAME",
                        help="Named config(s) from defense_pipeline_configs.yaml")
    parser.add_argument("--list-configs", action="store_true",
                        help="List all available configurations and exit")

    # Pipeline control
    parser.add_argument(
        "--steps", nargs="+", default=None, choices=ALL_STEPS,
        help=f"Pipeline steps to run (default: {' '.join(ALL_STEPS)})",
    )

    # Distill inputs
    parser.add_argument("--trajectory", type=str, default=None,
        help="Path to gen_res.json (attack evaluation trajectories) for distillation")
    parser.add_argument("--defense-file", type=str, default=None,
        help="Path to pre-distilled defense JSON (skips distill step)")
    parser.add_argument("--min-progress", type=int, default=None,
        help="Only distill from items with final_attack_progress >= N")
    parser.add_argument("--max-progress", type=int, default=None,
        help="Only distill from items with final_attack_progress <= N")
    parser.add_argument("--distill-envs", nargs="+", default=None,
        help="Only distill from these environments")
    parser.add_argument("--dataset", type=str, default=None,
        choices=["SHADE_Arena", "Agent_SafetyBench"],
        help="Only distill from items matching this dataset")
    parser.add_argument("--min-id", type=int, default=None,
        help="Only distill from items with id >= N")
    parser.add_argument("--max-id", type=int, default=None,
        help="Only distill from items with id < N")
    parser.add_argument("--no-resume", action="store_true",
        help="Start distillation fresh even if output exists")

    # Evaluate inputs
    parser.add_argument("--eval-input", type=str, default=None,
        help="Path to STAC-format JSON for evaluation")
    parser.add_argument("--evaluator", type=str, default=None,
        choices=["baseline", "stac"],
        help="Which evaluator to use (default: baseline)")
    parser.add_argument("--model_agent", type=str, default=None)
    parser.add_argument("--model_judge", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--region", type=str, default=None)

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    # ------------------------------------------------------------------
    # Config mode
    # ------------------------------------------------------------------
    if args.config:
        all_configs = load_configs()
        for config_name in args.config:
            if config_name not in all_configs:
                print(f"ERROR: Unknown config '{config_name}'. Use --list-configs to see available.")
                sys.exit(1)

            cfg = all_configs[config_name]
            print(f"\n{'#' * 60}")
            print(f"  Config: {config_name}")
            print(f"  {cfg.get('description', '')}")
            print(f"{'#' * 60}")

            # Build run_args from defaults -> config -> CLI overrides
            run_args = argparse.Namespace(**DEFAULTS)

            # Apply config values
            for key in DEFAULTS:
                if key in cfg:
                    setattr(run_args, key, cfg[key])

            # CLI overrides (only if explicitly provided)
            skip_keys = {"config", "list_configs", "no_resume"}
            for k, v in vars(args).items():
                if v is not None and k not in skip_keys:
                    setattr(run_args, k, v)
            if args.no_resume:
                run_args.no_resume = True

            # Set env vars
            old_env = {}
            if "env" in cfg and isinstance(cfg["env"], dict):
                print("\nSetting environment variables:")
                old_env = set_env_vars(cfg["env"])

            try:
                run_pipeline(run_args)
            finally:
                if old_env:
                    restore_env_vars(old_env)

        print(f"\n{'#' * 60}")
        print(f"  All configs complete: {', '.join(args.config)}")
        print(f"{'#' * 60}")
        return

    # ------------------------------------------------------------------
    # Direct CLI mode (no --config)
    # ------------------------------------------------------------------
    run_args = argparse.Namespace(**{
        k: (getattr(args, k) if getattr(args, k) is not None else v)
        for k, v in DEFAULTS.items()
    })
    # Handle store_true flag
    if args.no_resume:
        run_args.no_resume = True

    run_pipeline(run_args)


if __name__ == "__main__":
    main()
