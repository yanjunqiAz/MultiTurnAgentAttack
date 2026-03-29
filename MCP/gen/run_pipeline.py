"""Run the MCP attack generation pipeline (Steps 1-3).

Orchestrates:
  Step 1: Generate tool-chain attacks from benchmark tasks
  Step 2: Verify chains against live MCP servers
  Step 3: Reverse-engineer natural user prompts

Usage::

    # Full pipeline (OAS dataset)
    python -m MCP.gen.run_pipeline --model gpt-4.1 --dataset oas

    # Smoke test (2 tasks, 1 chain each)
    python -m MCP.gen.run_pipeline --model gpt-4.1 --dataset oas --max_tasks 2 --n_cases 1

    # Start from step 2 (reuse step 1 output)
    python -m MCP.gen.run_pipeline --model gpt-4.1 --dataset oas --start_step 2

    # Steps 1-2 only (skip prompt generation)
    python -m MCP.gen.run_pipeline --model gpt-4.1 --end_step 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # load .env from project root


def run_step(step_num: int, module: str, args: list[str]) -> None:
    """Run a pipeline step as a subprocess."""
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {module}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, "-m", module] + args
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Step {step_num} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP attack generation pipeline")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Model for generation/verification/prompt writing")
    parser.add_argument("--model_agent", type=str, default=None,
                        help="Model for agent execution in step 3 (defaults to --model)")
    parser.add_argument("--dataset", type=str, default="oas",
                        help="Benchmark dataset (oas, safearena, etc.)")
    parser.add_argument("--n_cases", type=int, default=3,
                        help="Attack chains per task (step 1)")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Limit tasks (step 1)")
    parser.add_argument("--max_chains", type=int, default=None,
                        help="Limit chains to verify/process (steps 2-3)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "benign", "malicious", "harm"])
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--end_step", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    model_agent = args.model_agent or args.model
    gen_model = args.model

    dataset = args.dataset

    # Verify data directory
    outdir = Path("data/MCP_gen") / gen_model / dataset / args.split
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"MCP Generation Pipeline")
    print(f"  Model: {args.model}")
    print(f"  Model (agent): {model_agent}")
    print(f"  Dataset: {dataset}")
    print(f"  Split: {args.split}")
    print(f"  Steps: {args.start_step} → {args.end_step}")
    print(f"  Output: {outdir}")

    # Step 1: Generate tool chains
    if args.start_step <= 1 <= args.end_step:
        step1_args = [
            "--model", args.model,
            "--dataset", dataset,
            "--n_cases", str(args.n_cases),
            "--split", args.split,
            "--batch_size", str(args.batch_size),
            "--temperature", str(args.temperature),
        ]
        if args.max_tasks:
            step1_args += ["--max_tasks", str(args.max_tasks)]
        run_step(1, "MCP.gen.step_1_gen_tool_chains", step1_args)

    # Step 2: Verify tool chains
    if args.start_step <= 2 <= args.end_step:
        step2_args = [
            "--model", args.model,
            "--gen_model", gen_model,
            "--dataset", dataset,
            "--split", args.split,
            "--batch_size", "1",
        ]
        if args.max_chains:
            step2_args += ["--max_chains", str(args.max_chains)]
        run_step(2, "MCP.gen.step_2_verify_tool_chain", step2_args)

    # Step 3: Generate prompts
    if args.start_step <= 3 <= args.end_step:
        step3_args = [
            "--model", args.model,
            "--model_agent", model_agent,
            "--gen_model", gen_model,
            "--dataset", dataset,
            "--split", args.split,
            "--batch_size", str(args.batch_size),
        ]
        if args.max_chains:
            step3_args += ["--max_chains", str(args.max_chains)]
        run_step(3, "MCP.gen.step_3_gen_prompts", step3_args)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete (steps {args.start_step}-{args.end_step})")
    print(f"  Output directory: {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
