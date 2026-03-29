"""Step 2: Verify MCP tool-chain attacks against live MCP servers.

Takes Step 1 output, creates MCPEnvironments, and uses the STAC Verifier
to interactively execute and validate each chain.

Usage::

    python -m MCP.gen.step_2_verify_tool_chain --model gpt-4.1 --dataset oas
    python -m MCP.gen.step_2_verify_tool_chain --model gpt-4.1 --input data/MCP_gen/gpt-4.1/oas/all/step1_tool_chains.json
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import yaml

from src.STAC import Verifier
from MCP.core.mcp_environment import MCPEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_VERIFIER_ROUNDS = 20


def load_server_registry() -> dict:
    """Load MCP server registry from mcp_servers.yml."""
    cfg_path = Path("MCP/mcp_servers.yml")
    if not cfg_path.exists():
        raise SystemExit(f"MCP server config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("servers", {})


def flatten_attacks(step1_data: list[dict]) -> list[dict]:
    """Flatten Step 1 output: one entry per (task, attack_chain) pair."""
    flat = []
    for task in step1_data:
        for chain_idx, attack in enumerate(task.get("tool_chain_attacks", [])):
            flat.append({
                "id": f"{task['id']}_chain{chain_idx}",
                "task_id": task["id"],
                "attack_goal": attack.get("objective", task.get("attack_goal", "")),
                "explanation": attack.get("explanation", ""),
                "mcp_servers": task.get("mcp_servers", ["filesystem"]),
                "tool_chain": attack.get("tool_chain", []),
                "generation_config": task.get("generation_config", {}),
                "_source": task.get("_source", {}),
            })
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Verify MCP tool chains")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Model for verification")
    parser.add_argument("--input", type=str, default=None,
                        help="Step 1 output file")
    parser.add_argument("--gen_model", type=str, default="gpt-4.1",
                        help="Model used in step 1 (for auto-detecting input)")
    parser.add_argument("--dataset", type=str, default="oas",
                        help="Benchmark dataset (for auto-detecting input path)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Process one chain at a time (Verifier is interactive)")
    parser.add_argument("--max_rounds", type=int, default=MAX_VERIFIER_ROUNDS)
    parser.add_argument("--max_chains", type=int, default=None)
    parser.add_argument("--split", type=str, default="all",
                        help="Split used in step 1 (for auto-detecting input path)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path("data/MCP_gen") / args.gen_model / args.dataset / args.split / "step1_tool_chains.json"
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    with open(input_path) as f:
        step1_data = json.load(f)
    print(f"Loaded {len(step1_data)} tasks from {input_path}")

    # Output path
    outdir = Path("data/MCP_gen") / args.gen_model / args.dataset / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = Path(args.output) if args.output else outdir / "step2_verified_chains.json"

    # Load existing for resume
    existing: list[dict] = []
    if outpath.exists():
        with open(outpath) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already verified")
    done_ids = {r["id"] for r in existing}

    # Flatten to one entry per chain
    chains = flatten_attacks(step1_data)
    chains = [c for c in chains if c["id"] not in done_ids]
    if args.max_chains:
        chains = chains[:args.max_chains]
    print(f"Chains to verify: {len(chains)}")

    if not chains:
        print("All chains already verified.")
        return

    # Server registry
    server_registry = load_server_registry()
    print(f"Server registry: {list(server_registry.keys())}")

    # Create Verifier
    verifier = Verifier(
        model_id=args.model,
        temperature=0.6,
        max_tokens=4096,
        sys_prompt_path="prompts/verifier.md",
    )

    all_results = list(existing)

    for idx, chain_entry in enumerate(chains):
        chain_id = chain_entry["id"]
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(chains)}] Verifying: {chain_id}")
        print(f"  Goal: {chain_entry['attack_goal'][:100]}")
        print(f"  Chain: {len(chain_entry['tool_chain'])} steps")

        # Build MCPEnvironment
        needed_servers = chain_entry.get("mcp_servers", ["filesystem"])
        server_configs = {}
        for s in needed_servers:
            if s in server_registry:
                server_configs[s] = server_registry[s]
            else:
                logger.warning("Server '%s' not in registry, skipping", s)

        if not server_configs:
            logger.error("No servers available for %s", chain_id)
            all_results.append({**chain_entry, "verified": False, "error": "No servers"})
            continue

        try:
            env = MCPEnvironment(server_configs, args.model, chain_entry)
        except Exception as exc:
            logger.error("MCPEnvironment init failed for %s: %s", chain_id, exc)
            all_results.append({**chain_entry, "verified": False, "error": str(exc)})
            continue

        try:
            # Format tool chain as string for verifier
            tool_chain_str = json.dumps(chain_entry["tool_chain"], indent=2)
            failure_mode = chain_entry["attack_goal"]

            # Reset verifier with this single environment
            verifier.reset(
                envs=[env],
                tool_chains=[tool_chain_str],
                failure_modes=[failure_mode],
            )

            # Run interactive verification loop
            for round_num in range(args.max_rounds):
                done = verifier.step(batch_size=1)
                print(f"  Round {round_num + 1}/{args.max_rounds}: running={verifier.running}")
                if done:
                    break

            # Collect result
            final_chain = verifier.final_tool_chains[0]
            verified = final_chain is not None
            messages = deepcopy(verifier.messages[0]) if verifier.messages else []

            result = {
                **chain_entry,
                "verified": verified,
                "final_tool_chain": final_chain,
                "messages": [
                    # Convert messages to serializable format
                    msg if isinstance(msg, dict) else str(msg)
                    for msg in messages
                ],
                "n_rounds": round_num + 1,
            }
            if verified:
                print(f"  VERIFIED in {round_num + 1} rounds")
            else:
                print(f"  FAILED to verify after {round_num + 1} rounds")

        except Exception as exc:
            logger.error("Verification failed for %s: %s", chain_id, exc, exc_info=True)
            result = {**chain_entry, "verified": False, "error": str(exc)}

        finally:
            try:
                env.reset()
                env.close()
            except Exception:
                pass

        all_results.append(result)

        # Checkpoint
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Summary
    n_verified = sum(1 for r in all_results if r.get("verified"))
    n_failed = sum(1 for r in all_results if not r.get("verified"))
    print(f"\n{'='*60}")
    print(f"Step 2 complete: {n_verified} verified, {n_failed} failed → {outpath}")


if __name__ == "__main__":
    main()
