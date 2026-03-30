"""Step 3: Reverse-engineer natural user prompts for verified MCP tool chains.

Takes Step 2 output (verified chains), uses PromptWriter to generate a
natural-sounding user prompt for each step in the chain, and constructs
the full multi-turn interaction history.

Usage::

    python -m MCP.gen.step_3_gen_prompts --model gpt-4.1 --dataset oas
    python -m MCP.gen.step_3_gen_prompts --model gpt-4.1 --input data/MCP_gen/gpt-4.1/oas/all/step2_verified_chains.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import yaml

from src.Agents import Agent
from src.STAC import PromptWriter
from src.utils import convert_message_between_APIs
from MCP.core.mcp_environment import MCPEnvironment
from prompts.agent import AgentSysemPrompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def load_server_registry() -> dict:
    """Load MCP server registry."""
    cfg_path = Path("MCP/mcp_servers.yml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("servers", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Generate prompts for MCP chains")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Model for prompt writing")
    parser.add_argument("--model_agent", type=str, default="gpt-4.1",
                        help="Model for the agent that executes tool calls")
    parser.add_argument("--input", type=str, default=None,
                        help="Step 2 output file")
    parser.add_argument("--gen_model", type=str, default="gpt-4.1",
                        help="Model used in steps 1-2 (for auto-detecting input)")
    parser.add_argument("--dataset", type=str, default="oas",
                        help="Benchmark dataset (for auto-detecting input path)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-chain timeout in seconds (0=no timeout)")
    parser.add_argument("--max_chains", type=int, default=None)
    parser.add_argument("--split", type=str, default="all",
                        help="Split used in steps 1-2 (for auto-detecting input path)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path("data/MCP_gen") / args.gen_model / args.dataset / args.split / "step2_verified_chains.json"
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    with open(input_path) as f:
        step2_data = json.load(f)

    # Filter to verified chains only
    verified = [d for d in step2_data if d.get("verified")]
    print(f"Loaded {len(step2_data)} chains, {len(verified)} verified")

    # Output path
    outdir = Path("data/MCP_gen") / args.gen_model / args.dataset / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = Path(args.output) if args.output else outdir / "step3_with_prompts.json"

    # Resume
    existing: list[dict] = []
    if outpath.exists():
        with open(outpath) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already processed")
    done_ids = {r["id"] for r in existing}

    chains = [c for c in verified if c["id"] not in done_ids]
    if args.max_chains:
        chains = chains[:args.max_chains]
    print(f"Chains to process: {len(chains)}")

    if not chains:
        print("All chains already processed.")
        return

    # Server registry
    server_registry = load_server_registry()

    # Create PromptWriter
    prompt_writer = PromptWriter(
        model_id=args.model,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
        sys_prompt_path="prompts/prompt_writer.md",
    )

    all_results = list(existing)

    for idx, chain_entry in enumerate(chains):
        chain_id = chain_entry["id"]
        # Use the verified chain if available, otherwise original
        tool_chain = chain_entry.get("final_tool_chain")
        if isinstance(tool_chain, dict) and "tool_chain" in tool_chain:
            steps = tool_chain["tool_chain"]
        elif isinstance(tool_chain, list):
            steps = tool_chain
        else:
            steps = chain_entry.get("tool_chain", [])

        if not steps:
            logger.warning("No tool chain steps for %s, skipping", chain_id)
            continue

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(chains)}] Generating prompts: {chain_id}")
        print(f"  Chain: {len(steps)} steps")

        # Build MCPEnvironment for the agent
        needed_servers = chain_entry.get("mcp_servers", ["filesystem"])
        server_configs = {s: server_registry[s] for s in needed_servers if s in server_registry}

        if not server_configs:
            logger.error("No servers for %s", chain_id)
            continue

        try:
            env = MCPEnvironment(server_configs, args.model_agent, chain_entry)
        except Exception as exc:
            logger.error("MCPEnvironment failed for %s: %s", chain_id, exc)
            continue

        try:
            # Create agent for executing tool calls
            agent = Agent(
                model_id=args.model_agent, envs=[env],
                temperature=0.0, top_p=0.95,
                max_tokens=2048, sys_prompt_path=None,
                n_agents=1,
            )
            prompts_obj = AgentSysemPrompts()
            mid = args.model_agent.lower()
            if "gpt" in mid or "o3" in mid or "o4" in mid:
                agent.extend_sys_prompt(prompts_obj.basic_sys_prompt_gpt)
            else:
                agent.extend_sys_prompt(prompts_obj.basic_sys_prompt)

            agent.reset([env])

            # Build interaction history step by step
            interaction_history: list[dict] = []
            chain_start = time.time()
            chain_timed_out = False

            for step_idx, step in enumerate(steps):
                # Check per-chain timeout
                if args.timeout and (time.time() - chain_start) > args.timeout:
                    print(f"  TIMEOUT: chain took >{args.timeout}s, skipping remaining steps")
                    chain_timed_out = True
                    break
                tool_name = step.get("tool_name", "")
                parameters = step.get("parameters", {})
                target_tool_call = json.dumps({
                    "tool_name": tool_name,
                    "parameters": parameters,
                })

                print(f"  Step {step_idx + 1}/{len(steps)}: {tool_name}")

                # Generate prompt for this step
                try:
                    generated_prompts = prompt_writer.step(
                        interaction_histories=[interaction_history],
                        target_tool_calls=[target_tool_call],
                        batch_size=1,
                    )
                    user_prompt = generated_prompts[0] if generated_prompts else None
                except Exception as exc:
                    logger.warning("PromptWriter failed at step %d: %s", step_idx + 1, exc)
                    user_prompt = None

                if not user_prompt:
                    logger.warning("No prompt generated for step %d, using fallback", step_idx + 1)
                    user_prompt = step.get("purpose", f"Please use {tool_name}")

                # Add user message to history
                interaction_history.append({
                    "role": "user",
                    "content": user_prompt,
                })

                # Execute the tool call via agent
                # Send the prompt to the agent and let it make tool calls
                agent.step([user_prompt], spotlighting=False)

                # Extract new messages from agent
                if agent.messages and agent.messages[0]:
                    # Get messages added since our last check
                    current_msgs = agent.messages[0]
                    # The agent appends to its message list, we need the new ones
                    # after our user prompt (which is len(interaction_history) position)
                    new_start = len(interaction_history)
                    for msg in current_msgs[new_start:]:
                        try:
                            converted = convert_message_between_APIs(msg, "gpt")
                            interaction_history.append(converted)
                        except Exception:
                            interaction_history.append(msg)

                print(f"    Prompt: {user_prompt[:80]}...")

            # Build result
            result = deepcopy(chain_entry)
            # Remove bulky verification messages
            result.pop("messages", None)
            result["interaction_history"] = interaction_history
            result["n_turns"] = sum(1 for m in interaction_history if m.get("role") == "user")

            all_results.append(result)
            print(f"  Done: {result['n_turns']} turns, {len(interaction_history)} messages")

        except Exception as exc:
            logger.error("Prompt generation failed for %s: %s", chain_id, exc, exc_info=True)
            result = deepcopy(chain_entry)
            result.pop("messages", None)
            result["error"] = str(exc)
            all_results.append(result)

        finally:
            try:
                env.reset()
                env.close()
            except Exception:
                pass

        # Checkpoint
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Summary
    n_ok = sum(1 for r in all_results if "interaction_history" in r)
    n_err = sum(1 for r in all_results if "error" in r)
    print(f"\n{'='*60}")
    print(f"Step 3 complete: {n_ok} with prompts, {n_err} errors → {outpath}")


if __name__ == "__main__":
    main()
