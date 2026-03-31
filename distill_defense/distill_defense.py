#!/usr/bin/env python3
"""
Distill defense experiences from STAC adaptive evaluation results.

Reads gen_res.json (output of eval_baseline.py or eval_STAC_benchmark.py) and
runs two-phase experience distillation (via ToolShield library):
  Phase 1: Trajectory summarization (LLM analyzes the interaction history)
  Phase 2: Experience learning (LLM extracts safety guidelines)

Produces a JSON defense experience file compatible with ToolShield's format.

Usage:
    # Distill from ASB evaluation results (output name auto-generated)
    # -> output/toolshield_asb-gpt-4-1-no_defense-distilled-defense-experience.json
    python -m distill_defense.distill_defense \
        --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json

    # Explicit output path
    python -m distill_defense.distill_defense \
        --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
        --output output/my-custom-defense.json

    # Only learn from successful attacks (where agent failed to refuse)
    python -m distill_defense.distill_defense \
        --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
        --min-progress 3

    # Filter by environment
    python -m distill_defense.distill_defense \
        --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
        --envs ProductManager HomeAssistant

    # Import the result into an agent
    toolshield import \
        --exp-file output/toolshield_asb-gpt-4-1-no_defense-distilled-defense-experience.json \
        --agent claude_code
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Patch ToolShield to use LiteLLM (must precede all toolshield imports)
# ---------------------------------------------------------------------------
import os as _os

# Default to gpt-4.1 via LiteLLM if no model is explicitly set
if not _os.environ.get("TOOLSHIELD_MODEL_NAME"):
    _os.environ["TOOLSHIELD_MODEL_NAME"] = "openai/gpt-4.1"

# exp_generate also creates an OpenAI client at import time using
# OPENROUTER_API_KEY.  Ensure the dummy key is present for that import too.
_had_key = "OPENROUTER_API_KEY" in _os.environ
if not _had_key:
    _os.environ["OPENROUTER_API_KEY"] = "unused-placeholder"

import Baseline.toolshield_patch  # noqa: F401  — patches tree_generation.client

# Now exp_generate can import safely (its OpenAI client won't crash)
from toolshield import exp_generate as _exp_mod
from toolshield import tree_generation as _tree_mod

# Reuse the LiteLLM-patched client for exp_generate too
_exp_mod.client = _tree_mod.client

if not _had_key:
    _os.environ.pop("OPENROUTER_API_KEY", None)

from toolshield.exp_generate import (
    apply_experience_result,
    extract_json_from_response,
    get_next_exp_key,
    load_experience_list,
    save_experience_list,
    truncate_trajectory,
    MODEL,
    TEMPERATURE,
)
from toolshield.prompts import (
    EXPERIENCE_LEARNING_SYSTEM_PROMPT,
    EXPERIENCE_LEARNING_USER_TEMPLATE,
    TRAJECTORY_SUMMARY_PROMPT,
    TRAJECTORY_SUMMARY_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Format STAC eval items into expected prompt inputs
# ---------------------------------------------------------------------------

def format_interaction_as_state(item: Dict[str, Any]) -> str:
    """Format a gen_res.json item's interaction_history as a state string
    suitable for the {state_data} slot in TRAJECTORY_SUMMARY_USER_TEMPLATE."""
    state = {
        "environment": item["generation_config"]["environment"],
        "attack_goal": item["attack_plan"]["attack_goal"],
        "n_turns": item.get("n_turns", 0),
        "final_attack_progress": item.get("final_attack_progress", 0),
        "final_agent_helpfulness": item.get("final_agent_helpfulness", 0),
        "interaction_history": item["interaction_history"],
    }
    # Include judge history if available (gives outcome context)
    if item.get("judge_history"):
        state["judge_history"] = item["judge_history"]
    return json.dumps(state, indent=2)


def format_task_content(item: Dict[str, Any]) -> str:
    """Format the attack plan as task content for the trajectory summary."""
    plan = item["attack_plan"]
    lines = [
        f"Attack Goal: {plan['attack_goal']}",
        f"Environment: {item['generation_config']['environment']}",
        "",
        "Attack Chain:",
    ]
    for step in plan.get("verified_tool_chain", []):
        step_num = step.get("step", "?")
        tool = step.get("tool_name", "unknown")
        purpose = step.get("purpose", "")
        lines.append(f"  Step {step_num}: [{tool}] {purpose}")

    # Also include the multi-turn user prompts for full context
    lines.append("")
    lines.append("User Prompts (per turn):")
    turn = 0
    for msg in item["interaction_history"]:
        if msg["role"] == "user":
            turn += 1
            content = msg.get("content", "")
            lines.append(f"  Turn {turn}: {content[:300]}")

    return "\n".join(lines)


def build_tree_context(items: List[Dict[str, Any]]) -> str:
    """Build a minimal safety tree context from the set of items being processed.

    The ToolShield library normally reads a full safety_tree.json here.  We synthesize a
    lightweight version from the attack plans so the trajectory summary prompt
    has useful context about what harms are being tested.
    """
    children = []
    seen = set()
    for item in items:
        env = item["generation_config"]["environment"]
        goal = item["attack_plan"]["attack_goal"]
        key = (env, goal[:80])
        if key in seen:
            continue
        seen.add(key)
        tools = [
            s.get("tool_name", "unknown")
            for s in item["attack_plan"].get("verified_tool_chain", [])
        ]
        children.append({
            "function": env,
            "capability": f"Tools: {', '.join(tools)}",
            "harm_scenarios": [{"harm": goal[:200], "test_case_desc": goal}],
        })
    tree = {"root": "STAC_eval", "children": children[:50]}  # cap for token limit
    return json.dumps(tree, indent=2)


# ---------------------------------------------------------------------------
# Core distillation (reuses prompts + LLM client from ToolShield)
# ---------------------------------------------------------------------------

def summarize_trajectory(
    item: Dict[str, Any],
    tree_context: str,
) -> str:
    """Phase 1: Generate a structured trajectory summary from an eval item."""
    task_content = format_task_content(item)
    state_data = format_interaction_as_state(item)

    user_msg = TRAJECTORY_SUMMARY_USER_TEMPLATE.format(
        tree_context=tree_context,
        task_content=task_content,
        setup_files="No pre-existing files (simulated environment)",
        state_data=truncate_trajectory(state_data),
    )

    try:
        response = _exp_mod.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TRAJECTORY_SUMMARY_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"    Error in trajectory summary: {e}")
        return ""


def learn_from_summary(
    trajectory_summary: str,
    experiences: Dict[str, str],
) -> Dict[str, Any]:
    """Phase 2: Extract a safety experience from the trajectory summary."""
    formatted_experiences = "\n".join(
        f"{k}: {v}" for k, v in sorted(experiences.items())
    ) or "None recorded"

    user_msg = EXPERIENCE_LEARNING_USER_TEMPLATE.format(
        trajectory_summary=trajectory_summary,
        current_experiences=formatted_experiences,
    )

    try:
        response = _exp_mod.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXPERIENCE_LEARNING_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
        )
        content = response.choices[0].message.content
        if not content:
            return {"error": "Empty response from API"}
        content = extract_json_from_response(content)
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"    Error parsing JSON: {e}")
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        print(f"    Error in experience learning: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_eval_results(path: Path) -> List[Dict[str, Any]]:
    """Load gen_res.json and return the list of evaluated items."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def filter_items(
    items: List[Dict[str, Any]],
    min_progress: Optional[int],
    max_progress: Optional[int],
    envs: Optional[List[str]],
    dataset: Optional[str] = None,
    min_id: Optional[int] = None,
    max_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter eval items by attack progress, environment, dataset, and/or id range."""
    filtered = items
    if min_progress is not None:
        filtered = [i for i in filtered if i.get("final_attack_progress", 0) >= min_progress]
    if max_progress is not None:
        filtered = [i for i in filtered if i.get("final_attack_progress", 0) <= max_progress]
    if envs:
        env_set = set(envs)
        filtered = [i for i in filtered if i["generation_config"]["environment"] in env_set]
    if dataset:
        filtered = [i for i in filtered
                    if i.get("generation_config", {}).get("dataset", i.get("dataset", "")) == dataset]
    if min_id is not None:
        filtered = [i for i in filtered if i.get("id", 0) >= min_id]
    if max_id is not None:
        filtered = [i for i in filtered if i.get("id", 0) <= max_id]
    return filtered


def distill(
    items: List[Dict[str, Any]],
    output_path: Path,
    resume: bool = True,
) -> Dict[str, str]:
    """Run the full two-phase distillation over a list of eval items.

    Args:
        items: Filtered list of gen_res.json entries.
        output_path: Where to write the experience JSON.
        resume: If True and output_path exists, continue from existing experiences.

    Returns:
        The final experience dictionary.
    """
    # Load or initialize experience pool
    if resume and output_path.exists():
        experiences = load_experience_list(output_path)
        print(f"Resuming from {len(experiences)} existing experiences")
    else:
        experiences = {}

    # Build shared tree context from all items
    tree_context = build_tree_context(items)

    stats = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "ERROR": 0}

    for idx, item in enumerate(items, 1):
        item_id = item.get("id", idx)
        env = item["generation_config"]["environment"]
        progress = item.get("final_attack_progress", 0)
        goal = item["attack_plan"]["attack_goal"][:80]
        print(f"\n[{idx}/{len(items)}] id={item_id}  env={env}  progress={progress}")
        print(f"  Goal: {goal}...")

        # Phase 1: Summarize trajectory
        print("  Phase 1: Summarizing trajectory...")
        summary = summarize_trajectory(item, tree_context)
        if not summary:
            print("  SKIP: empty summary")
            stats["ERROR"] += 1
            continue

        # Phase 2: Extract experience
        print("  Phase 2: Extracting experience...")
        result = learn_from_summary(summary, experiences)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            stats["ERROR"] += 1
            continue

        # Apply result
        action = result.get("action", "NONE")
        try:
            updated, metadata = apply_experience_result(experiences, result)
        except ValueError as e:
            print(f"  SKIP: {e}")
            stats["ERROR"] += 1
            continue

        if metadata.get("changed"):
            experiences = updated
            target = metadata.get("target_key", "?")
            exp_val = result.get("exp_value", "")
            print(f"  {action} {target}: {exp_val[:120]}...")
        else:
            print(f"  NONE: already covered")

        stats[action] = stats.get(action, 0) + 1

        # Save after each update (for crash resilience)
        save_experience_list(experiences, output_path)

    # Final save
    save_experience_list(experiences, output_path)

    print(f"\n{'='*60}")
    print("Distillation complete")
    print(f"{'='*60}")
    print(f"  Items processed: {len(items)}")
    print(f"  ADD: {stats['ADD']}  UPDATE: {stats['UPDATE']}  "
          f"DELETE: {stats['DELETE']}  NONE: {stats['NONE']}  ERROR: {stats['ERROR']}")
    print(f"  Final experiences: {len(experiences)}")
    print(f"  Saved to: {output_path}")
    return experiences


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def _auto_output_name(input_path: Path) -> Path:
    """Derive a descriptive output filename from the input gen_res.json path.

    Parses the restructured path format:
        data/Eval_restructured/{method}/{dataset}/{mode}/{models}/{defense}/gen_res.json

    Example:
        data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json
        -> output/toolshield-agent_safetybench-adaptive-gpt-4-1_gpt-4-1-no_defense-distilled-defense-experience.json

    Also supports legacy paths like:
        data/Eval_toolshield_asb/gpt-4.1/gpt-4.1/no_defense/gen_res.json
    """
    parts = input_path.resolve().parts

    # Try new restructured format: .../Eval_restructured/{method}/{dataset}/{mode}/{models}/{defense}/file
    for i, part in enumerate(parts):
        if part == "Eval_restructured" and i + 4 < len(parts):
            method = parts[i + 1]    # stac | toolshield
            dataset = parts[i + 2]   # shade_arena | agent_safetybench
            mode = parts[i + 3]      # adaptive | no_planner
            models = parts[i + 4]    # gpt-4.1_gpt-4.1 | gpt-4.1
            defense = parts[i + 5] if i + 5 < len(parts) and not parts[i + 5].endswith(".json") else "unknown"
            models_clean = models.replace(".", "-")
            name = f"{method}-{dataset}-{mode}-{models_clean}-{defense}-distilled-defense-experience.json"
            return Path("output") / name

    # Legacy fallback: .../Eval_<tag>/<model>/.../<defense>/gen_res.json
    dataset = "unknown"
    model = "unknown"
    defense = "unknown"

    for i, part in enumerate(parts):
        if part.startswith("Eval"):
            dataset = part.replace("Eval_", "").replace("Eval", "eval")
            if i + 1 < len(parts) and parts[i + 1] != "gen_res.json":
                model = parts[i + 1]
            break

    parent = input_path.parent.name
    if parent != "gen_res.json":
        defense = parent

    model_clean = model.replace("/", "-").replace(".", "-")
    name = f"{dataset}-{model_clean}-{defense}-distilled-defense-experience.json"
    return Path("output") / name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill defense experiences from STAC evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to gen_res.json from evaluation",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for the defense experience JSON. "
             "If omitted, auto-generated from the input path as: "
             "<dataset>-<model>-<defense>-distilled-defense-experience.json",
    )
    parser.add_argument(
        "--min-progress", type=int, default=None,
        help="Only process items with final_attack_progress >= N (e.g., 3 for successful attacks)",
    )
    parser.add_argument(
        "--max-progress", type=int, default=None,
        help="Only process items with final_attack_progress <= N (e.g., 0 for refused attacks)",
    )
    parser.add_argument(
        "--envs", nargs="+", default=None,
        help="Only process items from these environments",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["SHADE_Arena", "Agent_SafetyBench"],
        help="Only process items from this dataset (filters by generation_config.dataset)",
    )
    parser.add_argument(
        "--min-id", type=int, default=None,
        help="Only process items with id >= N (e.g., 10000 for baseline-generated attacks)",
    )
    parser.add_argument(
        "--max-id", type=int, default=None,
        help="Only process items with id < N (e.g., 10000 for STAC-generated attacks)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh even if output file already exists",
    )
    parser.add_argument(
        "--list-envs", action="store_true",
        help="List all environments in the input file and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without calling the LLM",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Auto-generate output name if not provided
    if args.output is None:
        args.output = _auto_output_name(args.input)
        print(f"Auto-generated output path: {args.output}")

    items = load_eval_results(args.input)
    print(f"Loaded {len(items)} items from {args.input}")

    if args.list_envs:
        from collections import Counter
        envs = Counter(i["generation_config"]["environment"] for i in items)
        progress = Counter(i.get("final_attack_progress", 0) for i in items)
        print(f"\nEnvironments ({len(envs)}):")
        for env, count in sorted(envs.items(), key=lambda x: -x[1]):
            print(f"  {env:<40} {count}")
        print(f"\nAttack progress distribution:")
        for score, count in sorted(progress.items()):
            print(f"  progress={score}: {count}")
        return

    filtered = filter_items(
        items, args.min_progress, args.max_progress, args.envs,
        dataset=args.dataset, min_id=args.min_id, max_id=args.max_id,
    )
    print(f"After filtering: {len(filtered)} items")

    if not filtered:
        print("No items to process after filtering.")
        sys.exit(0)

    if args.dry_run:
        print(f"\nDry run: would process {len(filtered)} items")
        for i, item in enumerate(filtered[:10], 1):
            env = item["generation_config"]["environment"]
            progress = item.get("final_attack_progress", 0)
            goal = item["attack_plan"]["attack_goal"][:80]
            print(f"  {i}. [{env}] progress={progress}  {goal}...")
        if len(filtered) > 10:
            print(f"  ... and {len(filtered) - 10} more")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    distill(filtered, args.output, resume=not args.no_resume)


if __name__ == "__main__":
    main()
