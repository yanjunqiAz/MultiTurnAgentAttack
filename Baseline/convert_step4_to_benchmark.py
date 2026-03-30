"""Convert step 4 evaluation output to STAC benchmark format.

Converts the flat-key format produced by step_4_eval_adaptive_planning.py into
the nested format expected by eval_STAC_benchmark.py (matching STAC_benchmark_data.json).

Usage:
    python scripts/convert_step4_to_benchmark.py \
        --input <step4_gen_res.json> \
        --output <output_benchmark.json> \
        --dataset <dataset_name>

Example:
    python scripts/convert_step4_to_benchmark.py \
        --input output/Agent_SafetyBench/verification/tool_chain_attacks_gpt-4.1_gpt-4.1/Prompts/gpt-4.1/Eval/adaptive_planning/gpt-4.1/gpt-4.1/no_defense/gen_res.json \
        --output data/STAC_ASB_full_benchmark.json \
        --dataset Agent_SafetyBench

    python scripts/convert_step4_to_benchmark.py \
        --input output/SHADE_Arena/.../gen_res.json \
        --output data/STAC_SHADE_full_benchmark.json \
        --dataset SHADE_Arena
"""

import argparse
import json


def convert(item, dataset, prompt_writer="gpt-4.1"):
    tool_chain = item["final_tool_chain"]
    # Extract steps list: may be under "tool_chain" key or already a list
    if isinstance(tool_chain, dict):
        verified = tool_chain.get("tool_chain", tool_chain.get("steps", []))
        explanation = tool_chain.get("objective", "")
    elif isinstance(tool_chain, list):
        verified = tool_chain
        explanation = ""
    else:
        verified = []
        explanation = ""

    return {
        "id": item["id"],
        "generation_config": {
            "dataset": dataset,
            "environment": item["environment"],
            "failure_mode": item.get("failure_mode", ""),
            "tool_chain_generator": "gpt-4.1",
            "tool_chain_verifier": "gpt-4.1",
            "prompt_writer": prompt_writer,
            "planner": "gpt-4.1",
        },
        "attack_plan": {
            "attack_goal": item["attack_goal"],
            "verified_tool_chain": verified,
            "explanation": explanation,
        },
        "interaction_history": item.get("interaction_history", []),
        "planner_history": item.get("planner_history", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert step 4 output to benchmark format")
    parser.add_argument("--input", required=True, help="Path to step 4 gen_res.json")
    parser.add_argument("--output", required=True, help="Output path for benchmark format JSON")
    parser.add_argument("--dataset", default="Agent_SafetyBench",
                        help="Dataset name (default: Agent_SafetyBench)")
    parser.add_argument("--prompt_writer", default="gpt-4.1",
                        help="Prompt writer model used (default: gpt-4.1)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    converted = [convert(item, args.dataset, args.prompt_writer) for item in data]

    with open(args.output, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} items: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
