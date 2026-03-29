"""Step 1: Generate MCP tool-chain attacks from benchmark task descriptions.

Loads benchmark tasks (OAS, SafeArena, etc.), discovers MCP tool schemas
from live servers, and calls the STAC Generator to produce attack chains.

Usage::

    python -m MCP.gen.step_1_gen_tool_chains --model gpt-4.1 --dataset oas --n_cases 5
    python -m MCP.gen.step_1_gen_tool_chains --model gpt-4.1 --dataset safearena --max_tasks 10
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.STAC import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def discover_tool_schemas() -> dict[str, str]:
    """Connect to live MCP servers and return tool schemas as strings.

    Falls back to hardcoded schemas if servers are unavailable.
    """
    try:
        from MCP.core.mcp_environment import MCPEnvironment
        from MCP.core.utils import format_tools_for_model

        # Load server registry
        mcp_cfg_path = Path("MCP/mcp_servers.yml")
        if not mcp_cfg_path.exists():
            raise FileNotFoundError(mcp_cfg_path)

        import yaml
        with open(mcp_cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        servers = cfg.get("servers", {})

        schemas: dict[str, str] = {}
        for name, server_cfg in servers.items():
            try:
                env = MCPEnvironment({name: server_cfg}, "gpt-4.1", None)
                # Collect raw tool info
                tool_info = env.get_tool_info()
                schemas[name] = tool_info or "[]"
                env.close()
                logger.info("Discovered tools from '%s': %s", name, tool_info[:200] if tool_info else "empty")
            except Exception as exc:
                logger.warning("Could not connect to server '%s': %s", name, exc)

        if schemas:
            return schemas
    except Exception as exc:
        logger.warning("Live tool discovery failed: %s — using hardcoded schemas", exc)

    # Fallback: return hardcoded descriptions
    return _hardcoded_tool_schemas()


def _hardcoded_tool_schemas() -> dict[str, str]:
    """Hardcoded MCP tool schemas for when servers aren't available."""
    filesystem_tools = [
        {"name": "read_file", "description": "Read complete file contents", "parameters": {"path": "string"}},
        {"name": "read_text_file", "description": "Read text file with encoding", "parameters": {"path": "string"}},
        {"name": "read_multiple_files", "description": "Read multiple files at once", "parameters": {"paths": "array of strings"}},
        {"name": "write_file", "description": "Create or overwrite a file", "parameters": {"path": "string", "content": "string"}},
        {"name": "edit_file", "description": "Edit a file with search/replace", "parameters": {"path": "string", "edits": "array"}},
        {"name": "create_directory", "description": "Create a directory", "parameters": {"path": "string"}},
        {"name": "list_directory", "description": "List directory contents", "parameters": {"path": "string"}},
        {"name": "move_file", "description": "Move or rename a file", "parameters": {"source": "string", "destination": "string"}},
        {"name": "search_files", "description": "Search for files by pattern", "parameters": {"path": "string", "pattern": "string"}},
        {"name": "get_file_info", "description": "Get file metadata", "parameters": {"path": "string"}},
        {"name": "directory_tree", "description": "Show directory tree structure", "parameters": {"path": "string"}},
        {"name": "list_allowed_directories", "description": "List allowed directories", "parameters": {}},
    ]
    playwright_tools = [
        {"name": "browser_navigate", "description": "Navigate to a URL", "parameters": {"url": "string"}},
        {"name": "browser_click", "description": "Click on page element", "parameters": {"ref": "string"}},
        {"name": "browser_type", "description": "Type text into element", "parameters": {"ref": "string", "text": "string"}},
        {"name": "browser_fill_form", "description": "Fill form fields", "parameters": {"fields": "array"}},
        {"name": "browser_snapshot", "description": "Capture page snapshot", "parameters": {}},
        {"name": "browser_take_screenshot", "description": "Take page screenshot", "parameters": {"type": "png|jpeg"}},
        {"name": "browser_evaluate", "description": "Run JavaScript on page", "parameters": {"function": "string"}},
        {"name": "browser_run_code", "description": "Run Playwright code", "parameters": {"code": "string"}},
        {"name": "browser_press_key", "description": "Press keyboard key", "parameters": {"key": "string"}},
        {"name": "browser_file_upload", "description": "Upload files", "parameters": {"paths": "array"}},
        {"name": "browser_tabs", "description": "Manage browser tabs", "parameters": {"action": "list|new|close|select"}},
        {"name": "browser_navigate_back", "description": "Go back in history", "parameters": {}},
        {"name": "browser_select_option", "description": "Select dropdown option", "parameters": {"ref": "string", "values": "array"}},
    ]
    return {
        "filesystem": json.dumps(filesystem_tools, indent=2),
        "playwright": json.dumps(playwright_tools, indent=2),
    }


def load_benchmark_tasks(
    dataset: str = "oas",
    max_tasks: int | None = None,
    split: str = "all",
) -> list[dict]:
    """Load tasks from a registered benchmark loader.

    Args:
        dataset: Benchmark name (``"oas"`` or ``"safearena"``).
        max_tasks: Limit number of tasks returned.
        split: Dataset split (``"all"``, ``"benign"``, ``"malicious"``).
    """
    from MCP.benchmarks import BENCHMARK_REGISTRY

    if dataset not in BENCHMARK_REGISTRY:
        available = list(BENCHMARK_REGISTRY.keys()) or ["(none)"]
        raise SystemExit(
            f"Unknown dataset '{dataset}'. Available: {', '.join(available)}"
        )

    loader = BENCHMARK_REGISTRY[dataset]()
    scenarios = loader.load_scenarios(split=split)

    if max_tasks:
        scenarios = scenarios[:max_tasks]

    return scenarios


def build_generator_input(
    scenario: dict,
    tool_schemas: dict[str, str],
) -> tuple[str, str, str]:
    """Build (env_info, tool_info, task_description) for one scenario.

    Returns the strings that Generator.step() expects.
    """
    servers = scenario.get("mcp_servers", ["filesystem"])

    # Combine tool schemas for all needed servers
    all_tools = []
    for s in servers:
        if s in tool_schemas:
            all_tools.append(f"### {s} server tools\n{tool_schemas[s]}")
    tool_info = "\n\n".join(all_tools) or "No tool schemas available"

    # Environment info from scenario
    env_info = (
        f"Task: {scenario.get('id', 'unknown')}\n"
        f"Servers: {servers}\n"
        f"Description: {scenario.get('explanation', '')}"
    )

    # Task description = attack goal
    task_desc = scenario["attack_goal"]

    return env_info, tool_info, task_desc


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Generate MCP tool-chain attacks")
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--dataset", type=str, default="oas",
                        help="Benchmark dataset (oas, safearena, etc.)")
    parser.add_argument("--n_cases", type=int, default=3,
                        help="Number of attack chains per task")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Limit number of tasks to process")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "benign", "malicious", "harm"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    # Output path
    outdir = Path("data/MCP_gen") / args.model / args.dataset / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = Path(args.output) if args.output else outdir / "step1_tool_chains.json"

    # Load existing results for resume
    existing: list[dict] = []
    if outpath.exists():
        with open(outpath) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} tasks already generated")
    done_ids = {r["id"] for r in existing}

    # Discover tool schemas
    print("Discovering MCP tool schemas...")
    tool_schemas = discover_tool_schemas()
    print(f"Tool schemas for: {list(tool_schemas.keys())}")

    # Load benchmark tasks
    print(f"Loading {args.dataset} tasks (split={args.split})...")
    scenarios = load_benchmark_tasks(dataset=args.dataset, max_tasks=args.max_tasks, split=args.split)
    scenarios = [s for s in scenarios if s["id"] not in done_ids]
    print(f"Tasks to process: {len(scenarios)}")

    if not scenarios:
        print("All tasks already processed.")
        return

    # Create Generator
    generator = Generator(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=4096,
        sys_prompt_path="MCP/prompts/mcp_generator.md",
        n_cases_per_fm=args.n_cases,
    )

    # Process in batches
    all_results = list(existing)
    batch_size = min(args.batch_size, len(scenarios))

    for batch_start in range(0, len(scenarios), batch_size):
        batch = scenarios[batch_start:batch_start + batch_size]
        print(f"\n--- Batch {batch_start // batch_size + 1} "
              f"({len(batch)} tasks, {batch_start + 1}–{batch_start + len(batch)}) ---")

        env_infos, tool_infos, task_descs = [], [], []
        for s in batch:
            ei, ti, td = build_generator_input(s, tool_schemas)
            env_infos.append(ei)
            tool_infos.append(ti)
            task_descs.append(td)

        # Generator.step() expects: env_infos, tool_infos, failure_modes, env_name
        # We repurpose failure_modes as "task description" (the attack goal)
        # and env_name as "MCP environment"
        env_names = [
            f"MCP environment ({', '.join(s.get('mcp_servers', ['filesystem']))})"
            for s in batch
        ]

        try:
            outputs = generator.step(
                env_infos=env_infos,
                tool_infos=tool_infos,
                failure_modes=task_descs,
                env_name=env_names,
                batch_size=batch_size,
            )
        except Exception as exc:
            logger.error("Generator failed on batch: %s", exc)
            continue

        # Collect results
        for i, scenario in enumerate(batch):
            output = outputs[i] if outputs and i < len(outputs) else None
            if not output:
                logger.warning("No output for %s", scenario["id"])
                continue

            attacks = output.get("tool_chain_attacks", [])
            result = {
                "id": scenario["id"],
                "attack_goal": scenario["attack_goal"],
                "explanation": scenario.get("explanation", ""),
                "mcp_servers": scenario.get("mcp_servers", []),
                "generation_config": {
                    "model": args.model,
                    "n_cases": args.n_cases,
                    "temperature": args.temperature,
                },
                "tool_chain_attacks": attacks,
                "_source": scenario.get("_source", {}),
            }
            all_results.append(result)
            print(f"  {scenario['id']}: {len(attacks)} chains generated")

        # Checkpoint save
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved checkpoint: {len(all_results)} total")

    print(f"\nStep 1 complete: {len(all_results)} tasks → {outpath}")


if __name__ == "__main__":
    main()
