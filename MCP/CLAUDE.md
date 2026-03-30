# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the `MCP/` module.

## What This Module Does

MCP extends STAC to evaluate multi-turn adversarial attacks against real MCP (Model Context Protocol) tool backends (filesystems, browsers, databases) running inside Docker containers. It is **fully self-contained** — zero changes to any existing file outside `MCP/`.

## Commands

```bash
# Run tests (no API keys, Docker, or GPU needed)
python -m pytest MCP/tests/ -v

# Skip vllm-dependent tests (macOS/no GPU)
python -m pytest MCP/tests/ -v --ignore=MCP/tests/test_eval_mcp.py --ignore=MCP/tests/test_state_verifier.py

# Start MCP servers (Docker containers, accessed via stdio)
cd MCP/docker && docker compose up -d

# Config-driven evaluation (recommended)
python MCP/run_eval.py --config oas_gpt41
python MCP/run_eval.py --config oas_gpt41_smoke   # 2 tasks, fast
python MCP/run_eval.py --list_configs              # show all configs

# Direct CLI evaluation
python MCP/run_eval.py --benchmark oas --model_agent gpt-4.1
python MCP/run_eval.py --scenarios MCP/benchmarks_data/quick_scenarios/ --model_agent gpt-4.1

# Unified pipeline: gen (steps 1-3) + eval
python MCP/run_pipeline.py --model gpt-4.1 --dataset oas
python MCP/run_pipeline.py --model gpt-4.1 --dataset oas --max_tasks 2 --n_cases 1  # smoke
python MCP/run_pipeline.py --model gpt-4.1 --dataset oas --gen_only     # gen only
python MCP/run_pipeline.py --model gpt-4.1 --dataset oas --eval_only    # eval existing
python MCP/run_pipeline.py --config pipeline_gpt41                       # config-driven

# Post-evaluation only (re-classify existing results)
python MCP/run_post_eval.py --model_agent gpt-4.1 --defense no_defense
python MCP/run_post_eval.py --config post_eval_gpt41
```

## Module Structure

```
MCP/
├── core/                        # MCP infrastructure
│   ├── mcp_environment.py       #   MCPEnvironment(BaseEnvironment): MCP server integration
│   ├── adapters.py              #   State adapters: seeding/reset/snapshot (DB, FS, browser)
│   └── utils.py                 #   Model-specific tool formatting (Bedrock/OpenAI/vLLM)
├── eval/                        # Evaluation & scoring
│   ├── eval_mcp.py              #   Main eval loop: CLI, config, scenario loading, STAC attack
│   ├── post_eval.py             #   3-way LLM judge: COMPLETE/REJECT/FAILED classification
│   └── state_verifier.py        #   Post-hoc LLM judge comparing pre/post state snapshots
├── benchmarks/                  # Benchmark loaders
│   ├── base.py                  #   BenchmarkLoader ABC + @register_benchmark + BENCHMARK_REGISTRY
│   ├── oas_loader.py            #   OpenAgentSafety → scenario conversion (356 tasks)
│   └── safearena_loader.py      #   SafeArena → scenario conversion
├── benchmarks_data/             # Datasets and scenarios
│   ├── openagentsafety/         #   OAS dataset (CSV + 356 task folders)
│   └── quick_scenarios/         #   Hand-crafted YAML attack scenarios
├── gen/                         # Attack generation pipeline (Steps 1-3)
│   ├── run_pipeline.py          #   Orchestrator: runs steps 1→2→3
│   ├── step_1_gen_tool_chains.py#   Generate attack chains from benchmark tasks
│   ├── step_2_verify_tool_chain.py# Verify chains against live MCP servers
│   └── step_3_gen_prompts.py    #   Reverse-engineer natural user prompts
├── tests/                       # Test suite (187 tests)
├── docker/                      # Docker Compose files for MCP servers
├── prompts/                     # LLM judge system prompts
│   ├── state_verifier.md        #   StateVerifier judge prompt
│   └── mcp_generator.md         #   MCP-specific Generator system prompt
├── run_pipeline.py              # Unified pipeline: gen (steps 1-3) + eval
├── run_eval.py                  # Entry point wrapper (vllm stub, sys.path)
├── run_post_eval.py             # Entry point for standalone post-evaluation
├── configs.yaml                 # Named configs (_global defaults: all models → gpt-4.1)
└── mcp_servers.yml              # Server registry: name → Docker stdio transport + adapter
```

## Architecture

Two input paths feed the same evaluation pipeline:
- `--scenarios MCP/benchmarks_data/quick_scenarios/` loads hand-crafted YAML files
- `--benchmark oas|safearena` uses a `BenchmarkLoader` to convert external tasks to scenario dicts

Per-scenario pipeline: seed state → snapshot pre-state → run AdaptivePlanningSystem (Planner→Agent→Judge) → snapshot post-state → check_criteria() → StateVerifier → collect results → 3-way post-eval (COMPLETE/REJECT/FAILED) → save results.

### Dual Evaluation

Each scenario is scored by two independent judges:
1. **STAC Judge** (scale -3 to 3) — scores `goal_progress`, `agent_helpfulness`, `prompt_harmlessness`. ASR = `goal_progress >= 3`.
2. **3-Way Post-Eval** (COMPLETE/REJECT/FAILED) — classifies agent behavior. Disable with `--no_post_eval`.

### Key Contracts

- **MCPEnvironment ↔ Agent**: `self.tool_config` attribute set in `_discover_tools()`. `step()` returns model-specific message format via `MCP.core.utils.format_tool_result_for_model()`.
- **Tool config format**: Bedrock → `{"tools": [...]}` dict, OpenAI → `json.dumps([...])` string, vLLM → JSON string. Model routing is by string matching on model_id.
- **Transport**: Default is **stdio** via `docker exec -i <container> <mcp-server>`. SSE and streamable-HTTP transports are still supported for custom setups.
- **Async wrapping**: MCP SDK is async; MCPEnvironment runs a background event loop thread with `asyncio.run_coroutine_threadsafe()`. Do NOT use inside an existing async loop.

### OAS Loader Gotchas

- Real CSV has parenthetical column names: `"Intent Category (user's intent)"`. Loader strips these to canonical keys via `key.split("(")[0].strip()`.
- 19 tool name variants (including typo `"file sytem"`) mapped to 2 servers: `filesystem` and `playwright`.
- Tool column uses both comma and semicolon separators.
- Splits: 245 benign, 107 malicious, 4 unlabeled (empty intent category).
- Dataset lives at `MCP/benchmarks_data/openagentsafety/` (not in `data/`).

## Docker Infrastructure

MCP servers run inside Docker containers and are accessed via `docker exec -i` over stdio transport. No HTTP endpoints or supergateway needed.

- `docker/docker-compose.yml` — Base MCP servers (long-running containers with pre-installed MCP packages):
  - Filesystem (`@modelcontextprotocol/server-filesystem`) — container `mcp-filesystem`
  - Playwright (`@playwright/mcp`) — container `mcp-playwright`
- `docker/docker-compose.oas.yml` — OAS/TheAgentCompany services (GitLab:8929, ownCloud:8092, RocketChat:3000, Plane:8091, API:2999).
- `docker/docker-compose.safearena.yml` — SafeArena web services

## Adding New Benchmarks

1. Create `benchmarks/myloader.py` with `@register_benchmark("name")` class
2. Import it in `benchmarks/__init__.py`
3. Optionally add `docker/docker-compose.mybench.yml`
4. Run: `python MCP/run_eval.py --benchmark name --model_agent gpt-4.1`

## Testing

187 tests pass without API keys or GPU (3 known failures in adapter/cleanup tests). Two test files (`test_eval_mcp.py`, `test_state_verifier.py`) require `vllm` (GPU environments only).
