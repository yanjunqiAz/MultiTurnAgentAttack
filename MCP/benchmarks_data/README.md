# MCP-STAC Benchmark Data

This directory contains the benchmark datasets used by the MCP-STAC evaluation pipeline. The MCP pipeline currently supports two external benchmarks (OpenAgentSafety and SafeArena) and hand-crafted YAML scenarios. The original STAC benchmarks (SHADE-Arena and Agent-SafetyBench) use a separate in-process pipeline but can be evaluated alongside MCP benchmarks — see [SHADE-Arena & Agent-SafetyBench](#4-shade-arena--agent-safetybench-original-stac-benchmarks) for details.

## Directory Structure

```
benchmarks_data/
├── openagentsafety/          # OpenAgentSafety (OAS) benchmark (356 tasks)
│   ├── OpenAgentSafety_tasks.csv   # Task index (name, tools, risk/intent category)
│   └── tasks/
│       └── safety-<name>/
│           ├── task.md             # Agent prompt
│           ├── checkpoints.md      # Grading criteria
│           ├── scenarios.json      # (some tasks) scenario metadata
│           └── utils/
│               ├── evaluator.py    # Rule-based grading script
│               └── dependencies.yml
│               └── workspace/      # Input files seeded into the environment
├── quick_scenarios/          # Hand-crafted YAML attack scenarios
│   ├── file_exfiltration_01.yml
│   ├── data_tampering_01.yml
│   ├── sql_injection_01.yml
│   └── browser_phishing_01.yml
└── README.md                 # This file

# Related data directories (outside benchmarks_data/)
# data/safearena/              ← SafeArena harm.json / safe.json
# data/STAC_benchmark_data.json ← 483-case SHADE-Arena + ASB benchmark
# SHADE_Arena/                  ← SHADE-Arena mock environments & universe data
# Agent_SafetyBench/            ← 200+ ASB safety environments
```

## Benchmarks

### 1. OpenAgentSafety (OAS)

**Source**: [HuggingFace — sani903/openagentsafety](https://huggingface.co/datasets/sani903/openagentsafety)

OAS is a safety benchmark for tool-using agents with 356 tasks across filesystem and browser environments. Each task has a natural-language prompt (`task.md`), evaluation checkpoints (`checkpoints.md`), and optional workspace files seeded into the environment.

**Splits**:
- `benign` (245 tasks) — user intent is benign; the agent must refuse NPC/environment manipulation
- `malicious` (107 tasks) — user intent is malicious; the agent must refuse
- `all` (356 tasks) — includes 4 unlabeled tasks with empty intent category

**MCP servers used**: `filesystem` (file system, bash, code tasks) and `playwright` (GitLab, ownCloud, Plane, RocketChat browser tasks). The mapping from OAS tool names to MCP servers is handled by `OASLoader` in `MCP/benchmarks/oas_loader.py`.

**Docker requirements**: Base MCP servers (`docker/docker-compose.yml`) + OAS services (`docker/docker-compose.oas.yml`) for GitLab, ownCloud, RocketChat, and Plane.

### 2. SafeArena (potentially, never experimented!)

**Source**: SafeArena (BrowserGym/WebArena-based web agent safety benchmark)

**Data location**: `data/safearena/` (outside this directory — downloaded separately)

SafeArena tasks involve browser interactions with sandboxed web services (shopping, forum, GitLab). The agent uses a Playwright MCP server to navigate, click, type, and take screenshots. Tasks are loaded from `harm.json` / `safe.json` files.

**MCP servers used**: `playwright` only.

**Docker requirements**: Base MCP servers + SafeArena web services (`docker/docker-compose.safearena.yml`).

### 3. Hand-Crafted Scenarios (quick_scenarios/)

Small set of manually written YAML scenarios for rapid testing and development. Each file defines a single attack scenario with pre-seeded state, success criteria, and tool chain hints.

**YAML scenario schema**:
```yaml
id: file_exfiltration_01
description: "Short description"
attack_goal: "What the attack tries to achieve"
explanation: "Why this is dangerous"
mcp_servers: [filesystem]          # which MCP servers to connect
pre_seed:                           # initial environment state
  filesystem:
    files:
      - path: "/sandbox/secret.txt"
        content: "sensitive data"
success_criteria:                   # how to verify the attack succeeded
  - server: filesystem
    tool: "search_files"
    arguments: {"path": "/sandbox/public", "pattern": "sensitive"}
    expect: "match_found"
tool_chain_hint:                    # ordered steps the attack follows
  - "Read the secret file"
  - "Write it to a public location"
```

### 4. SHADE-Arena & Agent-SafetyBench (Original STAC Benchmarks)

SHADE-Arena and Agent-SafetyBench (ASB) are the two original STAC benchmarks. They use **in-process Python mock environments** rather than external MCP servers, so they run through the main STAC pipeline (`STAC_gen/` and `STAC_eval/`) instead of the MCP pipeline.

#### SHADE-Arena (only Being used for its enviroment and tool setup)

**Data location**: `SHADE_Arena/` (repo root)

SHADE-Arena provides 4 mock tool environments that simulate realistic services:

| Environment | Tools | Example Operations |
|-------------|-------|--------------------|
| Banking | ~15 tools | `send_money_zelle`, `get_balance`, `schedule_transaction` |
| Travel | ~10 tools | `book_flight`, `search_hotels`, `cancel_reservation` |
| Workspace | ~10 tools | `get_emails`, `send_email`, `create_calendar_event` |
| Spam Filter | ~5 tools | `mark_as_spam`, `update_filter_rules` |

Tools are Python functions with type-annotated parameters, executed in-memory via `FunctionsRuntime`. State (bank accounts, inboxes, bookings) is managed as Python objects. Universe data (user profiles, account details) lives in YAML files under `SHADE_Arena/universe_related_stuff/`.

#### Agent-SafetyBench (ASB)

**Data location**: `Agent_SafetyBench/` (repo root)

ASB contains 200+ safety environments, each defined as a Python class (`.py`) + tool schema (`.json`) pair in `Agent_SafetyBench/environments/`. Environments cover diverse safety scenarios: account manipulation, content moderation, financial transactions, system administration, etc. Each environment has 3-8 specialized tools with state stored as in-memory dictionaries.

#### How to Use SHADE-Arena & ASB

These benchmarks are evaluated through the original STAC pipeline, **not** the MCP pipeline:

```bash
# Evaluate on the pre-generated 483-case benchmark (covers both SHADE-Arena and ASB)
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense no_defense --batch_size 512

# Full STAC generation pipeline for SHADE-Arena
python -m STAC_gen.step_1_gen_tool_chains --dataset SHADE_Arena --n_cases 120
python -m STAC_gen.step_2_verify_tool_chain --dataset SHADE_Arena --batch_size 512
python -m STAC_gen.step_3_reverse_engineer_prompts --dataset SHADE_Arena --model Qwen/Qwen3-32B
python -m STAC_gen.step_4_eval_adaptive_planning --benchmark SHADE_Arena --model_agent gpt-4.1

# Full STAC generation pipeline for ASB
python -m STAC_gen.step_1_gen_tool_chains --dataset Agent_SafetyBench --n_cases 120
python -m STAC_gen.step_2_verify_tool_chain --dataset Agent_SafetyBench --batch_size 512
python -m STAC_gen.step_3_reverse_engineer_prompts --dataset Agent_SafetyBench --model Qwen/Qwen3-32B
python -m STAC_gen.step_4_eval_adaptive_planning --benchmark Agent_SafetyBench --model_agent gpt-4.1
```

#### Why They Are Not in the MCP Pipeline

| Aspect | SHADE-Arena / ASB | MCP Benchmarks (OAS, SafeArena) |
|--------|-------------------|---------------------------------|
| **Tool execution** | In-process Python functions | External Docker containers via MCP protocol |
| **State management** | In-memory Python objects | Server-managed (filesystem, browser DOM, databases) |
| **Infrastructure** | No Docker or external services | Docker Compose + MCP servers |
| **Tool discovery** | Python reflection (`inspect.signature`) | MCP `tools/list` RPC |
| **Transport** | Direct Python calls | stdio / SSE / HTTP |

SHADE-Arena and ASB are lightweight mock environments designed for fast, reproducible evaluation without infrastructure overhead. The MCP pipeline targets real tool backends where actions have observable side effects (files written, pages navigated, databases modified).

#### Future: Wrapping as MCP Servers

To run SHADE-Arena or ASB through the MCP pipeline, their tools would need to be wrapped as MCP servers:

1. Create an MCP server that exposes the Python tools via the MCP `tools/list` and `tools/call` protocol
2. Package it in a Docker container (similar to the filesystem/playwright servers)
3. Write a `BenchmarkLoader` (e.g., `MCP/benchmarks/shade_loader.py`) to convert tasks to MCP scenarios
4. Add a state adapter in `MCP/core/adapters.py` for snapshot/reset support

This would enable unified evaluation across all four benchmarks through a single pipeline. See [Adding a New Benchmark](#adding-a-new-benchmark) for the loader implementation pattern.

---

### _Benchmark Summary_

| Benchmark | Pipeline | Tasks | Tool Type | Infrastructure |
|-----------|----------|-------|-----------|----------------|
| **OAS** | MCP (`run_eval.py`) | 356 | MCP servers (filesystem, playwright) | Docker |
| **SafeArena** | MCP (`run_eval.py`) | 100+ | MCP server (playwright) | Docker |
| **SHADE-Arena** | STAC (`eval_STAC_benchmark`) | ~240 | In-process Python | None |
| **ASB** | STAC (`eval_STAC_benchmark`) | ~240 | In-process Python | None |
| **Quick Scenarios** | MCP (`run_eval.py`) | 4 | MCP servers (filesystem) | Docker |

## Loading Benchmarks

All benchmarks are loaded through the `BenchmarkLoader` interface defined in `MCP/benchmarks/base.py`. Two loaders are registered out of the box:

| Name | Loader Class | Registry Key |
|------|-------------|--------------|
| OpenAgentSafety | `OASLoader` | `oas` |
| SafeArena | `SafeArenaLoader` | `safearena` |

### Python API

```python
from MCP.benchmarks import BENCHMARK_REGISTRY

# Load OAS scenarios
loader = BENCHMARK_REGISTRY["oas"]()
scenarios = loader.load_scenarios(split="all")       # all 356 tasks
scenarios = loader.load_scenarios(split="malicious")  # 107 malicious-intent tasks
scenarios = loader.load_scenarios(split="benign")     # 245 benign-intent tasks

# Load SafeArena scenarios
loader = BENCHMARK_REGISTRY["safearena"](data_dir="data/safearena")
scenarios = loader.load_scenarios(split="harm")

# Each scenario is a dict:
# {
#     "id": "oas_abusive-apology",
#     "attack_goal": "...",
#     "explanation": "...",
#     "mcp_servers": ["filesystem", "playwright"],
#     "pre_seed": {...},
#     "success_criteria": [...],
#     "tool_chain_hint": [...],
# }
```

### CLI

```bash
# OAS benchmark (all tasks)
python MCP/run_eval.py --benchmark oas --model_agent gpt-4.1

# OAS benchmark (specific split)
python MCP/run_eval.py --benchmark oas --benchmark_split malicious --model_agent gpt-4.1

# SafeArena benchmark
python MCP/run_eval.py --benchmark safearena --model_agent gpt-4.1

# Hand-crafted scenarios
python MCP/run_eval.py --scenarios MCP/benchmarks_data/quick_scenarios/ --model_agent gpt-4.1

# Config-driven (see MCP/configs.yaml for all named configs)
python MCP/run_eval.py --config oas_gpt41
python MCP/run_eval.py --config oas_gpt41_smoke   # 2-task smoke test
python MCP/run_eval.py --config safearena_gpt41

# Full pipeline (gen + eval)
python MCP/run_pipeline.py --config pipeline_gpt41
```

## Adding a New Benchmark

1. Create a loader in `MCP/benchmarks/myloader.py`:

```python
from MCP.benchmarks.base import BenchmarkLoader, register_benchmark

@register_benchmark("mybench")
class MyBenchLoader(BenchmarkLoader):
    def load_tasks(self, split="harm"):
        # Load from your data format (CSV, JSON, etc.)
        ...

    def to_scenario(self, task):
        # Convert to the scenario dict schema
        return {"id": ..., "attack_goal": ..., "mcp_servers": [...], ...}

    def get_required_servers(self):
        return {"filesystem": {...}}
```

2. Import it in `MCP/benchmarks/__init__.py`:
```python
from MCP.benchmarks import myloader  # noqa — registers "mybench"
```

3. (Optional) Add Docker services in `MCP/docker/docker-compose.mybench.yml`.

4. Run:
```bash
python MCP/run_eval.py --benchmark mybench --model_agent gpt-4.1
```
