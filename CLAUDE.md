# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAC (Sequential Tool Attack Chaining) is a research framework for generating and evaluating multi-turn adversarial attacks against LLM agents in tool-use environments. It targets two safety benchmarks: **SHADE-Arena** and **Agent-SafetyBench (ASB)**.

Paper: "STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents"

## Setup

```bash
conda env create -f environment.yml
conda activate STAC
export OPENAI_API_KEY="..."        # Required for most operations
export HF_TOKEN="..."              # For HuggingFace models
export AWS_ACCESS_KEY_ID="..."     # For Bedrock models
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."     # Expires every 12h
```

Python 3.12. All scripts run as modules from the repo root (`python -m ...`).

## Common Commands

### Benchmark evaluation (most common use case)
```bash
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense no_defense --batch_size 512
```

### Full STAC generation pipeline (4 sequential steps)
```bash
python -m STAC_gen.step_1_gen_tool_chains --dataset SHADE_Arena --n_cases 120
python -m STAC_gen.step_2_verify_tool_chain --dataset SHADE_Arena --batch_size 512
python -m STAC_gen.step_3_reverse_engineer_prompts --dataset SHADE_Arena --model Qwen/Qwen3-32B  # requires GPU
python -m STAC_gen.step_4_eval_adaptive_planning --benchmark SHADE_Arena --model_agent gpt-4.1
```

### Baseline (ToolShield) pipeline
```bash
python -m Baseline.run_baseline_pipeline --config shade_gpt41          # uses configs.yaml
python -m Baseline.run_baseline_pipeline --dataset shade               # direct args
python -m Baseline.eval_baseline --input_path data/toolshield_shade_stac.json --model_agent gpt-4.1
```

## Architecture

### Core modules (`src/`)

- **`LanguageModels.py`** — Three LM backends: `OpenAILM` (OpenAI API), `BedrockLM` (AWS Bedrock for Claude/Llama/DeepSeek), `VllmLM` (local GPU via vLLM+Ray). Model routing is by string matching on model_id (`gpt` → OpenAI, `claude`/`llama`/`deepseek` → Bedrock, else → vLLM).
- **`Environments.py`** — `BaseEnvironment` ABC with two implementations: `AgentSafetyBenchEnvironment` (wraps ASB's `EnvManager`) and `SHADEArenaEnvironment` (wraps SHADE-Arena environments via `FunctionsRuntime`). Both handle model-specific tool config formatting (Bedrock vs OpenAI vs vLLM).
- **`Agents.py`** — `Agent` class wrapping an LM + list of environments. The `step()` method runs one interaction cycle: send prompt → get completion → extract tool calls → execute in environment → update messages. Three separate code paths for Bedrock, OpenAI, and vLLM models.
- **`STAC.py`** — Core STAC pipeline classes, all inheriting from `BaseLM`:
  - `Generator` — produces tool chain attacks from failure modes
  - `Verifier` — interactively validates tool chains in environments
  - `Planner` — adaptive multi-turn attack planning
  - `Judge` — evaluates attack success (scores -3 to 3)
  - `PromptWriter` — generates prompts to elicit specific tool calls
  - `AdaptivePlanningSystem` — orchestrates Planner + Agent + Judge in a loop

### `STAC_gen/step_4` vs `STAC_eval/eval_STAC_benchmark`

Both run the same AdaptivePlanningSystem (Planner -> Agent -> Judge loop) but differ in input/scope:
- **`step_4_eval_adaptive_planning.py`** — pipeline step that evaluates newly generated attacks from Step 3 output. Requires `--benchmark` flag (one dataset at a time). Data uses `final_tool_chain`, `attack_goal`, `environment` fields.
- **`eval_STAC_benchmark.py`** — standalone benchmark evaluator for the released 483-case dataset. Auto-detects benchmark type from `generation_config.dataset`. Processes both SHADE-Arena and ASB in a single run. Data uses `attack_plan.verified_tool_chain`, `generation_config.environment` fields. Supports resuming with pre-existing `planner_history`.

The core logic (evaluate function, system initialization, defense setup) is nearly identical — forked from the same code.

### Pipeline steps (`STAC_gen/`)

Steps must run sequentially; each consumes the previous step's output:
1. `step_1_gen_tool_chains.py` — Generator creates attack chains
2. `step_2_verify_tool_chain.py` — Verifier validates chains in live environments
3. `step_3_reverse_engineer_prompts.py` — PromptWriter creates natural prompts (GPU required)
4. `step_4_eval_adaptive_planning.py` — AdaptivePlanningSystem runs multi-turn evaluation

### Benchmark evaluation (`STAC_eval/`)

`eval_STAC_benchmark.py` — Evaluates models on the pre-generated 483-case benchmark (`data/STAC_benchmark_data.json`). Handles both SHADE and ASB test cases.

### Baseline (`Baseline/`)

ToolShield-based attack generation as a comparison baseline. `run_baseline_pipeline.py` orchestrates: generate → convert → evaluate. Config-driven via `Baseline/configs.yaml`. `toolshield_patch.py` monkey-patches ToolShield to use LiteLLM.

### External benchmark integrations

- **`SHADE_Arena/`** — Mock environments (banking, travel, workspace, spam_filter) with tools, environment classes, and universe data (YAML files in `universe_related_stuff/`).
- **`Agent_SafetyBench/`** — 200+ safety environments (each a `.py` + `.json` pair in `environments/`), managed by `EnvManager`.

### Prompts (`prompts/`)

System prompts for each STAC component: `generator.md`, `verifier.md`, `planner.md`, `judge.md`, `prompt_writer.md`.

### Defense mechanisms

Five options passed via `--defense`: `no_defense`, `failure_modes`, `summarization`, `reasoning`, `spotlighting`. Applied in Agent/evaluation code.

## Key Patterns

- Tool config formatting differs per model type. Check `model_id` string to determine format (Bedrock dict vs OpenAI list vs vLLM XML-wrapped string).
- Environment `step()` returns model-specific message formats — Bedrock uses `toolResult` nested in `content` list, OpenAI uses flat `tool` role messages.
- Ray is used for vLLM models; `ensure_ray_initialized()` from `src/utils.py` must be called before creating `VllmLM` actors.
- Output files go under `data/Eval/` (STAC_eval), `data/` (generation outputs), and `output/` (baseline attack artifacts).
