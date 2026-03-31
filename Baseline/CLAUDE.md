# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the `Baseline/` module.

## What This Module Does

Baseline implements ToolShield-based attack generation as a comparison baseline for STAC. It generates attacks using ToolShield's tree generation, converts them to STAC format, and evaluates them against LLM agents. Defense distillation has been moved to the `distill_defense/` module.

## Commands

```bash
# Run tests (no API keys or GPU needed)
python -m pytest Baseline/tests/test_baseline.py -v

# Config-driven attack pipeline (generate + convert + evaluate)
python -m Baseline.run_baseline_pipeline --config shade_gpt41
python -m Baseline.run_baseline_pipeline --list-configs

# Direct CLI attack pipeline
python -m Baseline.run_baseline_pipeline --dataset shade
python -m Baseline.run_baseline_pipeline --dataset asb
python -m Baseline.run_baseline_pipeline --steps evaluate --input_path data/STAC_benchmark_data.json

# Evaluate without planner (no adaptive replanning)
python -m Baseline.eval_baseline --input_path data/toolshield_shade_stac.json --model_agent gpt-4.1

# Defense distillation (see distill_defense/ module)
python -m distill_defense.distill_defense --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json
python -m distill_defense.pipeline_distill_and_eval_defense --config dd_asb_adaptive_distill_eval_stac
```

## Module Structure

```
Baseline/
├── attack_gen/                          # Attack generation
│   ├── generate_shade_attacks.py        #   ToolShield attacks for SHADE-Arena
│   ├── attack_safetybench.py            #   ToolShield attacks for Agent_SafetyBench
│   └── shade_tool_extractor.py          #   Extract tool schemas from SHADE-Arena environments
├── tests/
│   └── test_baseline.py                 #   30+ tests: metrics, tool resolution, config, YAML
├── scripts/
│   └── restructure_eval_results.py      #   Reorganize eval outputs into Eval_restructured/
├── run_baseline_pipeline.py             #   One-command wrapper: generate → convert → evaluate
├── toolshield_attack_configs.yaml       #   Named configs for attack pipeline
├── eval_baseline.py                     #   Evaluate attacks without planner (agent + judge)
├── convert_to_stac.py                   #   Convert ToolShield output → STAC benchmark JSON
├── convert_step4_to_benchmark.py        #   Convert Step 4 output to benchmark format
├── toolshield_patch.py                  #   Monkey-patches ToolShield to use LiteLLM at runtime
└── README.md                            #   Full documentation with usage examples
```

## Architecture

### Attack Pipeline

`run_baseline_pipeline.py`: generate → convert → evaluate. Config: `toolshield_attack_configs.yaml`.

The **defense pipeline** (distill + evaluate) has moved to `distill_defense/`.

### Key Data Flow

```
ToolShield attack generation (attack_gen/)
    → convert_to_stac.py (ToolShield output → STAC JSON)
    → eval_baseline.py (no planner) OR STAC_eval/eval_STAC_benchmark.py (adaptive planner)
    → distill_defense/ module (gen_res.json → ToolShield defense experience JSON)
    → toolshield import --exp-file ... (inject into agent)
```

### Two Evaluation Paths

- **`eval_baseline.py`** — No adaptive planner. Sends pre-generated prompts directly to the agent. Simpler and faster.
- **`STAC_eval/eval_STAC_benchmark.py`** — Full STAC adaptive planning (Planner + Agent + Judge loop). Same evaluator used for the main STAC benchmark.

### Key Contracts

- **ToolShield patch**: `toolshield_patch.py` monkey-patches `toolshield.tree_generation.client` at runtime to use LiteLLM instead of ToolShield's hardcoded OpenRouter client. Attack gen scripts import this automatically.
- **STAC format**: `convert_to_stac.py` auto-detects SHADE vs ASB from environment names and resolves tool names against benchmark schemas.
- **Config YAML**: The attack pipeline supports named configs with `env:` blocks for API keys, `--list-configs` to show available configs, and CLI overrides.

### Eval Output Structure

Results are saved under `data/Eval_restructured/`:

```
data/Eval_restructured/
  ├── stac/{shade_arena,agent_safetybench}/{adaptive,no_planner}/...
  └── toolshield/{shade_arena,agent_safetybench}/{adaptive,no_planner}/...
```

Each leaf contains `gen_res.json` (adaptive) or `eval_results.json` (no planner) with full agent interaction trajectories and judge scores.

## Dependencies

- **ToolShield** (`pip install git+https://github.com/CHATS-lab/ToolShield.git`) — required for the `generate` step only.
- **LiteLLM** (`pip install litellm`) — required alongside ToolShield for attack generation.
- Evaluate-only workflows need only the base STAC dependencies.

## Testing

30+ tests pass without API keys or GPU. Coverage includes `compute_metrics`, tool name resolution, STAC format conversion, config loading, and YAML validation.
