# Defense Distillation

Distill defense experiences from STAC/Baseline evaluation trajectories,
then evaluate with the distilled defense applied. This module closes the loop:
attacks are evaluated against an agent, and the evaluation results are distilled
back into defensive guidelines that can be imported into an agent.

## Quick Start

```bash
# Distill defense from evaluation trajectories (output name auto-generated)
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json

# Config-driven defense pipeline (distill + evaluate)
python -m distill_defense.pipeline_distill_and_eval_defense --config dd_asb_adaptive_distill_eval_stac
python -m distill_defense.pipeline_distill_and_eval_defense --list-configs
```

## How It Works

### Phase 1: Trajectory Summarization

`distill_defense.py` reads `gen_res.json` (output of evaluation) and runs
two-phase experience distillation (via the ToolShield library):

1. **Trajectory summary** -- LLM analyzes the interaction history (attack prompts,
   tool calls, agent responses, judge scores)
2. **Experience learning** -- LLM extracts safety guidelines from the summary

The output is a JSON defense experience file compatible with ToolShield's format.

### Phase 2: Closed-Loop Pipeline

`pipeline_distill_and_eval_defense.py` wraps the distill + evaluate steps into a
single command, enabling experiments like "distill defense from ASB trajectories,
then evaluate its effectiveness on SHADE attacks."

## Commands

### Distill Defense

```bash
# Auto-generated output name from input path
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json

# Explicit output path
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --output output/my-custom-name.json

# Only learn from successful attacks (where agent failed to refuse)
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --min-progress 3

# Filter by specific environments
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --envs ProductManager HomeAssistant OS

# List environments and attack progress distribution
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --list-envs

# Dry-run (preview without LLM calls)
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --dry-run
```

| Option | Description |
|--------|-------------|
| `--min-progress N` | Only process items with `final_attack_progress >= N` |
| `--max-progress N` | Only process items with `final_attack_progress <= N` |
| `--envs E1 E2 ...` | Only process items from these environments |
| `--dataset D` | Only process items from `SHADE_Arena` or `Agent_SafetyBench` (by `generation_config.dataset`) |
| `--min-id N` | Only process items with `id >= N` (e.g., `10000` for baseline-generated attacks) |
| `--max-id N` | Only process items with `id < N` (e.g., `10000` for STAC-generated attacks) |
| `--no-resume` | Start fresh (default: append to existing output) |
| `--list-envs` | Print environments and progress distribution, then exit |
| `--dry-run` | Preview what would be processed without LLM calls |

### Closed-Loop Pipeline

```bash
# List all available configs
python -m distill_defense.pipeline_distill_and_eval_defense --list-configs

# Distill from STAC-generated SHADE attacks
python -m distill_defense.pipeline_distill_and_eval_defense --config distill_stac_shade

# Distill from Baseline ASB attacks, then evaluate on STAC benchmark
python -m distill_defense.pipeline_distill_and_eval_defense --config dd_asb_adaptive_distill_eval_stac

# Run multiple configs in sequence
python -m distill_defense.pipeline_distill_and_eval_defense --config distill_stac_shade distill_dd_asb_adaptive

# Direct CLI (no config needed)
python -m distill_defense.pipeline_distill_and_eval_defense \
    --trajectory data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --eval-input data/STAC_benchmark_data.json \
    --model_agent gpt-4.1

# Evaluate-only with a pre-distilled defense file
python -m distill_defense.pipeline_distill_and_eval_defense \
    --defense-file output/toolshield-agent_safetybench-adaptive-gpt-4-1_gpt-4-1-no_defense-distilled-defense-experience.json \
    --eval-input data/STAC_benchmark_data.json \
    --model_agent gpt-4.1 \
    --steps evaluate

# Use STAC adaptive planner instead of eval_baseline
python -m distill_defense.pipeline_distill_and_eval_defense \
    --trajectory data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --eval-input data/STAC_benchmark_data.json \
    --model_agent gpt-4.1 \
    --evaluator stac
```

### Defense Pipeline Configs

Predefined configurations in `defense_pipeline_configs.yaml` cover five distinct
trajectory data sources under `data/Eval_restructured/`:

| # | Source | Eval Type | ~Items |
|---|--------|-----------|--------|
| 1 | STAC/SHADE | Adaptive | 390 |
| 2 | STAC/ASB-gen | Adaptive | 422 |
| 3 | STAC/ASB-paper | Adaptive | 93 |
| 4 | Baseline/SHADE | Adaptive | 124 |
| 5 | Baseline/ASB | Adaptive | 200 |

## Dependencies

- **ToolShield** (`pip install git+https://github.com/CHATS-lab/ToolShield.git`) -- required for distillation (used as a library)
- **LiteLLM** (`pip install litellm`) -- required alongside ToolShield
- The ToolShield monkey-patch (`Baseline/toolshield_patch.py`) is imported at runtime

The evaluate step uses only base STAC dependencies.

## File Reference

| File | Purpose |
|------|---------|
| `distill_defense.py` | Distill eval trajectories into defense experience JSON |
| `pipeline_distill_and_eval_defense.py` | Closed-loop wrapper: distill + evaluate with defense applied |
| `defense_pipeline_configs.yaml` | Named configs for the distill + evaluate pipeline |
