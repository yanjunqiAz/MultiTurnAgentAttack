# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the `distill_defense/` module.

## What This Module Does

Distills defense experiences from STAC/Baseline evaluation trajectories, and orchestrates closed-loop defense experiments (distill defense from attack trajectories, then evaluate with the defense applied). Uses ToolShield as a library for the two-phase experience learning pipeline. Extracted from the Baseline module to stand alone.

## Commands

```bash
# Run tests (no API keys, GPU, or ToolShield install needed)
python -m pytest distill_defense/tests/test_distill_defense.py -v

# Distill defense from evaluation trajectories
python -m distill_defense.distill_defense \
    --input data/Eval_restructured/toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json

# Config-driven defense pipeline (distill + evaluate)
python -m distill_defense.pipeline_distill_and_eval_defense --config dd_asb_adaptive_distill_eval_stac
python -m distill_defense.pipeline_distill_and_eval_defense --list-configs

# Direct CLI defense pipeline
python -m distill_defense.pipeline_distill_and_eval_defense \
    --trajectory data/Eval_restructured/stac/shade_arena/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json \
    --eval-input data/STAC_benchmark_data.json \
    --model_agent gpt-4.1
```

## Module Structure

```
distill_defense/
├── distill_defense.py                   # Distill eval trajectories → defense experience JSON
├── pipeline_distill_and_eval_defense.py # Closed-loop: distill defense + evaluate with defense
├── defense_pipeline_configs.yaml        # Named configs for the defense pipeline
├── tests/
│   └── test_distill_defense.py          # 44 tests: filters, formatting, output naming, config validation
└── README.md                            # Full documentation with usage examples
```

## Architecture

### Two-Phase Distillation (`distill_defense.py`)

Uses ToolShield's experience learning pipeline as a library:
1. **Trajectory summarization** -- LLM analyzes interaction history
2. **Experience learning** -- LLM extracts safety guidelines

Output format is compatible with ToolShield's bundled experience files.

### Closed-Loop Pipeline (`pipeline_distill_and_eval_defense.py`)

Two stages, selectable per config:
1. **distill** -- Run `distill_defense.py` on a `gen_res.json`
2. **evaluate** -- Run `Baseline/eval_baseline.py` or `STAC_eval/eval_STAC_benchmark.py` with `--defense toolshield_experience`

Config: `defense_pipeline_configs.yaml`. Supports named configs with `env:` blocks for API keys, `--list-configs`, and CLI overrides.

### Key Contracts

- **ToolShield library**: Imports `Baseline.toolshield_patch` at runtime to monkey-patch ToolShield's client with LiteLLM. The patch lives in `Baseline/` since it is shared with attack generation scripts.
- **Input format**: Reads `gen_res.json` files from `data/Eval_restructured/` containing full agent interaction trajectories and judge scores.
- **Output format**: Produces JSON compatible with ToolShield's experience format. Auto-generates output filename from the input path.
- **Crash resilience**: Saves after each experience update. Supports `--no-resume` to start fresh.

## Dependencies

- **ToolShield** (`pip install git+https://github.com/CHATS-lab/ToolShield.git`) -- required for distillation (used as a library)
- **LiteLLM** (`pip install litellm`) -- required alongside ToolShield
- **Baseline module** -- `toolshield_patch.py` is imported from `Baseline/`; evaluation step calls `Baseline.eval_baseline` or `STAC_eval.eval_STAC_benchmark`

## Testing

44 tests pass without API keys, GPU, or ToolShield installed (ToolShield imports are stubbed). Coverage includes `filter_items`, `format_interaction_as_state`, `format_task_content`, `build_tree_context`, `_auto_output_name`, `load_eval_results`, YAML config validation, and env var helpers.
