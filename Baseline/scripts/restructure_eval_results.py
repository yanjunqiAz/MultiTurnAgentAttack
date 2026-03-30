#!/usr/bin/env python3
"""
Restructure evaluation results into a clean directory layout.

Eval scripts (eval_STAC_benchmark.py, eval_baseline.py) write results into flat
directories that mix multiple subsets in a single file. This script splits and
reorganizes those raw outputs into data/Eval_restructured/, where each file
contains exactly one method/dataset/mode combination — no runtime filters needed.

Run this after eval scripts produce new results to keep the restructured tree
up to date.

RAW (eval script output):
  data/Eval/{planner}/{agent}/{defense}/gen_res.json          (mixed subsets)
  data/Eval_toolshield_asb/{planner}/{agent}/{defense}/gen_res.json
  data/Eval_toolshield/{agent}/{defense}/eval_results.json    (mixed subsets)
  output/.../Eval/adaptive_planning/{planner}/{agent}/{defense}/gen_res.json

RESTRUCTURED (one file per subset):
  data/Eval_restructured/{method}/{dataset}/{mode}/{models}/{defense}/{file}.json

  data/Eval_restructured/
  ├── stac/
  │   ├── shade_arena/
  │   │   └── adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json      (390)
  │   └── agent_safetybench/
  │       └── adaptive/gpt-4.1_gpt-4.1/no_defense/
  │           ├── gen_res.json          (93, from paper benchmark)
  │           └── gen_res_full.json     (422, from full generation pipeline)
  │
  └── toolshield/
      ├── shade_arena/
      │   ├── adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json      (124)
      │   └── no_planner/gpt-4.1/no_defense/eval_results.json       (131)
      └── agent_safetybench/
          ├── adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json      (200)
          └── no_planner/gpt-4.1/no_defense/eval_results.json       (1886)

Usage:
  python -m Baseline.restructure_eval_results              # dry run (default)
  python -m Baseline.restructure_eval_results --execute    # actually copy/split files
"""

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent  # repo root (one level up from Baseline/)
NEW_BASE = ROOT / "data" / "Eval_restructured"


def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def save_json(data: list, path: Path, dry_run: bool):
    if dry_run:
        print(f"  [DRY RUN] Would write {len(data)} items -> {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {len(data)} items -> {path.relative_to(ROOT)}")


def copy_file(src: Path, dst: Path, dry_run: bool):
    if dry_run:
        print(f"  [DRY RUN] Would copy {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied -> {dst.relative_to(ROOT)}")


def migrate(dry_run: bool):
    print(f"{'DRY RUN' if dry_run else 'EXECUTING'} migration...\n")

    # ── Source 1: data/Eval/gpt-4.1/gpt-4.1/no_defense/gen_res.json ──
    # Contains 3 subsets that need splitting
    src1 = ROOT / "data/Eval/gpt-4.1/gpt-4.1/no_defense/gen_res.json"
    print(f"Source: {src1.relative_to(ROOT)}")
    data1 = load_json(src1)

    stac_shade = [x for x in data1 if x["id"] < 10000 and x["generation_config"]["dataset"] == "SHADE_Arena"]
    stac_asb   = [x for x in data1 if x["generation_config"]["dataset"] == "Agent_SafetyBench"]
    ts_shade   = [x for x in data1 if x["id"] >= 10000 and x["generation_config"]["dataset"] == "SHADE_Arena"]

    save_json(stac_shade, NEW_BASE / "stac/shade_arena/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json", dry_run)
    save_json(stac_asb,   NEW_BASE / "stac/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json", dry_run)
    save_json(ts_shade,   NEW_BASE / "toolshield/shade_arena/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json", dry_run)

    assert len(stac_shade) + len(stac_asb) + len(ts_shade) == len(data1), \
        f"Split mismatch: {len(stac_shade)}+{len(stac_asb)}+{len(ts_shade)} != {len(data1)}"
    print(f"  Split: STAC/SHADE={len(stac_shade)}, STAC/ASB={len(stac_asb)}, TS/SHADE={len(ts_shade)}\n")

    # ── Source 2: data/Eval_toolshield_asb/.../gen_res.json ──
    # Already a single subset, just copy
    src2 = ROOT / "data/Eval_toolshield_asb/gpt-4.1/gpt-4.1/no_defense/gen_res.json"
    print(f"Source: {src2.relative_to(ROOT)}")
    copy_file(src2, NEW_BASE / "toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json", dry_run)
    print(f"  Items: 200\n")

    # ── Source 3: data/Eval/gpt-4.1/no_defense/eval_results.json ──
    # Contains 2 subsets (no planner mode)
    src3 = ROOT / "data/Eval/gpt-4.1/no_defense/eval_results.json"
    print(f"Source: {src3.relative_to(ROOT)}")
    data3 = load_json(src3)

    ts_shade_np = [x for x in data3 if x["generation_config"]["dataset"] == "SHADE_Arena"]
    ts_asb_np   = [x for x in data3 if x["generation_config"]["dataset"] == "Agent_SafetyBench"]

    save_json(ts_shade_np, NEW_BASE / "toolshield/shade_arena/no_planner/gpt-4.1/no_defense/eval_results.json", dry_run)
    save_json(ts_asb_np,   NEW_BASE / "toolshield/agent_safetybench/no_planner/gpt-4.1/no_defense/eval_results.json", dry_run)

    assert len(ts_shade_np) + len(ts_asb_np) == len(data3), \
        f"Split mismatch: {len(ts_shade_np)}+{len(ts_asb_np)} != {len(data3)}"
    print(f"  Split: TS/SHADE={len(ts_shade_np)}, TS/ASB={len(ts_asb_np)}\n")

    # ── Source 4: output/.../gen_res.json (STAC/ASB full generation pipeline) ──
    src4 = ROOT / "output/Agent_SafetyBench/verification/tool_chain_attacks_gpt-4.1_gpt-4.1/Prompts/gpt-4.1/Eval/adaptive_planning/gpt-4.1/gpt-4.1/no_defense/gen_res.json"
    print(f"Source: {src4.relative_to(ROOT)}")
    copy_file(src4, NEW_BASE / "stac/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res_full.json", dry_run)
    print(f"  Items: 422\n")

    # ── Summary ──
    print("=" * 60)
    print("Migration summary:")
    print(f"  Output directory: {NEW_BASE.relative_to(ROOT)}/")
    print()
    print("  stac/shade_arena/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json          (390)")
    print("  stac/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json     (93)")
    print("  stac/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res_full.json(422)")
    print("  toolshield/shade_arena/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json     (124)")
    print("  toolshield/agent_safetybench/adaptive/gpt-4.1_gpt-4.1/no_defense/gen_res.json(200)")
    print("  toolshield/shade_arena/no_planner/gpt-4.1/no_defense/eval_results.json      (131)")
    print("  toolshield/agent_safetybench/no_planner/gpt-4.1/no_defense/eval_results.json (1886)")

    if dry_run:
        print("\nRe-run with --execute to perform the migration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate eval dirs to cleaner structure")
    parser.add_argument("--execute", action="store_true", help="Actually perform the migration (default is dry run)")
    args = parser.parse_args()
    migrate(dry_run=not args.execute)
