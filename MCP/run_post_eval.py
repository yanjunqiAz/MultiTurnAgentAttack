#!/usr/bin/env python3
"""Wrapper to run post-evaluation (3-way classification) on existing results.

Usage::

    python MCP/run_post_eval.py --model_agent gpt-4.1 --defense no_defense
    python MCP/run_post_eval.py --config post_eval_gpt41
"""

from __future__ import annotations

import os
import sys


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from MCP.eval.post_eval import main as post_eval_main
    post_eval_main()


if __name__ == "__main__":
    main()
