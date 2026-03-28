"""OpenAgentSafety (OAS) benchmark loader.

Loads tasks from the OpenAgentSafety dataset
(https://huggingface.co/datasets/sani903/openagentsafety).

Dataset structure::

    MCP/benchmarks_data/openagentsafety/
    ├── OpenAgentSafety_tasks.csv   # Index: Task Name, Description, Tools,
    │                               #   Risk Category, Intent Category,
    │                               #   Environment Intent, Path
    └── tasks/
        └── safety-<name>/
            ├── task.md             # Prompt given to the agent
            ├── checkpoints.md      # Expected behavior / grading criteria
            ├── evaluator.py        # Rule-based grading (optional)
            └── workspace/          # Input files for the task

Tools column in CSV maps to MCP servers:
  - "file system", "bash terminal", "code execution", "code generation" → filesystem
  - "Gitlab browsing", "Gitlab", "Owncloud browsing", "Owncloud"       → playwright
  - "Plane"                                                              → playwright
  - "ChatNPC"                                                            → playwright (RocketChat)

Usage::

    python -m MCP.eval_mcp --benchmark oas --model_agent gpt-4.1
    python -m MCP.eval_mcp --benchmark oas --benchmark_split benign --model_agent gpt-4.1
    python -m MCP.eval_mcp --benchmark oas --benchmark_data_dir /path/to/openagentsafety --model_agent gpt-4.1
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from MCP.benchmarks.base import BenchmarkLoader, register_benchmark

logger = logging.getLogger(__name__)

# Maps OAS tool strings (lowercased) to MCP server names.
# Real CSV values (inconsistent casing/naming):
#   file system, File system, filesystem, file sytem (typo),
#   bash terminal, Bash terminal, code execution, code generation,
#   Gitlab, Gitlab browser, Gitlab browsing,
#   Owncloud, Owncloud browser, Owncloud browsing, owncloud,
#   Browsing, Plane, ChatNPC
_TOOL_TO_SERVER = {
    # Filesystem / terminal / code
    "file system": "filesystem",
    "file sytem": "filesystem",       # typo in real CSV
    "filesystem": "filesystem",
    "bash terminal": "filesystem",
    "code execution": "filesystem",
    "code generation": "filesystem",
    # Gitlab
    "gitlab": "playwright",
    "gitlab browser": "playwright",
    "gitlab browsing": "playwright",
    # Owncloud
    "owncloud": "playwright",
    "owncloud browser": "playwright",
    "owncloud browsing": "playwright",
    # Browser (generic)
    "browsing": "playwright",
    # Project management
    "plane": "playwright",
    # Chat (RocketChat)
    "chatnpc": "playwright",
}


@register_benchmark("oas")
class OASLoader(BenchmarkLoader):
    """Load OpenAgentSafety tasks and convert to MCP scenarios."""

    def __init__(self, data_dir: Path = Path("MCP/benchmarks_data/openagentsafety")):
        self.data_dir = Path(data_dir)

    def load_tasks(self, split: str = "all") -> list[dict]:
        """Load tasks from the CSV index.

        Args:
            split: Filter by intent category.
                ``"all"`` — all 356 tasks
                ``"benign"`` — user intent is benign (agent must refuse NPC manipulation)
                ``"malicious"`` — user intent is malicious (agent must refuse)
                ``"harm"`` — alias for ``"all"`` (compat with other loaders)
        """
        csv_path = self.data_dir / "OpenAgentSafety_tasks.csv"
        tasks = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize column names — strip whitespace and parenthetical
                # suffixes. Real CSV has "Intent Category (user's intent)" etc.
                task = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    # Strip and remove parenthetical suffixes for canonical keys
                    key = k.strip()
                    canonical = key.split("(")[0].strip()
                    task[canonical] = v.strip() if isinstance(v, str) else (v or "")
                    # Also keep the full key for backwards compat
                    if canonical != key:
                        task[key] = task[canonical]
                tasks.append(task)

        if split in ("all", "harm"):
            return tasks
        elif split == "benign":
            return [t for t in tasks if t.get("Intent Category", "").lower() == "benign"]
        elif split == "malicious":
            return [t for t in tasks if t.get("Intent Category", "").lower() == "malicious"]
        else:
            logger.warning("Unknown split '%s', returning all tasks", split)
            return tasks

    def to_scenario(self, task: dict) -> dict:
        task_name = task.get("Task Name", "unknown")
        task_path = task.get("path", f"tasks/safety-{task_name}").strip()
        task_dir = self.data_dir / task_path

        # Read task.md (the prompt)
        task_md = self._read_file(task_dir / "task.md")
        # Read checkpoints.md (grading criteria)
        checkpoints = self._read_file(task_dir / "checkpoints.md")

        # Determine MCP servers from Tools column
        mcp_servers = self._map_tools_to_servers(task.get("Tools", ""))

        # Build pre_seed from workspace files
        pre_seed = self._build_pre_seed(task_dir, mcp_servers)

        return {
            "id": f"oas_{task_name}",
            "attack_goal": task_md or task.get("Description", ""),
            "explanation": (
                f"Risk: {task.get('Risk Category', 'unknown')}. "
                f"User intent: {task.get('Intent Category', 'unknown')}. "
                f"Environment intent: {task.get('Environment intent', 'none')}."
            ),
            "mcp_servers": mcp_servers,
            "pre_seed": pre_seed,
            "success_criteria": self._build_criteria(checkpoints, mcp_servers),
            "tool_chain_hint": [],
            "_source": {
                "benchmark": "openagentsafety",
                "task": task,
                "checkpoints": checkpoints,
            },
        }

    def _read_file(self, path: Path) -> str:
        """Read a file, returning empty string if missing."""
        try:
            return path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            return ""

    def _map_tools_to_servers(self, tools_str: str) -> list[str]:
        """Map OAS Tools column to MCP server names.

        The Tools column may use comma or semicolon as separator.
        """
        servers = set()
        # Split on both comma and semicolon
        import re
        parts = re.split(r"[,;]", tools_str)
        for tool in parts:
            tool = tool.strip().lower()
            if tool in _TOOL_TO_SERVER:
                servers.add(_TOOL_TO_SERVER[tool])
            elif tool:
                logger.debug("Unknown OAS tool '%s', mapping to filesystem", tool)
                servers.add("filesystem")
        return sorted(servers) or ["filesystem"]

    def _build_pre_seed(self, task_dir: Path, mcp_servers: list[str]) -> dict:
        """Build pre_seed from workspace/ files if filesystem is needed."""
        pre_seed = {}
        if "filesystem" not in mcp_servers:
            return pre_seed

        workspace = task_dir / "workspace"
        if not workspace.is_dir():
            return pre_seed

        files = []
        for f in sorted(workspace.rglob("*")):
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    content = f"(binary file: {f.name})"
                rel = f.relative_to(workspace)
                files.append({
                    "path": f"/workspace/{rel}",
                    "content": content,
                })

        if files:
            pre_seed["filesystem"] = {"files": files}
        return pre_seed

    def _build_criteria(self, checkpoints: str, mcp_servers: list[str]) -> list[dict]:
        """Build success criteria from checkpoints.

        OAS uses rule-based evaluator.py for precise grading, but we can't
        run arbitrary Python in MCP. Instead we rely on the StateVerifier
        (LLM judge) for outcome assessment and provide a minimal deterministic
        check: verify the filesystem is non-empty after the task.
        """
        criteria = []

        if "filesystem" in mcp_servers:
            criteria.append({
                "server": "filesystem",
                "tool": "list_directory",
                "arguments": {"path": "/workspace"},
                "expect": "non_empty",
            })

        if "playwright" in mcp_servers:
            criteria.append({
                "server": "playwright",
                "tool": "screenshot",
                "arguments": {},
                "expect": "non_empty",
            })

        # The real evaluation is done by StateVerifier comparing pre/post
        # state, with checkpoints as context in the attack_goal.
        return criteria

    def get_required_servers(self) -> dict:
        return {
            "filesystem": {
                "endpoint": "http://localhost:9090/mcp",
                "transport": "streamable_http",
                "adapter": "filesystem",
            },
            "playwright": {
                "endpoint": "http://localhost:9092/mcp",
                "transport": "streamable_http",
                "adapter": "browser",
            },
        }

    def get_docker_compose(self) -> str | None:
        return "MCP/docker/docker-compose.oas.yml"
