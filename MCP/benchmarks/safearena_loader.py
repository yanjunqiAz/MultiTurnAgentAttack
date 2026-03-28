"""SafeArena benchmark loader — converts SafeArena tasks to MCP scenarios.

SafeArena is a web agent safety benchmark built on BrowserGym/WebArena.
Agents interact with sandboxed web services (shopping, forum, gitlab, etc.)
via browser actions. The Playwright MCP server bridges BrowserGym's action
space to MCP tools (navigate, click, type, screenshot, etc.).

Prerequisites:
- SafeArena data downloaded: harm.json, safe.json
  (via huggingface_hub or manual download to data/safearena/)
- SafeArena Docker services running (see docker/docker-compose.safearena.yml)
- Playwright MCP server configured and connected to SafeArena's network
"""

from __future__ import annotations

import json
from pathlib import Path

from MCP.benchmarks.base import BenchmarkLoader, register_benchmark


@register_benchmark("safearena")
class SafeArenaLoader(BenchmarkLoader):
    """Load SafeArena tasks and convert to MCP scenarios."""

    def __init__(
        self,
        data_dir: Path = Path("data/safearena"),
        base_url: str = "http://sa-homepage:4399",
    ):
        self.data_dir = Path(data_dir)
        self.base_url = base_url
        self.service_urls = {
            "homepage": base_url,
            "shopping": "http://sa-shopping:7770",
            "shopping_admin": "http://sa-shopping-admin:7780",
            "forum": "http://sa-forum:9999",
            "gitlab": "http://sa-gitlab:8023",
        }

    def load_tasks(self, split: str = "harm") -> list[dict]:
        path = self.data_dir / f"{split}.json"
        with open(path) as f:
            return json.load(f)

    def to_scenario(self, task: dict) -> dict:
        task_id = task.get("task_id", task.get("id", "unknown"))
        target_services = self._detect_services(task)

        return {
            "id": f"safearena_{task_id}",
            "attack_goal": task["intent"],
            "explanation": task.get("reasoning", ""),
            "mcp_servers": ["playwright"],
            "pre_seed": {
                "playwright": {
                    "navigate_to": self._get_start_url(task, target_services),
                },
            },
            "success_criteria": self._convert_eval_criteria(task),
            "tool_chain_hint": self._extract_hints(task),
            "_source": {"benchmark": "safearena", "task": task},
        }

    def _detect_services(self, task: dict) -> list[str]:
        """Detect which SafeArena services a task needs from its intent."""
        if "sites" in task:
            return task["sites"]
        intent = task.get("intent", "").lower()
        services = []
        if any(w in intent for w in ["shop", "buy", "cart", "product", "order"]):
            services.append("shopping")
        if any(w in intent for w in ["post", "forum", "reddit", "comment", "thread"]):
            services.append("forum")
        if any(w in intent for w in ["gitlab", "repo", "commit", "code", "merge"]):
            services.append("gitlab")
        return services or ["homepage"]

    def _get_start_url(self, task: dict, services: list[str]) -> str:
        """Determine the starting URL for this task."""
        if "start_url" in task:
            return task["start_url"]
        primary = services[0] if services else "homepage"
        return self.service_urls.get(primary, self.base_url)

    def _convert_eval_criteria(self, task: dict) -> list[dict]:
        """Convert SafeArena's evaluation config to MCP success_criteria."""
        criteria = []

        if "eval" in task and "reference_url" in task["eval"]:
            criteria.append({
                "server": "playwright",
                "tool": "get_url",
                "arguments": {},
                "expect": f"contains:{task['eval']['reference_url']}",
            })

        if "eval" in task and "must_include" in task["eval"]:
            for text in task["eval"]["must_include"]:
                criteria.append({
                    "server": "playwright",
                    "tool": "get_page_content",
                    "arguments": {},
                    "expect": f"contains:{text}",
                })

        # Fallback: screenshot-based verification via StateVerifier
        if not criteria:
            criteria.append({
                "server": "playwright",
                "tool": "screenshot",
                "arguments": {},
                "expect": "non_empty",
            })

        return criteria

    def _extract_hints(self, task: dict) -> list[str]:
        """Extract step hints from task metadata if available."""
        if "steps" in task:
            return task["steps"]
        return []

    def get_required_servers(self) -> dict:
        return {
            "playwright": {
                "endpoint": "http://localhost:9092/mcp",
                "transport": "streamable_http",
                "adapter": "browser",
            },
        }

    def get_docker_compose(self) -> str | None:
        return "MCP/docker/docker-compose.safearena.yml"
