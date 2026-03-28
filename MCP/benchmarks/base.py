"""BenchmarkLoader ABC and registry for external benchmark integration."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BenchmarkLoader(ABC):
    """Converts an external benchmark's tasks into STAC MCP scenarios.

    Each loader handles:
    1. Loading tasks from the benchmark's native format
    2. Converting each task into a scenario dict (same schema as YAML scenarios)
    3. Declaring which MCP servers / Docker services are required
    4. Mapping the benchmark's evaluation criteria to MCP-based success_criteria
    """

    @abstractmethod
    def load_tasks(self, split: str = "harm") -> list[dict]:
        """Load tasks from the benchmark's native format.

        Args:
            split: which task split to load (e.g., "harm", "safe", "all")

        Returns:
            list of raw task dicts in the benchmark's native format
        """
        ...

    @abstractmethod
    def to_scenario(self, task: dict) -> dict:
        """Convert one benchmark task to a scenario dict.

        The returned dict must match the scenario YAML schema:
        {
            "id": str,
            "attack_goal": str,
            "explanation": str,           # optional
            "mcp_servers": list[str],
            "pre_seed": dict,             # optional
            "success_criteria": list,     # optional
            "tool_chain_hint": list[str], # optional
        }
        """
        ...

    @abstractmethod
    def get_required_servers(self) -> dict:
        """Return mcp_servers.yml entries this benchmark needs.

        Returns:
            dict of server configs to merge into the server registry.
            e.g., {"playwright": {"endpoint": "...", "adapter": "browser"}}
        """
        ...

    def get_docker_compose(self) -> str | None:
        """Return path to a docker-compose file for this benchmark's services.

        Optional — returns None if the benchmark doesn't need extra services
        beyond what's already in the base docker-compose.
        """
        return None

    def load_scenarios(self, split: str = "harm") -> list[dict]:
        """Convenience: load tasks and convert all to scenarios."""
        return [self.to_scenario(t) for t in self.load_tasks(split)]


# Registry of available benchmark loaders
BENCHMARK_REGISTRY: dict[str, type[BenchmarkLoader]] = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark loader."""
    def decorator(cls):
        BENCHMARK_REGISTRY[name] = cls
        return cls
    return decorator
