# Import all loader modules so @register_benchmark decorators fire.
from MCP.benchmarks.base import BenchmarkLoader, BENCHMARK_REGISTRY, register_benchmark  # noqa
from MCP.benchmarks import safearena_loader  # noqa — registers "safearena"
from MCP.benchmarks import oas_loader  # noqa — registers "oas"
