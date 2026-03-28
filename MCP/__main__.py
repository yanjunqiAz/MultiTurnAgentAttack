"""Entry point for ``python -m MCP``.

Delegates to ``MCP.run_eval.main()`` which handles the vllm stub
and sys.path fixups before importing the real eval pipeline.
"""

from MCP.run_eval import main

if __name__ == "__main__":
    main()
