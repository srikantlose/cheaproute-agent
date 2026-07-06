"""Entrypoint: `python -m cheaproute [--adapter stdio|http|cli] [--task ...]`.

The adapter also comes from config.yaml / CHEAPROUTE_ADAPTER, so the container
command line never needs to change — only an env var."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .router import build_router
from .tasklog import DecisionLogger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cheaproute")
    parser.add_argument("--adapter", choices=["stdio", "http", "cli", "batch"],
                        default=None)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--task", default=None, help="task text/JSON (cli adapter)")
    parser.add_argument("--task-file", default=None, help="file with task (cli adapter)")
    parser.add_argument("--input", default=None, help="tasks.json (batch adapter)")
    parser.add_argument("--output", default=None, help="results.json (batch adapter)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    adapter = args.adapter
    if adapter is None:
        if args.task or args.task_file:
            adapter = "cli"
        elif args.input or Path(cfg["batch"]["input_path"]).is_file():
            adapter = "batch"  # scoring environment: tasks file exists
        else:
            adapter = cfg["adapter"]
    router = build_router(cfg, logger=DecisionLogger(cfg.get("log_path")))

    if adapter == "batch":
        from .adapters import batch
        return batch.run(router, cfg, input_path=args.input,
                         output_path=args.output)
    if adapter == "http":
        from .adapters import http_server
        http_server.run(router, port=int(cfg["http_port"]))
        return 0
    if adapter == "cli":
        from .adapters import cli
        return cli.run(router, task_arg=args.task, file_arg=args.task_file)
    from .adapters import stdio
    stdio.run(router)
    return 0


if __name__ == "__main__":
    sys.exit(main())
