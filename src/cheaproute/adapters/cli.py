"""CLI adapter: answer one task passed as an argument or a file, then exit.
Useful for smoke tests and if the scorer invokes the container per-task."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import decision_to_response, parse_task


def run(router, task_arg: str | None = None, file_arg: str | None = None) -> int:
    if file_arg:
        try:
            payload = Path(file_arg).read_text(encoding="utf-8")
        except OSError as exc:
            print(json.dumps({"error": f"cannot read task file: {exc}"}))
            return 1
    else:
        payload = task_arg

    task = parse_task(payload)
    if task is None:
        print(json.dumps({"error": "no task provided"}))
        return 1
    decision = router.route(task)
    print(json.dumps(decision_to_response(task, decision), ensure_ascii=False))
    return 0
