"""stdio adapter: one task per stdin line (JSON or plain text), one JSON
answer per stdout line. Bad lines produce an error object, never a crash."""

from __future__ import annotations

import json
import sys

from .common import decision_to_response, parse_task


def run(router) -> None:
    # Decode stdin as UTF-8 and swallow a leading BOM (PowerShell pipes add
    # one; a scoring harness might too). errors="replace" keeps garbage bytes
    # from killing the loop.
    try:
        sys.stdin.reconfigure(encoding="utf-8-sig", errors="replace")
    except (AttributeError, ValueError):
        pass  # non-reconfigurable stream (e.g. test doubles): proceed as-is
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = parse_task(line)
        if task is None:
            print(json.dumps({"error": "empty or unparseable task"}), flush=True)
            continue
        decision = router.route(task)
        print(json.dumps(decision_to_response(task, decision), ensure_ascii=False),
              flush=True)
