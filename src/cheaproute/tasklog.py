"""Per-task decision logging: one JSON line per task to a file, plus a short
human-readable line to stderr (stdout is reserved for answers)."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path


class DecisionLogger:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Fresh log per process: per-task writes below append (a crash
                # mid-run keeps its partial log), but without this truncation
                # every run against the same mounted /output piles its rows
                # onto the previous run's file.
                self.path.write_text("", encoding="utf-8")
            except OSError:
                self.path = None  # unwritable filesystem: keep stderr only

    def log(self, decision) -> None:
        record = dataclasses.asdict(decision)
        line = json.dumps(record, ensure_ascii=False)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        cost = record["remote_tokens_in"] + record["remote_tokens_out"]
        print(
            f"[cheaproute] task={record['task_id']} route={record['route']} "
            f"conf={record['confidence']:.3f} remote_tokens={cost} "
            f"latency={record['latency_s']}s",
            file=sys.stderr, flush=True,
        )
