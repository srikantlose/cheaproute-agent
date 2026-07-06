"""Batch adapter — the Track 1 scoring interface.

Contract (Participant Guide):
  - read /input/tasks.json on startup:   [{"task_id": "...", "prompt": "..."}]
  - write /output/results.json on exit:  [{"task_id": "...", "answer": "..."}]
  - exit 0 on success, non-zero on failure
  - 10-minute hard runtime cap; results must be valid JSON or the run scores 0

Design:
  - tasks run concurrently on a thread pool (llama-server has matching
    --parallel slots; remote calls are I/O-bound anyway)
  - a global deadline watchdog guarantees results.json is written with EVERY
    task_id present even if some tasks never finished (empty answer beats an
    invalid or missing file)
  - an inference log with per-task routing decisions is written next to the
    results for transparency/debugging
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .common import parse_task


def run(router, cfg: dict, input_path: str | None = None,
        output_path: str | None = None) -> int:
    bc = cfg["batch"]
    in_path = Path(input_path or bc["input_path"])
    out_path = Path(output_path or bc["output_path"])
    deadline = time.time() + float(bc["runtime_budget_s"])

    entries = _read_tasks(in_path)
    if entries is None:
        # No readable input is a genuine failure — but still emit valid JSON.
        _write_json(out_path, [])
        return 1

    # Parse every entry up front so the output can echo every task_id even if
    # an entry is malformed or its task never runs.
    parsed = []
    for i, entry in enumerate(entries):
        task = parse_task(entry)
        tid = None
        if isinstance(entry, dict) and entry.get("task_id") is not None:
            tid = str(entry["task_id"])
        elif task is not None:
            tid = task.id
        parsed.append((tid or f"task-{i}", task))

    answers: dict[int, str] = {}
    log_rows: dict[int, dict] = {}

    def work(index: int, task) -> None:
        decision = router.route(task)
        answers[index] = decision.answer
        log_rows[index] = {
            "task_id": parsed[index][0], "route": decision.route,
            "task_type": decision.signals.get("task_type"),
            "confidence": decision.confidence,
            "remote_tokens_in": decision.remote_tokens_in,
            "remote_tokens_out": decision.remote_tokens_out,
            "latency_s": decision.latency_s, "error": decision.error,
        }

    workers = max(1, int(bc["workers"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, (_tid, task) in enumerate(parsed):
            if task is None:
                answers[i] = ""
                continue
            futures[pool.submit(work, i, task)] = i

        pending = set(futures)
        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"[batch] deadline reached with {len(pending)} tasks "
                      "unfinished — writing partial results",
                      file=sys.stderr, flush=True)
                for fut in pending:
                    fut.cancel()
                break
            done, pending = wait(pending, timeout=min(remaining, 5.0),
                                 return_when=FIRST_COMPLETED)

    results = []
    for i, (tid, _task) in enumerate(parsed):
        results.append({"task_id": tid, "answer": answers.get(i, "")})

    if not _write_json(out_path, results):
        return 1

    log_path = bc.get("inference_log_path")
    if log_path:
        _write_json(Path(log_path),
                    [log_rows.get(i, {"task_id": parsed[i][0],
                                      "route": "unfinished"})
                     for i in range(len(parsed))])

    n_done = sum(1 for i in range(len(parsed)) if i in answers and answers[i] != "")
    print(f"[batch] wrote {len(results)} results ({n_done} answered) "
          f"-> {out_path}", file=sys.stderr, flush=True)
    return 0


def _read_tasks(path: Path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[batch] cannot read tasks from {path}: {exc}",
              file=sys.stderr, flush=True)
        return None
    if isinstance(data, dict):  # tolerate {"tasks": [...]} wrapping
        data = data.get("tasks", [data])
    if not isinstance(data, list):
        return None
    return data


def _write_json(path: Path, obj) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        return True
    except OSError as exc:
        print(f"[batch] cannot write {path}: {exc}", file=sys.stderr, flush=True)
        return False
