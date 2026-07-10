"""Grade a container's /output/results.json against the practice set.

This is the harness-shaped counterpart to run_eval.py: instead of calling
the router in-process, it grades whatever a real `docker run` actually wrote
to results.json (+ optionally inference_log.json for route/latency detail).
Use it after a full container validation run to confirm the practice-set
accuracy bar before pushing an image.

Caveat: these are cheap string/execution graders (see graders.py), a proxy
for local validation. The real evaluator is an LLM judge scoring against
task intent, which can be more forgiving (paraphrases, extra detail) or
stricter (missing part of a multi-part question) than these graders. Treat
this as a floor check, not a promise of the judge's number.

Usage:
  python eval/grade_results.py --results out/results.json \
      --tasks eval/tasks/practice.jsonl [--log out/inference_log.json] \
      [--min-accuracy 0.90]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # for graders

from graders import grade  # noqa: E402


def _load_json(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to results.json")
    ap.add_argument("--tasks", default="eval/tasks/practice.jsonl")
    ap.add_argument("--log", default=None,
                    help="optional inference_log.json for route/latency detail")
    ap.add_argument("--min-accuracy", type=float, default=0.0,
                    help="exit 1 if overall accuracy falls below this")
    ap.add_argument("--out", default=None,
                    help="where to write graded.jsonl (default: alongside --results)")
    args = ap.parse_args()

    results_path = Path(args.results)
    results = _load_json(results_path)
    if isinstance(results, dict):
        results = results.get("results", [])
    by_id = {str(r.get("task_id")): r.get("answer", "") for r in results}

    tasks = _load_jsonl(Path(args.tasks))

    log_by_id: dict[str, dict] = {}
    if args.log:
        log_rows = _load_json(Path(args.log))
        for row in log_rows:
            tid = row.get("task_id")
            if tid is not None:
                log_by_id[str(tid)] = row

    graded = []
    missing = []
    for t in tasks:
        tid = str(t["id"])
        if tid not in by_id:
            missing.append(tid)
            answer = ""
        else:
            answer = by_id[tid]
        correct = grade(t, answer)
        row = {"id": tid, "type": t.get("type"), "answer": answer,
              "correct": correct}
        log_row = log_by_id.get(tid)
        if log_row:
            row["route"] = log_row.get("route")
            row["confidence"] = log_row.get("confidence")
            row["remote_tokens"] = (log_row.get("remote_tokens_in", 0) +
                                    log_row.get("remote_tokens_out", 0))
            row["latency_s"] = log_row.get("latency_s")
        graded.append(row)

    n = len(graded)
    n_correct = sum(1 for r in graded if r["correct"])
    acc = n_correct / n if n else 0.0

    by_type: dict[str, list] = {}
    for r in graded:
        by_type.setdefault(r["type"] or "?", []).append(r["correct"])

    print(f"=== {n} tasks | accuracy {n_correct}/{n} ({acc:.1%}) ===")
    for typ in sorted(by_type):
        oks = by_type[typ]
        print(f"  {typ:<16} {sum(oks)}/{len(oks)}")

    if missing:
        print(f"WARNING: {len(missing)} task_id(s) from {args.tasks} had no "
              f"matching result: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    if log_by_id:
        routes: dict[str, int] = {}
        total_remote_tokens = 0
        max_latency = 0.0
        for r in graded:
            route = r.get("route")
            if route:
                routes[route] = routes.get(route, 0) + 1
            total_remote_tokens += r.get("remote_tokens") or 0
            lat = r.get("latency_s")
            if lat is not None:
                max_latency = max(max_latency, lat)
        route_str = ", ".join(f"{k}={v}" for k, v in sorted(routes.items()))
        print(f"routes: {route_str}")
        print(f"total remote tokens: {total_remote_tokens}")
        print(f"max latency: {max_latency:.2f}s")

    out_dir = Path(args.out) if args.out else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    graded_path = out_dir / "graded.jsonl"
    with open(graded_path, "w", encoding="utf-8") as f:
        for r in graded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"per-task grading -> {graded_path}")

    if acc < args.min_accuracy:
        print(f"FAIL: accuracy {acc:.1%} below floor {args.min_accuracy:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
