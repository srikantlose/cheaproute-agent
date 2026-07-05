"""Evaluation harness: run the practice set through the router and report
accuracy, routing split, and remote-token spend.

Usage (from repo root, venv active):
  python eval/run_eval.py                          # route with current config
  python eval/run_eval.py --collect-both           # also query remote for EVERY
                                                   # task -> cache for the tuner
  python eval/run_eval.py --tasks eval/tasks/practice.jsonl --out eval/results

--collect-both costs remote tokens for every task; on the 80-task practice set
with real models that is small (~$0.05 of the $50 Fireworks credit), and it is
what lets tune_threshold.py sweep thresholds OFFLINE afterwards for free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))          # for graders
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from graders import grade  # noqa: E402

from cheaproute.confidence import (agreement_score, composite_confidence,
                                   extract_final, format_score, majority_index)
from cheaproute.config import load_config  # noqa: E402
from cheaproute.router import Router, build_router  # noqa: E402
from cheaproute.schema import Task  # noqa: E402


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="eval/tasks/practice.jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override routing threshold for this run")
    ap.add_argument("--collect-both", action="store_true",
                    help="query remote for every task and write the tuner cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg["routing"]["threshold"] = args.threshold

    router = build_router(cfg, logger=None)
    tasks = load_tasks(Path(args.tasks))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    rows, cache = [], []
    for t in tasks:
        task = Task(id=t["id"], text=t["task"], raw=t)
        t0 = time.time()

        if args.collect_both:
            row, cache_row = _collect_both(router, cfg, task, t)
            cache.append(cache_row)
        else:
            d = router.route(task)
            row = {
                "id": task.id, "type": t.get("type"), "route": d.route,
                "confidence": d.confidence, "answer": d.answer,
                "correct": grade(t, d.answer),
                "remote_tokens": d.remote_tokens_in + d.remote_tokens_out,
                "error": d.error,
            }
        row["latency_s"] = round(time.time() - t0, 3)
        rows.append(row)
        mark = "+" if row["correct"] else "-"
        print(f"  [{mark}] {task.id:<14} route={row['route']:<7} "
              f"conf={row['confidence']:.3f} rtok={row['remote_tokens']}",
              file=sys.stderr)

    # ---- summary ----------------------------------------------------------
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else 0.0
    local_n = sum(1 for r in rows if r["route"] == "local")
    rtok = sum(r["remote_tokens"] for r in rows)
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["type"] or "?", []).append(r["correct"])

    print(f"\n=== {n} tasks | accuracy {acc:.1%} | "
          f"local {local_n}/{n} ({local_n / n:.0%}) | remote tokens {rtok} ===")
    for typ, oks in sorted(by_type.items()):
        print(f"  {typ:<14} acc {sum(oks)}/{len(oks)}")

    results_path = out_dir / f"run-{stamp}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"per-task results -> {results_path}")

    if args.collect_both:
        cache_path = out_dir / "tuner_cache.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        print(f"tuner cache      -> {cache_path}  (feed to tune_threshold.py)")
    return 0


def _collect_both(router: Router, cfg: dict, task: Task, t: dict):
    """Generate the local answer + confidence AND the remote answer for one
    task, grade both, and return (display_row, cache_row)."""
    samples, _err = router._sample_local(task.text)
    if samples:
        finals = [extract_final(s.text) for s in samples]
        idx = majority_index(finals)
        best = samples[idx]
        local_answer = finals[idx]
        conf, _sig = composite_confidence(
            cfg["routing"], best.mean_logprob,
            agreement_score(finals),
            format_score(best.text, local_answer, best.finish_reason))
    else:
        local_answer, conf = "", 0.0

    try:
        remote = router.remote.generate(task.text)
        remote_answer = extract_final(remote.text) or remote.text.strip()
        remote_tokens = remote.tokens_in + remote.tokens_out
        remote_err = None
    except Exception as exc:
        remote_answer, remote_tokens, remote_err = None, 0, str(exc)

    local_ok = grade(t, local_answer)
    remote_ok = grade(t, remote_answer) if remote_answer is not None else False
    would_route = "local" if conf >= cfg["routing"]["threshold"] else "remote"
    routed_ok = local_ok if would_route == "local" else remote_ok

    row = {"id": task.id, "type": t.get("type"), "route": would_route,
           "confidence": round(conf, 4), "answer": local_answer,
           "correct": routed_ok,
           "remote_tokens": remote_tokens if would_route == "remote" else 0,
           "error": remote_err}
    cache_row = {"id": task.id, "type": t.get("type"),
                 "confidence": round(conf, 4),
                 "local_answer": local_answer, "local_correct": local_ok,
                 "remote_answer": remote_answer, "remote_correct": remote_ok,
                 "remote_tokens": remote_tokens, "remote_error": remote_err}
    return row, cache_row


if __name__ == "__main__":
    sys.exit(main())
