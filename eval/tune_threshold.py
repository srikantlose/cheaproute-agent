"""Threshold tuner: sweep τ over the cache produced by
`run_eval.py --collect-both` and print the full accuracy-vs-cost curve.

No model calls happen here — routing is re-simulated offline per τ, so trying
50 thresholds is free. At kickoff, once the real accuracy threshold is known,
rerun this and read the recommended τ straight off the table.

Usage:
  python eval/tune_threshold.py [--cache eval/results/tuner_cache.json]
                                [--min-accuracy 0.85] [--out eval/results]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def sweep(cache: list[dict], taus: list[float]) -> list[dict]:
    rows = []
    for tau in taus:
        correct = tokens = local_n = 0
        for c in cache:
            if c["confidence"] >= tau:
                local_n += 1
                correct += bool(c["local_correct"])
            else:
                correct += bool(c["remote_correct"])
                tokens += int(c["remote_tokens"])
        n = len(cache)
        rows.append({
            "tau": round(tau, 3),
            "accuracy": round(correct / n, 4) if n else 0.0,
            "pct_local": round(local_n / n, 4) if n else 0.0,
            "remote_tokens": tokens,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="eval/results/tuner_cache.json")
    ap.add_argument("--min-accuracy", type=float, default=0.85,
                    help="accuracy floor when recommending τ")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    with open(args.cache, encoding="utf-8") as f:
        cache = json.load(f)

    taus = [i / 50 for i in range(0, 51)]          # 0.00 .. 1.00 step 0.02
    rows = sweep(cache, taus)

    print(f"{'tau':>5} {'accuracy':>9} {'%local':>7} {'remote_tokens':>14}")
    for r in rows:
        print(f"{r['tau']:>5.2f} {r['accuracy']:>9.1%} "
              f"{r['pct_local']:>7.0%} {r['remote_tokens']:>14}")

    # Recommendation: cheapest τ that clears the accuracy floor.
    ok = [r for r in rows if r["accuracy"] >= args.min_accuracy]
    if ok:
        best = min(ok, key=lambda r: (r["remote_tokens"], -r["accuracy"]))
        print(f"\nRecommended: tau={best['tau']:.2f} -> accuracy "
              f"{best['accuracy']:.1%}, {best['pct_local']:.0%} local, "
              f"{best['remote_tokens']} remote tokens "
              f"(floor {args.min_accuracy:.0%})")
        print("Apply via config.yaml routing.threshold or "
              f"CHEAPROUTE_ROUTING_THRESHOLD={best['tau']:.2f}")
    else:
        peak = max(rows, key=lambda r: r["accuracy"])
        print(f"\nNo τ reaches {args.min_accuracy:.0%}. Peak accuracy is "
              f"{peak['accuracy']:.1%} at tau={peak['tau']:.2f} — a stronger "
              "remote model or better local prompt is needed.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / "threshold_curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "accuracy", "pct_local",
                                          "remote_tokens"])
        w.writeheader()
        w.writerows(rows)
    print(f"curve -> {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
