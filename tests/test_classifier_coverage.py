"""Data-driven guard: classify() must land in a sensible bucket for every
task in the practice set, and must exactly hit the categories that were
previously falling through to short_qa (extraction, language) or badly
under-detected (logic)."""

import json
from pathlib import Path

from cheaproute.tasktype import classify

PRACTICE_PATH = Path(__file__).resolve().parents[1] / "eval" / "tasks" / "practice.jsonl"

# Grading is decoupled from classification (eval/graders.py keys off the
# task's own `grader` field), so a classifier miss only costs prompt
# optimality -- these are the acceptable buckets per practice `type`, not a
# claim that classify() reproduces the practice taxonomy label-for-label.
ALLOWED = {
    "math": {"math", "logic", "short_qa"},
    "multiple_choice": {"mc"},
    "factual_qa": {"short_qa", "math"},
    "extraction": {"extraction"},
    "classification": {"classification", "sentiment"},
    "logic": {"logic", "math", "short_qa"},
    "summarization": {"summarization"},
    "code_gen": {"code_gen"},
    "code_debug": {"code_debug"},
    "language": {"language"},
    "ner": {"ner"},
}

MIN_EXACT = {"extraction": 10, "language": 5, "logic": 7}


def _load_practice():
    with open(PRACTICE_PATH, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_every_practice_task_lands_in_an_allowed_bucket():
    tasks = _load_practice()
    assert tasks, "practice.jsonl should not be empty"
    misses = []
    for t in tasks:
        got = classify(t["task"])
        allowed = ALLOWED.get(t["type"], {t["type"]})
        if got not in allowed:
            misses.append((t["id"], t["type"], got))
    assert not misses, f"classifier landed outside the allowed bucket: {misses}"


def test_minimum_exact_hits_for_previously_uncovered_categories():
    tasks = _load_practice()
    counts = {}
    for t in tasks:
        if t["type"] in MIN_EXACT:
            counts.setdefault(t["type"], 0)
            if classify(t["task"]) == t["type"]:
                counts[t["type"]] += 1
    for ptype, minimum in MIN_EXACT.items():
        assert counts.get(ptype, 0) >= minimum, (
            f"{ptype}: only {counts.get(ptype, 0)}/{minimum} exact classifier hits")
