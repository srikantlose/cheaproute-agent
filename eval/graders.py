"""Graders for practice tasks. The real scorer grades however it wants; these
exist so WE can measure accuracy before kickoff.

Grader types (per task, in practice.jsonl):
  exact    - normalized string equality with `expected`
  numeric  - parse both as numbers, small tolerance
  contains - answer contains any of `expected` (string or list), normalized
  mc       - multiple choice: answer names the right letter OR the option text
"""

from __future__ import annotations

import re

from cheaproute.confidence import answers_agree, normalize_answer

_LETTER_RE = re.compile(r"\b([a-d])\b")


def grade(task: dict, answer: str) -> bool:
    kind = task.get("grader", "exact")
    expected = task.get("expected", "")
    ans_norm = normalize_answer(answer or "")

    if kind == "numeric":
        return answers_agree(str(expected), answer or "")

    if kind == "contains":
        options = expected if isinstance(expected, list) else [expected]
        return any(normalize_answer(str(o)) in ans_norm for o in options)

    if kind == "mc":
        letter = normalize_answer(str(expected))
        text = normalize_answer(str(task.get("expected_text", "")))
        m = _LETTER_RE.search(ans_norm)
        if m and m.group(1) == letter:
            return True
        return bool(text) and text in ans_norm

    # exact
    return answers_agree(str(expected), answer or "")
