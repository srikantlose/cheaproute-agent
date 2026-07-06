"""Confidence estimation for local answers.

Three cheap signals, combined into one score in [0, 1]:

1. logprob_score  — how sure the model was token-by-token while writing the
                    answer (mean token logprob, normalized). Primary signal.
2. agreement      — self-consistency: fraction of k samples that agree on the
                    same final answer. Local samples are free in scoring.
3. format_ok      — did the model actually produce a well-formed final answer
                    (non-empty, not truncated, followed the Answer: convention)?

The composite is compared against a tunable threshold τ by the router.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final(text: str) -> str:
    """Pull the final answer out of a model response: the last 'Answer: ...'
    line if present, otherwise the last non-empty line."""
    if not text:
        return ""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_answer(ans: str) -> str:
    """Canonical form for comparing two answers."""
    s = ans.strip().lower()
    s = s.strip("\"'`*_ \t")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".!?,;:")
    return s


def _as_number(s: str) -> Optional[float]:
    m = _NUM_RE.fullmatch(s.replace(" ", ""))
    if not m:
        return None
    try:
        return float(s.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def answers_agree(a: str, b: str) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    fa, fb = _as_number(na), _as_number(nb)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))
    return False


def agreement_score(final_answers: Sequence[str]) -> float:
    """Fraction of samples in the largest agreeing cluster. 1 sample => 1.0
    (no evidence against itself); the logprob signal carries the weight then."""
    answers = [a for a in final_answers if a]
    if len(answers) <= 1:
        return 1.0 if answers else 0.0
    best = 1
    for i, a in enumerate(answers):
        size = sum(1 for b in answers if answers_agree(a, b))
        best = max(best, size)
    return best / len(answers)


def majority_index(final_answers: Sequence[str]) -> int:
    """Index of the sample whose answer has the most agreement; ties go to the
    earliest sample (the greedy, temperature-0 one)."""
    if not final_answers:
        return 0
    sizes = [
        sum(1 for b in final_answers if a and answers_agree(a, b))
        for a in final_answers
    ]
    return max(range(len(sizes)), key=lambda i: (sizes[i], -i))


def logprob_score(mean_logprob: Optional[float], floor: float, ceil: float) -> float:
    """Map mean token logprob linearly from [floor, ceil] onto [0, 1].
    Missing logprobs => neutral 0.5 (don't punish a backend that omits them)."""
    if mean_logprob is None:
        return 0.5
    if ceil <= floor:
        return 0.5
    return min(1.0, max(0.0, (mean_logprob - floor) / (ceil - floor)))


def format_score(raw_text: str, final_answer: str, finish_reason: str,
                 freeform: bool = False) -> float:
    """1.0 = well-formed and finished naturally; 0.5 = got something but
    sloppy; 0.0 = empty or truncated output. Free-form answers (summaries,
    code, NER lists) have no 'Answer:' convention — non-empty + finished is
    all we can check."""
    if not final_answer or finish_reason == "length":
        return 0.0
    if freeform:
        return 1.0
    if _ANSWER_RE.search(raw_text) and len(final_answer) <= 300:
        return 1.0
    return 0.5


def composite_confidence(routing_cfg: dict, mean_logprob: Optional[float],
                         agreement: Optional[float], fmt: float) -> tuple[float, dict]:
    """agreement=None (free-form or single-sample tasks) redistributes its
    weight onto the remaining signals instead of injecting a fake value."""
    lp = logprob_score(mean_logprob,
                       routing_cfg["logprob_floor"], routing_cfg["logprob_ceil"])
    if agreement is None:
        num = routing_cfg["w_logprob"] * lp + routing_cfg["w_format"] * fmt
        total_w = routing_cfg["w_logprob"] + routing_cfg["w_format"]
    else:
        num = (routing_cfg["w_logprob"] * lp
               + routing_cfg["w_agreement"] * agreement
               + routing_cfg["w_format"] * fmt)
        total_w = (routing_cfg["w_logprob"] + routing_cfg["w_agreement"]
                   + routing_cfg["w_format"])
    conf = num / total_w if total_w > 0 else 0.0
    return conf, {
        "logprob_score": round(lp, 4),
        "agreement": round(agreement, 4) if agreement is not None else None,
        "format_ok": fmt, "mean_logprob": mean_logprob,
    }
