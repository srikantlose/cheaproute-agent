"""The router: local-first, confidence-gated escalation to remote.

Flow per task:
  1. Sample k answers from the LOCAL model (1 greedy + k-1 at temperature>0).
     Local tokens are free in scoring, so k only costs latency.
  2. Score confidence = f(mean logprob, sample agreement, format check).
  3. confidence >= τ  -> return the majority local answer   (0 cost)
     confidence <  τ  -> call the REMOTE model              (tokens count)
  4. Any remote failure falls back to the best local answer. The router
     NEVER raises — a wrong answer scores better than a crashed container.
"""

from __future__ import annotations

import time
import traceback

from .confidence import (agreement_score, composite_confidence, extract_final,
                         format_score, majority_index)
from .remote_client import RemoteError
from .schema import Decision, GenResult, Task


class Router:
    def __init__(self, cfg: dict, local_client, remote_client, logger=None):
        self.cfg = cfg
        self.local = local_client
        self.remote = remote_client
        self.logger = logger

    def route(self, task: Task) -> Decision:
        t0 = time.time()
        try:
            decision = self._route_inner(task)
        except Exception:  # absolute last line of defense
            decision = Decision(
                task_id=task.id, route="error", answer="unknown",
                confidence=0.0, error=traceback.format_exc(limit=3),
            )
        decision.latency_s = round(time.time() - t0, 3)
        if self.logger is not None:
            self.logger.log(decision)
        return decision

    # -- internals ---------------------------------------------------------

    def _route_inner(self, task: Task) -> Decision:
        samples, local_error = self._sample_local(task.text)

        if not samples:
            # Local model completely unavailable: remote is the only option.
            return self._escalate(task, local_answer="", local_conf=0.0,
                                  signals={"local_error": local_error},
                                  tokens=(0, 0))

        finals = [extract_final(s.text) for s in samples]
        idx = majority_index(finals)
        best = samples[idx]
        local_answer = finals[idx]

        agreement = agreement_score(finals)
        fmt = format_score(best.text, local_answer, best.finish_reason)
        conf, signals = composite_confidence(
            self.cfg["routing"], best.mean_logprob, agreement, fmt)
        tin = sum(s.tokens_in for s in samples)
        tout = sum(s.tokens_out for s in samples)

        if conf >= self.cfg["routing"]["threshold"]:
            return Decision(
                task_id=task.id, route="local", answer=local_answer,
                confidence=round(conf, 4), signals=signals,
                local_answer=local_answer,
                local_tokens_in=tin, local_tokens_out=tout,
            )
        return self._escalate(task, local_answer, conf, signals, (tin, tout))

    def _sample_local(self, prompt: str) -> tuple[list[GenResult], str | None]:
        """1 greedy + (k-1) sampled generations; per-sample failures tolerated."""
        k = max(1, int(self.cfg["local"]["num_samples"]))
        temp = float(self.cfg["local"]["sample_temperature"])
        samples: list[GenResult] = []
        error: str | None = None
        for i in range(k):
            try:
                samples.append(self.local.generate(
                    prompt,
                    temperature=0.0 if i == 0 else temp,
                    seed=i,
                ))
            except Exception as exc:
                error = str(exc)
        return samples, error

    def _escalate(self, task: Task, local_answer: str, local_conf: float,
                  signals: dict, tokens: tuple[int, int]) -> Decision:
        tin, tout = tokens
        try:
            remote = self.remote.generate(task.text)
            remote_answer = extract_final(remote.text) or remote.text.strip()
            return Decision(
                task_id=task.id, route="remote",
                answer=remote_answer, confidence=round(local_conf, 4),
                signals=signals, local_answer=local_answer,
                remote_answer=remote_answer,
                local_tokens_in=tin, local_tokens_out=tout,
                remote_tokens_in=remote.tokens_in,
                remote_tokens_out=remote.tokens_out,
            )
        except (RemoteError, Exception) as exc:
            # Remote unavailable: the local answer (even a shaky one) is the
            # best remaining move. Never crash, never hang.
            return Decision(
                task_id=task.id, route="remote_failed_local",
                answer=local_answer or "unknown",
                confidence=round(local_conf, 4), signals=signals,
                local_answer=local_answer,
                local_tokens_in=tin, local_tokens_out=tout,
                error=f"remote escalation failed: {exc}",
            )


def build_router(cfg: dict, logger=None) -> Router:
    from .local_client import make_local_client
    from .remote_client import make_remote_client
    return Router(cfg, make_local_client(cfg), make_remote_client(cfg), logger)
