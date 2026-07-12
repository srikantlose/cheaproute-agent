"""The router. Two modes, selected by routing.mode (CHEAPROUTE_ROUTING_MODE):

local_first (default) — confidence-gated escalation:
  1. Sample k answers from the LOCAL model (1 greedy + k-1 at temperature>0).
     Local tokens are free in scoring, so k only costs latency.
  2. Score confidence = f(mean logprob, sample agreement, format check).
  3. confidence >= τ  -> return the majority local answer   (0 cost)
     confidence <  τ  -> call the REMOTE model              (tokens count)

remote_only — every answer comes from the remote model (organizer guidance:
  inference that counts must go through the Fireworks API). The task-type
  profile still shapes the prompt and max_tokens; the local model is used
  only as a last resort when the remote call fails.

In both modes any remote failure falls back to a local answer. The router
NEVER raises — a wrong answer scores better than a crashed container.
"""

from __future__ import annotations

import time
import traceback

from .confidence import (agreement_score, composite_confidence, extract_final,
                         format_score, majority_index)
from .remote_client import RemoteError
from .schema import Decision, GenResult, Task
from .tasktype import TypeProfile, classify, extract_answer, profile_for


class Router:
    def __init__(self, cfg: dict, local_client, remote_client, logger=None):
        self.cfg = cfg
        self.local = local_client
        self.remote = remote_client
        self.logger = logger

    def route(self, task: Task) -> Decision:
        t0 = time.time()
        try:
            decision = self._route_inner(task, t0)
        except Exception:  # absolute last line of defense
            decision = Decision(
                task_id=task.id, route="error", answer="unknown",
                confidence=0.0, error=traceback.format_exc(limit=3),
            )
        decision.latency_s = round(time.time() - t0, 3)
        if self.logger is not None:
            try:
                self.logger.log(decision)
            except Exception:
                pass  # logging must never be the reason route() raises
        return decision

    # -- internals ---------------------------------------------------------

    def _route_inner(self, task: Task, t0: float) -> Decision:
        ttype = classify(task.text)
        prof = profile_for(ttype)

        if self.cfg["routing"].get("mode", "local_first") == "remote_only":
            return self._escalate(task, ttype, prof, local_answer="",
                                  local_conf=0.0,
                                  signals={"task_type": ttype,
                                           "mode": "remote_only"},
                                  tokens=(0, 0), t0=t0)

        local_answer, conf, signals, tin, tout, local_error = \
            self.local_candidate(task.text, ttype, prof)

        if signals.get("local_unavailable"):
            # Local model completely unavailable: remote is the only option.
            return self._escalate(task, ttype, prof, local_answer="",
                                  local_conf=0.0,
                                  signals={"task_type": ttype,
                                           "local_error": local_error},
                                  tokens=(0, 0), t0=t0)

        if conf >= self.cfg["routing"]["threshold"]:
            if signals.get("format_ok", 0.0) <= 0.0:
                # A truncated or empty local answer (format_ok == 0) is
                # known-bad no matter how confident the token logprobs look.
                # When agreement is unavailable the logprob weight alone can
                # carry the composite past tau (seen: an answer cut off
                # mid-markdown-list shipped at conf 0.74) -- veto the gate
                # and escalate instead. Mirrored by the offline sweep in
                # eval/run_eval.py + eval/tune_threshold.py.
                signals["format_veto"] = True
            else:
                return Decision(
                    task_id=task.id, route="local", answer=local_answer,
                    confidence=round(conf, 4), signals=signals,
                    local_answer=local_answer,
                    local_tokens_in=tin, local_tokens_out=tout,
                )
        return self._escalate(task, ttype, prof, local_answer, conf, signals,
                              (tin, tout), t0=t0)

    def local_candidate(self, task_text: str, ttype: str, prof: TypeProfile
                        ) -> tuple[str, float, dict, int, int, str | None]:
        """Sample + score the local model exactly as the local_first routing
        path does. Returns (local_answer, confidence, signals, tokens_in,
        tokens_out, local_error). `signals["local_unavailable"]` is set when
        no sample could be drawn at all (local model down). Shared by
        _route_inner and eval/run_eval.py's --collect-both so offline
        threshold tuning sees the same confidence the router would compute
        in production, not an approximation of it."""
        samples, local_error = self._sample_local(task_text, prof)

        if not samples:
            return "", 0.0, {"local_unavailable": True}, 0, 0, local_error

        finals = [extract_answer(ttype, s.text, extract_final) for s in samples]
        # Free-form outputs (summaries, code) never string-match: keep the
        # greedy sample and let logprobs+format carry the confidence.
        idx = 0 if prof.freeform else majority_index(finals)
        best = samples[idx]
        local_answer = finals[idx]

        agreement = (None if prof.freeform or len(samples) == 1
                     else agreement_score(finals))
        fmt = format_score(best.text, local_answer, best.finish_reason,
                           freeform=prof.freeform)
        conf, signals = composite_confidence(
            self.cfg["routing"], best.mean_logprob, agreement, fmt)
        signals["task_type"] = ttype
        tin = sum(s.tokens_in for s in samples)
        tout = sum(s.tokens_out for s in samples)
        return local_answer, conf, signals, tin, tout, local_error

    def _sample_local(self, prompt: str,
                      prof: TypeProfile) -> tuple[list[GenResult], str | None]:
        """1 greedy + up to (k-1) sampled generations; per-sample failures are
        tolerated and sampling stops early when the time budget is spent."""
        k = max(1, min(int(prof.samples), int(self.cfg["local"]["num_samples"])))
        temp = float(self.cfg["local"]["sample_temperature"])
        budget = float(self.cfg["local"].get("sample_budget_s", 18))
        system = f"You are a careful assistant. {prof.local_style}"
        t0 = time.time()
        samples: list[GenResult] = []
        error: str | None = None
        for i in range(k):
            if i > 0 and time.time() - t0 > budget:
                # Stay inside the per-task budget even if every attempt so
                # far failed/timed out (an empty `samples` must not defeat
                # this check, or a stalled first call blows the deadline).
                break
            try:
                samples.append(self.local.generate(
                    prompt,
                    temperature=0.0 if i == 0 else temp,
                    seed=i,
                    system_prompt=system,
                    max_tokens=prof.local_max_tokens,
                ))
            except Exception as exc:
                error = str(exc)
        return samples, error

    def _escalate(self, task: Task, ttype: str, prof: TypeProfile,
                  local_answer: str, local_conf: float,
                  signals: dict, tokens: tuple[int, int], t0: float) -> Decision:
        tin, tout = tokens
        try:
            remote = self.remote.generate(
                task.text,
                system_prompt=prof.remote_style,
                max_tokens=prof.remote_max_tokens,
            )
            remote_answer = (extract_answer(ttype, remote.text, extract_final)
                             or remote.text.strip())
            if not remote_answer.strip():
                # HTTP 200 + finish_reason=stop but no visible content (e.g. a
                # reasoning model burned its budget on hidden reasoning_content)
                # is functionally a failure -- fall through to local rescue
                # rather than shipping an empty answer under route="remote".
                raise RemoteError("remote returned an empty answer")
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
            # best remaining move. In remote_only mode no local sample exists
            # yet, so take a single greedy one now. Never crash, never hang.
            if not local_answer:
                remaining = (self.cfg["routing"].get("task_deadline_s", 28)
                             - (time.time() - t0))
                local_answer, ltin, ltout = self._local_last_resort(
                    task.text, ttype, prof, remaining)
                tin, tout = tin + ltin, tout + ltout
            return Decision(
                task_id=task.id, route="remote_failed_local",
                answer=local_answer or "unknown",
                confidence=round(local_conf, 4), signals=signals,
                local_answer=local_answer,
                local_tokens_in=tin, local_tokens_out=tout,
                error=f"remote escalation failed: {exc}",
            )

    # Minimum time left worth attempting one more local generate() call for;
    # below this a rescue attempt would almost certainly be cut off anyway
    # and just burns time better spent returning "unknown" within budget.
    _MIN_RESCUE_S = 3.0

    def _local_last_resort(self, prompt: str, ttype: str, prof: TypeProfile,
                           remaining_s: float) -> tuple[str, int, int]:
        """One greedy local sample, best-effort: failure returns no answer.
        Bounded to whatever's left of the per-task deadline (not a fresh
        `local.timeout_s`) so a stalled first local sample followed by a
        remote failure can't stack two uncapped timeouts and blow past the
        judge's 30s/request limit -- see routing.task_deadline_s."""
        if remaining_s < self._MIN_RESCUE_S:
            return "", 0, 0
        try:
            r = self.local.generate(
                prompt, temperature=0.0,
                system_prompt=f"You are a careful assistant. {prof.local_style}",
                max_tokens=prof.local_max_tokens,
                timeout_s=remaining_s,
            )
            return (extract_answer(ttype, r.text, extract_final),
                    r.tokens_in, r.tokens_out)
        except Exception:
            return "", 0, 0


def build_router(cfg: dict, logger=None) -> Router:
    from .local_client import make_local_client
    from .remote_client import make_remote_client
    return Router(cfg, make_local_client(cfg), make_remote_client(cfg), logger)
