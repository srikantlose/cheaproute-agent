"""Shared data structures passed between the adapter, router, and clients."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Task:
    """A single task to answer. `text` is the normalized prompt; `raw` keeps the
    original payload so adapters can echo back ids/fields the scorer expects."""

    id: str
    text: str
    raw: Any = None


@dataclass
class GenResult:
    """One generation from a model (local or remote)."""

    text: str
    mean_logprob: Optional[float] = None  # None when the backend gave no logprobs
    min_logprob: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str = "stop"


@dataclass
class Decision:
    """The router's full record for one task — this is what gets logged per task."""

    task_id: str
    route: str                      # "local" | "remote" | "remote_failed_local"
    answer: str
    confidence: float
    signals: dict = field(default_factory=dict)  # logprob_score, agreement, format_ok
    local_answer: str = ""
    remote_answer: Optional[str] = None
    local_tokens_in: int = 0
    local_tokens_out: int = 0
    remote_tokens_in: int = 0       # the only tokens that COST anything in scoring
    remote_tokens_out: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None


def task_id_for(text: str) -> str:
    """Stable fallback id when the incoming payload has none."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
