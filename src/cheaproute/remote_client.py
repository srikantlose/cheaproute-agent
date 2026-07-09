"""Remote (Fireworks AI) client — the only tokens that cost anything in scoring.

Design constraints:
- Compact prompt (input tokens likely count too) and capped max_tokens.
- Bounded retries with backoff; a total failure raises RemoteError and the
  router falls back to the local answer rather than crashing or hanging.
- API key comes exclusively from the FIREWORKS_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests

from .schema import GenResult

REMOTE_SYSTEM_PROMPT = (
    "Answer the task accurately. Do not show reasoning, chain-of-thought, or "
    "restate the question. Respond with ONLY the final result, on a single "
    "line, formatted exactly as:\nAnswer: <final answer>"
)


class RemoteError(Exception):
    pass


class FireworksClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 45,
                 max_tokens: int = 400, retries: int = 2, backoff_s: float = 1.0,
                 api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.retries = retries
        self.backoff_s = backoff_s
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")

    def generate(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int | None = None) -> GenResult:
        if not self.api_key:
            raise RemoteError("FIREWORKS_API_KEY is not set")
        budget = max_tokens or self.max_tokens
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt or REMOTE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": budget,
            }
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout_s,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"retryable status {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                message = choice.get("message") or {}
                text = message.get("content") or ""
                finish_reason = choice.get("finish_reason", "stop")
                # Some hosted "reasoning" models put hidden chain-of-thought in
                # a separate reasoning_content field; a tight max_tokens can
                # exhaust the whole budget on that before any visible content
                # forms. Retry once with a much bigger budget rather than
                # repeating the same starved call and falling back to local.
                if not text and message.get("reasoning_content") and \
                        finish_reason == "length" and attempt < self.retries:
                    last_exc = RemoteError("budget exhausted by reasoning_content")
                    budget = max(budget * 4, 256)
                    continue
                return GenResult(
                    text=text,
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    finish_reason=finish_reason,
                )
            except (requests.RequestException, json.JSONDecodeError,
                    KeyError, IndexError, TypeError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
        raise RemoteError(f"remote call failed after {self.retries + 1} attempts: {last_exc}")


class MockRemoteClient:
    """Stand-in strong model for GPU/key-free development. Optionally scripted
    to fail, so tests can exercise the fallback path."""

    def __init__(self, fail: bool = False, **_ignored):
        self.model = "mock-remote"
        self.fail = fail
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int | None = None) -> GenResult:
        self.calls += 1
        if self.fail:
            raise RemoteError("mock remote configured to fail")
        return GenResult(
            text="Detailed expert response.\nAnswer: REMOTE",
            tokens_in=max(1, len(prompt) // 4),
            tokens_out=12,
        )


def pick_remote_model(allowed_csv: str | None, preference: list[str],
                      fallback: str) -> str:
    """Choose the remote model from the harness-provided ALLOWED_MODELS list
    (never hardcoded — calls to non-allowed models invalidate the submission).

    Matching per preference: exact basename first ('gemma-4-31b-it' ==
    'accounts/fireworks/models/gemma-4-31b-it'.split('/')[-1]), then substring
    — exact-first prevents 'gemma-4-31b-it' accidentally selecting
    'gemma-4-31b-it-nvfp4'. Falls back to the first allowed model, or the
    configured default when the env var is absent (dev environments).
    """
    if not allowed_csv or not allowed_csv.strip():
        return fallback
    allowed = [m.strip() for m in allowed_csv.split(",") if m.strip()]
    if not allowed:
        return fallback
    for pref in preference:
        p = pref.lower()
        for model in allowed:
            if model.lower().split("/")[-1] == p:
                return model
        for model in allowed:
            if p in model.lower():
                return model
    return allowed[0]


def make_remote_client(cfg: dict):
    rc = cfg["remote"]
    if rc.get("backend") == "mock":
        return MockRemoteClient()
    # Harness-injected environment takes priority over any configured value:
    # ALL calls must go through FIREWORKS_BASE_URL or they are not recorded.
    base_url = os.environ.get("FIREWORKS_BASE_URL") or rc["base_url"]
    model = pick_remote_model(
        os.environ.get("ALLOWED_MODELS"),
        rc.get("model_preference") or [],
        rc["model"],
    )
    print(f"[cheaproute] remote model: {model} via {base_url}",
          file=sys.stderr, flush=True)
    return FireworksClient(
        base_url=base_url, model=model, timeout_s=rc["timeout_s"],
        max_tokens=rc["max_tokens"], retries=rc["retries"], backoff_s=rc["backoff_s"],
    )
