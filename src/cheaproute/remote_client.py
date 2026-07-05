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
import time

import requests

from .schema import GenResult

REMOTE_SYSTEM_PROMPT = (
    "Answer the task accurately and concisely. "
    "Give the final result on the last line as:\nAnswer: <final answer>"
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

    def generate(self, prompt: str) -> GenResult:
        if not self.api_key:
            raise RemoteError("FIREWORKS_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REMOTE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
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
                return GenResult(
                    text=choice["message"]["content"] or "",
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
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

    def generate(self, prompt: str) -> GenResult:
        self.calls += 1
        if self.fail:
            raise RemoteError("mock remote configured to fail")
        return GenResult(
            text="Detailed expert response.\nAnswer: REMOTE",
            tokens_in=max(1, len(prompt) // 4),
            tokens_out=12,
        )


def make_remote_client(cfg: dict):
    rc = cfg["remote"]
    if rc.get("backend") == "mock":
        return MockRemoteClient()
    return FireworksClient(
        base_url=rc["base_url"], model=rc["model"], timeout_s=rc["timeout_s"],
        max_tokens=rc["max_tokens"], retries=rc["retries"], backoff_s=rc["backoff_s"],
    )
