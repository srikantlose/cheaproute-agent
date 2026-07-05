"""Local model client.

Production: an OpenAI-compatible /chat/completions endpoint served by vLLM (ROCm)
with logprobs enabled — logprobs are our primary confidence signal.
Development: MockLocalClient, which needs no GPU and produces synthetic logprobs,
so the router, tests, and eval harness all run on any machine.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Optional

import requests

from .schema import GenResult

SYSTEM_PROMPT = (
    "You are a careful assistant. Solve the task. Think briefly if needed, "
    "then give only the final result on the last line in the form:\n"
    "Answer: <final answer>"
)


class LocalClientError(Exception):
    pass


class OpenAICompatClient:
    """Talks to any OpenAI-compatible server (vLLM in production)."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 60,
                 max_tokens: int = 512, api_key: str = "not-needed"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.api_key = api_key

    def generate(self, prompt: str, temperature: float = 0.0,
                 seed: Optional[int] = None) -> GenResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True,
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise LocalClientError(f"local model call failed: {exc}") from exc

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LocalClientError(f"unexpected local response shape: {exc}") from exc

        mean_lp, min_lp = _logprob_stats(choice)
        usage = data.get("usage") or {}
        return GenResult(
            text=text,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )


def _logprob_stats(choice: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract mean/min token logprob from an OpenAI-format choice, tolerating
    servers that omit logprobs entirely."""
    content = (choice.get("logprobs") or {}).get("content") or []
    lps = [t["logprob"] for t in content if isinstance(t, dict) and "logprob" in t]
    if not lps:
        return None, None
    return sum(lps) / len(lps), min(lps)


def _det_rng(prompt: str, temperature: float, seed: Optional[int]) -> random.Random:
    """Deterministic RNG across processes (built-in hash() is randomized)."""
    key = f"{prompt}|{temperature:.3f}|{seed}".encode("utf-8", errors="replace")
    return random.Random(int.from_bytes(hashlib.md5(key).digest()[:8], "big"))


class MockLocalClient:
    """GPU-free stand-in. Answers simple arithmetic confidently (high logprobs,
    all samples agree); anything else gets a vague answer with low logprobs and
    seed-dependent disagreement, exercising the escalation path. Fully
    deterministic for a given (prompt, temperature, seed)."""

    _GUESSES = ["It might be A.", "Possibly B, hard to say.",
                "I believe the answer is C."]

    def __init__(self, **_ignored):
        self.model = "mock-local"

    def generate(self, prompt: str, temperature: float = 0.0,
                 seed: Optional[int] = None) -> GenResult:
        rng = _det_rng(prompt, temperature, seed)
        m = re.search(r"(-?\d+)\s*([+\-*])\s*(-?\d+)", prompt)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            val = {"+": a + b, "-": a - b, "*": a * b}[op]
            # Opt-in disagreement for tests: prompts containing "flaky" get a
            # perturbed answer on sampled (temperature>0) generations.
            if temperature > 0 and "flaky" in prompt.lower():
                val += rng.choice([-1, 1])
            return GenResult(
                text=f"Answer: {val}",
                mean_logprob=-0.12 + rng.uniform(-0.03, 0.03),
                min_logprob=-0.9,
                tokens_in=len(prompt) // 4,
                tokens_out=6,
            )
        # Unknown territory: low-confidence waffle; different seeds pick
        # different guesses, so k samples disagree and confidence drops.
        guess = self._GUESSES[(seed or 0) % len(self._GUESSES)]
        return GenResult(
            text=f"{guess}\nAnswer: {guess.split()[-1].rstrip('.,')}",
            mean_logprob=-1.9 + rng.uniform(-0.4, 0.4),
            min_logprob=-4.2,
            tokens_in=len(prompt) // 4,
            tokens_out=18,
        )


def make_local_client(cfg: dict):
    lc = cfg["local"]
    if lc.get("backend") == "mock":
        return MockLocalClient()
    return OpenAICompatClient(
        base_url=lc["base_url"], model=lc["model"],
        timeout_s=lc["timeout_s"], max_tokens=lc["max_tokens"],
    )
