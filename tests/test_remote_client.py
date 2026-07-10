"""Tests for FireworksClient retry/fail-fast behavior against HTTP status codes."""

import requests

from cheaproute.remote_client import FireworksClient, RemoteError


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_non_retryable_4xx_fails_fast(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        return FakeResponse(404, text="model not found")

    monkeypatch.setattr(requests, "post", fake_post)
    client = FireworksClient(base_url="https://api.fireworks.ai/inference/v1",
                             model="accounts/fireworks/models/does-not-exist",
                             api_key="test-key", retries=2)
    try:
        client.generate("hello")
        assert False, "expected RemoteError"
    except RemoteError as exc:
        assert "404" in str(exc)
    assert len(calls) == 1  # no retries burned on a doomed call


def test_retryable_5xx_retries_then_succeeds(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            return FakeResponse(503)
        return FakeResponse(200, payload={
            "choices": [{"message": {"content": "Answer: 42"},
                        "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = FireworksClient(base_url="https://api.fireworks.ai/inference/v1",
                             model="some-model", api_key="test-key",
                             retries=2, backoff_s=0.01)
    result = client.generate("hello")
    assert result.text == "Answer: 42"
    assert len(calls) == 2
