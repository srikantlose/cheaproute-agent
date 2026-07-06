"""Tests for scoring-harness compliance: model selection from ALLOWED_MODELS
and base-URL override from FIREWORKS_BASE_URL."""

from cheaproute.remote_client import (FireworksClient, make_remote_client,
                                      pick_remote_model)

PREFS = ["gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4",
         "minimax-m3", "kimi-k2p7-code"]


def test_pick_prefers_gemma():
    allowed = "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it"
    assert pick_remote_model(allowed, PREFS, "fb") == "gemma-4-31b-it"


def test_pick_exact_beats_substring_variant():
    allowed = "gemma-4-31b-it-nvfp4,gemma-4-31b-it"
    assert pick_remote_model(allowed, PREFS, "fb") == "gemma-4-31b-it"


def test_pick_matches_fully_qualified_ids():
    allowed = ("accounts/fireworks/models/minimax-m3,"
               "accounts/fireworks/models/gemma-4-31b-it")
    assert pick_remote_model(allowed, PREFS, "fb") == \
        "accounts/fireworks/models/gemma-4-31b-it"


def test_pick_falls_back_to_first_allowed_when_no_preference_matches():
    assert pick_remote_model("some-new-model,other", PREFS, "fb") == "some-new-model"


def test_pick_uses_configured_default_without_env():
    assert pick_remote_model(None, PREFS, "fb") == "fb"
    assert pick_remote_model("  ", PREFS, "fb") == "fb"


def test_make_remote_client_uses_harness_env(monkeypatch):
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://proxy.judge.example/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "minimax-m3,gemma-4-26b-a4b-it")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    cfg = {"remote": {
        "backend": "fireworks",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/gemma-4-31b-it",
        "model_preference": PREFS,
        "max_tokens": 400, "timeout_s": 25, "retries": 2, "backoff_s": 1.0,
    }}
    client = make_remote_client(cfg)
    assert isinstance(client, FireworksClient)
    assert client.base_url == "https://proxy.judge.example/v1"
    assert client.model == "gemma-4-26b-a4b-it"
