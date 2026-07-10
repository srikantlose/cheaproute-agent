from cheaproute.config import load_config


def test_int_env_override_tolerates_float_string(monkeypatch):
    monkeypatch.setenv("CHEAPROUTE_REMOTE_MAX_TOKENS", "30.0")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["remote"]["max_tokens"] == 30


def test_int_env_override_plain_int_still_works(monkeypatch):
    monkeypatch.setenv("CHEAPROUTE_BATCH_WORKERS", "8")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["batch"]["workers"] == 8
