"""Config loading: config.yaml values, overridable per-key via environment variables.

Env overrides exist so the container can be re-tuned at kickoff (threshold, model
ids, adapter mode) without rebuilding the image. Naming: CHEAPROUTE_<SECTION>_<KEY>,
e.g. CHEAPROUTE_ROUTING_THRESHOLD=0.7, CHEAPROUTE_LOCAL_MODEL=google/gemma-3-1b-it.
Secrets (FIREWORKS_API_KEY) come ONLY from the environment, never from the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "local": {
        "backend": "mock",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "local",  # llama-server serves whatever GGUF it loaded
        "max_tokens": 512,
        "timeout_s": 25,
        "num_samples": 3,
        "sample_temperature": 0.7,
        # Stop drawing extra self-consistency samples once this much wall time
        # is spent on a task ("<30s per request" rule; CPU inference is slow).
        "sample_budget_s": 18,
    },
    "remote": {
        "backend": "mock",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/gemma-4-31b-it",
        # Ranked preferences matched against the harness's ALLOWED_MODELS env
        # var at runtime (exact basename first, then substring). Gemma first
        # (accuracy + Gemma prize), quantized variant before non-Gemma.
        "model_preference": [
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b-it-nvfp4",
            "minimax-m3",
            "kimi-k2p7-code",
        ],
        "max_tokens": 400,
        "timeout_s": 25,
        "retries": 2,
        "backoff_s": 1.0,
    },
    "routing": {
        # "local_first": sample the local model, escalate below threshold.
        # "remote_only": every answer comes from the remote model; local is
        # kept solely as a fallback when the remote API is unreachable.
        "mode": "local_first",
        "threshold": 0.62,
        "w_logprob": 0.45,
        "w_agreement": 0.40,
        "w_format": 0.15,
        "logprob_floor": -2.5,
        "logprob_ceil": -0.05,
    },
    "batch": {
        "input_path": "/input/tasks.json",
        "output_path": "/output/results.json",
        "inference_log_path": "/output/inference_log.json",
        "workers": 4,
        # Leave headroom inside the harness's 10-minute hard cap.
        "runtime_budget_s": 540,
    },
    "adapter": "stdio",
    "http_port": 8080,
    "log_path": "logs/decisions.jsonl",
}


def _coerce(value: str, like: Any) -> Any:
    """Cast an env-var string to the type of the default it overrides."""
    if isinstance(like, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Merge defaults <- config.yaml <- environment variables."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}

    candidates = [Path(path)] if path else [
        Path(os.environ.get("CHEAPROUTE_CONFIG", "")),
        Path("config.yaml"),
        Path(__file__).resolve().parents[2] / "config.yaml",
    ]
    for cand in candidates:
        if cand and str(cand) and cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            for key, val in file_cfg.items():
                if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                    cfg[key].update(val)
                else:
                    cfg[key] = val
            break

    # Environment overrides: flat keys, then section keys.
    for key, default in _DEFAULTS.items():
        if isinstance(default, dict):
            for sub, subdefault in default.items():
                if isinstance(subdefault, (list, dict)):
                    continue  # structured values: config file only
                env = os.environ.get(f"CHEAPROUTE_{key.upper()}_{sub.upper()}")
                if env is not None:
                    cfg[key][sub] = _coerce(env, subdefault)
        else:
            env = os.environ.get(f"CHEAPROUTE_{key.upper()}")
            if env is not None:
                cfg[key] = _coerce(env, default)

    return cfg
