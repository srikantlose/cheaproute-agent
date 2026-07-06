"""Task parsing shared by all adapters: accept any reasonable payload shape
and never raise on garbage input."""

from __future__ import annotations

import json
from typing import Any, Optional

from ..schema import Task, task_id_for

# Field names the scorer might plausibly use for the task text / id.
_TEXT_KEYS = ("task", "question", "prompt", "input", "text", "query", "instruction")
_ID_KEYS = ("id", "task_id", "uid", "name")


def parse_task(payload: Any) -> Optional[Task]:
    """Turn an incoming payload (JSON dict, JSON string, or raw text) into a
    Task. Returns None only when there is genuinely nothing to answer."""
    if payload is None:
        return None

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")

    if isinstance(payload, str):
        # Strip BOM / zero-width characters that break json.loads (seen with
        # Windows pipes; a scoring harness could emit them too).
        stripped = payload.strip().lstrip("﻿​").strip()
        if not stripped:
            return None
        try:
            return parse_task(json.loads(stripped))
        except (json.JSONDecodeError, RecursionError):
            return Task(id=task_id_for(stripped), text=stripped, raw=payload)

    if isinstance(payload, dict):
        text = None
        saw_text_key = False
        for key in _TEXT_KEYS:
            if key in payload:
                saw_text_key = True
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
        if text is None:
            if saw_text_key:
                # A recognized task field exists but is empty — there is
                # genuinely nothing to answer.
                return None
            # Unknown schema: serialize the whole object as the task text so
            # the models at least see everything the scorer sent.
            text = json.dumps(payload, ensure_ascii=False)
        tid = None
        for key in _ID_KEYS:
            if payload.get(key) is not None:
                tid = str(payload[key])
                break
        return Task(id=tid or task_id_for(text), text=text, raw=payload)

    if isinstance(payload, (int, float, bool)):
        text = str(payload)
        return Task(id=task_id_for(text), text=text, raw=payload)

    if isinstance(payload, list):
        text = json.dumps(payload, ensure_ascii=False)
        return Task(id=task_id_for(text), text=text, raw=payload)

    return None


def decision_to_response(task: Task, decision) -> dict:
    """The JSON object every adapter returns for one task."""
    return {
        "id": task.id,
        "answer": decision.answer,
        "route": decision.route,
        "confidence": decision.confidence,
        "remote_tokens": decision.remote_tokens_in + decision.remote_tokens_out,
    }
