import io
import json

from cheaproute.adapters.common import parse_task
from cheaproute.adapters import stdio
from cheaproute.config import load_config
from cheaproute.local_client import MockLocalClient
from cheaproute.remote_client import MockRemoteClient
from cheaproute.router import Router


def test_parse_task_json_variants():
    for key in ("task", "question", "prompt", "input", "text", "query"):
        t = parse_task({key: "What is 2+2?", "id": "x1"})
        assert t is not None and t.text == "What is 2+2?" and t.id == "x1"


def test_parse_task_plain_string_and_json_string():
    t = parse_task("just a plain question")
    assert t is not None and t.text == "just a plain question"
    t = parse_task('{"question": "embedded json?"}')
    assert t is not None and t.text == "embedded json?"


def test_parse_task_unknown_schema_serializes_object():
    t = parse_task({"weird_field": [1, 2, 3], "meta": "x"})
    assert t is not None and "weird_field" in t.text


def test_parse_task_strips_bom():
    t = parse_task('﻿{"task": "What is 2+2?", "id": "bom-1"}')
    assert t is not None and t.id == "bom-1" and t.text == "What is 2+2?"


def test_parse_task_garbage():
    assert parse_task(None) is None
    assert parse_task("") is None
    assert parse_task("   ") is None
    t = parse_task(b"\xff\xfebytes?")
    assert t is not None  # undecodable bytes still become a replaced-char task


def test_stdio_loop_survives_bad_lines(monkeypatch, capsys):
    cfg = load_config(path="__no_such_file__.yaml")
    router = Router(cfg, MockLocalClient(), MockRemoteClient())
    stdin = io.StringIO('{"id": "a", "task": "What is 2 + 3?"}\n'
                        "\n"
                        "not json at all but still a task\n")
    monkeypatch.setattr("sys.stdin", stdin)
    stdio.run(router)
    out_lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert len(out_lines) == 2  # empty line skipped
    assert out_lines[0]["id"] == "a"
    assert out_lines[0]["answer"] == "5"
    assert "answer" in out_lines[1]
