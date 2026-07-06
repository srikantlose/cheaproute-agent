import json

from cheaproute.adapters import batch
from cheaproute.config import load_config
from cheaproute.local_client import MockLocalClient
from cheaproute.remote_client import MockRemoteClient
from cheaproute.router import Router


def make_router(cfg):
    return Router(cfg, MockLocalClient(), MockRemoteClient())


def make_cfg(tmp_path, **batch_overrides):
    cfg = load_config(path="__no_such_file__.yaml")
    cfg["batch"]["input_path"] = str(tmp_path / "tasks.json")
    cfg["batch"]["output_path"] = str(tmp_path / "results.json")
    cfg["batch"]["inference_log_path"] = str(tmp_path / "inference_log.json")
    cfg["batch"].update(batch_overrides)
    return cfg


def write_tasks(tmp_path, tasks):
    (tmp_path / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")


def read_results(tmp_path):
    return json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))


def test_batch_happy_path(tmp_path):
    write_tasks(tmp_path, [
        {"task_id": "t1", "prompt": "What is 17 + 25?"},
        {"task_id": "t2", "prompt": "Which philosopher wrote the Zibaldone?"},
    ])
    cfg = make_cfg(tmp_path)
    rc = batch.run(make_router(cfg), cfg)
    assert rc == 0
    results = read_results(tmp_path)
    assert [r["task_id"] for r in results] == ["t1", "t2"]
    by_id = {r["task_id"]: r["answer"] for r in results}
    assert by_id["t1"] == "42"
    assert by_id["t2"]  # answered somehow (mock remote)
    assert set(results[0]) == {"task_id", "answer"}  # exact contract shape
    log = json.loads((tmp_path / "inference_log.json").read_text(encoding="utf-8"))
    assert len(log) == 2 and log[0]["task_id"] == "t1"


def test_batch_malformed_entry_still_covered(tmp_path):
    write_tasks(tmp_path, [
        {"task_id": "ok", "prompt": "What is 2 + 2?"},
        {"task_id": "empty", "prompt": "   "},
        "just a bare string task",
    ])
    cfg = make_cfg(tmp_path)
    assert batch.run(make_router(cfg), cfg) == 0
    results = read_results(tmp_path)
    assert len(results) == 3
    assert results[1]["task_id"] == "empty" and results[1]["answer"] == ""


def test_batch_missing_input_writes_empty_and_fails(tmp_path):
    cfg = make_cfg(tmp_path)
    assert batch.run(make_router(cfg), cfg) == 1
    assert read_results(tmp_path) == []


def test_batch_deadline_zero_still_valid_json(tmp_path):
    write_tasks(tmp_path, [
        {"task_id": f"t{i}", "prompt": f"What is {i} + {i}?"} for i in range(5)
    ])
    cfg = make_cfg(tmp_path, runtime_budget_s=0)
    assert batch.run(make_router(cfg), cfg) == 0
    results = read_results(tmp_path)
    assert [r["task_id"] for r in results] == [f"t{i}" for i in range(5)]
    # every task present; answers may be empty but the JSON is valid


def test_batch_tolerates_tasks_wrapper_object(tmp_path):
    (tmp_path / "tasks.json").write_text(
        json.dumps({"tasks": [{"task_id": "w1", "prompt": "What is 1 + 1?"}]}),
        encoding="utf-8")
    cfg = make_cfg(tmp_path)
    assert batch.run(make_router(cfg), cfg) == 0
    assert read_results(tmp_path)[0]["task_id"] == "w1"
