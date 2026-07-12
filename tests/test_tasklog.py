import json

from cheaproute.schema import Decision
from cheaproute.tasklog import DecisionLogger


def _decision(task_id: str) -> Decision:
    return Decision(task_id=task_id, route="local", answer="42", confidence=1.0)


def test_appends_within_one_run(tmp_path):
    path = tmp_path / "decisions.jsonl"
    logger = DecisionLogger(str(path))
    logger.log(_decision("a"))
    logger.log(_decision("b"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["task_id"] for l in lines] == ["a", "b"]


def test_new_run_starts_a_fresh_log(tmp_path):
    # Each container/eval run constructs its own DecisionLogger against the
    # same mounted output path; rows from earlier runs must not accumulate.
    path = tmp_path / "decisions.jsonl"
    DecisionLogger(str(path)).log(_decision("stale"))
    logger = DecisionLogger(str(path))
    logger.log(_decision("fresh"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["task_id"] for l in lines] == ["fresh"]


def test_no_path_logs_to_stderr_only(tmp_path, capsys):
    logger = DecisionLogger(None)
    logger.log(_decision("x"))  # must not raise
    assert "task=x" in capsys.readouterr().err
