"""routing.mode=remote_only: every answer comes from the remote model; the
local model is touched only when the remote call fails."""

from cheaproute.config import load_config
from cheaproute.local_client import MockLocalClient
from cheaproute.remote_client import MockRemoteClient
from cheaproute.router import Router
from cheaproute.schema import Task


def make_cfg(**routing_overrides):
    cfg = load_config(path="__no_such_file__.yaml")  # pure defaults
    cfg["routing"].update(routing_overrides)
    return cfg


class CountingLocal(MockLocalClient):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, *a, **k):
        self.calls += 1
        return super().generate(*a, **k)


class BrokenLocal:
    def generate(self, *a, **k):
        raise RuntimeError("gpu on fire")


def test_default_mode_is_local_first():
    assert make_cfg()["routing"]["mode"] == "local_first"


def test_mode_env_override(monkeypatch):
    monkeypatch.setenv("CHEAPROUTE_ROUTING_MODE", "remote_only")
    assert make_cfg()["routing"]["mode"] == "remote_only"


def test_remote_only_never_samples_local():
    # a prompt the local mock answers confidently — local_first would keep it
    local, remote = CountingLocal(), MockRemoteClient()
    router = Router(make_cfg(mode="remote_only"), local, remote)
    d = router.route(Task(id="r1", text="What is 12 + 30?"))
    assert d.route == "remote"
    assert d.answer == "REMOTE"
    assert local.calls == 0
    assert remote.calls == 1
    assert d.local_tokens_in + d.local_tokens_out == 0


def test_remote_only_routes_every_task_type_remote():
    remote = MockRemoteClient()
    router = Router(make_cfg(mode="remote_only"), CountingLocal(), remote)
    prompts = ["What is 2 + 2?",
               "Summarize this paragraph in one sentence: the cat sat.",
               "Which philosopher wrote the Zibaldone?",
               "Write a function that reverses a string."]
    for i, p in enumerate(prompts):
        d = router.route(Task(id=f"r{i}", text=p))
        assert d.route == "remote"
        assert d.signals.get("mode") == "remote_only"
    assert remote.calls == len(prompts)


def test_remote_only_failure_takes_single_local_sample():
    local = CountingLocal()
    router = Router(make_cfg(mode="remote_only"), local, MockRemoteClient(fail=True))
    d = router.route(Task(id="r5", text="What is 6 * 7?"))
    assert d.route == "remote_failed_local"
    assert d.answer == "42"
    assert local.calls == 1  # one greedy last-resort sample, not k
    assert d.local_tokens_out > 0


def test_remote_only_everything_down_still_answers():
    router = Router(make_cfg(mode="remote_only"), BrokenLocal(),
                    MockRemoteClient(fail=True))
    d = router.route(Task(id="r6", text="anything"))
    assert d.route == "remote_failed_local"
    assert d.answer == "unknown"
    assert d.error and "remote escalation failed" in d.error
