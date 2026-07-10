from cheaproute.config import load_config
from cheaproute.local_client import MockLocalClient
from cheaproute.remote_client import MockRemoteClient, RemoteError
from cheaproute.router import Router
from cheaproute.schema import GenResult, Task
from cheaproute.tasktype import profile_for


def make_cfg(**routing_overrides):
    cfg = load_config(path="__no_such_file__.yaml")  # pure defaults
    cfg["routing"].update(routing_overrides)
    return cfg


def test_confident_local_answer_stays_local():
    router = Router(make_cfg(), MockLocalClient(), MockRemoteClient())
    d = router.route(Task(id="t1", text="What is 12 + 30?"))
    assert d.route == "local"
    assert d.answer == "42"
    assert d.remote_tokens_in + d.remote_tokens_out == 0


def test_low_confidence_escalates_to_remote():
    # MockLocalClient's confidence for this question is ~0.35; explicitly set
    # a threshold above that so the test exercises escalation regardless of
    # wherever the production default happens to be tuned.
    remote = MockRemoteClient()
    router = Router(make_cfg(threshold=0.62), MockLocalClient(), remote)
    d = router.route(Task(id="t2", text="Which philosopher wrote the Zibaldone?"))
    assert d.route == "remote"
    assert d.answer == "REMOTE"
    assert remote.calls == 1
    assert d.remote_tokens_in + d.remote_tokens_out > 0


def test_remote_failure_falls_back_to_local_answer():
    router = Router(make_cfg(threshold=0.62), MockLocalClient(),
                    MockRemoteClient(fail=True))
    d = router.route(Task(id="t3", text="Which philosopher wrote the Zibaldone?"))
    assert d.route == "remote_failed_local"
    assert d.answer  # still returned something
    assert d.error and "remote escalation failed" in d.error


class BrokenLocal:
    def generate(self, *a, **k):
        raise RuntimeError("gpu on fire")


def test_local_failure_goes_straight_to_remote():
    remote = MockRemoteClient()
    router = Router(make_cfg(), BrokenLocal(), remote)
    d = router.route(Task(id="t4", text="anything"))
    assert d.route == "remote"
    assert remote.calls == 1


def test_both_backends_down_still_returns_answer():
    router = Router(make_cfg(), BrokenLocal(), MockRemoteClient(fail=True))
    d = router.route(Task(id="t5", text="anything"))
    assert d.route == "remote_failed_local"
    assert d.answer == "unknown"


def test_threshold_is_respected():
    # threshold 0 -> everything stays local, even the shaky answers
    router = Router(make_cfg(threshold=0.0), MockLocalClient(), MockRemoteClient())
    d = router.route(Task(id="t6", text="Which philosopher wrote the Zibaldone?"))
    assert d.route == "local"
    # threshold 1.01 -> everything escalates
    router = Router(make_cfg(threshold=1.01), MockLocalClient(), MockRemoteClient())
    d = router.route(Task(id="t7", text="What is 2 + 2?"))
    assert d.route == "remote"


class EmptyRemote:
    def generate(self, *a, **k):
        return GenResult(text="", finish_reason="stop", tokens_in=5, tokens_out=0)


def test_empty_remote_answer_falls_back_to_local():
    router = Router(make_cfg(mode="remote_only"), MockLocalClient(), EmptyRemote())
    d = router.route(Task(id="t9", text="What is 12 + 30?"))
    assert d.route == "remote_failed_local"
    assert d.answer and d.answer != ""


class TruncatedLocal:
    def generate(self, *a, **k):
        return GenResult(text="Answer: blah", mean_logprob=-0.1,
                         finish_reason="length")


def test_truncated_local_output_lowers_confidence():
    remote = MockRemoteClient()
    router = Router(make_cfg(), TruncatedLocal(), remote)
    d = router.route(Task(id="t8", text="2 + 2?"))
    # format score is 0 for truncation; with high agreement+logprob it may
    # still pass, so assert the signal is present and zeroed
    assert d.signals.get("format_ok") == 0.0


class RaisingLogger:
    def log(self, decision):
        raise RuntimeError("disk full")


def test_local_candidate_matches_route_local_first_confidence():
    router = Router(make_cfg(), MockLocalClient(), MockRemoteClient())
    ttype = "math"
    prof = profile_for(ttype)
    answer, conf, signals, tin, tout, err = router.local_candidate(
        "What is 12 + 30?", ttype, prof)
    assert answer == "42"
    assert conf > 0
    assert signals.get("task_type") == "math"
    assert not signals.get("local_unavailable")
    assert tin + tout > 0
    assert err is None


def test_local_candidate_reports_unavailable_when_local_is_down():
    router = Router(make_cfg(), BrokenLocal(), MockRemoteClient())
    answer, conf, signals, tin, tout, err = router.local_candidate(
        "anything", "short_qa", profile_for("short_qa"))
    assert signals.get("local_unavailable") is True
    assert answer == "" and conf == 0.0 and tin == 0 and tout == 0
    assert err  # the underlying exception message is preserved


def test_raising_logger_does_not_crash_route():
    router = Router(make_cfg(), MockLocalClient(), MockRemoteClient(),
                    logger=RaisingLogger())
    d = router.route(Task(id="t10", text="What is 12 + 30?"))
    assert d.route == "local"
    assert d.answer == "42"
