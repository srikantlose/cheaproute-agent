import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))

from graders import grade  # noqa: E402


def test_numeric_tolerates_currency_units_and_unicode_minus():
    assert grade({"grader": "numeric", "expected": "42"}, "$42")
    assert grade({"grader": "numeric", "expected": "42"}, "42 kg")
    assert grade({"grader": "numeric", "expected": "-5"}, "−5")  # unicode minus
    assert not grade({"grader": "numeric", "expected": "42"}, "41")  # still wrong
    assert not grade({"grader": "numeric", "expected": "42"}, "$41")


def test_contains_all():
    task = {"grader": "contains_all", "expected": ["merkel", "paris"]}
    assert grade(task, "PERSON: Angela Merkel, LOCATION: Paris")
    assert not grade(task, "Only Merkel is mentioned here")


def test_word_limit():
    task = {"grader": "word_limit",
            "expected": {"max_words": 10, "any_of": ["wright", "flight"]}}
    assert grade(task, "The Wright brothers made the first powered flight.")
    assert not grade(task, "The Wright brothers made the very first powered "
                           "airplane flight near Kitty Hawk in December 1903.")
    assert not grade(task, "A short unrelated sentence about nothing much.")
    assert not grade(task, "")


def test_py_exec_passes_correct_code():
    task = {"grader": "py_exec",
            "test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    assert grade(task, good)
    assert grade(task, "def add(a, b):\n    return a + b")  # unfenced too


def test_py_exec_fails_buggy_and_hanging_code():
    task = {"grader": "py_exec", "test_code": "assert add(2, 3) == 5"}
    assert not grade(task, "def add(a, b):\n    return a - b")
    hang = ("def add(a, b):\n    while True:\n        pass\n")
    assert not grade({"grader": "py_exec", "test_code": "assert add(1, 1) == 2"},
                     hang)  # timeout kills it


def test_py_exec_rejects_empty():
    assert not grade({"grader": "py_exec", "test_code": "assert True"}, "")
