from cheaproute.confidence import extract_final
from cheaproute.tasktype import classify, extract_answer, profile_for


def test_classify_eight_categories():
    cases = {
        "What is 15% of 240?": "math",
        "Which is the largest planet? A) Earth B) Jupiter C) Mars D) Venus.": "mc",
        "Classify the sentiment of this review as positive or negative: 'Great!'": "sentiment",
        "Summarise the following text in one sentence: 'Lorem ipsum...'": "summarization",
        "Extract all person names and locations from: 'Alice went to Rome.'": "ner",
        "This function has a bug, fix it:\n\ndef add(a, b):\n    return a - b": "code_debug",
        "Write a Python function is_even(n) that returns True if n is even.": "code_gen",
        "What is the next number in the sequence 2, 4, 8?": "logic",
        "What is the capital of France?": "short_qa",
    }
    for prompt, expected in cases.items():
        assert classify(prompt) == expected, f"{prompt!r} -> {classify(prompt)}"


def test_profiles_freeform_vs_shortform():
    assert profile_for("summarization").freeform
    assert profile_for("code_gen").freeform
    assert not profile_for("math").freeform
    assert profile_for("math").samples > 1
    assert profile_for("code_gen").samples == 1
    # remote budgets stay tight for label-like answers
    assert profile_for("mc").remote_max_tokens <= 32


def test_extract_answer_code_prefers_fenced_block():
    raw = "Here is the fix:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    out = extract_answer("code_debug", raw, extract_final)
    assert out == "def add(a, b):\n    return a + b"


def test_extract_answer_shortform_uses_answer_line():
    raw = "Thinking...\nAnswer: 42"
    assert extract_answer("math", raw, extract_final) == "42"


def test_extract_answer_freeform_returns_full_text():
    raw = "The passage describes honeybee communication."
    assert extract_answer("summarization", raw, extract_final) == raw
