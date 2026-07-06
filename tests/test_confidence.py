from cheaproute.confidence import (agreement_score, answers_agree,
                                   composite_confidence, extract_final,
                                   format_score, logprob_score, majority_index,
                                   normalize_answer)

ROUTING = {"w_logprob": 0.45, "w_agreement": 0.40, "w_format": 0.15,
           "logprob_floor": -2.5, "logprob_ceil": -0.05}


def test_extract_final_answer_marker():
    assert extract_final("Let me think...\nAnswer: 42") == "42"
    assert extract_final("answer: Paris\nmore text\nAnswer: London") == "London"


def test_extract_final_fallback_last_line():
    assert extract_final("some reasoning\nthe result is 7") == "the result is 7"
    assert extract_final("") == ""


def test_normalize_answer():
    assert normalize_answer("  Paris.  ") == "paris"
    assert normalize_answer('"42"') == "42"
    assert normalize_answer("A)  the   Answer!") == "a) the answer"


def test_numeric_agreement():
    assert answers_agree("1,000", "1000")
    assert answers_agree("3.14", "3.14")
    assert not answers_agree("3.14", "2.71")
    assert answers_agree("Paris", "paris.")
    assert not answers_agree("Paris", "London")


def test_agreement_score():
    assert agreement_score(["42", "42", "42"]) == 1.0
    assert abs(agreement_score(["42", "42", "7"]) - 2 / 3) < 1e-9
    assert agreement_score(["a", "b", "c"]) == 1 / 3
    assert agreement_score(["only"]) == 1.0
    assert agreement_score([]) == 0.0


def test_majority_index_prefers_greedy_on_ties():
    # two clusters of one each: earliest (greedy) sample wins
    assert majority_index(["a", "b"]) == 0
    # larger cluster wins even if it excludes the greedy sample
    assert majority_index(["a", "b", "b"]) in (1, 2)


def test_logprob_score_mapping():
    assert logprob_score(-0.05, -2.5, -0.05) == 1.0
    assert logprob_score(-2.5, -2.5, -0.05) == 0.0
    assert logprob_score(-5.0, -2.5, -0.05) == 0.0   # clamped
    assert logprob_score(None, -2.5, -0.05) == 0.5   # missing => neutral


def test_format_score():
    assert format_score("Answer: 42", "42", "stop") == 1.0
    assert format_score("just some text", "just some text", "stop") == 0.5
    assert format_score("Answer: 42", "42", "length") == 0.0
    assert format_score("", "", "stop") == 0.0


def test_composite_in_unit_range():
    conf, signals = composite_confidence(ROUTING, -0.1, 1.0, 1.0)
    assert 0.9 <= conf <= 1.0
    conf_low, _ = composite_confidence(ROUTING, -2.4, 0.33, 0.5)
    assert conf_low < 0.4
    assert set(signals) >= {"logprob_score", "agreement", "format_ok"}


def test_composite_agreement_none_redistributes_weight():
    # free-form/single-sample: agreement weight moves to logprob+format
    conf_none, signals = composite_confidence(ROUTING, -0.1, None, 1.0)
    assert signals["agreement"] is None
    assert 0.9 <= conf_none <= 1.0
    # low logprob without agreement backup should stay low
    conf_low, _ = composite_confidence(ROUTING, -2.4, None, 0.5)
    assert conf_low < 0.35


def test_format_score_freeform():
    assert format_score("A fine summary.", "A fine summary.", "stop",
                        freeform=True) == 1.0
    assert format_score("Truncated tex", "Truncated tex", "length",
                        freeform=True) == 0.0
