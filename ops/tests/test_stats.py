from pytest import approx

from vektr_ops.stats import add, decide, elo_estimate, expected_score, llr


def test_expected_score_is_logistic():
    assert expected_score(0) == approx(0.5)
    assert expected_score(190.8485) == approx(0.75, abs=1e-6)


def test_llr_reference_values():
    assert llr((10, 40, 100, 60, 20)) == approx(1.14, abs=1e-3)
    assert llr((20, 60, 100, 40, 10)) == approx(-1.3457, abs=1e-3)
    assert llr((100, 400, 1000, 480, 120)) == approx(3.0951, abs=1e-3)


def test_llr_is_zero_without_information():
    assert llr((0, 0, 0, 0, 0)) == 0.0
    assert llr((0, 0, 30, 0, 0)) == 0.0  # zero variance


def test_decide_uses_the_five_percent_bounds():
    assert decide(2.95) == "accept"
    assert decide(-2.95) == "reject"
    assert decide(2.94) == "continue"


def test_elo_estimate_reference():
    elo, margin = elo_estimate((10, 40, 100, 60, 20))
    assert elo == approx(30.29, abs=0.01)
    assert margin == approx(21.8, abs=0.3)


def test_add_sums_counts():
    assert add((1, 2, 3, 4, 5), (5, 4, 3, 2, 1)) == (6, 6, 6, 6, 6)
