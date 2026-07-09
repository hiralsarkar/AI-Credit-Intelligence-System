"""Behavioural contracts for the decision engine. No data files required."""
from signal_aggregator import (
    build_module_a_signal, build_module_b_signal, build_module_c_signal,
    aggregate_signals, assign_risk_band,
)
from engine import make_decision
from constants import (
    DECISION_APPROVE, DECISION_DECLINE, DECISION_MANUAL_REVIEW, DECISION_REPRICE,
)

VALID = {DECISION_APPROVE, DECISION_DECLINE, DECISION_MANUAL_REVIEW, DECISION_REPRICE}


def _c(pd_a, pd_b, **b):
    a = build_module_a_signal(pd_score=pd_a, credit_score=680, ead=200000.0)
    bb = build_module_b_signal(delinquency_prob=pd_b, behavioural_risk_score=60.0, **b)
    c = build_module_c_signal(market_pd=0.12, loan_grade_signal="C",
                              suggested_rate=12.0, pricing_adequacy="Fair",
                              concentration_flag=False)
    return aggregate_signals(a, bb, c)


def test_composite_pd_is_weighted_average_and_bounded():
    comp = _c(0.10, 0.20)
    assert 0.0 <= comp["composite_pd"] <= 1.0
    assert abs(comp["composite_pd"] - (0.60 * 0.10 + 0.40 * 0.20)) < 1e-9


def test_risk_band_is_monotonic():
    assert assign_risk_band(0.01) == 1
    assert assign_risk_band(0.50) == 5
    assert assign_risk_band(0.01) <= assign_risk_band(0.50)


def test_decision_is_always_one_of_four_with_reasons():
    out = make_decision(_c(0.06, 0.05))
    assert out["decision"] in VALID
    assert out["reason_codes"]


def test_extreme_pd_declines():
    assert make_decision(_c(0.5, 0.5))["decision"] == DECISION_DECLINE


def test_behavioural_hard_override_forces_manual_review():
    out = make_decision(_c(0.05, 0.05, high_delinquency_score=True))
    assert out["decision"] == DECISION_MANUAL_REVIEW
    assert "high_delinquency_score" in out["overrides_triggered"]
