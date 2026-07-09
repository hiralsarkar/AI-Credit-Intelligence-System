"""Behavioural contracts for the decision engine. No data files required."""
from signal_aggregator import (
    build_module_a_signal, build_module_b_signal, build_module_c_signal,
    build_module_d_signal, aggregate_signals, assign_risk_band,
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


def test_module_d_provisioning_is_attached_and_advisory_only():
    a = build_module_a_signal(pd_score=0.05, credit_score=720, ead=200000.0)
    b = build_module_b_signal(delinquency_prob=0.04, behavioural_risk_score=70.0)
    d = build_module_d_signal(ifrs9_stage=2, lifetime_pd=0.0587,
                              lifetime_ecl_rate=0.0245, twelve_month_ecl_rate=0.0006,
                              sicr_flag=True)
    with_d = make_decision(aggregate_signals(a, b, None, d))
    without_d = make_decision(aggregate_signals(a, b, None))
    assert with_d["provisioning"]["available"] is True
    assert with_d["provisioning"]["ifrs9_stage"] == 2
    assert without_d["provisioning"]["available"] is False
    # provisioning is advisory: adding D must not change the origination decision
    assert with_d["decision"] == without_d["decision"]
