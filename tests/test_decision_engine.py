"""Behavioural contracts for the decision engine (v2). No data files required."""
import itertools

import constants
import engine
from signal_aggregator import (
    build_module_a_signal, build_module_b_signal, build_module_c_signal,
    build_module_d_signal, aggregate_signals, assign_risk_band, deal_economics,
)
from engine import make_decision
from constants import (
    DECISION_APPROVE, DECISION_DECLINE, DECISION_MANUAL_REVIEW, DECISION_REPRICE,
    SIGNAL_WEIGHTS, HURDLE_RATE, MAX_LENDING_RATE,
)

VALID = {DECISION_APPROVE, DECISION_DECLINE, DECISION_MANUAL_REVIEW, DECISION_REPRICE}


def _c(pd_a, pd_b, rate=12.0, adequacy="Fair", c_rate=12.0, ead=200000.0, **b):
    a = build_module_a_signal(pd_score=pd_a, credit_score=680, ead=ead, offered_rate=rate)
    bb = None if pd_b is None else build_module_b_signal(
        delinquency_prob=pd_b, behavioural_risk_score=60.0, **b)
    c = build_module_c_signal(market_pd=0.12, loan_grade_signal="C",
                              suggested_rate=c_rate, pricing_adequacy=adequacy,
                              concentration_flag=False)
    return aggregate_signals(a, bb, c)


# ── Weights and bands ─────────────────────────────────────────────────────────

def test_signal_weights_are_a_valid_governance_choice():
    wa, wb = SIGNAL_WEIGHTS["module_a"], SIGNAL_WEIGHTS["module_b"]
    assert abs(wa + wb - 1.0) < 1e-9          # normalised
    assert wa >= wb                            # Module A anchored primary at origination


def test_composite_pd_uses_configured_weights_and_is_bounded():
    wa, wb = SIGNAL_WEIGHTS["module_a"], SIGNAL_WEIGHTS["module_b"]
    comp = _c(0.10, 0.20)
    assert 0.0 <= comp["composite_pd"] <= 1.0
    assert abs(comp["composite_pd"] - (wa * 0.10 + wb * 0.20)) < 1e-9


def test_risk_band_is_monotonic():
    assert assign_risk_band(0.01) == 1
    assert assign_risk_band(0.50) == 5
    assert assign_risk_band(0.01) <= assign_risk_band(0.50)


# ── Core routing ──────────────────────────────────────────────────────────────

def test_extreme_pd_declines():
    assert make_decision(_c(0.5, 0.5))["decision"] == DECISION_DECLINE


def test_behavioural_hard_override_forces_manual_review():
    out = make_decision(_c(0.05, 0.05, high_delinquency_score=True))
    assert out["decision"] == DECISION_MANUAL_REVIEW
    assert "high_delinquency_score" in out["overrides_triggered"]


def test_module_d_provisioning_is_attached_and_advisory_only():
    a = build_module_a_signal(pd_score=0.05, credit_score=720, ead=200000.0, offered_rate=12.0)
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


def test_every_decision_has_a_reason_across_the_input_space():
    """v1 could DECLINE with an empty reason list; v2 must never do that."""
    pds = [0.01, 0.06, 0.09, 0.12, 0.18, 0.25, 0.4]
    rates = [6.0, 9.0, 12.0, 18.0, 26.0]
    for pd_a, pd_b, rate, adequacy in itertools.product(pds, pds + [None], rates,
                                                        ["Fair", "Underpriced"]):
        out = make_decision(_c(pd_a, pd_b, rate=rate, adequacy=adequacy, c_rate=14.0))
        assert out["decision"] in VALID
        assert out["reason_codes"], (pd_a, pd_b, rate, adequacy)
        assert out["primary_reason"]
        if out["decision"] != DECISION_APPROVE:
            assert out["what_would_change"] or out["decision"] == DECISION_DECLINE


# ── v2: price-aware RAROC ─────────────────────────────────────────────────────

def test_raroc_depends_on_the_offered_rate():
    low = make_decision(_c(0.06, 0.05, rate=7.0))["capital_impact"]["raroc"]
    high = make_decision(_c(0.06, 0.05, rate=14.0))["capital_impact"]["raroc"]
    assert high > low


def test_breakeven_rate_exactly_meets_the_hurdle():
    e = deal_economics(0.08, 300000.0, 0.12)
    at_breakeven = deal_economics(0.08, 300000.0, e["breakeven_rate"])
    assert abs(at_breakeven["raroc"] - HURDLE_RATE) < 1e-9


def test_teaser_rate_loan_is_repriced_not_approved():
    out = make_decision(_c(0.07, 0.05, rate=6.5))
    assert out["decision"] == DECISION_REPRICE
    suggested = out["pricing"]["suggested_rate"] / 100
    assert suggested >= out["capital_impact"]["breakeven_rate"] / 100
    # repricing to the suggested rate turns it into a clean approval
    assert make_decision(_c(0.07, 0.05, rate=suggested * 100))["decision"] == DECISION_APPROVE


def test_rate_above_ceiling_declines_instead_of_repricing():
    out = make_decision(_c(0.15, 0.15, rate=10.0, adequacy="Underpriced", c_rate=30.0))
    assert out["decision"] == DECISION_DECLINE
    assert any(r["code"] == "unpriceable" for r in out["reason_codes"])
    assert 30.0 / 100 > MAX_LENDING_RATE


def test_missing_offered_rate_is_disclosed():
    a = build_module_a_signal(pd_score=0.05, credit_score=700, ead=100000.0)
    out = make_decision(aggregate_signals(a, None, None))
    assert any(r["code"] == "rate_assumed" for r in out["reason_codes"])


# ── v2: thin file ─────────────────────────────────────────────────────────────

def test_thin_file_uses_no_synthetic_behavioural_values():
    comp = _c(0.165, None, rate=13.0)
    assert comp["signal_b"] is None and comp["thin_file"] is True
    assert comp["composite_pd"] == 0.165
    out = make_decision(comp)
    assert out["decision"] == DECISION_MANUAL_REVIEW
    assert any(r["code"] == "thin_file" for r in out["reason_codes"])
    assert "6.70%" not in out["decision_basis"]


# ── v2: constants drive every explanation ─────────────────────────────────────

def test_narrative_follows_constants(monkeypatch):
    monkeypatch.setattr(engine, "HURDLE_RATE", 0.20)
    out = make_decision(_c(0.04, 0.03, rate=12.0))
    assert "20%" in out["primary_reason"] or "20%" in out["decision_basis"]
    assert constants.HURDLE_RATE == 0.14  # module constant itself untouched


# ── Demo profiles land where the story says ───────────────────────────────────

def test_demo_profiles_match_expected_decisions():
    from demo import APPLICANT_PROFILES, run_single_applicant
    for profile in APPLICANT_PROFILES.values():
        out = run_single_applicant(profile, log_decisions=False)
        assert out["decision"] == profile["expected_decision"], profile["name"]
