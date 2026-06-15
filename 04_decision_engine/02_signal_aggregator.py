"""
signal_aggregator.py — AI Credit Intelligence System
==================================================
Loads outputs from Module A, B, and C and computes a weighted composite
risk score for the Decision Engine.

Each module produces a standardised signal dictionary. This aggregator:
  1. Accepts pre-computed signal dicts (real-time path)
  2. Can also load them from saved CSV outputs (batch path)
  3. Computes the weighted composite PD
  4. Handles graceful degradation when a module signal is unavailable

Architecture
------------
  Real-time path (new application):
    Features → ModuleA.score() → signal_a
    Features → ModuleB.score() → signal_b  (if behavioural data available)
    Features → ModuleC.score() → signal_c
    → aggregate_signals(a, b, c)

  Batch path (portfolio review):
    load_module_a_signal(csv_path) → signal_a
    load_module_b_signal(csv_path) → signal_b
    load_module_c_signal(csv_path) → signal_c
    → aggregate_signals(a, b, c)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from constants import (
    SIGNAL_WEIGHTS,
    MODULE_B_FALLBACK_DELINQUENCY_PROB,
    MODULE_B_FALLBACK_RISK_BAND,
    RISK_BAND_THRESHOLDS,
    RISK_BAND_LABELS,
)


# ── Signal Schema Definitions ─────────────────────────────────────────────────
# These define the canonical keys each module must populate.
# The engine reads only these keys — modules can add extra keys freely.

MODULE_A_REQUIRED_KEYS = {
    "pd_score",       # float 0-1  — XGBoost PD from Module A
    "risk_band",      # int 1-5    — risk band assignment
    "credit_score",   # int 300-900 — WoE scorecard score
    "risk_grade",     # str        — "Very Low Risk" to "Very High Risk"
    "el",             # float      — Expected Loss (₹)
    "el_rate",        # float      — EL as fraction of EAD
    "raroc",          # float      — Risk-Adjusted Return on Capital
    "ead",            # float      — Exposure at Default (₹)
    "economic_capital", # float    — Economic Capital (₹)
}

MODULE_B_REQUIRED_KEYS = {
    "delinquency_prob",       # float 0-1  — XGBoost delinquency probability
    "behavioural_risk_band",  # int 1-5
    "behavioural_risk_score", # float 0-100 (higher = safer)
    "stress_flag",            # bool — general distress
    "critical_utilisation",   # bool — revolving util >= 80%
    "escalating_delinquency", # bool — has 60/90DPD events
    "high_delinquency_score", # bool — delinquency_score >= 5
}

MODULE_C_REQUIRED_KEYS = {
    "market_pd",           # float 0-1  — market-implied PD from grade model
    "loan_grade_signal",   # str "A"-"G"
    "suggested_rate",      # float      — risk-appropriate rate (%)
    "pricing_adequacy",    # str        — "Underpriced"/"Fair"/"Overpriced"
    "concentration_flag",  # bool
}


# ── Individual Module Signal Builders ─────────────────────────────────────────

def build_module_a_signal(
    pd_score: float,
    credit_score: int,
    ead: float,
    risk_band: Optional[int] = None,
    el: Optional[float] = None,
    raroc: Optional[float] = None,
    economic_capital: Optional[float] = None,
) -> dict:
    """
    Construct a validated Module A signal dictionary from raw model outputs.

    In a production system, this is called after running the XGBoost PD model
    and the financial calculations from Module A NB04.

    Parameters
    ----------
    pd_score       : Probability of default (0-1) from XGBoost champion model.
    credit_score   : WoE scorecard score (300-900).
    ead            : Exposure at default — loan amount in ₹.
    risk_band      : Integer 1-5. Computed from pd_score if not provided.
    el             : Expected Loss. Computed if not provided.
    raroc          : Risk-Adjusted Return on Capital. Computed if not provided.
    economic_capital: Economic Capital. Computed if not provided.
    """
    from constants import (
        LGD_UNSECURED, RISK_WEIGHTS, CAPITAL_RATIO,
        INTEREST_RATE, OPERATING_COST, HURDLE_RATE,
    )

    # Assign risk band from PD if not provided
    if risk_band is None:
        risk_band = assign_risk_band(pd_score)

    # Compute financial metrics if not provided
    lgd = LGD_UNSECURED
    if el is None:
        el      = pd_score * lgd * ead
    el_rate     = pd_score * lgd

    rw = RISK_WEIGHTS.get(risk_band, 1.00)
    rwa = ead * rw
    if economic_capital is None:
        economic_capital = rwa * CAPITAL_RATIO

    revenue    = ead * INTEREST_RATE
    opex       = ead * OPERATING_COST
    net_income = revenue - el - opex
    if raroc is None:
        raroc = net_income / economic_capital if economic_capital > 0 else 0.0

    # Score to risk grade label
    risk_grade = score_to_grade_label(credit_score)

    return {
        "pd_score":         float(pd_score),
        "risk_band":        int(risk_band),
        "credit_score":     int(credit_score),
        "risk_grade":       risk_grade,
        "el":               float(el),
        "el_rate":          float(el_rate),
        "raroc":            float(raroc),
        "ead":              float(ead),
        "economic_capital": float(economic_capital),
        "rwa":              float(rwa),
        "above_hurdle":     raroc >= HURDLE_RATE,
    }


def build_module_b_signal(
    delinquency_prob: float,
    behavioural_risk_score: float,
    stress_flag: bool = False,
    critical_utilisation: bool = False,
    escalating_delinquency: bool = False,
    high_delinquency_score: bool = False,
    behavioural_risk_band: Optional[int] = None,
) -> dict:
    """
    Construct a validated Module B signal dictionary.

    Parameters
    ----------
    delinquency_prob       : P(90-day delinquency in 2 years) from XGBoost.
    behavioural_risk_score : 0-100 score (higher = safer).
    stress_flag            : True if delinquency_score >= 3 or has 90DPD.
    critical_utilisation   : True if revolving utilisation >= 80%.
    escalating_delinquency : True if has 60DPD or 90DPD events.
    high_delinquency_score : True if delinquency_score >= 5.
    behavioural_risk_band  : 1-5. Computed from delinquency_prob if not provided.
    """
    if behavioural_risk_band is None:
        behavioural_risk_band = assign_behavioural_band(delinquency_prob)

    return {
        "delinquency_prob":       float(delinquency_prob),
        "behavioural_risk_band":  int(behavioural_risk_band),
        "behavioural_risk_score": float(behavioural_risk_score),
        "stress_flag":            bool(stress_flag),
        "critical_utilisation":   bool(critical_utilisation),
        "escalating_delinquency": bool(escalating_delinquency),
        "high_delinquency_score": bool(high_delinquency_score),
    }


def build_module_c_signal(
    market_pd: float,
    loan_grade_signal: str,
    suggested_rate: float,
    pricing_adequacy: str,
    concentration_flag: bool,
) -> dict:
    """
    Construct a validated Module C signal dictionary.

    Parameters
    ----------
    market_pd          : Market-implied P(default) derived from predicted grade.
    loan_grade_signal  : Predicted LendingClub-equivalent grade (A-G).
    suggested_rate     : Risk-appropriate interest rate (% p.a.).
    pricing_adequacy   : "Underpriced" / "Fair" / "Overpriced".
    concentration_flag : True if high-risk purpose or 60-month term.
    """
    return {
        "market_pd":           float(market_pd),
        "loan_grade_signal":   str(loan_grade_signal),
        "suggested_rate":      float(suggested_rate),
        "pricing_adequacy":    str(pricing_adequacy),
        "concentration_flag":  bool(concentration_flag),
    }


# ── Signal Aggregator ─────────────────────────────────────────────────────────

def aggregate_signals(
    signal_a: dict,
    signal_b: Optional[dict],
    signal_c: Optional[dict],
    weights: Optional[dict] = None,
) -> dict:
    """
    Compute weighted composite risk score from Module A and B signals.

    Module C is NOT included in the composite PD — it provides the pricing
    and concentration signals that modify the decision path downstream.

    Parameters
    ----------
    signal_a : Module A signal dictionary (required).
    signal_b : Module B signal dictionary (optional — graceful fallback).
    signal_c : Module C signal dictionary (optional — affects routing).
    weights  : Override default weights {"module_a": 0.60, "module_b": 0.40}.

    Returns
    -------
    dict with keys:
        composite_pd       : Weighted PD combining Modules A and B
        composite_risk_band: Risk band derived from composite_pd
        signal_a           : Module A signal (passed through)
        signal_b           : Module B signal (or fallback)
        signal_c           : Module C signal (or None)
        weights_used       : Actual weights applied
        b_available        : Whether real Module B data was used
        c_available        : Whether real Module C data was used
        module_pd_comparison: Dict showing each module's PD contribution
    """
    w = weights or SIGNAL_WEIGHTS

    # ── Module B: use fallback if not available ───────────────────────────────
    b_available = signal_b is not None
    if not b_available:
        signal_b = _module_b_fallback()
        # When B is unavailable, put full weight on A
        w_a, w_b = 1.00, 0.00
    else:
        w_a = w.get("module_a", 0.60)
        w_b = w.get("module_b", 0.40)

    # ── Composite PD ─────────────────────────────────────────────────────────
    pd_a = signal_a["pd_score"]
    pd_b = signal_b["delinquency_prob"]

    composite_pd = w_a * pd_a + w_b * pd_b
    composite_pd = float(np.clip(composite_pd, 0.0, 1.0))

    # ── Composite risk band ───────────────────────────────────────────────────
    composite_risk_band = assign_risk_band(composite_pd)

    # ── Market PD divergence check ────────────────────────────────────────────
    c_available = signal_c is not None
    market_pd_divergence = None
    if c_available:
        divergence = abs(signal_c["market_pd"] - composite_pd)
        market_pd_divergence = {
            "composite_pd":  round(composite_pd, 4),
            "market_pd":     round(signal_c["market_pd"], 4),
            "divergence":    round(divergence, 4),
            "flag":          divergence > 0.15,  # >15pp divergence is notable
        }

    return {
        "composite_pd":         composite_pd,
        "composite_risk_band":  composite_risk_band,
        "composite_risk_label": RISK_BAND_LABELS.get(composite_risk_band, "Unknown"),
        "signal_a":             signal_a,
        "signal_b":             signal_b,
        "signal_c":             signal_c if c_available else None,
        "weights_used":         {"module_a": w_a, "module_b": w_b},
        "b_available":          b_available,
        "c_available":          c_available,
        "module_pd_comparison": {
            "module_a_pd":    round(pd_a, 4),
            "module_b_pd":    round(pd_b, 4),
            "composite_pd":   round(composite_pd, 4),
            "market_pd":      round(signal_c["market_pd"], 4) if c_available else None,
        },
        "market_pd_divergence": market_pd_divergence,
    }


# ── Batch Loaders (portfolio review path) ─────────────────────────────────────

def load_module_a_signal_from_csv(scored_path: str, row_index: int = 0) -> dict:
    """
    Load a Module A signal from scorecard_output.csv.

    In portfolio review mode, this is called for each row in the scored output.

    Parameters
    ----------
    scored_path : Path to scorecard_output.csv produced by Module A NB04.
    row_index   : Row to load (0-indexed). Default: first row.
    """
    df = pd.read_csv(scored_path)
    row = df.iloc[row_index]

    return build_module_a_signal(
        pd_score         = float(row.get("PD", row.get("PD_XGB", 0.10))),
        credit_score     = int(row.get("CREDIT_SCORE", 600)),
        ead              = float(row.get("EAD", row.get("AMT_CREDIT", 100000))),
        risk_band        = int(row.get("RISK_BAND", 3)),
        el               = float(row.get("EL", 0)),
        raroc            = float(row.get("RAROC", 0)),
        economic_capital = float(row.get("CAPITAL_STRESSED",
                                         row.get("ECONOMIC_CAPITAL", 0))),
    )


def load_module_b_signal_from_csv(scored_path: str, row_index: int = 0) -> dict:
    """
    Load a Module B signal from scorecard_output_b.csv.

    Parameters
    ----------
    scored_path : Path to scorecard_output_b.csv produced by Module B NB03.
    row_index   : Row to load (0-indexed).
    """
    df = pd.read_csv(scored_path)
    row = df.iloc[row_index]

    return build_module_b_signal(
        delinquency_prob        = float(row.get("DELINQUENCY_PROB", 0.067)),
        behavioural_risk_score  = float(row.get("BEHAVIOURAL_RISK_SCORE", 50)),
        stress_flag             = bool(row.get("STRESS_FLAG", False)),
        critical_utilisation    = bool(row.get("CRITICAL_UTILISATION", False)),
        escalating_delinquency  = bool(row.get("ESCALATING_DELINQUENCY", False)),
        high_delinquency_score  = bool(row.get("HIGH_DELINQUENCY_SCORE", False)),
        behavioural_risk_band   = int(row.get("BEHAVIOURAL_RISK_BAND", 2)),
    )


def load_module_c_signal_from_csv(scored_path: str, row_index: int = 0) -> dict:
    """
    Load a Module C signal from scored_test_c.csv.

    Parameters
    ----------
    scored_path : Path to scored_test_c.csv produced by Module C NB02.
    row_index   : Row to load (0-indexed).
    """
    df = pd.read_csv(scored_path)
    row = df.iloc[row_index]

    return build_module_c_signal(
        market_pd          = float(row.get("MARKET_PD", 0.20)),
        loan_grade_signal  = str(row.get("PREDICTED_GRADE", "C")),
        suggested_rate     = float(row.get("SUGGESTED_RATE", 12.0)),
        pricing_adequacy   = str(row.get("PRICING_ADEQUACY", "Fair")),
        concentration_flag = bool(row.get("CONCENTRATION_FLAG", False)),
    )


# ── Helper Functions ──────────────────────────────────────────────────────────

def assign_risk_band(pd_score: float) -> int:
    """Map probability of default to integer risk band 1-5."""
    if pd_score < 0.05:  return 1
    if pd_score < 0.10:  return 2
    if pd_score < 0.20:  return 3
    if pd_score < 0.35:  return 4
    return 5


def assign_behavioural_band(delinquency_prob: float) -> int:
    """Map delinquency probability to behavioural risk band 1-5."""
    if delinquency_prob < 0.03:  return 1
    if delinquency_prob < 0.07:  return 2
    if delinquency_prob < 0.15:  return 3
    if delinquency_prob < 0.30:  return 4
    return 5


def score_to_grade_label(credit_score: int) -> str:
    """Map WoE scorecard score (300-900) to risk grade label."""
    if credit_score >= 700:  return "Very Low Risk"
    if credit_score >= 625:  return "Low Risk"
    if credit_score >= 550:  return "Medium Risk"
    if credit_score >= 450:  return "High Risk"
    return "Very High Risk"


def _module_b_fallback() -> dict:
    """
    Fallback Module B signal for thin-file applicants with no behavioural history.

    Conservative assumption: population average delinquency probability,
    no active stress flags.
    """
    return build_module_b_signal(
        delinquency_prob        = MODULE_B_FALLBACK_DELINQUENCY_PROB,
        behavioural_risk_score  = 50.0,
        stress_flag             = False,
        critical_utilisation    = False,
        escalating_delinquency  = False,
        high_delinquency_score  = False,
        behavioural_risk_band   = MODULE_B_FALLBACK_RISK_BAND,
    )
