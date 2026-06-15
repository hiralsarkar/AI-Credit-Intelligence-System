"""
engine.py — AI Credit Intelligence System
=======================================
The core decision brain. Takes the aggregated composite signal and applies
a layered set of policy rules to produce the final lending decision.

Decision Framework
------------------
The engine applies rules in strict order. Earlier rules take precedence.

  Layer 1 — Hard Overrides (stress flags from Module B)
    Certain behavioural patterns override all quantitative scores.
    These represent tail risks the model underestimates.

  Layer 2 — Composite PD hard limits
    Decline if composite PD exceeds the hard ceiling.
    Approve directly if composite PD is very low AND RAROC above hurdle.

  Layer 3 — Module C routing (pricing and concentration)
    Reprice if underpriced but otherwise acceptable.
    Manual Review if concentration + high market PD.

  Layer 4 — RAROC gate
    Decline if RAROC below hurdle (loan is value-destructive).

  Layer 5 — Default path
    Remaining cases routed to Manual Review for human judgement.

Regulatory Context
------------------
This decision framework mirrors the credit committee process at a regulated
Indian lender under RBI guidelines. Every decision is:
  - Explainable (reason codes, not black-box scores)
  - Auditable (full input/output logged by audit_logger.py)
  - Consistent (same inputs always produce same output)
  - Compliant (ECOA/FCRA-style adverse action notice generation)
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime

from constants import (
    COMPOSITE_PD_APPROVE_MAX,
    COMPOSITE_PD_REVIEW_MIN,
    COMPOSITE_PD_DECLINE_MIN,
    HURDLE_RATE,
    HARD_OVERRIDE_FLAGS,
    REPRICE_ADEQUACY_TRIGGER,
    CONCENTRATION_PD_THRESHOLD,
    MARKET_PD_DIVERGENCE_THRESHOLD,
    DECISION_APPROVE,
    DECISION_DECLINE,
    DECISION_REPRICE,
    DECISION_MANUAL_REVIEW,
    RISK_BAND_LABELS,
)


# ── Decision Colours (for display formatting) ─────────────────────────────────
DECISION_METADATA = {
    DECISION_APPROVE:       {"emoji": "✅", "colour": "green",  "severity": 0},
    DECISION_REPRICE:       {"emoji": "💰", "colour": "yellow", "severity": 1},
    DECISION_MANUAL_REVIEW: {"emoji": "🔍", "colour": "orange", "severity": 2},
    DECISION_DECLINE:       {"emoji": "❌", "colour": "red",    "severity": 3},
}


def make_decision(composite_signal: dict) -> dict:
    """
    Apply the full decision framework to a composite signal and return
    a complete lending decision with reason codes and capital impact.

    Parameters
    ----------
    composite_signal : Output of signal_aggregator.aggregate_signals().
                       Must contain signal_a, signal_b, signal_c keys.

    Returns
    -------
    dict with keys:
        decision            : str — APPROVE / DECLINE / REPRICE / MANUAL REVIEW
        decision_code       : str — short code for logging
        composite_pd        : float — the composite probability of default
        composite_risk_band : int
        composite_risk_label: str
        primary_reason      : str — the single most important reason
        reason_codes        : list[dict] — ordered list of decision reasons
        capital_impact      : dict — EL, RWA, RAROC, economic_capital
        pricing             : dict — current rate, suggested rate (if reprice)
        signal_summary      : dict — one-line summary of each module's signal
        overrides_triggered : list — any hard override flags that fired
        decision_basis      : str — full plain-English explanation
        market_pd_note      : str | None — note if market PD diverges from composite
        timestamp           : str — ISO format decision timestamp
    """
    sig_a = composite_signal["signal_a"]
    sig_b = composite_signal["signal_b"]
    sig_c = composite_signal.get("signal_c")
    composite_pd = composite_signal["composite_pd"]
    composite_risk_band = composite_signal["composite_risk_band"]

    reason_codes      = []
    overrides_triggered = []

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 1 — Hard Override Checks (Module B stress flags)
    # ─────────────────────────────────────────────────────────────────────────
    for flag in HARD_OVERRIDE_FLAGS:
        if sig_b.get(flag, False):
            overrides_triggered.append(flag)

    if overrides_triggered:
        reason_codes.extend([
            _make_reason(flag, "hard_override", _override_description(flag))
            for flag in overrides_triggered
        ])
        # Still continue to collect all reasons — don't return early.
        # The override fires Manual Review in the routing logic below.

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 2 — Composite PD Hard Limits
    # ─────────────────────────────────────────────────────────────────────────
    if composite_pd >= COMPOSITE_PD_DECLINE_MIN:
        reason_codes.append(_make_reason(
            "composite_pd_decline",
            "pd_threshold",
            f"Composite probability of default ({composite_pd:.1%}) exceeds "
            f"hard decline threshold ({COMPOSITE_PD_DECLINE_MIN:.0%})"
        ))

    # Low-PD conditions
    low_pd        = composite_pd < COMPOSITE_PD_APPROVE_MAX
    above_hurdle  = sig_a.get("above_hurdle", sig_a.get("raroc", 0) >= HURDLE_RATE)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 3 — Module C Routing (pricing and concentration)
    # ─────────────────────────────────────────────────────────────────────────
    reprice_triggered      = False
    concentration_review   = False
    market_pd_divergence_flag = False

    if sig_c:
        if sig_c["pricing_adequacy"] == REPRICE_ADEQUACY_TRIGGER:
            reprice_triggered = True
            reason_codes.append(_make_reason(
                "underpriced",
                "pricing",
                f"Proposed rate is below the risk-appropriate level. "
                f"Suggested rate: {sig_c['suggested_rate']:.2f}% p.a. "
                f"(based on predicted grade {sig_c['loan_grade_signal']})"
            ))

        if sig_c["concentration_flag"] and sig_c["market_pd"] > CONCENTRATION_PD_THRESHOLD:
            concentration_review = True
            reason_codes.append(_make_reason(
                "concentration_risk",
                "portfolio",
                f"Loan adds portfolio concentration risk. "
                f"Market-implied PD ({sig_c['market_pd']:.1%}) exceeds "
                f"concentration threshold ({CONCENTRATION_PD_THRESHOLD:.0%})"
            ))

        div = composite_signal.get("market_pd_divergence", {})
        if div and div.get("flag", False):
            market_pd_divergence_flag = True

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 4 — RAROC Gate
    # ─────────────────────────────────────────────────────────────────────────
    raroc = sig_a.get("raroc", 0)
    if raroc < HURDLE_RATE and composite_pd >= COMPOSITE_PD_APPROVE_MAX:
        reason_codes.append(_make_reason(
            "raroc_below_hurdle",
            "capital",
            f"Risk-adjusted return on capital ({raroc:.1%}) is below the "
            f"hurdle rate ({HURDLE_RATE:.0%}). Loan is not value-accretive."
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTING LOGIC — Apply layers in priority order
    # ─────────────────────────────────────────────────────────────────────────

    # Priority 1: Hard decline (extreme PD)
    if composite_pd >= COMPOSITE_PD_DECLINE_MIN:
        decision = DECISION_DECLINE
        primary_reason = reason_codes[0]["description"] if reason_codes else "PD above threshold"

    # Priority 2: Hard override → Manual Review
    elif overrides_triggered:
        decision = DECISION_MANUAL_REVIEW
        primary_reason = _override_description(overrides_triggered[0])

    # Priority 3: Concentration + high market PD → Manual Review
    elif concentration_review:
        decision = DECISION_MANUAL_REVIEW
        primary_reason = reason_codes[-1]["description"]

    # Priority 4: Underpriced → Reprice (only if PD is in acceptable range)
    elif reprice_triggered and composite_pd < COMPOSITE_PD_REVIEW_MIN:
        decision = DECISION_REPRICE
        primary_reason = f"Loan is acceptable but requires repricing to {sig_c['suggested_rate']:.2f}%"

    # Priority 5: Clean approve (low PD + above hurdle)
    elif low_pd and above_hurdle:
        decision = DECISION_APPROVE
        reason_codes.append(_make_reason(
            "pd_low_raroc_adequate",
            "approval",
            f"Composite PD ({composite_pd:.1%}) within approval band. "
            f"RAROC ({raroc:.1%}) above hurdle rate ({HURDLE_RATE:.0%})."
        ))
        primary_reason = reason_codes[-1]["description"]

    # Priority 6: RAROC below hurdle (but PD not in hard decline zone)
    elif raroc < HURDLE_RATE:
        decision = DECISION_DECLINE
        primary_reason = f"RAROC ({raroc:.1%}) below hurdle rate ({HURDLE_RATE:.0%})"

    # Priority 7: Medium PD zone → Manual Review
    else:
        decision = DECISION_MANUAL_REVIEW
        reason_codes.append(_make_reason(
            "pd_review_band",
            "review",
            f"Composite PD ({composite_pd:.1%}) is in the manual review band "
            f"({COMPOSITE_PD_APPROVE_MAX:.0%}-{COMPOSITE_PD_DECLINE_MIN:.0%}). "
            f"Human judgement required."
        ))
        primary_reason = reason_codes[-1]["description"]

    # ─────────────────────────────────────────────────────────────────────────
    # Build output
    # ─────────────────────────────────────────────────────────────────────────
    capital_impact = {
        "ead":              sig_a.get("ead", 0),
        "el":               sig_a.get("el", 0),
        "el_rate":          sig_a.get("el_rate", 0),
        "economic_capital": sig_a.get("economic_capital", 0),
        "raroc":            raroc,
        "above_hurdle":     above_hurdle,
        "value_decision":   (
            "Value Accretive" if raroc >= HURDLE_RATE else
            "Marginal"        if raroc >= HURDLE_RATE * 0.85 else
            "Value Destructive"
        ),
    }

    pricing = {
        "current_rate_pct": None,   # Populated by caller if known
        "suggested_rate":   sig_c["suggested_rate"] if sig_c else None,
        "pricing_adequacy": sig_c["pricing_adequacy"] if sig_c else "Unknown",
        "loan_grade":       sig_c["loan_grade_signal"] if sig_c else "N/A",
        "market_pd":        sig_c["market_pd"] if sig_c else None,
    }

    signal_summary = _build_signal_summary(composite_signal)

    market_pd_note = None
    if market_pd_divergence_flag and sig_c:
        div = composite_signal["market_pd_divergence"]
        market_pd_note = (
            f"Note: Internal model PD ({div['composite_pd']:.1%}) diverges from "
            f"market-implied PD ({div['market_pd']:.1%}) by "
            f"{div['divergence']:.1%}. This warrants additional review."
        )

    decision_basis = _build_decision_narrative(
        decision, composite_pd, reason_codes, sig_a, sig_b, sig_c,
        overrides_triggered, composite_signal.get("b_available", True)
    )

    return {
        "decision":             decision,
        "decision_code":        decision.upper().replace(" ", "_"),
        "decision_metadata":    DECISION_METADATA.get(decision, {}),
        "composite_pd":         round(composite_pd, 4),
        "composite_risk_band":  composite_risk_band,
        "composite_risk_label": RISK_BAND_LABELS.get(composite_risk_band, "Unknown"),
        "primary_reason":       primary_reason,
        "reason_codes":         reason_codes,
        "capital_impact":       capital_impact,
        "pricing":              pricing,
        "signal_summary":       signal_summary,
        "overrides_triggered":  overrides_triggered,
        "decision_basis":       decision_basis,
        "market_pd_note":       market_pd_note,
        "timestamp":            datetime.now().isoformat(),
    }


# ── Helper Functions ──────────────────────────────────────────────────────────

def _make_reason(code: str, category: str, description: str) -> dict:
    """Create a standardised reason code dictionary."""
    return {
        "code":        code,
        "category":    category,
        "description": description,
    }


def _override_description(flag: str) -> str:
    """Map a hard override flag name to a plain-English description."""
    descriptions = {
        "high_delinquency_score":  "Severe repeated delinquency pattern (5+ events in 2 years)",
        "critical_utilisation":    "Credit card utilisation at critical level (80%+ of limit)",
        "escalating_delinquency":  "Escalating delinquency severity (60+ or 90+ day late events on record)",
    }
    return descriptions.get(flag, f"Behavioural risk flag: {flag}")


def _build_signal_summary(composite_signal: dict) -> dict:
    """Build a concise one-line summary of each module's contribution."""
    sig_a = composite_signal["signal_a"]
    sig_b = composite_signal["signal_b"]
    sig_c = composite_signal.get("signal_c")

    return {
        "module_a": (
            f"PD={sig_a['pd_score']:.1%} | "
            f"Score={sig_a['credit_score']} | "
            f"Band={sig_a['risk_band']} ({sig_a['risk_grade']}) | "
            f"RAROC={sig_a['raroc']:.1%}"
        ),
        "module_b": (
            f"Delinq PD={sig_b['delinquency_prob']:.1%} | "
            f"BRS={sig_b['behavioural_risk_score']:.0f}/100 | "
            f"Band={sig_b['behavioural_risk_band']} | "
            f"Flags={'⚠ ' if any([sig_b.get('stress_flag'), sig_b.get('critical_utilisation'), sig_b.get('high_delinquency_score')]) else '✓ clean'}"
        ),
        "module_c": (
            f"Market PD={sig_c['market_pd']:.1%} | "
            f"Grade={sig_c['loan_grade_signal']} | "
            f"Rate={sig_c['suggested_rate']:.2f}% | "
            f"Pricing={sig_c['pricing_adequacy']} | "
            f"Conc={'⚠' if sig_c['concentration_flag'] else '✓'}"
        ) if sig_c else "Not available",
    }


def _build_decision_narrative(
    decision: str,
    composite_pd: float,
    reason_codes: list,
    sig_a: dict,
    sig_b: dict,
    sig_c: Optional[dict],
    overrides: list,
    b_available: bool,
) -> str:
    """
    Generate a full plain-English decision narrative.

    This is what a credit analyst or borrower would read to understand
    why a decision was made. Structured to satisfy regulatory adverse
    action notice requirements.
    """
    lines = []
    lines.append(f"DECISION: {decision}")
    lines.append("")

    lines.append("RISK ASSESSMENT SUMMARY")
    lines.append(f"  Composite Probability of Default : {composite_pd:.2%}")
    lines.append(f"  Module A (Application Risk) PD  : {sig_a['pd_score']:.2%}")
    lines.append(f"  Module B (Behavioural Risk) PD  : {sig_b['delinquency_prob']:.2%}"
                 + (" [estimated — no behavioural history]" if not b_available else ""))
    if sig_c:
        lines.append(f"  Module C (Market-Implied) PD    : {sig_c['market_pd']:.2%}")
    lines.append(f"  Credit Score (Module A)         : {sig_a['credit_score']} / 900")
    lines.append(f"  Behavioural Risk Score          : {sig_b['behavioural_risk_score']:.0f} / 100")
    lines.append(f"  RAROC                           : {sig_a['raroc']:.2%}")
    lines.append("")

    if overrides:
        lines.append("HARD OVERRIDE FLAGS TRIGGERED")
        for flag in overrides:
            lines.append(f"  ⚠  {_override_description(flag)}")
        lines.append("")

    lines.append("PRIMARY REASON CODES")
    for i, rc in enumerate(reason_codes[:3], 1):
        lines.append(f"  {i}. [{rc['category'].upper()}] {rc['description']}")
    lines.append("")

    # Decision-specific narrative
    if decision == DECISION_APPROVE:
        lines.append("APPROVAL BASIS")
        lines.append(
            f"  Composite PD of {composite_pd:.2%} is within the automatic approval band "
            f"(<{0.10:.0%}). RAROC of {sig_a['raroc']:.2%} exceeds the hurdle rate of 14%, "
            f"confirming this loan creates shareholder value. "
            f"No adverse behavioural flags are present."
        )
    elif decision == DECISION_DECLINE:
        lines.append("DECLINE BASIS")
        lines.append(
            f"  This application does not meet the minimum credit standards required "
            f"for approval. The composite PD of {composite_pd:.2%} and/or the "
            f"risk-adjusted return ({sig_a['raroc']:.2%} RAROC) do not satisfy "
            f"the institution's credit policy requirements."
        )
        lines.append("")
        lines.append("ADVERSE ACTION NOTICE (ECOA/FCRA / RBI Fair Practices Code)")
        lines.append("  The primary reasons for this decision are listed above.")
        lines.append("  The applicant may request a copy of this report within 60 days.")
    elif decision == DECISION_REPRICE:
        lines.append("REPRICE BASIS")
        lines.append(
            f"  This applicant's risk profile is acceptable, but the proposed "
            f"interest rate does not adequately compensate for the credit risk. "
            f"A rate of {sig_c['suggested_rate']:.2f}% p.a. is required for this loan "
            f"to meet the institution's risk-return requirements."
        )
    elif decision == DECISION_MANUAL_REVIEW:
        lines.append("MANUAL REVIEW BASIS")
        lines.append(
            f"  This application requires human judgement. The quantitative models "
            f"have identified risk factors that do not resolve to an automatic "
            f"approve or decline. A credit analyst should review the full application "
            f"file, particularly the flagged items above."
        )

    return "\n".join(lines)
