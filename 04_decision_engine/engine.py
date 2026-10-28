"""
engine.py - AI Credit Intelligence System (decision engine v2)
=============================================================
The core decision brain. Takes the aggregated composite signal and applies
a layered set of policy rules to produce the final lending decision.

Decision Framework (applied in strict order; the first rule that fires wins)
---------------------------------------------------------------------------
  1. Hard decline      composite PD at or above COMPOSITE_PD_DECLINE_MIN
  2. Hard override     Module B stress flags send the file to an analyst
  3. Concentration     Module C concentration flag with high market PD -> analyst
  4. Price check       inside the repricing band (PD < COMPOSITE_PD_REVIEW_MIN):
                         offered rate below breakeven, or Module C says Underpriced
                         -> REPRICE to the higher of the two rates,
                            or DECLINE if that rate is above MAX_LENDING_RATE
  5. Approve           PD inside the auto-approve band AND RAROC at the
                       offered rate meets the hurdle
  6. RAROC gate        value-destructive at the offered rate -> DECLINE
                       (thin-file applicants go to an analyst instead)
  7. Review band       everything else -> MANUAL REVIEW

What changed from v1
--------------------
  * RAROC is computed at the rate actually offered, on the composite PD.
    v1 used a fixed 12% revenue rate, so RAROC depended on PD alone.
  * Every route appends its own reason code, so an adverse-action notice is
    never empty.
  * Thin-file applicants carry no synthetic behavioural values.
  * Every threshold in reasons and narratives is read from constants.py.
  * Each non-approval states the smallest change that would move it.

Regulatory Context
------------------
Every decision is explainable (reason codes), auditable (audit_logger.py),
consistent (same inputs, same decision) and carries an adverse-action notice
in line with the RBI Fair Practices Code and ECOA/FCRA conventions.
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime

from constants import (
    ENGINE_VERSION,
    COMPOSITE_PD_APPROVE_MAX,
    COMPOSITE_PD_REVIEW_MIN,
    COMPOSITE_PD_DECLINE_MIN,
    HURDLE_RATE,
    HARD_OVERRIDE_FLAGS,
    REPRICE_ADEQUACY_TRIGGER,
    CONCENTRATION_PD_THRESHOLD,
    MAX_LENDING_RATE,
    THIN_FILE_REASON,
    SIGNAL_WEIGHTS,
    DECISION_APPROVE,
    DECISION_DECLINE,
    DECISION_REPRICE,
    DECISION_MANUAL_REVIEW,
    RISK_BAND_LABELS,
)
from signal_aggregator import deal_economics, round_rate_up


DECISION_METADATA = {
    DECISION_APPROVE:       {"emoji": "✅", "colour": "green",  "severity": 0},
    DECISION_REPRICE:       {"emoji": "💰", "colour": "yellow", "severity": 1},
    DECISION_MANUAL_REVIEW: {"emoji": "🔍", "colour": "orange", "severity": 2},
    DECISION_DECLINE:       {"emoji": "❌", "colour": "red",    "severity": 3},
}


def make_decision(composite_signal: dict) -> dict:
    """
    Apply the decision framework to a composite signal and return a complete
    lending decision with reason codes, deal economics and next steps.

    Parameters
    ----------
    composite_signal : Output of signal_aggregator.aggregate_signals().

    Returns
    -------
    dict with keys decision, decision_code, composite_pd, composite_risk_band,
    composite_risk_label, primary_reason, reason_codes, capital_impact, pricing,
    provisioning, signal_summary, overrides_triggered, what_would_change,
    decision_basis, market_pd_note, thin_file, engine_version, timestamp.
    """
    sig_a = composite_signal["signal_a"]
    sig_b = composite_signal.get("signal_b")
    sig_c = composite_signal.get("signal_c")
    sig_d = composite_signal.get("signal_d")
    composite_pd = composite_signal["composite_pd"]
    composite_risk_band = composite_signal["composite_risk_band"]
    thin_file = sig_b is None

    reason_codes: list[dict] = []
    overrides_triggered: list[str] = []
    what_would_change: list[str] = []

    # ── Deal economics at the offered rate, on the composite PD ──────────────
    offered = sig_a.get("offered_rate", 12.0) / 100.0
    econ = deal_economics(composite_pd, sig_a["ead"], offered, composite_risk_band)
    raroc = econ["raroc"]
    breakeven = econ["breakeven_rate"]
    if not sig_a.get("rate_supplied", False):
        reason_codes.append(_make_reason(
            "rate_assumed", "data",
            f"No offered rate supplied; the base rate of {offered:.2%} was assumed for pricing."
        ))

    # ── Data and flag checks ─────────────────────────────────────────────────
    if thin_file:
        reason_codes.append(_make_reason("thin_file", "data", THIN_FILE_REASON))
    else:
        for flag in HARD_OVERRIDE_FLAGS:
            if sig_b.get(flag, False):
                overrides_triggered.append(flag)
                reason_codes.append(_make_reason(flag, "hard_override", _override_description(flag)))

    concentration_review = bool(
        sig_c and sig_c["concentration_flag"] and sig_c["market_pd"] > CONCENTRATION_PD_THRESHOLD
    )
    if concentration_review:
        reason_codes.append(_make_reason(
            "concentration_risk", "portfolio",
            f"Loan adds portfolio concentration risk. Market-implied PD ({sig_c['market_pd']:.1%}) "
            f"exceeds the concentration threshold ({CONCENTRATION_PD_THRESHOLD:.0%})."
        ))

    # ── Pricing requirement ──────────────────────────────────────────────────
    c_says_under = bool(sig_c and sig_c["pricing_adequacy"] == REPRICE_ADEQUACY_TRIGGER)
    c_rate = (sig_c["suggested_rate"] / 100.0) if sig_c else 0.0
    under_by_econ = offered < breakeven
    under_by_c = c_says_under and offered < c_rate
    required_rate = round_rate_up(max(breakeven, c_rate if c_says_under else 0.0))

    # ── Routing ──────────────────────────────────────────────────────────────
    if composite_pd >= COMPOSITE_PD_DECLINE_MIN:
        decision = DECISION_DECLINE
        primary_reason = (
            f"Composite probability of default ({composite_pd:.1%}) is at or above the "
            f"hard decline threshold ({COMPOSITE_PD_DECLINE_MIN:.0%})."
        )
        reason_codes.insert(0, _make_reason("composite_pd_decline", "pd_threshold", primary_reason))

    elif overrides_triggered:
        decision = DECISION_MANUAL_REVIEW
        primary_reason = _override_description(overrides_triggered[0])

    elif concentration_review:
        decision = DECISION_MANUAL_REVIEW
        primary_reason = "Concentration risk needs portfolio sign-off before this loan is booked."

    elif composite_pd < COMPOSITE_PD_REVIEW_MIN and (under_by_econ or under_by_c):
        if required_rate > MAX_LENDING_RATE:
            decision = DECISION_DECLINE
            primary_reason = (
                f"The rate needed to meet the {HURDLE_RATE:.0%} hurdle ({required_rate:.2%}) is above "
                f"the {MAX_LENDING_RATE:.0%} policy rate ceiling."
            )
            reason_codes.append(_make_reason("unpriceable", "capital", primary_reason))
        else:
            decision = DECISION_REPRICE
            primary_reason = (
                f"Reprice from {offered:.2%} to {required_rate:.2%}. At the offered rate RAROC is "
                f"{raroc:.1%} against a {HURDLE_RATE:.0%} hurdle."
            )
            reason_codes.append(_make_reason(
                "underpriced", "pricing",
                f"Offered rate {offered:.2%} is below the breakeven rate {breakeven:.2%} "
                f"(the rate at which RAROC meets the {HURDLE_RATE:.0%} hurdle)."
                if under_by_econ else
                f"Module C predicts grade {sig_c['loan_grade_signal']}, which prices at "
                f"{c_rate:.2%}; offered {offered:.2%}."
            ))

    elif composite_pd < COMPOSITE_PD_APPROVE_MAX and raroc >= HURDLE_RATE:
        decision = DECISION_APPROVE
        primary_reason = (
            f"Composite PD ({composite_pd:.1%}) is inside the auto-approve band "
            f"(below {COMPOSITE_PD_APPROVE_MAX:.0%}) and RAROC at the offered rate "
            f"({raroc:.1%}) meets the {HURDLE_RATE:.0%} hurdle."
        )
        reason_codes.append(_make_reason("pd_low_raroc_adequate", "approval", primary_reason))

    elif raroc < HURDLE_RATE and not thin_file:
        decision = DECISION_DECLINE
        primary_reason = (
            f"RAROC at the offered rate ({raroc:.1%}) is below the {HURDLE_RATE:.0%} hurdle, "
            f"and composite PD ({composite_pd:.1%}) is outside the repricing band "
            f"(below {COMPOSITE_PD_REVIEW_MIN:.0%})."
        )
        reason_codes.append(_make_reason("raroc_below_hurdle", "capital", primary_reason))

    else:
        decision = DECISION_MANUAL_REVIEW
        if thin_file:
            primary_reason = (
                "Thin-file applicant outside the auto-approve band. An analyst should verify "
                "income and bank statements before deciding."
            )
        else:
            primary_reason = (
                f"Composite PD ({composite_pd:.1%}) is in the manual review band "
                f"({COMPOSITE_PD_APPROVE_MAX:.0%} to {COMPOSITE_PD_DECLINE_MIN:.0%}). "
                f"Human judgement required."
            )
        reason_codes.append(_make_reason("pd_review_band", "review", primary_reason))

    # ── What would change this decision ──────────────────────────────────────
    if decision != DECISION_APPROVE:
        what_would_change = _counterfactuals(
            decision, composite_signal, raroc, required_rate, overrides_triggered
        )

    # ── IFRS 9 provisioning view (advisory, never gates origination) ─────────
    provisioning = _build_provisioning(sig_d)
    if sig_d and (sig_d.get("ifrs9_stage", 1) >= 2 or sig_d.get("sicr_flag")):
        reason_codes.append(_make_reason(
            "ifrs9_provisioning", "provisioning",
            f"Expected IFRS 9 Stage {sig_d['ifrs9_stage']}: provision lifetime ECL "
            f"{sig_d['lifetime_ecl_rate']:.2%} rather than 12-month "
            f"{sig_d['twelve_month_ecl_rate']:.2%}."
        ))

    capital_impact = {
        "ead":              sig_a["ead"],
        "el":               econ["el"],
        "el_rate":          econ["el_rate"],
        "economic_capital": econ["economic_capital"],
        "net_income":       econ["net_income"],
        "raroc":            raroc,
        "above_hurdle":     raroc >= HURDLE_RATE,
        "offered_rate":     offered * 100.0,
        "breakeven_rate":   breakeven * 100.0,
        "value_decision":   (
            "Value Accretive" if raroc >= HURDLE_RATE else
            "Marginal"        if raroc >= HURDLE_RATE * 0.85 else
            "Value Destructive"
        ),
    }

    pricing = {
        "current_rate_pct": offered * 100.0,
        "breakeven_rate":   breakeven * 100.0,
        "suggested_rate":   (required_rate * 100.0 if decision == DECISION_REPRICE
                             else (sig_c["suggested_rate"] if sig_c else None)),
        "pricing_adequacy": sig_c["pricing_adequacy"] if sig_c else "Unknown",
        "loan_grade":       sig_c["loan_grade_signal"] if sig_c else "N/A",
        "market_pd":        sig_c["market_pd"] if sig_c else None,
    }

    market_pd_note = None
    div = composite_signal.get("market_pd_divergence")
    if div and div.get("flag"):
        market_pd_note = (
            f"Note: internal composite PD ({div['composite_pd']:.1%}) diverges from the "
            f"market-implied PD ({div['market_pd']:.1%}) by {div['divergence']:.1%}. "
            f"This warrants additional review."
        )

    decision_basis = _build_decision_narrative(
        decision, composite_pd, reason_codes, sig_a, sig_b, sig_c,
        overrides_triggered, econ, required_rate, what_would_change,
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
        "provisioning":         provisioning,
        "signal_summary":       _build_signal_summary(composite_signal),
        "overrides_triggered":  overrides_triggered,
        "what_would_change":    what_would_change,
        "decision_basis":       decision_basis,
        "market_pd_note":       market_pd_note,
        "thin_file":            thin_file,
        "engine_version":       ENGINE_VERSION,
        "timestamp":            datetime.now().isoformat(),
    }


# ── Helper Functions ──────────────────────────────────────────────────────────

def _counterfactuals(decision, composite_signal, raroc, required_rate, overrides) -> list[str]:
    """The smallest change to each driver that would move the decision."""
    out = []
    composite_pd = composite_signal["composite_pd"]
    sig_b = composite_signal.get("signal_b")
    w = composite_signal.get("weights_used", SIGNAL_WEIGHTS)
    w_a, w_b = w.get("module_a", 1.0), w.get("module_b", 0.0)

    if composite_pd >= COMPOSITE_PD_DECLINE_MIN:
        out.append(f"Composite PD would need to fall below {COMPOSITE_PD_DECLINE_MIN:.0%} "
                   f"to leave the decline zone.")
    if COMPOSITE_PD_APPROVE_MAX <= composite_pd < COMPOSITE_PD_DECLINE_MIN:
        b_part = w_b * sig_b["delinquency_prob"] if sig_b else 0.0
        need_a = (COMPOSITE_PD_APPROVE_MAX - b_part) / w_a if w_a > 0 else -1
        if need_a > 0:
            out.append(f"An application PD below {need_a:.1%} would bring composite PD under "
                       f"{COMPOSITE_PD_APPROVE_MAX:.0%}.")
        else:
            out.append(f"The behavioural PD alone keeps composite PD above {COMPOSITE_PD_APPROVE_MAX:.0%}.")
    if (raroc < HURDLE_RATE or decision == DECISION_REPRICE) and required_rate <= MAX_LENDING_RATE \
            and composite_pd < COMPOSITE_PD_DECLINE_MIN:
        out.append(f"A rate of {required_rate:.2%} meets the {HURDLE_RATE:.0%} hurdle at this PD.")
    if overrides:
        out.append("Clearing the behavioural flags removes the hard override.")
    return out


def _build_provisioning(sig_d: Optional[dict]) -> dict:
    """IFRS 9 provisioning view from the Module D signal (advisory, not a gate)."""
    if not sig_d:
        return {"available": False, "note": "Module D provisioning signal not supplied"}
    stage = sig_d.get("ifrs9_stage", 1)
    label = {
        1: "Stage 1 (performing, 12-month ECL)",
        2: "Stage 2 (SICR, lifetime ECL)",
        3: "Stage 3 (credit-impaired, lifetime ECL)",
    }.get(stage, "Unknown")
    return {
        "available":             True,
        "ifrs9_stage":           stage,
        "stage_label":           label,
        "twelve_month_ecl_rate": sig_d.get("twelve_month_ecl_rate"),
        "lifetime_ecl_rate":     sig_d.get("lifetime_ecl_rate"),
        "stage2_cliff_multiple": sig_d.get("stage2_cliff_multiple"),
        "sicr_flag":             sig_d.get("sicr_flag", False),
        "lgd_basis":             "Module D measured LGD (41.8%, Freddie Mac panel)",
    }


def _make_reason(code: str, category: str, description: str) -> dict:
    """Create a standardised reason code dictionary."""
    return {"code": code, "category": category, "description": description}


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
    sig_b = composite_signal.get("signal_b")
    sig_c = composite_signal.get("signal_c")

    if sig_b:
        flagged = any(sig_b.get(k) for k in ("stress_flag", "critical_utilisation", "high_delinquency_score"))
        b_line = (
            f"Delinq PD={sig_b['delinquency_prob']:.1%} | "
            f"BRS={sig_b['behavioural_risk_score']:.0f}/100 | "
            f"Band={sig_b['behavioural_risk_band']} | "
            f"Flags={'⚠ ' if flagged else '✓ clean'}"
        )
    else:
        b_line = "No behavioural history (thin file) - not used"

    return {
        "module_a": (
            f"PD={sig_a['pd_score']:.1%} | Score={sig_a['credit_score']} | "
            f"Band={sig_a['risk_band']} ({sig_a['risk_grade']}) | "
            f"Offered={sig_a.get('offered_rate', 12.0):.2f}%"
        ),
        "module_b": b_line,
        "module_c": (
            f"Market PD={sig_c['market_pd']:.1%} | Grade={sig_c['loan_grade_signal']} | "
            f"Rate={sig_c['suggested_rate']:.2f}% | Pricing={sig_c['pricing_adequacy']} | "
            f"Conc={'⚠' if sig_c['concentration_flag'] else '✓'}"
        ) if sig_c else "Not available",
    }


def _build_decision_narrative(
    decision, composite_pd, reason_codes, sig_a, sig_b, sig_c,
    overrides, econ, required_rate, what_would_change,
) -> str:
    """Plain-English decision narrative, structured for adverse-action notices."""
    lines = [f"DECISION: {decision}", "", "RISK ASSESSMENT SUMMARY"]
    lines.append(f"  Composite Probability of Default : {composite_pd:.2%}")
    lines.append(f"  Module A (Application Risk) PD  : {sig_a['pd_score']:.2%}")
    lines.append("  Module B (Behavioural Risk) PD  : "
                 + (f"{sig_b['delinquency_prob']:.2%}" if sig_b else "not available (thin file)"))
    if sig_c:
        lines.append(f"  Module C (Market-Implied) PD    : {sig_c['market_pd']:.2%}")
    lines.append(f"  Credit Score (Module A)         : {sig_a['credit_score']} / 900")
    if sig_b:
        lines.append(f"  Behavioural Risk Score          : {sig_b['behavioural_risk_score']:.0f} / 100")
    lines.append(f"  Offered rate                    : {econ['rate']:.2%}")
    lines.append(f"  RAROC at offered rate           : {econ['raroc']:.2%}")
    lines.append(f"  Breakeven rate (meets hurdle)   : {econ['breakeven_rate']:.2%}")
    lines.append("")

    if overrides:
        lines.append("HARD OVERRIDE FLAGS TRIGGERED")
        lines.extend(f"  ⚠  {_override_description(f)}" for f in overrides)
        lines.append("")

    lines.append("PRIMARY REASON CODES")
    for i, rc in enumerate(reason_codes[:3], 1):
        lines.append(f"  {i}. [{rc['category'].upper()}] {rc['description']}")
    lines.append("")

    if decision == DECISION_APPROVE:
        lines.append("APPROVAL BASIS")
        lines.append(
            f"  Composite PD of {composite_pd:.2%} is within the automatic approval band "
            f"(below {COMPOSITE_PD_APPROVE_MAX:.0%}). RAROC of {econ['raroc']:.2%} at the offered rate "
            f"meets the {HURDLE_RATE:.0%} hurdle, so this loan creates shareholder value."
        )
    elif decision == DECISION_DECLINE:
        lines.append("DECLINE BASIS")
        lines.append(
            "  This application does not meet the minimum credit standards for approval. "
            "The reasons above are the principal reasons for this decision."
        )
        lines.append("")
        lines.append("ADVERSE ACTION NOTICE (RBI Fair Practices Code / ECOA / FCRA)")
        lines.append("  The primary reasons for this decision are listed above.")
        lines.append("  The applicant may request a copy of this report within 60 days.")
    elif decision == DECISION_REPRICE:
        lines.append("REPRICE BASIS")
        lines.append(
            f"  The risk profile is acceptable, but the offered rate of {econ['rate']:.2%} does not "
            f"compensate for the risk. A rate of {required_rate:.2%} p.a. meets the institution's "
            f"risk-return requirement."
        )
    else:
        lines.append("MANUAL REVIEW BASIS")
        lines.append(
            "  This application needs human judgement. The models identified risk factors "
            "that do not resolve to an automatic approve or decline. Review the flagged items above."
        )

    if what_would_change:
        lines.append("")
        lines.append("WHAT WOULD CHANGE THIS DECISION")
        lines.extend(f"  - {m}" for m in what_would_change)

    return "\n".join(lines)
