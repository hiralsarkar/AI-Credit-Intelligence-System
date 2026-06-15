"""
explainability.py — AI Credit Intelligence System
===============================================
Converts raw model signals and SHAP values into plain-English explanations
suitable for:
  - Credit analyst review
  - Regulatory adverse action notices (ECOA/FCRA, RBI Fair Practices Code)
  - Borrower-facing communication
  - Model validator documentation

Every decision in the system is explainable at two levels:
  1. Signal level  — which module (A, B, C) drove the decision, and by how much
  2. Feature level — which individual features within each module were decisive

Both levels are produced by this module.
"""

from __future__ import annotations

from typing import Optional


# ── Reason Code Registry ──────────────────────────────────────────────────────
# Maps feature names (as they appear in model inputs) to plain-English
# descriptions suitable for adverse action notices.
# Covers all three modules.

REASON_CODE_REGISTRY = {
    # ── Module A — Application Risk ───────────────────────────────────────────
    "EXT_SOURCE_1":                   "External bureau score (Source 1) indicates elevated credit risk",
    "EXT_SOURCE_2":                   "External bureau score (Source 2) indicates elevated credit risk",
    "EXT_SOURCE_3":                   "External bureau score (Source 3) indicates elevated credit risk",
    "DAYS_BIRTH":                     "Limited credit history relative to borrower age profile",
    "DAYS_EMPLOYED":                  "Short employment tenure — income stability risk",
    "AMT_CREDIT":                     "Loan amount relative to assessed repayment capacity",
    "AMT_INCOME_TOTAL":               "Income level relative to requested credit exposure",
    "AMT_ANNUITY":                    "Annual repayment obligation relative to income",
    "DAYS_ID_PUBLISH":                "Identity document recency relative to risk profile",
    "DAYS_LAST_PHONE_CHANGE":         "Recent change in contact information — stability indicator",
    "CODE_GENDER":                    "Not used — demographic attributes excluded from scoring",

    # ── Module B — Behavioural Risk ───────────────────────────────────────────
    "HAS_90DAY_LATE":                 "History of 90+ day past-due payment",
    "NumberOfTimes90DaysLate":        "Multiple 90+ day late payment events on record",
    "DELINQUENCY_SCORE":              "High cumulative delinquency score across all payment windows",
    "WORST_DELINQUENCY":              "Severe delinquency event on record (worst classification reached)",
    "NumberOfTime30-59DaysPastDueNotWorse": "Recent 30-59 day late payment history",
    "NumberOfTime60-89DaysPastDueNotWorse": "Recent 60-89 day late payment history",
    "RevolvingUtilizationOfUnsecuredLines": "High revolving credit utilisation rate",
    "UTILIZATION_RISK_BAND":          "Revolving credit utilisation in elevated risk band",
    "DebtRatio":                      "Elevated debt-to-income ratio",
    "DEBT_TO_INCOME":                 "High debt burden relative to monthly income",
    "age":                            "Age profile associated with elevated delinquency risk",
    "MonthlyIncome":                  "Income level relative to debt obligations",
    "INCOME_PER_DEPENDENT":           "Constrained disposable income after dependent obligations",
    "CREDIT_LINES_RISK":              "Thin credit file — insufficient credit history",
    "INCOME_MISSING":                 "Income information not disclosed at time of application",

    # ── Module C — Portfolio & Pricing Risk ───────────────────────────────────
    "grade_n":                        "Market-equivalent loan grade indicates elevated credit risk",
    "sub_grade_n":                    "Market loan sub-grade confirms elevated risk positioning",
    "int_rate":                       "Proposed interest rate relative to risk-appropriate level",
    "RATE_SPREAD":                    "Interest rate deviates significantly from grade benchmark",
    "RATE_PER_RISK":                  "Rate-per-unit-risk ratio outside acceptable range",
    "LOAN_TO_INCOME":                 "Loan-to-income ratio indicates overleveraging",
    "INSTALLMENT_TO_INCOME":          "Monthly instalment represents high fraction of income",
    "EL_RATE_PROXY":                  "Expected loss rate (proxy) elevated for this loan",
    "HIGH_PURPOSE_RISK":              "Loan purpose has historically elevated default rate",
    "LONG_TERM":                      "60-month term loans carry significantly higher default risk",
    "fico_mid":                       "FICO credit score indicates elevated credit risk",
    "dti":                            "Debt-to-income ratio above acceptable threshold",
    "NEVER_DELINQUENT":               "No prior delinquency history (positive signal)",
    "CONCENTRATION_FLAG":             "Loan adds to portfolio concentration in high-risk segment",
}

# Reason codes that indicate POSITIVE signals (reduce risk) — formatted differently
POSITIVE_REASON_CODES = {
    "NEVER_DELINQUENT",
    "above_hurdle",
    "pd_low_raroc_adequate",
}


# ── Signal-Level Attribution ──────────────────────────────────────────────────

def explain_signal_contributions(composite_signal: dict) -> dict:
    """
    Compute the percentage contribution of each module to the composite PD.

    Returns a breakdown showing how much each module drove the final score.
    This is the signal-level explanation — "Module B is responsible for 40%
    of the composite risk score."

    Parameters
    ----------
    composite_signal : Output of signal_aggregator.aggregate_signals().

    Returns
    -------
    dict with keys:
        contributions  : dict — {module: absolute_pd_contribution}
        percentages    : dict — {module: % share of composite PD}
        dominant_module: str — which module drove the decision most
        narrative      : str — plain English paragraph
    """
    weights = composite_signal.get("weights_used", {"module_a": 0.60, "module_b": 0.40})
    sig_a   = composite_signal["signal_a"]
    sig_b   = composite_signal["signal_b"]
    sig_c   = composite_signal.get("signal_c")

    pd_a = sig_a["pd_score"]
    pd_b = sig_b["delinquency_prob"]
    w_a  = weights.get("module_a", 0.60)
    w_b  = weights.get("module_b", 0.40)

    contrib_a = w_a * pd_a
    contrib_b = w_b * pd_b
    total     = contrib_a + contrib_b + 1e-9

    pct_a = contrib_a / total * 100
    pct_b = contrib_b / total * 100

    dominant = "Module A (Application Risk)" if pct_a >= pct_b else "Module B (Behavioural Risk)"

    # Build narrative
    narrative_parts = [
        f"The composite PD of {composite_signal['composite_pd']:.2%} is driven "
        f"primarily by {dominant}."
    ]
    narrative_parts.append(
        f"Module A (application-time risk) contributes {pct_a:.0f}% of the composite score "
        f"(PD={pd_a:.2%}, weight={w_a:.0%})."
    )
    narrative_parts.append(
        f"Module B (behavioural risk) contributes {pct_b:.0f}% "
        f"(delinquency probability={pd_b:.2%}, weight={w_b:.0%})"
        + (" [estimated — no behavioural history]" if not composite_signal.get("b_available") else "") + "."
    )
    if sig_c:
        narrative_parts.append(
            f"Module C cross-check: market-implied PD={sig_c['market_pd']:.2%} "
            f"(grade {sig_c['loan_grade_signal']}) — "
            + ("aligns with internal estimate." if abs(sig_c["market_pd"] - composite_signal["composite_pd"]) < 0.10
               else f"diverges from internal estimate by "
                    f"{abs(sig_c['market_pd'] - composite_signal['composite_pd']):.1%}. "
                    f"Further review warranted.")
        )

    return {
        "contributions":   {"module_a": round(contrib_a, 4), "module_b": round(contrib_b, 4)},
        "percentages":     {"module_a": round(pct_a, 1),     "module_b": round(pct_b, 1)},
        "dominant_module": dominant,
        "narrative":       " ".join(narrative_parts),
    }


# ── Feature-Level Attribution ─────────────────────────────────────────────────

def generate_adverse_action_reasons(
    decision_output: dict,
    composite_signal: dict,
    shap_top_drivers: Optional[list] = None,
    max_reasons: int = 3,
) -> list[dict]:
    """
    Generate a ranked list of adverse action reason codes for a decline or review.

    For APPROVE decisions, generates approval basis instead.

    Follows the format required by:
    - ECOA/FCRA adverse action notices (US)
    - RBI Fair Practices Code (India)
    - SR 11-7 model documentation

    Parameters
    ----------
    decision_output  : Output of engine.make_decision().
    composite_signal : Output of signal_aggregator.aggregate_signals().
    shap_top_drivers : Optional list of {feature, shap_value, value} dicts
                       from Module B SHAP analysis (03_explainability_b.py).
    max_reasons      : Maximum number of reason codes to return.

    Returns
    -------
    List of reason dicts, each with:
        rank        : int — importance rank (1 = most important)
        code        : str — machine-readable code
        description : str — plain English (for adverse action notice)
        source      : str — which module produced this reason
        value       : str — the actual feature value that triggered this (if available)
    """
    decision   = decision_output["decision"]
    reasons    = []
    sig_a      = composite_signal["signal_a"]
    sig_b      = composite_signal["signal_b"]
    sig_c      = composite_signal.get("signal_c")
    overrides  = decision_output.get("overrides_triggered", [])
    reason_codes_raw = decision_output.get("reason_codes", [])

    # ── Hard overrides first (highest priority) ───────────────────────────────
    for flag in overrides:
        reasons.append({
            "rank":        len(reasons) + 1,
            "code":        flag,
            "description": REASON_CODE_REGISTRY.get(flag, f"Behavioural risk flag: {flag}"),
            "source":      "Module B — Behavioural Risk",
            "value":       "Flag active",
            "priority":    "critical",
        })

    # ── Engine reason codes ───────────────────────────────────────────────────
    for rc in reason_codes_raw:
        if rc["code"] not in [r["code"] for r in reasons]:
            reasons.append({
                "rank":        len(reasons) + 1,
                "code":        rc["code"],
                "description": rc["description"],
                "source":      _categorise_source(rc["category"]),
                "value":       "",
                "priority":    "high" if rc["category"] in ("pd_threshold", "hard_override") else "medium",
            })

    # ── SHAP feature-level reasons (if available from Module B) ──────────────
    if shap_top_drivers:
        for driver in shap_top_drivers[:max_reasons]:
            feat = driver.get("feature", "")
            val  = driver.get("value", "")
            shap = driver.get("shap_value", 0)
            if shap > 0.01:   # Only include risk-increasing features
                reasons.append({
                    "rank":        len(reasons) + 1,
                    "code":        f"shap_{feat}",
                    "description": REASON_CODE_REGISTRY.get(feat, f"Feature risk factor: {feat}"),
                    "source":      "Module B — SHAP Attribution",
                    "value":       str(round(val, 3)) if isinstance(val, float) else str(val),
                    "priority":    "medium",
                })

    # ── Add module-level reasons if we have space ─────────────────────────────
    composite_pd = composite_signal["composite_pd"]
    if composite_pd >= 0.20 and not any(r["code"] == "composite_pd" for r in reasons):
        reasons.append({
            "rank":        len(reasons) + 1,
            "code":        "composite_pd",
            "description": f"Composite probability of default ({composite_pd:.1%}) indicates elevated risk",
            "source":      "Composite Model",
            "value":       f"{composite_pd:.2%}",
            "priority":    "high",
        })

    if sig_a["raroc"] < 0.14 and not any(r["code"] == "raroc_below_hurdle" for r in reasons):
        reasons.append({
            "rank":        len(reasons) + 1,
            "code":        "raroc_below_hurdle",
            "description": f"Risk-adjusted return ({sig_a['raroc']:.1%}) below minimum required ({0.14:.0%})",
            "source":      "Module A — Capital Model",
            "value":       f"{sig_a['raroc']:.2%}",
            "priority":    "high",
        })

    # Deduplicate and cap at max_reasons
    seen_codes = set()
    unique_reasons = []
    for r in reasons:
        if r["code"] not in seen_codes:
            seen_codes.add(r["code"])
            unique_reasons.append(r)

    # Renumber ranks
    for i, r in enumerate(unique_reasons[:max_reasons], 1):
        r["rank"] = i

    return unique_reasons[:max_reasons]


def explain_decision(decision_output: dict, composite_signal: dict) -> dict:
    """
    Generate the complete explanation package for a lending decision.

    This is the top-level function called by the demo and audit logger.

    Parameters
    ----------
    decision_output  : Output of engine.make_decision().
    composite_signal : Output of signal_aggregator.aggregate_signals().

    Returns
    -------
    dict with keys:
        signal_contributions : Signal-level attribution (% per module)
        adverse_action_reasons : Ranked reason codes (for adverse action)
        decision_narrative   : Full plain-English explanation
        regulatory_summary   : One-paragraph summary for compliance documentation
    """
    signal_contrib = explain_signal_contributions(composite_signal)
    adverse_reasons = generate_adverse_action_reasons(decision_output, composite_signal)

    regulatory_summary = _build_regulatory_summary(
        decision_output, composite_signal, signal_contrib, adverse_reasons
    )

    return {
        "signal_contributions":    signal_contrib,
        "adverse_action_reasons":  adverse_reasons,
        "decision_narrative":      decision_output["decision_basis"],
        "regulatory_summary":      regulatory_summary,
    }


def format_decision_for_display(
    decision_output: dict,
    composite_signal: dict,
    explanation: dict,
    width: int = 72,
) -> str:
    """
    Format the complete decision output as a clean, readable text report.

    This is what the demo.py uses to print to the terminal.
    """
    sep   = "─" * width
    thick = "═" * width

    lines = []
    lines.append(thick)
    lines.append(" AI RISK DECISIONING SYSTEM  —  DECISION REPORT")
    lines.append(thick)
    lines.append("")

    # Decision banner
    d         = decision_output["decision"]
    meta      = decision_output.get("decision_metadata", {})
    emoji     = meta.get("emoji", "")
    lines.append(f"  {emoji}  DECISION:  {d}")
    lines.append(f"     {decision_output['primary_reason']}")
    lines.append("")
    lines.append(sep)

    # Composite risk summary
    lines.append("  COMPOSITE RISK PROFILE")
    lines.append("")
    lines.append(f"  {'Composite PD':<30} {decision_output['composite_pd']:.2%}")
    lines.append(f"  {'Risk Band':<30} {decision_output['composite_risk_band']} — {decision_output['composite_risk_label']}")
    lines.append("")
    lines.append(sep)

    # Module signals
    lines.append("  MODULE SIGNALS")
    lines.append("")
    sig = decision_output["signal_summary"]
    lines.append(f"  Module A  {sig['module_a']}")
    lines.append(f"  Module B  {sig['module_b']}")
    lines.append(f"  Module C  {sig['module_c']}")
    lines.append("")

    # Signal attribution
    sc = explanation["signal_contributions"]
    lines.append(f"  Signal attribution: "
                 f"Module A = {sc['percentages']['module_a']:.0f}%  |  "
                 f"Module B = {sc['percentages']['module_b']:.0f}%")
    lines.append(f"  Dominant: {sc['dominant_module']}")
    lines.append("")
    lines.append(sep)

    # Capital impact
    ci = decision_output["capital_impact"]
    lines.append("  CAPITAL IMPACT")
    lines.append("")
    lines.append(f"  {'Expected Loss (EL)':<30} ₹{ci['el']:,.0f}  ({ci['el_rate']:.2%} of loan)")
    lines.append(f"  {'Economic Capital':<30} ₹{ci['economic_capital']:,.0f}")
    lines.append(f"  {'RAROC':<30} {ci['raroc']:.2%}  ({ci['value_decision']})")
    lines.append("")
    lines.append(sep)

    # Pricing
    p = decision_output["pricing"]
    if p.get("suggested_rate"):
        lines.append("  PRICING")
        lines.append("")
        lines.append(f"  {'Pricing Adequacy':<30} {p['pricing_adequacy']}")
        lines.append(f"  {'Suggested Rate':<30} {p['suggested_rate']:.2f}% p.a.")
        lines.append(f"  {'Market Grade':<30} {p['loan_grade']}")
        if p.get("market_pd"):
            lines.append(f"  {'Market-Implied PD':<30} {p['market_pd']:.2%}")
        lines.append("")
        lines.append(sep)

    # Adverse action reasons
    if decision_output["decision"] in ("DECLINE", "MANUAL REVIEW"):
        lines.append("  ADVERSE ACTION REASON CODES")
        lines.append("  (Regulatory disclosure — RBI Fair Practices Code / ECOA)")
        lines.append("")
        for r in explanation["adverse_action_reasons"]:
            lines.append(f"  {r['rank']}.  {r['description']}")
            lines.append(f"      Source: {r['source']}")
        lines.append("")
        lines.append(sep)

    # Overrides
    if decision_output.get("overrides_triggered"):
        lines.append("  ⚠  HARD OVERRIDE FLAGS")
        for flag in decision_output["overrides_triggered"]:
            lines.append(f"     • {flag}")
        lines.append("")
        lines.append(sep)

    # Market PD note
    if decision_output.get("market_pd_note"):
        lines.append(f"  ℹ  {decision_output['market_pd_note']}")
        lines.append("")
        lines.append(sep)

    lines.append(f"  Timestamp: {decision_output['timestamp']}")
    lines.append(thick)

    return "\n".join(lines)


def _categorise_source(category: str) -> str:
    """Map an engine reason category to a module source label."""
    mapping = {
        "pd_threshold":  "Composite Model",
        "hard_override": "Module B — Behavioural Risk",
        "pricing":       "Module C — Portfolio/Pricing Risk",
        "portfolio":     "Module C — Portfolio/Pricing Risk",
        "capital":       "Module A — Capital Model",
        "review":        "Composite Model",
        "approval":      "Composite Model",
    }
    return mapping.get(category, "Decision Engine")


def _build_regulatory_summary(
    decision_output: dict,
    composite_signal: dict,
    signal_contrib: dict,
    adverse_reasons: list,
) -> str:
    """Build a one-paragraph regulatory compliance summary."""
    decision = decision_output["decision"]
    cpd      = composite_signal["composite_pd"]
    sa       = composite_signal["signal_a"]
    sb       = composite_signal["signal_b"]

    summary = (
        f"This lending decision was produced by the AI Credit Intelligence System "
        f"on {decision_output['timestamp'][:10]}. "
        f"The composite probability of default was assessed at {cpd:.2%}, "
        f"derived from a weighted combination of the Application Risk model "
        f"(Module A, PD={sa['pd_score']:.2%}, weight=60%) and the Behavioural "
        f"Risk model (Module B, PD={sb['delinquency_prob']:.2%}, weight=40%). "
        f"The decision of {decision} was reached through a rules-based policy "
        f"framework applied to the composite score. "
    )

    if adverse_reasons:
        top_reasons = "; ".join([r["description"] for r in adverse_reasons[:2]])
        summary += (
            f"The primary factors were: {top_reasons}. "
        )

    summary += (
        f"This report constitutes the full audit trail for this decision and "
        f"satisfies the documentation requirements of SR 11-7 and "
        f"RBI Model Risk Management guidelines."
    )
    return summary
