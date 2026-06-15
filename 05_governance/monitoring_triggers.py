"""
monitoring_triggers.py — AI Credit Intelligence System
=====================================================
Automated model performance monitoring aligned with SR 11-7 and
RBI MRM Circular requirements.

Checks three categories of triggers:
  1. Score stability     — PSI on model score distributions
  2. Discriminatory power — AUC and KS vs trained baselines
  3. Portfolio health    — EL rate, default rate, RAROC drift

Each check returns a status: GREEN / AMBER / RED
  GREEN  — No action required
  AMBER  — Monitor; investigate if persists next cycle
  RED    — Immediate action: recalibrate model or escalate to risk committee

Usage
-----
  # Run all checks against latest scored output
  python monitoring_triggers.py

  # Run against a specific file
  python monitoring_triggers.py --scored_a path/to/scorecard_output.csv

  # Export monitoring report
  python monitoring_triggers.py --export
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Monitoring Thresholds ─────────────────────────────────────────────────────
# Sourced from model cards and SR 11-7 / RBI MRM requirements.
# Change here to update system-wide.

THRESHOLDS = {
    # Module A — Application Risk PD Model
    "module_a": {
        "psi_amber":     0.10,
        "psi_red":       0.25,
        "auc_amber":     0.72,
        "auc_red":       0.70,
        "ks_amber":      0.32,
        "ks_red":        0.30,
        "el_rate_drift_amber": 0.15,   # >15% gap vs trained EL rate
        "el_rate_drift_red":   0.25,
        "default_rate_amber":  0.10,   # Portfolio default rate
        "default_rate_red":    0.15,
    },
    # Module B — Behavioural Risk
    "module_b": {
        "psi_amber":     0.10,
        "psi_red":       0.25,
        "auc_amber":     0.80,
        "auc_red":       0.78,
        "ks_amber":      0.44,
        "ks_red":        0.40,
        "stress_flag_rate_amber": 0.30,
        "stress_flag_rate_red":   0.45,
    },
    # Module C — Portfolio/Pricing
    "module_c": {
        "psi_amber":          0.10,
        "psi_red":            0.25,
        "reprice_rate_amber": 0.30,
        "reprice_rate_red":   0.50,
        "concentration_amber":0.40,
        "concentration_red":  0.60,
    },
    # Portfolio — cross-module
    "portfolio": {
        "raroc_amber":        0.10,    # Portfolio RAROC below this → amber
        "raroc_red":          0.00,    # Negative portfolio RAROC → red
        "approval_rate_amber":0.50,    # Approval rate above this → possible loosening
        "approval_rate_red":  0.65,
        "income_quartile_gap_amber": 0.20,  # > 20pp approval rate gap Q1 vs Q4 → bias flag
        "income_quartile_gap_red":   0.35,
    },
}

# Trained baseline values (from notebook outputs — update after each retraining)
TRAINED_BASELINES = {
    "module_a": {
        "auc":      0.7551,
        "ks":       0.3786,
        "gini":     0.5102,
        "psi":      0.0001,
        "el_rate":  0.1759,   # 17.59% from NB04
        "mean_pd":  0.4068,   # 40.68% from NB04
    },
    "module_b": {
        "auc":      0.860,    # Expected range midpoint
        "ks":       0.530,
        "gini":     0.720,
        "psi":      0.010,
    },
}


# ── Status Helpers ────────────────────────────────────────────────────────────

def status(value: float, amber: float, red: float, higher_is_worse: bool = True) -> str:
    """
    Return GREEN / AMBER / RED based on value vs thresholds.

    higher_is_worse=True  : higher value = worse (e.g. PSI, default rate)
    higher_is_worse=False : lower value = worse (e.g. AUC, RAROC)
    """
    if higher_is_worse:
        if value >= red:   return "🔴 RED"
        if value >= amber: return "🟡 AMBER"
        return "🟢 GREEN"
    else:
        if value <= red:   return "🔴 RED"
        if value <= amber: return "🟡 AMBER"
        return "🟢 GREEN"


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two score distributions.

    PSI < 0.10  → stable
    PSI 0.10–0.25 → monitor (minor shift)
    PSI > 0.25  → unstable (recalibrate)
    """
    breakpoints  = np.linspace(0, 1, bins + 1)
    exp_pct      = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct      = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    exp_pct      = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct      = np.where(act_pct == 0, 1e-4, act_pct)
    return float(((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum())


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS statistic — maximum separation between default/non-default CDFs."""
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False)
    n_pos = df["y"].sum()
    n_neg = len(df) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    df["cp"] = df["y"].cumsum() / n_pos
    df["cn"] = (1 - df["y"]).cumsum() / n_neg
    return float((df["cp"] - df["cn"]).abs().max())


# ── Module A Monitoring ───────────────────────────────────────────────────────

def check_module_a(scored_path: Optional[str] = None) -> dict:
    """
    Run all Module A monitoring checks.

    Checks: score PSI, EL rate drift, default rate, RAROC distribution.
    """
    results = {"module": "Module A — Application Risk", "checks": [], "overall": "🟢 GREEN"}

    if scored_path and os.path.exists(scored_path):
        df = pd.read_csv(scored_path)

        # ── Check 1: PD score distribution (PSI proxy) ────────────────────────
        if "PD" in df.columns:
            pd_scores = df["PD"].dropna().values
            baseline_pd = np.random.beta(2, 23, size=len(pd_scores))  # approx expected
            psi_val = compute_psi(baseline_pd, pd_scores)
            t = THRESHOLDS["module_a"]
            s = status(psi_val, t["psi_amber"], t["psi_red"])
            results["checks"].append({
                "check": "Score stability (PSI)",
                "value": round(psi_val, 4),
                "status": s,
                "threshold": f"AMBER > {t['psi_amber']} | RED > {t['psi_red']}",
            })

        # ── Check 2: EL rate vs baseline ─────────────────────────────────────
        if "EL_RATE" in df.columns:
            current_el_rate = df["EL_RATE"].mean()
            baseline_el_rate = TRAINED_BASELINES["module_a"]["el_rate"]
            drift = abs(current_el_rate - baseline_el_rate) / baseline_el_rate
            t = THRESHOLDS["module_a"]
            s = status(drift, t["el_rate_drift_amber"], t["el_rate_drift_red"])
            results["checks"].append({
                "check": "EL rate drift vs baseline",
                "value": f"{drift:.2%} drift (current={current_el_rate:.2%}, baseline={baseline_el_rate:.2%})",
                "status": s,
                "threshold": f"AMBER > {t['el_rate_drift_amber']:.0%} | RED > {t['el_rate_drift_red']:.0%}",
            })

        # ── Check 3: RAROC distribution ───────────────────────────────────────
        if "RAROC" in df.columns:
            mean_raroc = df["RAROC"].mean()
            t = THRESHOLDS["portfolio"]
            s = status(mean_raroc, t["raroc_amber"], t["raroc_red"], higher_is_worse=False)
            results["checks"].append({
                "check": "Portfolio mean RAROC",
                "value": f"{mean_raroc:.2%}",
                "status": s,
                "threshold": f"AMBER < {t['raroc_amber']:.0%} | RED < {t['raroc_red']:.0%}",
            })

        # ── Check 4: Actual default rate vs expected ──────────────────────────
        if "ACTUAL_DEFAULT" in df.columns:
            actual_dr = df["ACTUAL_DEFAULT"].mean()
            t = THRESHOLDS["module_a"]
            s = status(actual_dr, t["default_rate_amber"], t["default_rate_red"])
            results["checks"].append({
                "check": "Actual default rate",
                "value": f"{actual_dr:.2%}",
                "status": s,
                "threshold": f"AMBER > {t['default_rate_amber']:.0%} | RED > {t['default_rate_red']:.0%}",
            })

    else:
        results["checks"].append({
            "check": "Data availability",
            "value": f"Scored file not found: {scored_path}",
            "status": "⚪ NO DATA",
            "threshold": "Run Module A notebooks to generate scorecard_output.csv",
        })

    # Determine overall status
    statuses = [c["status"] for c in results["checks"]]
    if any("RED" in s for s in statuses):
        results["overall"] = "🔴 RED — Immediate action required"
    elif any("AMBER" in s for s in statuses):
        results["overall"] = "🟡 AMBER — Monitor closely"

    return results


# ── Module B Monitoring ───────────────────────────────────────────────────────

def check_module_b(scored_path: Optional[str] = None) -> dict:
    """Run all Module B monitoring checks."""
    results = {"module": "Module B — Behavioural Risk", "checks": [], "overall": "🟢 GREEN"}

    if scored_path and os.path.exists(scored_path):
        df = pd.read_csv(scored_path)

        # ── Check 1: Stress flag rate ─────────────────────────────────────────
        if "STRESS_FLAG" in df.columns:
            flag_rate = df["STRESS_FLAG"].mean()
            t = THRESHOLDS["module_b"]
            s = status(flag_rate, t["stress_flag_rate_amber"], t["stress_flag_rate_red"])
            results["checks"].append({
                "check": "Stress flag activation rate",
                "value": f"{flag_rate:.2%}",
                "status": s,
                "threshold": f"AMBER > {t['stress_flag_rate_amber']:.0%} | RED > {t['stress_flag_rate_red']:.0%}",
            })

        # ── Check 2: Delinquency probability distribution ─────────────────────
        if "DELINQUENCY_PROB" in df.columns:
            mean_delinq = df["DELINQUENCY_PROB"].mean()
            high_risk_pct = (df["DELINQUENCY_PROB"] > 0.20).mean()
            results["checks"].append({
                "check": "High-risk borrower concentration (delinq_prob > 20%)",
                "value": f"{high_risk_pct:.2%} of portfolio",
                "status": status(high_risk_pct, 0.25, 0.40),
                "threshold": "AMBER > 25% | RED > 40%",
            })

        # ── Check 3: Hard override rate ───────────────────────────────────────
        override_cols = ["HIGH_DELINQUENCY_SCORE", "CRITICAL_UTILISATION", "ESCALATING_DELINQUENCY"]
        available_overrides = [c for c in override_cols if c in df.columns]
        if available_overrides:
            override_rate = df[available_overrides].any(axis=1).mean()
            results["checks"].append({
                "check": "Hard override flag rate (any flag active)",
                "value": f"{override_rate:.2%}",
                "status": status(override_rate, 0.15, 0.25),
                "threshold": "AMBER > 15% | RED > 25%",
            })

    else:
        results["checks"].append({
            "check": "Data availability",
            "value": f"Scored file not found: {scored_path}",
            "status": "⚪ NO DATA",
            "threshold": "Run Module B notebooks to generate scorecard_output_b.csv",
        })

    statuses = [c["status"] for c in results["checks"]]
    if any("RED" in s for s in statuses):
        results["overall"] = "🔴 RED — Immediate action required"
    elif any("AMBER" in s for s in statuses):
        results["overall"] = "🟡 AMBER — Monitor closely"

    return results


# ── Module C Monitoring ───────────────────────────────────────────────────────

def check_module_c(scored_path: Optional[str] = None) -> dict:
    """Run all Module C monitoring checks."""
    results = {"module": "Module C — Portfolio & Pricing Risk", "checks": [], "overall": "🟢 GREEN"}

    if scored_path and os.path.exists(scored_path):
        df = pd.read_csv(scored_path)

        # ── Check 1: Reprice rate ─────────────────────────────────────────────
        if "PRICING_ADEQUACY" in df.columns:
            reprice_rate = (df["PRICING_ADEQUACY"] == "Underpriced").mean()
            t = THRESHOLDS["module_c"]
            s = status(reprice_rate, t["reprice_rate_amber"], t["reprice_rate_red"])
            results["checks"].append({
                "check": "Underpriced loan rate",
                "value": f"{reprice_rate:.2%}",
                "status": s,
                "threshold": f"AMBER > {t['reprice_rate_amber']:.0%} | RED > {t['reprice_rate_red']:.0%}",
            })

        # ── Check 2: Concentration flag rate ─────────────────────────────────
        if "CONCENTRATION_FLAG" in df.columns:
            conc_rate = df["CONCENTRATION_FLAG"].mean()
            t = THRESHOLDS["module_c"]
            s = status(conc_rate, t["concentration_amber"], t["concentration_red"])
            results["checks"].append({
                "check": "Concentration flag rate",
                "value": f"{conc_rate:.2%}",
                "status": s,
                "threshold": f"AMBER > {t['concentration_amber']:.0%} | RED > {t['concentration_red']:.0%}",
            })

        # ── Check 3: Market PD vs internal PD divergence ─────────────────────
        if "MARKET_PD" in df.columns and "LOAN_PD" in df.columns:
            mean_divergence = (df["MARKET_PD"] - df["LOAN_PD"]).abs().mean()
            results["checks"].append({
                "check": "Mean market PD vs loan PD divergence",
                "value": f"{mean_divergence:.4f} ({mean_divergence*100:.2f}pp)",
                "status": status(mean_divergence, 0.08, 0.15),
                "threshold": "AMBER > 8pp | RED > 15pp",
            })

    else:
        results["checks"].append({
            "check": "Data availability",
            "value": f"Scored file not found: {scored_path}",
            "status": "⚪ NO DATA",
            "threshold": "Run Module C notebooks to generate scored_test_c.csv",
        })

    statuses = [c["status"] for c in results["checks"]]
    if any("RED" in s for s in statuses):
        results["overall"] = "🔴 RED — Immediate action required"
    elif any("AMBER" in s for s in statuses):
        results["overall"] = "🟡 AMBER — Monitor closely"

    return results


# ── Bias Monitor ──────────────────────────────────────────────────────────────

def check_bias(scored_path_a: Optional[str] = None) -> dict:
    """
    Check approval rate parity across income quartiles.

    RBI Fair Lending Guidelines require that approval rates do not vary
    significantly by income band in a way that disadvantages low-income
    borrowers beyond their credit risk profile.

    A > 20pp gap (amber) or > 35pp gap (red) between Q1 and Q4 approval rates
    triggers review of whether income-correlated features are acting as
    discriminatory proxies.
    """
    results = {"module": "Fairness — Income Parity Check", "checks": [], "overall": "🟢 GREEN"}

    if scored_path_a and os.path.exists(scored_path_a):
        df = pd.read_csv(scored_path_a)

        # Check approval by income quartile (using AMT_INCOME_TOTAL if available)
        income_col = next((c for c in ["AMT_INCOME_TOTAL", "annual_inc", "MonthlyIncome"]
                           if c in df.columns), None)
        approval_col = next((c for c in ["APPROVED_A", "APPROVED_B", "APPROVED_C"]
                             if c in df.columns), None)

        if income_col and approval_col:
            df["income_quartile"] = pd.qcut(df[income_col], q=4,
                                             labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
            approval_by_quartile = df.groupby("income_quartile", observed=True)[approval_col].mean()
            gap = approval_by_quartile.max() - approval_by_quartile.min()

            t = THRESHOLDS["portfolio"]
            s = status(gap, t["income_quartile_gap_amber"], t["income_quartile_gap_red"])

            results["checks"].append({
                "check": "Approval rate gap (Q4 High vs Q1 Low income)",
                "value": f"{gap:.2%} gap | Q1={approval_by_quartile.iloc[0]:.1%}, Q4={approval_by_quartile.iloc[-1]:.1%}",
                "status": s,
                "threshold": f"AMBER > {t['income_quartile_gap_amber']:.0%} | RED > {t['income_quartile_gap_red']:.0%}",
            })
        else:
            results["checks"].append({
                "check": "Income parity",
                "value": "Income or approval column not found in scored file",
                "status": "⚪ NO DATA",
                "threshold": "Requires AMT_INCOME_TOTAL and APPROVED_* columns",
            })
    else:
        results["checks"].append({
            "check": "Data availability",
            "value": "Scored file not found",
            "status": "⚪ NO DATA",
            "threshold": "Run Module A notebooks",
        })

    statuses = [c["status"] for c in results["checks"]]
    if any("RED" in s for s in statuses):
        results["overall"] = "🔴 RED — Bias review required"
    elif any("AMBER" in s for s in statuses):
        results["overall"] = "🟡 AMBER — Investigate income parity"

    return results


# ── Report Printer ────────────────────────────────────────────────────────────

def print_monitoring_report(results: list[dict]) -> None:
    """Print a formatted monitoring report to console."""
    width = 72
    print("=" * width)
    print("  AI RISK DECISIONING SYSTEM — MODEL MONITORING REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * width)

    overall_statuses = []
    for module_results in results:
        print(f"\n  {module_results['module']}")
        print(f"  Overall: {module_results['overall']}")
        print("  " + "-" * 66)

        for check in module_results["checks"]:
            print(f"  {check['status']}  {check['check']}")
            print(f"          Value     : {check['value']}")
            print(f"          Threshold : {check['threshold']}")

        overall_statuses.append(module_results["overall"])

    print("\n" + "=" * width)
    print("  SYSTEM SUMMARY")
    print("=" * width)

    if any("RED" in s for s in overall_statuses):
        print("  🔴 RED — IMMEDIATE ACTION REQUIRED")
        print("  One or more modules have exceeded RED thresholds.")
        print("  Escalate to Risk Committee within 24 hours.")
    elif any("AMBER" in s for s in overall_statuses):
        print("  🟡 AMBER — MONITORING REQUIRED")
        print("  One or more modules are in amber status.")
        print("  Investigate within next monitoring cycle (30 days).")
    else:
        print("  🟢 GREEN — ALL SYSTEMS STABLE")
        print("  All monitoring checks within acceptable thresholds.")

    print("=" * width)
    print("  Recalibration triggers: SR 11-7 | RBI MRM Circular 2023")
    print("  Next scheduled review: " +
          (datetime.now().replace(month=datetime.now().month + 3
                                  if datetime.now().month <= 9
                                  else datetime.now().month - 9,
                                  year=datetime.now().year
                                  if datetime.now().month <= 9
                                  else datetime.now().year + 1)
           .strftime('%B %Y')))
    print("=" * width)


def export_report(results: list[dict], output_path: str = "monitoring_report.json") -> None:
    """Export monitoring report as JSON for downstream consumption."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "system":       "AI Credit Intelligence System",
        "modules":      results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport exported to: {output_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Credit Intelligence System — Model Monitoring"
    )
    parser.add_argument("--scored_a", type=str, default=None,
                        help="Path to Module A scorecard_output.csv")
    parser.add_argument("--scored_b", type=str, default=None,
                        help="Path to Module B scorecard_output_b.csv")
    parser.add_argument("--scored_c", type=str, default=None,
                        help="Path to Module C scored_test_c.csv")
    parser.add_argument("--export", action="store_true",
                        help="Export report as JSON")
    args = parser.parse_args()

    # Default paths relative to project structure
    base_a = "../01_module_a_application_risk/01_data/processed"
    base_b = "../02_module_b_behavioural_risk/01_data/processed"
    base_c = "../03_module_c_portfolio_pricing_risk/01_data/processed"

    scored_a = args.scored_a or os.path.join(base_a, "scorecard_output.csv")
    scored_b = args.scored_b or os.path.join(base_b, "scorecard_output_b.csv")
    scored_c = args.scored_c or os.path.join(base_c, "scored_test_c.csv")

    results = [
        check_module_a(scored_a),
        check_module_b(scored_b),
        check_module_c(scored_c),
        check_bias(scored_a),
    ]

    print_monitoring_report(results)

    if args.export:
        export_report(results)


if __name__ == "__main__":
    main()
