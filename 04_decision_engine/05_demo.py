"""
demo.py — AI Credit Intelligence System
======================================
A self-contained runnable demo of the complete three-signal decision engine.

Usage
-----
    # Run with default applicant profiles
    python 05_demo.py

    # Run with a custom applicant (key=value pairs)
    python 05_demo.py \\
        --pd_a 0.08 --credit_score 642 --ead 250000 --raroc 0.18 \\
        --pd_b 0.05 --brs 72 \\
        --market_pd 0.12 --grade B --rate 12.5 --pricing Fair

    # Run all preset profiles and show comparison
    python 05_demo.py --all

    # Batch mode: process a scored output CSV
    python 05_demo.py --batch path/to/scorecard_output.csv

This demo works WITHOUT the trained model .pkl files. It uses pre-computed
signal dictionaries directly. After running notebooks A1-C3, replace the
preset values with actual model outputs.

Architecture
------------
The demo exercises the full engine stack:

  signal_aggregator.aggregate_signals()
    → engine.make_decision()
      → explainability.explain_decision()
        → audit_logger.log_decision()
          → formatted output
"""

import sys
import os
import argparse
from datetime import datetime

# Add the engine directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_aggregator import (
    build_module_a_signal,
    build_module_b_signal,
    build_module_c_signal,
    aggregate_signals,
)
from engine import make_decision
from explainability import explain_decision, format_decision_for_display
from audit_logger import log_decision, print_audit_summary


# ── Preset Applicant Profiles ─────────────────────────────────────────────────
# Four profiles representing the four possible decision outcomes.
# Values are calibrated to the real model outputs from the trained notebooks.

APPLICANT_PROFILES = {

    "ANAND_MEHTA": {
        "name": "Anand Mehta",
        "description": "Salaried professional, stable income, clean credit history",
        "expected_decision": "APPROVE",
        "signal_a": {
            "pd_score":     0.048,
            "credit_score": 724,
            "ead":          350000.0,
        },
        "signal_b": {
            "delinquency_prob":       0.031,
            "behavioural_risk_score": 81.0,
            "stress_flag":            False,
            "critical_utilisation":   False,
            "escalating_delinquency": False,
            "high_delinquency_score": False,
        },
        "signal_c": {
            "market_pd":           0.054,
            "loan_grade_signal":   "A",
            "suggested_rate":      7.2,
            "pricing_adequacy":    "Fair",
            "concentration_flag":  False,
        },
    },

    "PRIYA_SHARMA": {
        "name": "Priya Sharma",
        "description": "Self-employed borrower, low rate offered, loan needs repricing",
        "expected_decision": "REPRICE",
        "signal_a": {
            "pd_score":     0.088,
            "credit_score": 648,
            "ead":          180000.0,
        },
        "signal_b": {
            "delinquency_prob":       0.062,
            "behavioural_risk_score": 58.0,
            "stress_flag":            False,
            "critical_utilisation":   False,
            "escalating_delinquency": False,
            "high_delinquency_score": False,
        },
        "signal_c": {
            "market_pd":           0.224,   # Market sees this as Grade C
            "loan_grade_signal":   "C",
            "suggested_rate":      14.8,    # Offered 11% — market says 14.8%
            "pricing_adequacy":    "Underpriced",
            "concentration_flag":  False,
        },
    },

    "VIKRAM_NAIR": {
        "name": "Vikram Nair",
        "description": "Borrower with escalating delinquency history, high utilisation",
        "expected_decision": "MANUAL REVIEW",
        "signal_a": {
            "pd_score":     0.156,
            "credit_score": 582,
            "ead":          120000.0,
        },
        "signal_b": {
            "delinquency_prob":       0.218,
            "behavioural_risk_score": 28.0,
            "stress_flag":            True,
            "critical_utilisation":   True,   # Hard override
            "escalating_delinquency": True,   # Hard override
            "high_delinquency_score": False,
        },
        "signal_c": {
            "market_pd":           0.328,
            "loan_grade_signal":   "D",
            "suggested_rate":      17.5,
            "pricing_adequacy":    "Underpriced",
            "concentration_flag":  True,
        },
    },

    "RAHUL_VERMA": {
        "name": "Rahul Verma",
        "description": "High-risk applicant, repeated delinquencies, value-destructive RAROC",
        "expected_decision": "DECLINE",
        "signal_a": {
            "pd_score":     0.412,
            "credit_score": 431,
            "ead":          95000.0,
        },
        "signal_b": {
            "delinquency_prob":       0.348,
            "behavioural_risk_score": 12.0,
            "stress_flag":            True,
            "critical_utilisation":   False,
            "escalating_delinquency": True,
            "high_delinquency_score": True,   # Hard override
        },
        "signal_c": {
            "market_pd":           0.519,
            "loan_grade_signal":   "F",
            "suggested_rate":      24.1,
            "pricing_adequacy":    "Underpriced",
            "concentration_flag":  True,
        },
    },
}


# ── Core Runner ───────────────────────────────────────────────────────────────

def run_single_applicant(profile: dict, log_decisions: bool = True) -> dict:
    """
    Run the complete decision engine pipeline for one applicant profile.

    Returns the full decision output dict.
    """
    # ── Build signals ─────────────────────────────────────────────────────────
    sig_a = build_module_a_signal(**profile["signal_a"])
    sig_b = build_module_b_signal(**profile["signal_b"])
    sig_c = build_module_c_signal(**profile["signal_c"])

    # ── Aggregate ─────────────────────────────────────────────────────────────
    composite = aggregate_signals(sig_a, sig_b, sig_c)

    # ── Decision ──────────────────────────────────────────────────────────────
    decision_output = make_decision(composite)

    # ── Explain ───────────────────────────────────────────────────────────────
    explanation = explain_decision(decision_output, composite)

    # ── Log ───────────────────────────────────────────────────────────────────
    if log_decisions:
        decision_id = log_decision(
            decision_output, composite, explanation,
            applicant_id=profile.get("name", "UNKNOWN").replace(" ", "_").upper()
        )
        decision_output["decision_id"] = decision_id

    # ── Display ───────────────────────────────────────────────────────────────
    report = format_decision_for_display(decision_output, composite, explanation)
    print(report)

    # ── Validate against expected outcome ────────────────────────────────────
    if "expected_decision" in profile:
        expected = profile["expected_decision"]
        actual   = decision_output["decision"]
        status   = "✅ MATCHES EXPECTED" if actual == expected else f"⚠ Expected {expected}"
        print(f"\n  Validation: {status}")
        print()

    return decision_output


def run_custom_applicant(args) -> dict:
    """Run the engine with command-line supplied values."""
    profile = {
        "name": "Custom Applicant",
        "signal_a": {
            "pd_score":     args.pd_a,
            "credit_score": args.credit_score,
            "ead":          args.ead,
        },
        "signal_b": {
            "delinquency_prob":       args.pd_b,
            "behavioural_risk_score": args.brs,
            "stress_flag":            args.stress_flag,
            "critical_utilisation":   args.critical_util,
            "escalating_delinquency": args.escalating,
            "high_delinquency_score": args.high_delinq,
        },
        "signal_c": {
            "market_pd":          args.market_pd,
            "loan_grade_signal":  args.grade,
            "suggested_rate":     args.rate,
            "pricing_adequacy":   args.pricing,
            "concentration_flag": args.concentration,
        },
    }
    return run_single_applicant(profile)


def run_batch_from_csv(csv_path: str, n_rows: int = 10) -> None:
    """
    Process the first n_rows from a scored output CSV (e.g. scorecard_output.csv).
    Demonstrates batch portfolio decisioning mode.
    """
    import pandas as pd
    from signal_aggregator import (
        load_module_a_signal_from_csv,
        load_module_b_signal_from_csv,
        load_module_c_signal_from_csv,
    )

    print(f"\nBATCH MODE: Processing {n_rows} loans from {csv_path}")
    print("=" * 72)

    df = pd.read_csv(csv_path)
    n  = min(n_rows, len(df))

    results = {"APPROVE": 0, "DECLINE": 0, "REPRICE": 0, "MANUAL REVIEW": 0}

    for i in range(n):
        try:
            sig_a = load_module_a_signal_from_csv(csv_path, i)

            # Module B fallback (would normally load from scorecard_output_b.csv)
            sig_b = build_module_b_signal(
                delinquency_prob=0.067,
                behavioural_risk_score=50.0,
            )
            # Module C fallback
            sig_c = build_module_c_signal(
                market_pd=sig_a["pd_score"],
                loan_grade_signal="C",
                suggested_rate=12.0,
                pricing_adequacy="Fair",
                concentration_flag=False,
            )

            composite = aggregate_signals(sig_a, sig_b, sig_c)
            decision_output = make_decision(composite)
            results[decision_output["decision"]] += 1

            pd_val = decision_output["composite_pd"]
            d      = decision_output["decision"]
            score  = sig_a["credit_score"]
            raroc  = decision_output["capital_impact"]["raroc"]
            print(f"  Loan {i+1:>3}:  PD={pd_val:.2%}  Score={score}  RAROC={raroc:.1%}  → {d}")

        except Exception as e:
            print(f"  Loan {i+1:>3}:  Error: {e}")

    print("=" * 72)
    print("BATCH SUMMARY")
    total = sum(results.values())
    for d, n_ in results.items():
        print(f"  {d:<20} {n_:>4} ({n_/total*100:.1f}%)")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Credit Intelligence System — Decision Engine Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_demo.py                      # Run all 4 preset profiles
  python 05_demo.py --profile ANAND_MEHTA  # Run one profile
  python 05_demo.py --pd_a 0.12 --credit_score 610 --ead 200000 --pd_b 0.08 --brs 55
  python 05_demo.py --batch ../01_module_a_application_risk/01_data/processed/scorecard_output.csv
  python 05_demo.py --summary            # Show audit log summary
        """
    )

    # Mode selection
    parser.add_argument("--all",     action="store_true", help="Run all preset profiles")
    parser.add_argument("--profile", choices=list(APPLICANT_PROFILES.keys()),
                        help="Run a specific preset profile")
    parser.add_argument("--batch",   type=str, metavar="CSV_PATH",
                        help="Batch mode: process rows from a scored CSV")
    parser.add_argument("--summary", action="store_true",
                        help="Print audit log summary")

    # Custom applicant inputs (Module A)
    parser.add_argument("--pd_a",         type=float, default=0.10, help="Module A PD (0-1)")
    parser.add_argument("--credit_score", type=int,   default=620,  help="WoE scorecard score (300-900)")
    parser.add_argument("--ead",          type=float, default=200000, help="Loan amount (EAD) in Rs")

    # Custom applicant inputs (Module B)
    parser.add_argument("--pd_b",         type=float, default=0.067, help="Module B delinquency prob (0-1)")
    parser.add_argument("--brs",          type=float, default=50.0,  help="Behavioural risk score (0-100)")
    parser.add_argument("--stress_flag",  action="store_true", help="Stress flag active")
    parser.add_argument("--critical_util",action="store_true", help="Critical utilisation flag")
    parser.add_argument("--escalating",   action="store_true", help="Escalating delinquency flag")
    parser.add_argument("--high_delinq",  action="store_true", help="High delinquency score flag")

    # Custom applicant inputs (Module C)
    parser.add_argument("--market_pd",    type=float, default=0.20,   help="Module C market PD")
    parser.add_argument("--grade",        type=str,   default="C",    help="Predicted loan grade A-G")
    parser.add_argument("--rate",         type=float, default=13.25,  help="Suggested rate (%)")
    parser.add_argument("--pricing",      type=str,   default="Fair",
                        choices=["Fair","Underpriced","Overpriced"], help="Pricing adequacy")
    parser.add_argument("--concentration",action="store_true", help="Concentration flag active")

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    print()
    print("  AI Credit Intelligence System")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    if args.summary:
        print_audit_summary()
        return

    if args.batch:
        run_batch_from_csv(args.batch)
        return

    if args.profile:
        profile = APPLICANT_PROFILES[args.profile]
        run_single_applicant(profile)
        return

    # Default or --all: run all four preset profiles
    has_custom = any([
        args.pd_a != 0.10, args.credit_score != 620,
        args.pd_b != 0.067, args.brs != 50.0,
        args.stress_flag, args.critical_util,
    ])

    if has_custom:
        run_custom_applicant(args)
    else:
        # Run all four preset profiles
        print("  Running all preset applicant profiles...")
        print("  Each demonstrates a different decision outcome.")
        print()

        for profile_name, profile in APPLICANT_PROFILES.items():
            print(f"\n  {'─'*68}")
            print(f"  Applicant: {profile['name']}  ({profile['description']})")
            print(f"  {'─'*68}\n")
            run_single_applicant(profile)

        # Final summary
        print("\n" + "═" * 72)
        print("  DEMO COMPLETE — 4 applicants processed, 4 decisions logged")
        print("  Run  python 05_demo.py --summary  to see the audit log")
        print("═" * 72)


if __name__ == "__main__":
    main()
