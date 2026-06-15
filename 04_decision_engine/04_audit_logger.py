"""
audit_logger.py — AI Credit Intelligence System
=============================================
Logs every decision with its complete input signals and output.

In a production system, this writes to a database. Here it writes to:
  - A JSON log file (one record per decision, append mode)
  - A CSV summary (for analysis and monitoring dashboards)

Every logged record satisfies:
  - SR 11-7 audit trail requirements
  - RBI MRM circular documentation requirements
  - ECOA record retention requirements (25 months minimum)

Log Schema
----------
Each record captures:
  - Timestamp and unique decision ID
  - Full composite signal (all three modules)
  - Decision output (decision, reason codes, capital impact)
  - Model versions (for future recalibration tracking)
  - Weights used (for A/B testing of weight changes)
"""

from __future__ import annotations

import json
import csv
import os
import uuid
from datetime import datetime
from typing import Optional


# ── Log file paths ─────────────────────────────────────────────────────────────
DEFAULT_LOG_DIR     = "04_decision_engine/logs"
JSON_LOG_FILE       = "decision_log.jsonl"    # One JSON object per line
CSV_SUMMARY_FILE    = "decision_summary.csv"  # Tabular summary for analysis


# ── Model version registry ────────────────────────────────────────────────────
# Update these when models are retrained. Tracked in every log record.
MODEL_VERSIONS = {
    "module_a_pd_model":          "xgb_v1.0_homecredit",
    "module_a_scorecard":         "woe_scorecard_v1.0",
    "module_b_delinquency_model": "xgb_v1.0_givemecredit",
    "module_b_scorecard":         "woe_scorecard_v1.0",
    "module_c_grade_model":       "xgb_multiclass_v1.0_lendingclub",
    "module_c_default_model":     "xgb_v1.0_lendingclub",
    "decision_engine":            "v1.0",
}


def log_decision(
    decision_output: dict,
    composite_signal: dict,
    explanation: dict,
    applicant_id: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
) -> str:
    """
    Log a complete decision record to both JSON and CSV files.

    Parameters
    ----------
    decision_output  : Output of engine.make_decision().
    composite_signal : Output of signal_aggregator.aggregate_signals().
    explanation      : Output of explainability.explain_decision().
    applicant_id     : Optional external applicant identifier.
    log_dir          : Directory to write log files. Created if absent.

    Returns
    -------
    str : The unique decision_id assigned to this record.
    """
    os.makedirs(log_dir, exist_ok=True)

    decision_id = str(uuid.uuid4())[:8].upper()
    timestamp   = decision_output.get("timestamp", datetime.now().isoformat())

    # ── Build the full log record ─────────────────────────────────────────────
    record = {
        "decision_id":      decision_id,
        "applicant_id":     applicant_id or f"ANON_{decision_id}",
        "timestamp":        timestamp,

        # Decision
        "decision":         decision_output["decision"],
        "decision_code":    decision_output["decision_code"],
        "primary_reason":   decision_output["primary_reason"],

        # Risk scores
        "composite_pd":     decision_output["composite_pd"],
        "composite_risk_band": decision_output["composite_risk_band"],
        "module_a_pd":      composite_signal["signal_a"]["pd_score"],
        "module_b_pd":      composite_signal["signal_b"]["delinquency_prob"],
        "market_pd":        composite_signal["signal_c"]["market_pd"] if composite_signal.get("signal_c") else None,
        "credit_score":     composite_signal["signal_a"]["credit_score"],
        "behavioural_score": composite_signal["signal_b"]["behavioural_risk_score"],

        # Capital
        "ead":              decision_output["capital_impact"]["ead"],
        "el":               decision_output["capital_impact"]["el"],
        "el_rate":          decision_output["capital_impact"]["el_rate"],
        "economic_capital": decision_output["capital_impact"]["economic_capital"],
        "raroc":            decision_output["capital_impact"]["raroc"],
        "value_decision":   decision_output["capital_impact"]["value_decision"],

        # Pricing
        "suggested_rate":   decision_output["pricing"]["suggested_rate"],
        "pricing_adequacy": decision_output["pricing"]["pricing_adequacy"],
        "loan_grade":       decision_output["pricing"]["loan_grade"],

        # Flags
        "stress_flag":          composite_signal["signal_b"].get("stress_flag", False),
        "critical_utilisation": composite_signal["signal_b"].get("critical_utilisation", False),
        "high_delinquency":     composite_signal["signal_b"].get("high_delinquency_score", False),
        "concentration_flag":   composite_signal["signal_c"]["concentration_flag"] if composite_signal.get("signal_c") else False,
        "overrides_triggered":  decision_output.get("overrides_triggered", []),

        # Attribution
        "signal_weight_a":  composite_signal["weights_used"]["module_a"],
        "signal_weight_b":  composite_signal["weights_used"]["module_b"],
        "dominant_module":  explanation["signal_contributions"]["dominant_module"],
        "b_available":      composite_signal.get("b_available", True),
        "c_available":      composite_signal.get("c_available", True),

        # Reason codes (top 3)
        "reason_1": explanation["adverse_action_reasons"][0]["description"] if len(explanation["adverse_action_reasons"]) > 0 else "",
        "reason_2": explanation["adverse_action_reasons"][1]["description"] if len(explanation["adverse_action_reasons"]) > 1 else "",
        "reason_3": explanation["adverse_action_reasons"][2]["description"] if len(explanation["adverse_action_reasons"]) > 2 else "",

        # Model provenance
        "model_versions":   MODEL_VERSIONS,

        # Full narrative (for compliance retrieval)
        "regulatory_summary": explanation["regulatory_summary"],
    }

    # ── Write JSON log (append mode) ──────────────────────────────────────────
    json_path = os.path.join(log_dir, JSON_LOG_FILE)
    with open(json_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    # ── Write / append CSV summary ────────────────────────────────────────────
    csv_path    = os.path.join(log_dir, CSV_SUMMARY_FILE)
    csv_columns = [
        "decision_id", "applicant_id", "timestamp", "decision",
        "composite_pd", "composite_risk_band", "module_a_pd", "module_b_pd",
        "market_pd", "credit_score", "behavioural_score",
        "ead", "el", "el_rate", "raroc", "value_decision",
        "suggested_rate", "pricing_adequacy", "loan_grade",
        "stress_flag", "critical_utilisation", "high_delinquency", "concentration_flag",
        "dominant_module", "reason_1", "reason_2",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    return decision_id


def load_decision_log(log_dir: str = DEFAULT_LOG_DIR) -> list[dict]:
    """
    Load all logged decisions from the JSON log file.

    Returns
    -------
    list of decision record dicts, in chronological order.
    """
    json_path = os.path.join(log_dir, JSON_LOG_FILE)
    if not os.path.exists(json_path):
        return []

    records = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_decision_by_id(decision_id: str, log_dir: str = DEFAULT_LOG_DIR) -> Optional[dict]:
    """Retrieve a specific decision record by its ID."""
    for record in load_decision_log(log_dir):
        if record.get("decision_id") == decision_id:
            return record
    return None


def print_audit_summary(log_dir: str = DEFAULT_LOG_DIR) -> None:
    """
    Print a summary of all logged decisions.
    Useful for monitoring dashboard and model performance review.
    """
    records = load_decision_log(log_dir)
    if not records:
        print("No decisions logged yet.")
        return

    from collections import Counter
    import statistics

    decisions    = [r["decision"] for r in records]
    composite_pds = [r["composite_pd"] for r in records if r.get("composite_pd")]
    rarocs       = [r["raroc"] for r in records if r.get("raroc")]

    decision_counts = Counter(decisions)
    total = len(records)

    print("=" * 60)
    print(f"DECISION LOG SUMMARY  ({total} decisions)")
    print("=" * 60)
    for d, n in sorted(decision_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:<20} {n:>5} ({n/total*100:.1f}%)")
    print("-" * 60)
    if composite_pds:
        print(f"  Mean Composite PD  : {statistics.mean(composite_pds):.4f}")
        print(f"  Median RAROC       : {statistics.median(rarocs):.4f}" if rarocs else "")
    print("=" * 60)
