"""
constants.py — AI Credit Intelligence System
==========================================
Single source of truth for every threshold, weight, and parameter used
across the Decision Engine. Change a value here and it propagates everywhere.

All values are documented with their regulatory or business basis so the
system remains fully auditable. This file is the starting point for any
recalibration exercise.
"""

# ── Decision Thresholds ───────────────────────────────────────────────────────

# Composite PD thresholds (applied after signal aggregation)
COMPOSITE_PD_APPROVE_MAX  = 0.10   # Auto-approve if composite PD below this
COMPOSITE_PD_REVIEW_MIN   = 0.20   # Route to Manual Review above this
COMPOSITE_PD_DECLINE_MIN  = 0.35   # Hard decline above this

# RAROC hurdle — minimum acceptable risk-adjusted return on capital
# Basis: indicative hurdle rate for Indian retail lending (RBI capital norms)
HURDLE_RATE = 0.14   # 14%

# ── Signal Aggregation Weights ────────────────────────────────────────────────
# Weighted composite PD = w_A * pd_A + w_B * pd_B
# Module C is NOT included in the composite PD — it modifies the decision path
# (Reprice) rather than the risk score itself.
#
# Weight rationale:
#   Module A (0.60): Application-time PD is the primary origination signal.
#                    Based on validated XGBoost model with AUC ~0.77.
#   Module B (0.40): Behavioural signal is complementary — strong when history
#                    exists, but many applicants may lack 2yr behavioural data.
#
# These weights are configurable and should be re-calibrated quarterly against
# observed default rates by decision segment (SR 11-7 ongoing monitoring).
SIGNAL_WEIGHTS = {
    "module_a": 0.60,
    "module_b": 0.40,
}

# ── Hard Override Rules ───────────────────────────────────────────────────────
# These flags trigger Manual Review regardless of composite PD or RAROC.
# They represent patterns where the model's point estimate may understate
# the true risk due to non-linearity or tail risk.
HARD_OVERRIDE_FLAGS = [
    "high_delinquency_score",   # Module B: delinquency_score >= 5 (repeated pattern)
    "critical_utilisation",     # Module B: revolving utilisation >= 80%
    "escalating_delinquency",   # Module B: has 60DPD or 90DPD events
]

# ── Module C Routing Rules ────────────────────────────────────────────────────
# Conditions that trigger REPRICE or MANUAL REVIEW based on pricing/portfolio signals
REPRICE_ADEQUACY_TRIGGER     = "Underpriced"   # Pricing adequacy from Module C
CONCENTRATION_PD_THRESHOLD   = 0.25            # market_pd above which concentration flag → Review
MARKET_PD_DIVERGENCE_THRESHOLD = 0.15          # |market_pd - composite_pd| triggers review note

# ── Capital Parameters ────────────────────────────────────────────────────────
# Mirrors 04_expected_loss_capital_model.ipynb exactly
LGD_UNSECURED  = 0.45    # Basel II IRB floor for unsecured retail
LGD_SECURED    = 0.25    # Collateralised loans
CAPITAL_RATIO  = 0.105   # RBI CET1 (4.5%) + conservation buffer (2.5%) + buffer
INTEREST_RATE  = 0.12    # Base lending rate (12% p.a.)
OPERATING_COST = 0.03    # Operating cost ratio (3% of EAD)

# Basel III risk weights by risk band
RISK_WEIGHTS = {1: 0.75, 2: 0.75, 3: 1.00, 4: 1.50, 5: 1.50}

# Risk-based pricing rates by band (Module C)
RBP_RATES = {1: 0.10, 2: 0.12, 3: 0.15, 4: 0.18, 5: 0.22}

# ── Risk Band Definitions ─────────────────────────────────────────────────────
# Maps PD to integer risk band 1-5 (consistent across all modules)
RISK_BAND_THRESHOLDS = {
    1: (0.00, 0.05),   # Very Low
    2: (0.05, 0.10),   # Low
    3: (0.10, 0.20),   # Medium
    4: (0.20, 0.35),   # High
    5: (0.35, 1.00),   # Very High
}
RISK_BAND_LABELS = {
    1: "Very Low",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Very High",
}

# ── Decision Labels ───────────────────────────────────────────────────────────
DECISION_APPROVE        = "APPROVE"
DECISION_DECLINE        = "DECLINE"
DECISION_REPRICE        = "REPRICE"
DECISION_MANUAL_REVIEW  = "MANUAL REVIEW"

# ── Module B Defaults (when Module B signal not available) ────────────────────
# Used when behavioural history is absent (e.g. thin-file applicant)
# Conservative assumption: treat as population average
MODULE_B_FALLBACK_DELINQUENCY_PROB = 0.067   # Dataset default rate
MODULE_B_FALLBACK_RISK_BAND        = 2        # Low risk (benefit of doubt)

# ── Monitoring Triggers (SR 11-7) ─────────────────────────────────────────────
PSI_AMBER_THRESHOLD = 0.10   # Monitor
PSI_RED_THRESHOLD   = 0.25   # Recalibrate
AUC_FLOOR           = 0.70   # Recalibrate if AUC drops below
KS_FLOOR            = 0.30   # Recalibrate if KS drops below
